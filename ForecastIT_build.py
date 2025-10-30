import math
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, numpy as np

class _AttnRecordMixin:
    def __init__(self):
        self._record_attn = False
        self._attn_records = []  

    def set_recording(self, flag: bool = True, clear: bool = True):
        self._record_attn = bool(flag)
        if clear:
            self._attn_records = []

    def _push_attn(self, attn_tensor):
        if not self._record_attn:
            return
        a = attn_tensor.detach().float()
        
        if a.dim() == 3:
            a = a.unsqueeze(1) 
        elif a.dim() == 5:
            pass  
        elif a.dim() != 4:
            return
            
        self._attn_records.append(a.cpu().numpy())

    def pop_all_records(self):
        out = self._attn_records
        self._attn_records = []
        return out




from typing import List, Optional

def make_sym_norm_P(A: torch.Tensor, add_self_loop: bool = True) -> torch.Tensor:
    if not torch.is_tensor(A): A = torch.tensor(A, dtype=torch.float32)
    A = A.float()
    A = 0.5 * (A + A.t())
    if add_self_loop:
        A = A + torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    d = A.sum(-1).clamp_min(1e-12)
    D_inv_sqrt = torch.diag(torch.pow(d, -0.5))
    return D_inv_sqrt @ A @ D_inv_sqrt

class UndirectedAptDiffusion(nn.Module):
    def __init__(self, in_dim, out_dim, K, P_undirected: torch.Tensor,
                 apt_channels=16, aapt_norm="sym", use_bias=True):
        super().__init__()
        self.K = int(K)
        self.N = P_undirected.size(0)
        self.register_buffer("P", P_undirected.float())
        self.lin_rw  = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=use_bias) for _ in range(self.K+1)])
        self.lin_apt = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=use_bias) for _ in range(self.K+1)])
        self.E1 = nn.Parameter(torch.randn(self.N, apt_channels) * 0.01)
        self.E2 = nn.Parameter(torch.randn(self.N, apt_channels) * 0.01)
        self.aapt_norm = aapt_norm

    @staticmethod
    def _row_norm(A: torch.Tensor, eps=1e-12):
        return A / (A.sum(-1, keepdim=True).clamp_min(eps))

    @staticmethod
    def _sym_norm(A: torch.Tensor, eps=1e-12):
        d = A.sum(-1).clamp_min(eps)
        D_inv_sqrt = torch.diag(torch.pow(d, -0.5))
        return D_inv_sqrt @ A @ D_inv_sqrt

    def _build_Aapt(self):
        A = F.relu(self.E1 @ self.E2.t())         
        A = 0.5 * (A + A.t())                      
        A = self._sym_norm(A) if self.aapt_norm == "sym" else self._row_norm(A)
        return A

    def forward(self, x):                          
        B,T,N,C = x.shape
        X = x.view(B*T, N, C)                      
        out_rw = 0
        Ik = torch.eye(N, device=x.device, dtype=x.dtype)
        Pk = Ik
        for k in range(self.K+1):
            Zk = (Pk @ X)                         
            out_rw = out_rw + self.lin_rw[k](Zk)   
            Pk = Pk @ self.P

        Aapt = self._build_Aapt()
        out_apt = 0
        Ak = Ik
        for k in range(self.K+1):
            Zk = (Ak @ X)
            out_apt = out_apt + self.lin_apt[k](Zk)
            Ak = Ak @ Aapt

        return (out_rw + out_apt).view(B, T, N, -1)


class MSPFPriorProjector(nn.Module):
    def __init__(self, d_model: int, proj_type: str = "linear", hidden_ratio: int = 2):
        super().__init__()
        if proj_type == "linear":
            self.proj = nn.Linear(d_model, d_model)
        elif proj_type == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(d_model, d_model*hidden_ratio), nn.GELU(), nn.Linear(d_model*hidden_ratio, d_model)
            )
        else:
            raise ValueError(proj_type)
    def forward(self, x): return self.proj(x)

class TemporalRefineBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ff_ratio: int, layers: int, dropout: float, causal: bool):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model, heads, d_model*ff_ratio, dropout,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.causal = causal
    def forward(self, x):  # x:[B,T,Nf,d]
        B,T,Nf,D = x.shape
        xx = x.permute(0,2,1,3).contiguous().view(B*Nf,T,D)
        mask = None
        if self.causal:
            mask = torch.triu(torch.ones(T,T, device=x.device), diagonal=1).bool()
        y = self.encoder(xx, mask=mask)
        return y.view(B,Nf,T,D).permute(0,2,1,3).contiguous()

class SpatialRefineBlock(nn.Module):
    def __init__(self, d_model: int, heads: int = 2, dropout: float = 0.1, use_adj_bias: bool = True):
        super().__init__()

        self.q = nn.Linear(d_model, d_model); self.k = nn.Linear(d_model, d_model); self.v = nn.Linear(d_model, d_model)
        self.h = heads; self.dk = d_model//heads; self.o = nn.Linear(d_model,d_model)
        self.drop = nn.Dropout(dropout); self.use_adj_bias = use_adj_bias
    def forward(self, x, adj_sub: Optional[torch.Tensor]=None):  # x:[B,T,Nf,d]
        B,T,Nf,D = x.shape
        q,k,v = self.q(x), self.k(x), self.v(x)
        def split(t): return t.view(B,T,Nf,self.h,self.dk).permute(0,1,3,2,4)
        qh,kh,vh = split(q), split(k), split(v)
        scores = torch.einsum("bthid,bthjd->bthij", qh, kh) / (self.dk**0.5)
        if self.use_adj_bias and adj_sub is not None:
            scores = scores.masked_fill((adj_sub<=0).unsqueeze(0).unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = self.drop(torch.softmax(scores, dim=-1))
        out = torch.einsum("bthij,bthjd->bthid", attn, vh).permute(0,1,3,2,4).contiguous().view(B,T,Nf,D)
        return self.o(out)





class FineGrainBlock(nn.Module):

    def __init__(self, d_model: int, cfg: dict):
        super().__init__()
        pj, tr, sr, gt = cfg.get("projector",{}), cfg.get("temporal_refine",{}), cfg.get("spatial_refine",{}), cfg.get("gate",{})
        self.prior_value_proj = nn.Linear(1, d_model)  
        self.prior_proj = MSPFPriorProjector(d_model, pj.get("type","linear"), pj.get("hidden_ratio",2))
        fcfg = cfg.get("fusion", {})
        self.fusion_type = str(fcfg.get("type", "concat")).lower()
        if self.fusion_type == "concat":
            self.fuse_in = nn.Linear(d_model*2, d_model)
            self.fuse_gate = None
        elif self.fusion_type == "gate":
            hr = int(fcfg.get("hidden_ratio", 2))
            self.fuse_in = None
            self.fuse_gate = nn.Sequential(
                nn.Linear(d_model*2, d_model*hr), nn.GELU(),
                nn.Linear(d_model*hr, d_model), nn.Sigmoid()
            )
        else:
            raise ValueError(f"fusion.type must be concat|gate, got {self.fusion_type}")


        self.temporal_type = tr.get("type", "self")  # "self" | "cross_parent"
        if self.temporal_type == "cross_parent":
            self.temporal_cross = TemporalCrossParentAttn(
                d_model=d_model,
                n_heads=tr.get("attn_heads", 4),
                dropout=tr.get("dropout", 0.1)
            )
            self.temporal = None
        else:
            self.temporal = TemporalRefineBlock(d_model, tr.get("attn_heads",4), tr.get("ff_ratio",2),
                                                tr.get("layers",2), tr.get("dropout",0.1), tr.get("causal",False))
            self.temporal_cross = None
        self.spatial  = SpatialRefineBlock(d_model, sr.get("heads",2), sr.get("dropout",0.1), sr.get("use_adj_bias",True))
        self.to_delta = nn.Linear(d_model, 1)
        self.gate_mlp = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(),
                                      nn.Linear(d_model//2,1),
                                      nn.Sigmoid() if gt.get("method","sigmoid")=="sigmoid" else nn.Tanh())
        self.use_delta_res = gt.get("use_delta_residual", True)

        cfg_sp = cfg.get("spatial_refine", {})
        self.spatial_type = cfg_sp.get("type", "attn")      
        self.kv_source   = cfg_sp.get("kv_source", "hidden")
        if self.spatial_type == "cross_parent":
            self.cross_parent = CrossParentAttn(
                d_model=d_model, 
                n_heads=int(cfg_sp.get("heads", 2)), 
                dropout=float(cfg_sp.get("dropout", 0.1))
            )
        else:
            self.cross_parent = None


        tr = cfg.get("temporal_refine", {})
        self.temporal_type = tr.get("type", "self")  # "self" | "cross_parent"

        if self.temporal_type == "cross_parent":
            self.temporal_cross = TemporalCrossParentAttn(
                d_model=d_model,
                n_heads=tr.get("attn_heads", 4),
                dropout=tr.get("dropout", 0.1)
            )
        else:
            self.temporal_refine = TemporalRefineBlock(
                d_model=d_model,
                heads=tr.get("attn_heads", 4),
                ff_ratio=tr.get("ff_ratio", 2),
                layers=tr.get("layers", 2),
                dropout=tr.get("dropout", 0.1),
                causal=tr.get("causal", False)
            )

    @staticmethod
    def _row_norm(M: torch.Tensor, eps=1e-8): return M/(M.sum(1, keepdim=True)+eps)

    def _prior_from_hidden(self, h_all, fine_idx, use_multi, parent_1to1, coarse_pool, M):
        if use_multi:
            parent_pool = h_all[:,:,coarse_pool,:]      # [B,T,Nc,d]
            M = self._row_norm(M)                       # [Nf,Nc]
            return torch.einsum("btjd,fj->btfd", parent_pool, M)
        else:
            return h_all[:,:,parent_1to1,:]             # [B,T,Nf,d]

    def forward(
        self,
        h_all,                 
        y_base,               
        fg_idx,                
        prior_external=None,   
    ):

        dev = h_all.device
        if not torch.is_tensor(fg_idx):
            fg_idx = torch.tensor(fg_idx, dtype=torch.long, device=dev)

        x_fg = h_all.index_select(dim=2, index=fg_idx)    # [B, T, Nf, C]

        if prior_external is None:
            prior_external = getattr(self, "_y_prior_external", None)
            
        if prior_external is not None:
            p_raw = prior_external  # [B, Tout, Nf, 1]
            T_in = x_fg.size(1)    
            T_out = p_raw.size(1)  
            
            if T_in != T_out:
                if T_out == 1:
                    p_raw = p_raw.repeat(1, T_in, 1, 1)
                else:
                    p_raw = p_raw[:, :T_in, :, :]  
            
            p_feat = self.prior_value_proj(p_raw)           # [B, T, Nf, C]
            
            hp = torch.cat([x_fg, p_feat], dim=-1)          # [B, T, Nf, 2*C]
            if self.fusion_type == "concat":
                x_fg = self.fuse_in(hp)
            else:  # gate
                g_in = self.fuse_gate(hp)
                x_fg = g_in * x_fg + (1.0 - g_in) * p_feat   

        if getattr(self, "fg_coarse_pool", None) is not None and self.fg_coarse_pool.numel() > 0:
            pool_idx = self.fg_coarse_pool
        else:
            if getattr(self, "fg_parent_1to1", None) is None or self.fg_parent_1to1.numel() == 0:
                raise RuntimeError("FineGrainBlock: neither coarse_pool nor parent_1to1 is set.")
            pool_idx = torch.unique(self.fg_parent_1to1)
        
        if hasattr(self, "temporal_type") and self.temporal_type == "cross_parent" and hasattr(self, "temporal_cross"):
            x_parent = h_all.index_select(dim=2, index=pool_idx)
            x_fg = self.temporal_cross(x_fg, x_parent)
        elif hasattr(self, "temporal_refine") and self.temporal_refine is not None:
            x_fg = self.temporal_refine(x_fg)
        elif hasattr(self, "temporal") and self.temporal is not None:
            x_fg = self.temporal(x_fg)

        if self.spatial_type == "cross_parent" and self.cross_parent is not None:
            if self.kv_source == "hidden":
                ksrc = h_all.index_select(dim=2, index=pool_idx)
                x_fg = self.cross_parent(x_fg, ksrc)
        elif getattr(self, "spatial_type", "none") == "attn":
            if hasattr(self, "spatial_refine"):
                x_fg = self.spatial_refine(x_fg)

        z_fg = x_fg
        if hasattr(self, "decode_fg") and self.decode_fg is not None:
            z_fg = self.decode_fg(z_fg)                             # [B, Tout, Nf, C]


        if hasattr(self, "out_head"):
            y_fg = self.out_head(z_fg)                              # [B, Tout, Nf, 1]
        else:
            if not hasattr(self, "_temp_out_proj"):
                self._temp_out_proj = nn.Linear(z_fg.size(-1), 1).to(z_fg.device)
            y_fg = self._temp_out_proj(z_fg)

        if getattr(self, "use_delta", True):
            y_fg = y_fg + y_base.index_select(dim=2, index=fg_idx)

        y_out = y_base.clone()
        y_out.index_copy_(dim=2, index=fg_idx, source=y_fg)

        self.last_consistency_loss = None
        lam = float(getattr(self, "lambda_consistency", 0.0))
        if lam > 0 and prior_external is not None:
            y_fine_mean   = y_fg.mean(dim=1)                        
            y_parent_mean = prior_external.mean(dim=1)             
            self.last_consistency_loss = F.mse_loss(y_fine_mean, y_parent_mean)

        return y_out

# -----------------------------
# Utilities
# -----------------------------
def _to_long_tensor(x):
    if isinstance(x, list):
        return torch.tensor(x, dtype=torch.long)
    if isinstance(x, torch.Tensor):
        return x.long()
    raise TypeError("cluster ids must be list[int] or torch.Tensor")


# -----------------------------
# Diffusion GCN (optional)
# -----------------------------
class DiffusionGCN(nn.Module):

    def __init__(self, a_in: int, a_out: int, K: int, A: torch.Tensor):
        super().__init__()
        self.K = K
        self.register_buffer("A", A)      # [N,N]
        self.lin_phy = nn.ModuleList([nn.Linear(a_in, a_out) for _ in range(K)])

    def forward(self, x):                  # x: [B,T,N,C]
        B, T, N, C = x.shape
        row_sum = self.A.sum(-1, keepdim=True).clamp_min(1e-12)
        P = self.A / row_sum               # [N,N]
        X = x.reshape(B * T, N, C)         # [BT,N,C]

        outs = []
        Pk = torch.eye(N, device=x.device, dtype=x.dtype)  # P^0
        for k in range(self.K):
            if k == 0:
                Z = X                      
            else:
                Pk = torch.matmul(Pk, P)   
                Z = torch.matmul(Pk, X)    # [BT,N,F] = (P^k @ X)
            Z = Z.view(B, T, N, C)         
            outs.append(self.lin_phy[k](Z))
        return torch.stack(outs, dim=0).sum(0)  # [B,T,N,a_out]



class TemporalSelfAttentionClusterKey(nn.Module, _AttnRecordMixin):

    def __init__(self, d_model: int, n_heads: int, num_clusters: int,
                 cluster_ids: torch.Tensor, attn_dropout: float = 0.0):
        super().__init__()
        _AttnRecordMixin.__init__(self)   
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.num_clusters = num_clusters

        cid = _to_long_tensor(cluster_ids)
        self.register_buffer("cluster_ids", cid)  


        self.Wq = nn.Linear(d_model, d_model, bias=True)
        self.Wv = nn.Linear(d_model, d_model, bias=True)


        self.Wk = nn.Parameter(torch.Tensor(num_clusters, d_model, d_model))
        self.bk = nn.Parameter(torch.Tensor(num_clusters, d_model))

        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Wq.weight); nn.init.zeros_(self.Wq.bias)
        nn.init.xavier_uniform_(self.Wv.weight); nn.init.zeros_(self.Wv.bias)
        nn.init.xavier_uniform_(self.proj.weight); nn.init.zeros_(self.proj.bias)

        for c in range(self.num_clusters):
            nn.init.xavier_uniform_(self.Wk[c])
            nn.init.zeros_(self.bk[c])

    def _cluster_linear_K(self, x):

        B,T,N,C = x.shape
        K = x.new_zeros(B,T,N,C)
        for c in range(self.num_clusters):
            mask = (self.cluster_ids == c)          # [N]
            if not mask.any(): 
                continue
            xc = x[:,:,mask,:]                      # [B,T,Nc,D]
            yc = torch.matmul(xc, self.Wk[c]) + self.bk[c]
            K[:,:,mask,:] = yc
        return K                                     # [B,T,N,D]

    def forward(self, x):  # x: [B,T,N,D]
        B,T,N,D = x.shape
        Q = self.Wq(x)                               # [B,T,N,D]
        V = self.Wv(x)                               # [B,T,N,D]
        K = self._cluster_linear_K(x)               

        def split_heads(z):
            z = z.view(B, T, N, self.n_heads, self.d_head)
            return z.permute(0, 2, 3, 1, 4)         # [B,N,H,T,dh]
        Qh, Kh, Vh = map(split_heads, (Q, K, V))

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = torch.softmax(scores, dim=-1)
        self._push_attn(attn)           
        attn = self.attn_dropout(attn)
        self._push_attn(attn)            
        out_h = torch.matmul(attn, Vh)
        out = out_h.permute(0,3,1,2,4).contiguous().view(B, T, N, D)
        return self.proj(out)


class SpatialSelfAttention(nn.Module, _AttnRecordMixin):
    def __init__(self, d_model: int, n_heads: int, attn_dropout: float = 0.0):
        super().__init__()
        _AttnRecordMixin.__init__(self)
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(attn_dropout)

    def forward(self, x):  # x: [B,T,N,C]
        B, T, N, C = x.shape
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        def split_heads(z):
            # [B,T,N,C] -> [B,T,H,N,dh]
            z = z.view(B, T, N, self.n_heads, self.d_head).permute(0, 1, 3, 2, 4).contiguous()
            return z

        Qh, Kh, Vh = (split_heads(Q), split_heads(K), split_heads(V))  # [B,T,H,N,dh]
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B,T,H,N,N]
        attn = torch.softmax(scores, dim=-1)
        self._push_attn(attn)             # [B,T,H,N,N]
        attn = self.drop(attn)
        self._push_attn(attn)             # [B,T,H,N,N]
        out_h = torch.matmul(attn, Vh)  # [B,T,H,N,dh]

        out = out_h.permute(0, 1, 3, 2, 4).contiguous().view(B, T, N, C)
        return self.proj(out)

class ClusterGatedSpatialSelfAttention(nn.Module, _AttnRecordMixin):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        cluster_ids,                 
        attn_dropout: float = 0.0,
        allow_self_attn: bool = False,   
        gate_bias_init: float = 0.0,     
    ):
        super().__init__()
        _AttnRecordMixin.__init__(self)
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(attn_dropout)
        self.gate_fc = nn.Linear(d_model * 2, d_model)
        nn.init.constant_(self.gate_fc.bias, gate_bias_init)

        cid = torch.as_tensor(cluster_ids, dtype=torch.long).view(-1)   # [N]
        same = (cid[None, :] == cid[:, None]).float()                   # [N,N]
        if not allow_self_attn: same.fill_diagonal_(0.0)
        diff = 1.0 - same
        if not allow_self_attn: diff.fill_diagonal_(0.0)
        self.register_buffer("M_intra", same, persistent=False)         # [N,N]
        self.register_buffer("M_inter", diff, persistent=False)         # [N,N]

    def forward(self, x):  # x: [B,T,N,D]
        B, T, N, D = x.shape

        Q = self.Wq(x); K = self.Wk(x); V = self.Wv(x)

        def split_heads(z):
            # [B,T,N,D] -> [B,T,H,N,dh]
            return z.view(B, T, N, self.n_heads, self.d_head).permute(0,1,3,2,4).contiguous()
        Qh, Kh, Vh = (split_heads(Q), split_heads(K), split_heads(V))   # [B,T,H,N,dh]

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(self.d_head)
        Mi = self.M_intra[None, None, :, :]    # [1,1,N,N]
        Mo = self.M_inter[None, None, :, :]    # [1,1,N,N]
        Mi = Mi.expand(B, T, self.n_heads, N, N).reshape(B*T, self.n_heads, N, N)
        Mo = Mo.expand(B, T, self.n_heads, N, N).reshape(B*T, self.n_heads, N, N)
        S  = scores.reshape(B*T, self.n_heads, N, N)

        def masked_softmax(sc, mask):
            sc = sc.masked_fill(mask==0, float('-inf'))
            row_all_inf = torch.isinf(sc).all(dim=-1, keepdim=True)
            if row_all_inf.any():
                eye = torch.eye(sc.size(-1), device=sc.device).unsqueeze(0).unsqueeze(0)
                sc = torch.where(row_all_inf, eye*0.0, sc)
            attn = torch.softmax(sc, dim=-1)
            return attn

        Aintra = masked_softmax(S, Mi)    # [BT,H,N,N]
        Ainter = masked_softmax(S, Mo)
        Aintra_bth = Aintra.view(B, T, self.n_heads, N, N)
        Ainter_bth = Ainter.view(B, T, self.n_heads, N, N)
        self._push_attn(Aintra_bth)
        self._push_attn(Ainter_bth)

        Vh_bt  = Vh.reshape(B*T, self.n_heads, N, self.d_head)
        Zintra = torch.matmul(Aintra, Vh_bt)  # [BT,H,N,dh]
        Zinter = torch.matmul(Ainter, Vh_bt)

        def merge_heads(z):
            return z.view(B, T, self.n_heads, N, self.d_head).permute(0,1,3,2,4).contiguous().view(B,T,N,D)
        Zintra = merge_heads(Zintra); Zinter = merge_heads(Zinter)

        Zcat = torch.cat([Zintra, Zinter], dim=-1)  # [B,T,N,2D]
        G = torch.sigmoid(self.gate_fc(Zcat))       # [B,T,N,D]
        Z = G * Zintra + (1.0 - G) * Zinter         # [B,T,N,D]
        out = self.proj(Z)
        return out




class CrossParentAttn(nn.Module, _AttnRecordMixin):
    def __init__(self, d_model: int, n_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        _AttnRecordMixin.__init__(self)
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.h = n_heads
        self.dh = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(d_model, d_model)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        B, T, Nf, C = q_in.shape
        Np = kv_in.size(2)
        q = self.Wq(q_in).view(B, T, Nf, self.h, self.dh).transpose(2, 3)  # [B,T,H,Nf,Dh]
        k = self.Wk(kv_in).view(B, T, Np, self.h, self.dh).transpose(2, 3)  # [B,T,H,Np,Dh]
        v = self.Wv(kv_in).view(B, T, Np, self.h, self.dh).transpose(2, 3)  # [B,T,H,Np,Dh]
        scores = torch.einsum("bthqd,bthkd->bthqk", q, k) / (self.dh ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        self._push_attn(attn)  # [B,T,H,Nf,Np]
        attn = self.dropout(attn)
        out_h = torch.einsum("bthqk,bthkd->bthqd", attn, v)                 # [B,T,H,Nf,Dh]
        out = out_h.transpose(2, 3).contiguous().view(B, T, Nf, C)          # [B,T,Nf,C]
        return self.proj_out(out)


class TemporalCrossParentAttn(nn.Module, _AttnRecordMixin):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        _AttnRecordMixin.__init__(self)
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.h = n_heads
        self.dh = d_model // n_heads
        
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(d_model, d_model)
        
    def forward(self, q_fine: torch.Tensor, kv_parent: torch.Tensor) -> torch.Tensor:
        B, T, Nf, C = q_fine.shape
        Np = kv_parent.size(2)
        
        Q = self.Wq(q_fine)     # [B, T, Nf, C]
        K = self.Wk(kv_parent)  # [B, T, Np, C]  
        V = self.Wv(kv_parent)  # [B, T, Np, C]
        
        def reshape_for_scores(x, N):
            return x.view(B, T, N, self.h, self.dh).permute(0, 3, 1, 2, 4)
        
        Q = reshape_for_scores(Q, Nf)  # [B, H, T, Nf, dh]
        K = reshape_for_scores(K, Np)  # [B, H, T, Np, dh]
        V = reshape_for_scores(V, Np)  # [B, H, T, Np, dh]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dh)
        attn_weights = torch.softmax(scores, dim=-1)
        self._push_attn(attn_weights)  # [B,H,T,Nf,Np]
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.permute(0, 2, 3, 1, 4).contiguous()
        context = context.view(B, T, Nf, C)

        output = self.proj_out(context)
        return output




# -----------------------------
# Encoder Block
# -----------------------------
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, num_clusters, cluster_ids,
                 attn_dropout=0.0, ffn_ratio=4.0, dropout=0.1,
                 use_cluster_spa: bool = True,   
                 gate_bias_init: float = 0.0):   

        super().__init__()
        self.temporal = TemporalSelfAttentionClusterKey(
            d_model=d_model, n_heads=n_heads,
            num_clusters=num_clusters, cluster_ids=cluster_ids,
            attn_dropout=attn_dropout
        )
        self.spatial = ClusterGatedSpatialSelfAttention(
            d_model=d_model, n_heads=n_heads, cluster_ids=cluster_ids,
            attn_dropout=attn_dropout, gate_bias_init=gate_bias_init
        ) if use_cluster_spa else SpatialSelfAttention(d_model=d_model, n_heads=n_heads, attn_dropout=attn_dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, int(ffn_ratio*d_model)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(ffn_ratio*d_model), d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):  # [B,T,N,D]
        # Temporal
        xt = self.temporal(x)
        x = x + self.drop(xt)
        x = self.norm1(x)
        # Spatial
        xs = self.spatial(x)
        x = x + self.drop(xs)
        x = self.norm2(x)
        # FFN
        xf = self.ffn(x)
        x = x + self.drop(xf)
        return x


class ForecastIT(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        pred_len: int = 12,
        steps_per_day: int = 288,
        tod_emb_dim: int = 8,
        dow_emb_dim: int = 4,
        use_gcn: bool = True,
        gcn_order: int = 2,
        gcn_out: int = 32,
        A: Optional[torch.Tensor] = None,
        num_clusters: int = 4,
        cluster_ids: Optional[List[int]] = None,
        dropout: float = 0.1,
        use_cluster_spa: bool = True,             
        cluster_spa_gate_bias: float = 0.0,       
    ):
        super().__init__()
        self.N = num_nodes
        self.pred_len = pred_len
        self.steps_per_day = steps_per_day
        self.tod_emb_dim = tod_emb_dim
        self.dow_emb_dim = dow_emb_dim
        self.input_dim = input_dim

        self.flow_proj = nn.Linear(1, d_model // 2)

        if tod_emb_dim > 0:
            self.tod_emb = nn.Embedding(steps_per_day, tod_emb_dim)
        else:
            self.tod_emb = None
        if dow_emb_dim > 0:
            self.dow_emb = nn.Embedding(7, dow_emb_dim)
        else:
            self.dow_emb = None

        self.use_gcn = use_gcn
        self.udiff = None
        if use_gcn:
            assert A is not None, "use_gcn=True 需要传入邻接矩阵 A [N,N]"
            aapt_cfg = getattr(self, "aapt_cfg", None) 
            undirected = bool(getattr(self, "use_undirected_diffusion", False))

            if undirected:
                P = make_sym_norm_P(A)  
                self.udiff = UndirectedAptDiffusion(
                    in_dim=1, out_dim=gcn_out, K=gcn_order,
                    P_undirected=P, apt_channels=(aapt_cfg or {}).get("channels", 16),
                    aapt_norm=(aapt_cfg or {}).get("norm", "sym")
                )
                gcn_dim = gcn_out
            else:
                self.gcn = DiffusionGCN(a_in=1, a_out=gcn_out, K=gcn_order, A=A)
                gcn_dim = gcn_out
        else:
            gcn_dim = 0

        fuse_in = (d_model // 2) + (tod_emb_dim + dow_emb_dim) + gcn_dim
        self.fuse_proj = nn.Linear(fuse_in, d_model)
        cid = torch.tensor(cluster_ids if cluster_ids is not None else [0]*num_nodes, dtype=torch.long)
        self.encoders = nn.ModuleList([
            EncoderBlock(d_model=d_model, n_heads=n_heads,
                        num_clusters=num_clusters, cluster_ids=cid,
                        attn_dropout=dropout, ffn_ratio=4.0, dropout=dropout,
                        use_cluster_spa=use_cluster_spa,
                        gate_bias_init=cluster_spa_gate_bias)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model * 1, pred_len) 
        self.out_proj = nn.Linear(1, 1)

        self.fg_enable = False; self.fg_cfg = {}; self.lambda_consistency = 0.0
        self.register_buffer("fg_idx", torch.empty(0, dtype=torch.long))
        self.register_buffer("fg_parent_1to1", torch.empty(0, dtype=torch.long))
        self.register_buffer("fg_coarse_pool", torch.empty(0, dtype=torch.long))
        self.register_buffer("fg_M", torch.empty(0))
        self.register_buffer("adj_mx", torch.empty(0))    
        self.fg_block: Optional[FineGrainBlock] = None


    def set_adj(self, A: torch.Tensor):
        if not torch.is_tensor(A): A = torch.tensor(A, dtype=torch.float32)
        self.register_buffer("adj_mx", A.float())

    def enable_fine_grain_full(
        self,
        fine_indices: List[int],
        fg_cfg: dict,
        parent_for_fine: Optional[List[int]] = None,
        coarse_pool: Optional[List[int]] = None,
        M_weights: Optional[List[List[float]]] = None,
    ):
        dev = next(self.parameters()).device

        if not fg_cfg.get("enable", False):
            print("[fine-grain] skipped (enable=false)")
            self.fg_enable = False
            self.fg_block = None
            self.fg_idx = torch.empty(0, dtype=torch.long, device=dev)
            self.fg_coarse_pool = torch.empty(0, dtype=torch.long, device=dev)
            self.fg_parent_1to1 = torch.empty(0, dtype=torch.long, device=dev)
            self.fg_M = torch.empty(0, device=dev)
            self.lambda_consistency = 0.0
            return
        self.fg_block = FineGrainBlock(
            d_model=self.fuse_proj.out_features,  
            cfg=fg_cfg
        ).to(dev)  

        self.fg_enable = True
        self.fg_cfg = fg_cfg
        self.lambda_consistency = float(fg_cfg.get("lambda_consistency", 0.0))
        self.fg_idx = torch.tensor(list(map(int, fine_indices)), dtype=torch.long, device=dev)
        
        use_multi = bool(fg_cfg.get("parents", {}).get("use_multi", False))
        if use_multi:
            assert coarse_pool is not None and M_weights is not None, \
                "use_multi=True 时需要提供 coarse_pool 与 M_weights"
            self.fg_coarse_pool = torch.tensor(list(map(int, coarse_pool)), dtype=torch.long, device=dev)
            M = torch.tensor(M_weights, dtype=torch.float32, device=dev)
            assert M.dim() == 2 and M.size(0) == self.fg_idx.numel() and M.size(1) == self.fg_coarse_pool.numel(), \
                f"M shape={tuple(M.shape)} 与 fine={self.fg_idx.numel()}、pool={self.fg_coarse_pool.numel()} 不匹配"

            M = M / (M.sum(dim=1, keepdim=True) + 1e-8)
            self.fg_M = M
            self.fg_parent_1to1 = torch.empty(0, dtype=torch.long, device=dev)
            
            self.fg_block.fg_coarse_pool = self.fg_coarse_pool
            self.fg_block.fg_parent_1to1 = torch.empty(0, dtype=torch.long, device=dev)
        else:
            assert parent_for_fine is not None, "use_multi=False 时需要提供 parent_for_fine（与 fine_indices 对应的一对一父节点）"
            p = torch.tensor(list(map(int, parent_for_fine)), dtype=torch.long, device=dev)
            assert p.numel() == self.fg_idx.numel(), \
                f"parent_for_fine 数量 {p.numel()} 必须与 fine_indices 数量 {self.fg_idx.numel()} 一致"
            self.fg_parent_1to1 = p
            self.fg_coarse_pool = torch.empty(0, dtype=torch.long, device=dev)
            self.fg_M = torch.empty(0, device=dev)
            
            self.fg_block.fg_coarse_pool = torch.empty(0, dtype=torch.long, device=dev)
            self.fg_block.fg_parent_1to1 = self.fg_parent_1to1

        print(f"[fine-grain] enabled on {dev} | fine={self.fg_idx.tolist()} "
            f"| mode={'M' if use_multi else '1to1'}")


        
    def _discretize_tod_dow(self, x, idx):
        tod_idx = None
        if self.tod_emb is not None:
            tod_cont = x[..., idx]
            tod_idx = (tod_cont * self.steps_per_day).floor().clamp(0, self.steps_per_day - 1).long()
            idx += 1
        dow_idx = None
        if self.dow_emb is not None:
            dow_cont = x[..., idx]
            dow_idx = (dow_cont * 7).floor().clamp(0, 6).long()
            idx += 1
        return tod_idx, dow_idx, idx

    def forward(self, x):
        B,T,N,C = x.shape
        assert N == self.N
        flow = x[..., 0:1]                             # [B,T,N,1]
        flow_feat = self.flow_proj(flow)               # [B,T,N,d_model/2]
        idx = 1
        tod_idx, dow_idx, idx = self._discretize_tod_dow(x, idx)
        time_feats = []
        if self.tod_emb is not None:
            time_feats.append(self.tod_emb(tod_idx))   # [B,T,N,tod_dim]
        if self.dow_emb is not None:
            time_feats.append(self.dow_emb(dow_idx))   # [B,T,N,dow_dim]
        time_feat = torch.cat(time_feats, dim=-1) if len(time_feats)>0 else x.new_zeros(B,T,N,0)
        if self.use_gcn and (self.udiff is not None):
            gcn_feat = self.udiff(flow)              
        elif self.use_gcn:
            gcn_feat = self.gcn(flow)              
        else:
            gcn_feat = x.new_zeros(B,T,N,0)
        h = torch.cat([flow_feat, time_feat, gcn_feat], dim=-1)   # [B,T,N,fuse_in]
        h = self.fuse_proj(h)                                     # [B,T,N,d_model]
        for enc in self.encoders:
            h = enc(h)                                           # [B,T,N,d_model]
        self._h_all = h                                         # [B,T,N,d_model]

        z = h.mean(dim=1, keepdim=True)                 # [B, 1, N, d_model]
        y_base = self.head(z).permute(0, 3, 2, 1)            # [B, pred_len, N, 1]

        if self.fg_enable and (self.fg_block is not None) and (self.fg_idx.numel()>0):
            y_prior_external = getattr(self, "_y_prior_external", None)
            if y_prior_external is not None:
                y_prior_external = y_prior_external.to(self._h_all.device)  # ★
            adj_sub = None
            if self.adj_mx.numel()>0 and self.fg_cfg.get("spatial_refine",{}).get("use_adj_bias", True):
                adj_sub = self.adj_mx[self.fg_idx][:, self.fg_idx]
            y_out = self.fg_block(self._h_all, y_base, self.fg_idx,
                                  prior_external=y_prior_external)
            self.last_consistency_loss = None
            lam = self.lambda_consistency
            if lam>0:
                y_fine_mean = y_out[:,:,self.fg_idx,:].mean(dim=1)
                if self.fg_coarse_pool.numel()>0:
                    M = self.fg_M / (self.fg_M.sum(dim=1, keepdim=True)+1e-8)
                    y_parent_pool = y_base[:,:,self.fg_coarse_pool,:]
                    y_parent_map = torch.einsum("btjd,fj->btfd", y_parent_pool, M)
                else:
                    y_parent_map = y_base[:,:,self.fg_parent_1to1,:]
                y_parent_mean = y_parent_map.mean(dim=1)
                self.last_consistency_loss = F.mse_loss(y_fine_mean, y_parent_mean)
            return y_out

        return y_base
        
    def enable_attn_recording(self, flag: bool = True, clear: bool = True):
        for _, m in self.named_modules():
            if hasattr(m, "set_recording"):
                m.set_recording(flag, clear=clear)

    @torch.no_grad()
    def dump_attn_records(self, out_path: str):
        store = {}
        for name, m in self.named_modules():
            if hasattr(m, "pop_all_records"):
                arrs = m.pop_all_records()
                if len(arrs) == 0:
                    continue
                try:
                    stacked = np.concatenate(arrs, axis=0)  
                except Exception:
                    stacked = np.array(arrs, dtype=object)
                store[name] = stacked
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, **store)
        print(f"[OK] saved attention to {out_path}")
        return out_path


if __name__ == "__main__":
    B,T,N,C,K,a_out = 64,12,33,1,2,32
    x = torch.randn(B,T,N,C).cuda()
    A = torch.rand(N,N).cuda()
    gcn = DiffusionGCN(a_in=C, a_out=a_out, K=K, A=A).cuda()
    y = gcn(x)
    print(y.shape) 


if __name__ == "__main__":
    B,T,N,C,K,a_out = 64,12,33,1,2,32
    x = torch.randn(B,T,N,C).cuda()
    A = torch.rand(N,N).cuda()
    gcn = DiffusionGCN(a_in=C, a_out=a_out, K=K, A=A).cuda()
    y = gcn(x)
    print(y.shape)  

