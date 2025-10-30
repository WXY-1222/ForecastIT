# -*- coding: utf-8 -*-
import os, sys, argparse, json, time, math, random, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib.pyplot as plt
import seaborn as sns 


import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel.enable_flash(False)
    sdp_kernel.enable_mem_efficient(True)
    sdp_kernel.enable_math(False)
except Exception:
    pass
import yaml


from data_utils import SeqDataset, load_indices, StandardScaler
from metrics import RMSE as RMSE, MAE as MAE, WAPE as WAPE, SMAPE as SMAPE

def load_clusters(assign_path: str):
    import os
    if not os.path.exists(assign_path):
        raise FileNotFoundError(f"[load_clusters] file not found: {assign_path}")
    with open(assign_path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    tokens = []
    for seg in txt.replace(",", " ").split():
        try:
            tokens.append(int(seg))
        except ValueError:
            pass
    if len(tokens) == 0:
        raise ValueError(f"[load_clusters] no integers found in {assign_path}")
    return tokens 

# -------------------- Utils --------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def to_dev(items, dev):
    if isinstance(items, (list, tuple)):
        return [x.to(dev) for x in items]
    return items.to(dev)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True); return path

class EarlyStopper:
    def __init__(self, patience=10, mode="min", min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.bad = 0

    def step(self, value):
        if self.best is None:
            self.best = value; return False
        improve = (value < self.best - self.min_delta) if self.mode == "min" else (value > self.best + self.min_delta)
        if improve:
            self.best = value; self.bad = 0; return False
        self.bad += 1
        return self.bad > self.patience

def ddp_is_on(args): return args.ddp and dist.is_available()

def ddp_setup(args):
    if not args.ddp: return 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)
    return local_rank, torch.device(f"cuda:{local_rank}")

def ddp_cleanup():
    if dist.is_initialized(): dist.destroy_process_group()

# -------------------- Core --------------------
def build_dataloaders(cfg, scaler, device, ddp=False):
    root = cfg["data"]["root"]
    data_npz = np.load(os.path.join(root, cfg["data"]["data_file"]))                        
    idx_npz  = np.load(os.path.join(root, cfg["data"]["index_file"]))
    idx_dict = load_indices(idx_npz)  # {'train','val','test'}

    Tin  = int(idx_dict["train"][0][1] - idx_dict["train"][0][0])
    Tout = int(idx_dict["train"][0][2] - idx_dict["train"][0][1])

    train_set = SeqDataset(data_npz, idx_dict["train"], scaler, (Tin, Tout))
    val_set   = SeqDataset(data_npz, idx_dict["val"], scaler, (Tin, Tout))
    test_set  = SeqDataset(data_npz, idx_dict["test"], scaler, (Tin, Tout))

    num_workers = cfg["train"].get("num_workers", 0)
    batch_size  = cfg["train"]["batch_size"]

    if ddp:
        train_sampler = DistributedSampler(train_set, shuffle=True)
        val_sampler   = DistributedSampler(val_set,   shuffle=False)
        test_sampler  = DistributedSampler(test_set,  shuffle=False)
    else:
        train_sampler = val_sampler = test_sampler = None

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=num_workers, drop_last=True, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              sampler=val_sampler,   num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              sampler=test_sampler,  num_workers=num_workers, pin_memory=True)
    return data_npz, idx_dict, (train_loader, val_loader, test_loader)


def load_adj(path, device):
    import pickle, numpy as np, torch
    with open(path, "rb") as f:
        sensor_ids, id2ind, A = pickle.load(f)
    A = A.astype(np.float32)                       
    return torch.tensor(A, dtype=torch.float32, device=device), sensor_ids



def _has_str(x):
    if isinstance(x, (list, tuple)):
        return any(_has_str(xx) for xx in x)
    return isinstance(x, str)

def _is_numeric_2d(x):
    import numpy as np
    if isinstance(x, np.ndarray):
        return x.ndim == 2 and np.issubdtype(x.dtype, np.number)
    if isinstance(x, (list, tuple)) and len(x) > 0:
        if isinstance(x[0], (list, tuple)) and len(x[0]) > 0:
            return not _has_str(x)  
    return False


def inverse_tensor(scaler: StandardScaler, y: torch.Tensor) -> torch.Tensor:
    return scaler.inverse(y)

def build_models(cfg, device, N_full, Ns, clusters_full, clusters_sub, A_sub):
    import torch
    from ForecastIT_build import ForecastIT, make_sym_norm_P, UndirectedAptDiffusion

    mc = cfg["model"]
    Tin      = int(cfg["data"]["in_channels"])
    Tout     = int(mc.get("pred_len", cfg["data"]["pred_len"]))
    spd      = int(cfg["data"].get("steps_per_day", 288))
    d_model  = int(mc["d_model"])
    n_heads  = int(mc["n_heads"])
    n_layers = int(mc["n_layers"])
    dropout  = float(mc.get("dropout", 0.1))

    coarse = ForecastIT(
        num_nodes=Ns,
        input_dim=Tin,                
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        pred_len=Tout,               
        steps_per_day=spd,
        tod_emb_dim=int(mc.get("tod_emb_dim", 8)),
        dow_emb_dim=int(mc.get("dow_emb_dim", 4)),
        use_gcn=bool(mc.get("use_gcn", True)),   
        gcn_order=int(mc.get("gcn_order", 2)),
        gcn_out=int(mc.get("gcn_out", 64)),
        A=A_sub,                                   
        num_clusters=int(mc["time_key_cluster"]["num_clusters"]),
        cluster_ids=clusters_sub,
        dropout=dropout,
        use_cluster_spa=bool(mc.get("cluster_spa", {}).get("use", True)),  
        cluster_spa_gate_bias=float(mc.get("cluster_spa_gate_bias", 0.0)),
    ).to(device)


    if bool(mc.get("use_gcn", True)) and bool(mc.get("undirected_diffusion", True)):
        P_undirected = make_sym_norm_P(A_sub if isinstance(A_sub, torch.Tensor)
                                       else torch.tensor(A_sub, dtype=torch.float32, device=device))
        aapt_cfg = mc.get("aapt", {"channels": 16, "norm": "sym"})
        coarse.udiff = UndirectedAptDiffusion(
            in_dim=1,
            out_dim=int(mc.get("gcn_out", 64)),
            K=int(mc.get("gcn_order", 2)),
            P_undirected=P_undirected,
            apt_channels=int(aapt_cfg.get("channels", 16)),
            aapt_norm=str(aapt_cfg.get("norm", "sym")),
        ).to(device)
        coarse.gcn = None                         
        coarse.use_undirected_diffusion = True    


    fine = ForecastIT(
        num_nodes=N_full,
        input_dim=Tin,               
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        pred_len=Tout,
        steps_per_day=spd,
        tod_emb_dim=int(mc.get("tod_emb_dim", 8)),
        dow_emb_dim=int(mc.get("dow_emb_dim", 4)),
        use_gcn=False,               
        gcn_order=0,
        gcn_out=0,
        A=None,
        num_clusters=int(mc["time_key_cluster"]["num_clusters"]),
        cluster_ids=clusters_full,
        dropout=dropout,
        use_cluster_spa=False, 
        cluster_spa_gate_bias=float(mc.get("cluster_spa_gate_bias", 0.0)),
    ).to(device)

    return coarse, fine






import time

def train_one_epoch(cfg, coarse, fine, loaders, scaler, device, fine_idx, coarse_idx, pool_on_sub, M_or_parent, use_multi,
                    opt_c, opt_f, loss_main, ddp=False):
    train_loader, _, _ = loaders
    coarse.train(); fine.train()
    tr_c, tr_f, nb = 0.0, 0.0, 0
    Tout = cfg["data"]["pred_len"]

    for it, (X, Y) in enumerate(train_loader):
        t0 = time.time()


        X, Y = to_dev((X, Y), device) 
        torch.cuda.synchronize(); t1 = time.time()


        Xc = X[:, :, coarse_idx, :]
        Yc = Y[:, :, coarse_idx].unsqueeze(-1)

        opt_c.zero_grad(set_to_none=True)
        Yc_hat = coarse(Xc)
        loss_c = loss_main(inverse_tensor(scaler, Yc_hat).squeeze(-1), Y[:, :, coarse_idx])
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(coarse.parameters(), cfg["train"].get("grad_clip", 5.0))
        opt_c.step()
        torch.cuda.synchronize(); t2 = time.time()

        opt_f.zero_grad(set_to_none=True)
        idx_f = torch.tensor(fine_idx, dtype=torch.long, device=device)
        Nf = len(fine_idx); Tout = cfg["data"]["pred_len"]

        with torch.no_grad():
            if use_multi:
                pool_on_sub = M_or_parent["pool_on_sub"]                 # [Npool]
                y_pool_all = Yc_hat[:, :, pool_on_sub, :]                # [B,Tout,Npool,1]
                y_pool = y_pool_all.squeeze(-1)                          # [B,Tout,Npool]
                M = M_or_parent["M"]                                     # [Nf,Npool]
                # (B,T,Npool) × (Nf,Npool)^T -> (B,T,Nf)
                y_prior = torch.einsum("btj,fj->btf", y_pool, M)         # [B,Tout,Nf]
                y_prior_ext = y_prior.unsqueeze(-1)                      # [B,Tout,Nf,1]
            else:
                parent_on_sub = M_or_parent["parent_on_sub"]             # [Nf]
                y_prior_ext = Yc_hat[:, :, parent_on_sub, :]             # [B,Tout,Nf,1]

        fine._y_prior_external = y_prior_ext
        Yf_hat = fine(X)                                                # [B,Tout,N,1]
        del fine._y_prior_external


        y_pred = inverse_tensor(scaler, Yf_hat).squeeze(-1)[:, :, idx_f]   # [B,Tout,Nf]
        y_true = Y[:, :, idx_f]                                            # [B,Tout,Nf]
        loss_f_total = loss_main(y_pred, y_true)                          
        loss_f_total.backward()
        torch.nn.utils.clip_grad_norm_(fine.parameters(), cfg["train"].get("grad_clip", 5.0))
        opt_f.step()

        torch.cuda.synchronize(); t3 = time.time()


        if it < 3:
            print(f"[Timing][batch {it}] H2D={(t1-t0):.3f}s | coarse={(t2-t1):.3f}s | fine={(t3-t2):.3f}s")

        tr_c += loss_c.item() * X.size(0)
        tr_f += loss_f_total.item() * X.size(0)
        nb   += X.size(0)

    return tr_c/nb, tr_f/nb



@torch.no_grad()
def evaluate(cfg, coarse, fine, loaders, scaler, device, fine_idx, coarse_idx, _unused_pool_on_sub,
             M_or_parent, use_multi, ddp=False, return_details: bool=False):
    _, val_loader, _ = loaders
    coarse.eval(); fine.eval()


    yh_c, y_c = [], []
    for Xv, Yv in val_loader:
        Xv, Yv = to_dev((Xv, Yv), device)
        Yc_hat = coarse(Xv[:, :, coarse_idx, :])                     
        yh_c.append(inverse_tensor(scaler, Yc_hat).squeeze(-1).cpu())
        y_c.append(Yv[:, :, coarse_idx].cpu())
    YHc = torch.cat(yh_c, 0).numpy()
    Yc  = torch.cat(y_c , 0).numpy()
    rmse_c = float(RMSE(Yc, YHc))
    mae_c  = float(MAE (Yc, YHc))
    wape_c = float(WAPE(Yc, YHc))
    smape_c= float(SMAPE(Yc, YHc))


    yh_f, y_f = [], []
    for Xv, Yv in val_loader:
        Xv, Yv = to_dev((Xv, Yv), device)
        Yc_hat = coarse(Xv[:, :, coarse_idx, :])                      

        if use_multi:
            pool_on_sub = M_or_parent["pool_on_sub"]
            y_pool_all = Yc_hat[:, :, pool_on_sub, :]                # [B,Tout,Npool,1]
            y_pool = y_pool_all.squeeze(-1)                          # [B,Tout,Npool]
            M = M_or_parent["M"]                                     # [Nf,Npool]
            y_prior = torch.einsum("btj,fj->btf", y_pool, M)         # [B,Tout,Nf]
            y_prior_ext_all = y_prior.unsqueeze(-1)                  # [B,Tout,Nf,1]
        else:
            parent_on_sub = M_or_parent["parent_on_sub"]             # [Nf]
            y_prior_ext_all = Yc_hat[:, :, parent_on_sub, :]         # [B,Tout,Nf,1]

        fine._y_prior_external = y_prior_ext_all
        Yf_hat = fine(Xv)                                            # [B,Tout,N,1]
        del fine._y_prior_external

        yh_f.append(inverse_tensor(scaler, Yf_hat).squeeze(-1)[:, :, fine_idx].cpu())
        y_f.append(Yv[:, :, fine_idx].cpu())

    YHf_t = torch.cat(yh_f, 0)          # [B_all, Tout, Nf] (torch)
    Yf_t  = torch.cat(y_f , 0)          # [B_all, Tout, Nf] (torch)
    YHf   = YHf_t.numpy()
    Yf    = Yf_t.numpy()

    rmse_f = float(RMSE(Yf, YHf))
    mae_f  = float(MAE (Yf, YHf))
    wape_f = float(WAPE(Yf, YHf))
    smape_f= float(SMAPE(Yf, YHf))

    if not return_details:
        return (rmse_c, mae_c, wape_c, smape_c), (rmse_f, mae_f, wape_f, smape_f)


    fine_pernode = []
    for k, nid in enumerate(fine_idx):
        yk = Yf[:, :, k]; yh = YHf[:, :, k]
        fine_pernode.append({
            "id": int(nid),
            "rmse": float(RMSE(yk, yh)),
            "mae":  float(MAE (yk, yh)),
        })
    return (rmse_c, mae_c, wape_c, smape_c), (rmse_f, mae_f, wape_f, smape_f), fine_pernode


def evaluate_by_horizon(cfg, coarse, fine, loaders, scaler, device, fine_idx, coarse_idx, _unused_pool_on_sub,
             M_or_parent, use_multi, horizon, ddp=False, return_details: bool=False):

    _, _, test_loader = loaders  
    coarse.eval(); fine.eval()

    with torch.no_grad():  
        yh_c, y_c = [], []
        for Xv, Yv in test_loader:
            Xv, Yv = to_dev((Xv, Yv), device)
            Yc_hat = coarse(Xv[:, :, coarse_idx, :])                      # [B,Tout,Nc,1]
            yh_c.append(inverse_tensor(scaler, Yc_hat[:, :horizon, :, :]).squeeze(-1).detach().cpu())
            y_c.append(Yv[:, :horizon, coarse_idx].cpu())
        YHc = torch.cat(yh_c, 0).numpy()
        Yc  = torch.cat(y_c , 0).numpy()
        rmse_c = float(RMSE(Yc, YHc))
        mae_c  = float(MAE (Yc, YHc))
        wape_c = float(WAPE(Yc, YHc))
        smape_c= float(SMAPE(Yc, YHc))


        yh_f, y_f = [], []
        for Xv, Yv in test_loader:
            Xv, Yv = to_dev((Xv, Yv), device)
            Yc_hat = coarse(Xv[:, :, coarse_idx, :])                      # [B,Tout,Nc,1]

            if use_multi:
                pool_on_sub = M_or_parent["pool_on_sub"]
                y_pool_all = Yc_hat[:, :, pool_on_sub, :]                # [B,Tout,Npool,1]
                y_pool = y_pool_all.squeeze(-1)                          # [B,Tout,Npool]
                M = M_or_parent["M"]                                     # [Nf,Npool]
                y_prior = torch.einsum("btj,fj->btf", y_pool, M)         # [B,Tout,Nf]
                y_prior_ext_all = y_prior.unsqueeze(-1)                  # [B,Tout,Nf,1]
            else:
                parent_on_sub = M_or_parent["parent_on_sub"]             # [Nf]
                y_prior_ext_all = Yc_hat[:, :, parent_on_sub, :]         # [B,Tout,Nf,1]

            fine._y_prior_external = y_prior_ext_all
            Yf_hat = fine(Xv)                                            # [B,Tout,N,1]
            del fine._y_prior_external


            yh_f.append(inverse_tensor(scaler, Yf_hat[:, :horizon, :, :]).squeeze(-1)[:, :, fine_idx].detach().cpu())
            y_f.append(Yv[:, :horizon, fine_idx].cpu())

        YHf_t = torch.cat(yh_f, 0)          # [B_all, horizon, Nf] (torch)
        Yf_t  = torch.cat(y_f , 0)          # [B_all, horizon, Nf] (torch)
        YHf   = YHf_t.numpy()
        Yf    = Yf_t.numpy()

        rmse_f = float(RMSE(Yf, YHf))
        mae_f  = float(MAE (Yf, YHf))
        wape_f = float(WAPE(Yf, YHf))
        smape_f= float(SMAPE(Yf, YHf))

        if not return_details:
            return (rmse_c, mae_c, wape_c, smape_c), (rmse_f, mae_f, wape_f, smape_f)


        fine_pernode = []
        for k, nid in enumerate(fine_idx):
            yk = Yf[:, :, k]; yh = YHf[:, :, k]
            fine_pernode.append({
                "id": int(nid),
                "rmse": float(RMSE(yk, yh)),
                "mae":  float(MAE (yk, yh)),
            })
        return (rmse_c, mae_c, wape_c, smape_c), (rmse_f, mae_f, wape_f, smape_f), fine_pernode


def evaluate_single_step(cfg, coarse, fine, loaders, scaler, device, fine_idx, coarse_idx, _unused_pool_on_sub,
             M_or_parent, use_multi, step, ddp=False, return_details: bool=False):

    _, _, test_loader = loaders  
    coarse.eval(); fine.eval()

    with torch.no_grad(): 

        yh_c, y_c = [], []
        for Xv, Yv in test_loader:
            Xv, Yv = to_dev((Xv, Yv), device)
            Yc_hat = coarse(Xv[:, :, coarse_idx, :])                      # [B,Tout,Nc,1]
            yh_c.append(inverse_tensor(scaler, Yc_hat[:, step-1:step, :, :]).squeeze(-1).detach().cpu())
            y_c.append(Yv[:, step-1:step, coarse_idx].cpu())
        YHc = torch.cat(yh_c, 0).numpy()
        Yc  = torch.cat(y_c , 0).numpy()
        rmse_c = float(RMSE(Yc, YHc))
        mae_c  = float(MAE (Yc, YHc))
        wape_c = float(WAPE(Yc, YHc))
        smape_c= float(SMAPE(Yc, YHc))


        yh_f, y_f = [], []
        for Xv, Yv in test_loader:
            Xv, Yv = to_dev((Xv, Yv), device)
            Yc_hat = coarse(Xv[:, :, coarse_idx, :])                      # [B,Tout,Nc,1]

            if use_multi:
                pool_on_sub = M_or_parent["pool_on_sub"]
                y_pool_all = Yc_hat[:, :, pool_on_sub, :]                # [B,Tout,Npool,1]
                y_pool = y_pool_all.squeeze(-1)                          # [B,Tout,Npool]
                M = M_or_parent["M"]                                     # [Nf,Npool]
                y_prior = torch.einsum("btj,fj->btf", y_pool, M)         # [B,Tout,Nf]
                y_prior_ext_all = y_prior.unsqueeze(-1)                  # [B,Tout,Nf,1]
            else:
                parent_on_sub = M_or_parent["parent_on_sub"]             # [Nf]
                y_prior_ext_all = Yc_hat[:, :, parent_on_sub, :]         # [B,Tout,Nf,1]

            fine._y_prior_external = y_prior_ext_all
            Yf_hat = fine(Xv)                                            # [B,Tout,N,1]
            del fine._y_prior_external


            yh_f.append(inverse_tensor(scaler, Yf_hat[:, step-1:step, :, :]).squeeze(-1)[:, :, fine_idx].detach().cpu())
            y_f.append(Yv[:, step-1:step, fine_idx].cpu())

        YHf_t = torch.cat(yh_f, 0)          # [B_all, 1, Nf] (torch)
        Yf_t  = torch.cat(y_f , 0)          # [B_all, 1, Nf] (torch)
        YHf   = YHf_t.numpy()
        Yf    = Yf_t.numpy()

        rmse_f = float(RMSE(Yf, YHf))
        mae_f  = float(MAE (Yf, YHf))
        wape_f = float(WAPE(Yf, YHf))
        smape_f= float(SMAPE(Yf, YHf))

        if not return_details:
            return (rmse_c, mae_c, wape_c, smape_c), (rmse_f, mae_f, wape_f, smape_f)


        fine_pernode = []
        for k, nid in enumerate(fine_idx):
            yk = Yf[:, :, k]; yh = YHf[:, :, k]
            fine_pernode.append({
                "id": int(nid),
                "rmse": float(RMSE(yk, yh)),
                "mae":  float(MAE (yk, yh)),
            })
        return (rmse_c, mae_c, wape_c, smape_c), (rmse_f, mae_f, wape_f, smape_f), fine_pernode

@torch.no_grad()
def evaluate_stepwise_pernode_fine(cfg, coarse, fine, loaders, scaler, device,
                                   fine_idx, coarse_idx, M_or_parent, use_multi, ddp=False):
    _, _, test_loader = loaders
    coarse.eval(); fine.eval()


    yh_f, y_f = [], [] 
    for Xv, Yv in test_loader:
        Xv, Yv = to_dev((Xv, Yv), device)
        Yc_hat = coarse(Xv[:, :, coarse_idx, :])  # [B,Tout,Nc,1]

        if use_multi:
            pool_on_sub = M_or_parent["pool_on_sub"]
            y_pool_all = Yc_hat[:, :, pool_on_sub, :]                # [B,Tout,Npool,1]
            y_pool = y_pool_all.squeeze(-1)                          # [B,Tout,Npool]
            M = M_or_parent["M"]                                     # [Nf,Npool]
            y_prior = torch.einsum("btj,fj->btf", y_pool, M)         # [B,Tout,Nf]
            y_prior_ext_all = y_prior.unsqueeze(-1)                  # [B,Tout,Nf,1]
        else:
            parent_on_sub = M_or_parent["parent_on_sub"]             # [Nf]
            y_prior_ext_all = Yc_hat[:, :, parent_on_sub, :]         # [B,Tout,Nf,1]

        fine._y_prior_external = y_prior_ext_all
        Yf_hat = fine(Xv)                                            # [B,Tout,N,1]
        del fine._y_prior_external


        yh_f.append(inverse_tensor(scaler, Yf_hat).squeeze(-1)[:, :, fine_idx].detach().cpu())
        y_f.append(Yv[:, :, fine_idx].detach().cpu())
    YH = torch.cat(yh_f, 0).numpy()   # [S, T, Nf]
    Y  = torch.cat(y_f , 0).numpy()   # [S, T, Nf]

    err = YH - Y                      # [S, T, Nf]
    rmse = np.sqrt((err**2).mean(axis=0))             # [T, Nf]
    mae  = np.abs(err).mean(axis=0)                   # [T, Nf]

    denom = np.where(np.abs(Y) > 1e-6, np.abs(Y), np.nan)
    mape = np.nanmean(np.abs(err) / denom, axis=0) * 100.0  # [T, Nf]

    mape = np.where(np.isnan(mape), 0.0, mape)


    return {
        "steps": list(range(1, Y.shape[1] + 1)),
        "node_ids": list(map(int, fine_idx)),
        "rmse": rmse.T,   # (Nf, T)
        "mae" : mae.T,    # (Nf, T)
        "mape": mape.T,   # (Nf, T)
    }


def plot_stepwise_pernode(metrics, save_dir, title_prefix="Fine Node"):

    import csv
    os.makedirs(save_dir, exist_ok=True)
    steps    = metrics["steps"]            # [1..T]
    node_ids = metrics["node_ids"]         # [Nf]
    RMSE = metrics["rmse"]                 # (Nf, T)
    MAE  = metrics["mae"]                  # (Nf, T)
    MAPE = metrics["mape"]                 # (Nf, T)


    for i, nid in enumerate(node_ids):
        plt.figure(figsize=(8,5))
        plt.plot(steps, RMSE[i].tolist(), 'o-', label='RMSE')
        plt.plot(steps, MAE[i].tolist(),  's-', label='MAE')
        plt.plot(steps, MAPE[i].tolist(), '^-', label='MAPE (%)')
        plt.xlabel('Prediction Step'); plt.ylabel('Metric Value')
        plt.title(f'{title_prefix} {nid}: Step-wise Metrics')
        plt.legend(); plt.grid(True); plt.tight_layout()
        fpath = os.path.join(save_dir, f'fine_node{nid}_stepwise_metrics.png')
        plt.savefig(fpath, dpi=300); plt.close()
        print(f"[Plot] saved: {fpath}")


    csv_path = os.path.join(save_dir, "fine_stepwise_metrics_pernode.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "step", "rmse", "mae", "mape(%)"])
        for i, nid in enumerate(node_ids):
            for j, s in enumerate(steps):
                w.writerow([nid, s, float(RMSE[i, j]), float(MAE[i, j]), float(MAPE[i, j])])
    print(f"[CSV] saved: {csv_path}")


def save_ckpt(path, coarse, fine, opt_c, opt_f, epoch, best_metric, coarse_idx):
    state = {
        "epoch": epoch,
        "best": best_metric,
        "coarse": coarse.state_dict(),
        "fine": fine.state_dict(),
        "opt_c": opt_c.state_dict(),
        "opt_f": opt_f.state_dict(),
        "coarse_idx": coarse_idx,
    }
    torch.save(state, path)

def load_ckpt(path, coarse, fine, opt_c, opt_f, map_location):
    ck = torch.load(path, map_location=map_location)
    coarse.load_state_dict(ck["coarse"])
    fine.load_state_dict(ck["fine"])
    if opt_c is not None and opt_f is not None:
        opt_c.load_state_dict(ck["opt_c"])
        opt_f.load_state_dict(ck["opt_f"])
    return ck.get("epoch", 0), ck.get("best", None), ck.get("coarse_idx", None)

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description="Joint training (step-wise prior) for coarse & fine")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument("--save-dir", type=str, default=None, help="Override save dir from config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    rank, device = ddp_setup(args)
    is_main = (rank == 0)

    with open(args.config, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"].get("seed", args.seed))

    save_dir = ensure_dir(args.save_dir if args.save_dir else cfg["train"].get("save_dir", "saved_models/joint"))

    data_npz, idx_dict, loaders = build_dataloaders(cfg, scaler=None, device=device, ddp=args.ddp)

    print("[Scaler] Calculating mean/std over training set...")

    data_arr = data_npz["data"].astype(np.float32, copy=False)  # shape [T, N, C]
    flow = data_arr[:, :, 0]                                   

    total_sum = 0.0
    total_sum_sq = 0.0
    total_count = 0

    train_ranges = idx_dict["train"]
    num_ranges = len(train_ranges)
    log_every = max(1, num_ranges // 10)

    for i, (s, e, *_) in enumerate(train_ranges):
        seg = flow[s:e]                
        total_sum    += float(seg.sum())
        total_sum_sq += float((seg * seg).sum())
        total_count  += int(seg.size)
        if (i + 1) % log_every == 0 or (i + 1) == num_ranges:
            print(f"  progress: {i+1}/{num_ranges} ranges processed")

    mean = total_sum / max(total_count, 1)
    var  = max(total_sum_sq / max(total_count, 1) - mean * mean, 0.0)
    std  = var ** 0.5
    if std < 1e-6:
        std = 1.0

    scaler = StandardScaler(mean, std)

    del flow, data_arr

    print("[Scaler] Calculation finished!")

    _, _, loaders = build_dataloaders(cfg, scaler, device, ddp=args.ddp)
    train_loader, val_loader, test_loader = loaders

    T, N, C = data_npz["data"].shape
    fine_idx = list(map(int, cfg["model"]["fine_grain"]["fine_indices"]))
    coarse_idx = [i for i in range(N) if i not in fine_idx]
    pos_on_sub = {n: i for i, n in enumerate(coarse_idx)}

    clusters_full = load_clusters(cfg["model"]["time_key_cluster"]["assign_path"])
    clusters_sub  = [clusters_full[i] for i in coarse_idx]
    num_clusters  = int(cfg["model"]["time_key_cluster"]["num_clusters"])
    A_sub_raw, node_ids_sub = load_adj(cfg["model"]["adj_sub_path"], device=device)
    Ns = int(A_sub_raw.size(0))
    coarse_idx = list(range(Ns))
    coarse, fine = build_models(cfg, device, N, len(coarse_idx), clusters_full, clusters_sub, A_sub_raw)

    fg = cfg["model"]["fine_grain"]
    use_multi = bool(fg.get("parents", {}).get("use_multi", False))

    if use_multi:
        parents_cfg = fg.get("parents", {})
        assert "coarse_pool" in parents_cfg and "M" in parents_cfg, \
            "use_multi=true 时需要 fine_grain.parents.coarse_pool 和 fine_grain.parents.M"
        coarse_pool = list(map(int, parents_cfg["coarse_pool"]))        
        M = torch.tensor(parents_cfg["M"], dtype=torch.float32, device=device)
        assert M.dim()==2 and M.size(0)==len(fine_idx) and M.size(1)==len(coarse_pool), \
            f"M shape={tuple(M.shape)} 与 fine={len(fine_idx)}、pool={len(coarse_pool)} 不匹配"
        M = M / (M.sum(dim=1, keepdim=True) + 1e-8)

        fine.enable_fine_grain_full(
            fine_indices=fine_idx, fg_cfg=fg,
            parent_for_fine=None,
            coarse_pool=coarse_pool, M_weights=parents_cfg["M"]
        )
        M_or_parent = {"M": M, "pool_on_sub": coarse_pool}

    else:
        parents_cfg = fg.get("parents", {})
        pff = parents_cfg.get("parent_for_fine", fg.get("parent_for_fine", None))
        if pff is None:
            raise KeyError("use_multi=false 时必须提供 parent_for_fine（可放在 fine_grain.parents.parent_for_fine 或 fine_grain.parent_for_fine）")
        parent_on_sub = list(map(int, pff))
        assert len(parent_on_sub) == len(fine_idx), \
            f"parent_for_fine 数量 {len(parent_on_sub)} 必须与 fine_indices 数量 {len(fine_idx)} 一致"

        fine.enable_fine_grain_full(
            fine_indices=fine_idx, fg_cfg=fg,
            parent_for_fine=parent_on_sub, coarse_pool=None, M_weights=None
        )
        M_or_parent = {"parent_on_sub": parent_on_sub}
    lr = cfg["train"]["lr"]
    wd = cfg["train"].get("weight_decay", 3e-4)
    opt_c = torch.optim.AdamW(coarse.parameters(), lr=lr, weight_decay=wd)
    params_fine = []
    main_params = [p for n, p in fine.named_parameters() if n != "fg_M"]
    if main_params:
        params_fine.append({"params": main_params, "lr": lr, "weight_decay": wd})
    lr_M = cfg["model"]["fine_grain"]["parents"].get("lr_M", lr * 5.0)
    if hasattr(fine, "fg_M") and isinstance(fine.fg_M, torch.nn.Parameter):
        params_fine.append({"params": [fine.fg_M], "lr": lr_M, "weight_decay": 0.0})
        if (rank == 0):  # 只主进程打印
            print(f"[opt] fg_M is learnable: lr_M={lr_M:g}, no weight_decay")

    opt_f = torch.optim.AdamW(params_fine)
    loss_name = cfg["train"].get("loss", "huber").lower()
    if loss_name in ("mse", "l2"):
        loss_main = torch.nn.MSELoss()
    elif loss_name in ("l1", "mae"):
        loss_main = torch.nn.L1Loss()
    else:
        loss_main = torch.nn.SmoothL1Loss(beta=0.5)


    start_ep, best_metric = 0, None
    ckpt_path = os.path.join(save_dir, "best.ckpt")

    if args.resume and os.path.exists(ckpt_path):
        ep0, best_metric, ck_coarse_idx = load_ckpt(ckpt_path, coarse, fine, opt_c, opt_f, map_location=device)
        start_ep = ep0 + 1
        if is_main:
            print(f"[resume] loaded epoch={ep0}, best={best_metric}, coarse_idx_len={len(ck_coarse_idx) if ck_coarse_idx else 'NA'}")

    if ddp_is_on(args):
        coarse = DDP(coarse, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        fine   = DDP(fine,   device_ids=[rank], output_device=rank, find_unused_parameters=False)
    stopper = EarlyStopper(patience=cfg["train"].get("patience", 10), mode="min", min_delta=cfg["train"].get("min_delta", 0.0))

    E = int(cfg["train"]["epochs"])
    for ep in range(start_ep, E):
        if ddp_is_on(args):
            train_loader.sampler.set_epoch(ep)

        tr_c, tr_f = train_one_epoch(
            cfg, coarse if not ddp_is_on(args) else coarse.module,
            fine if not ddp_is_on(args) else fine.module,
            loaders, scaler, device, fine_idx, coarse_idx, None, M_or_parent, use_multi,
            opt_c, opt_f, loss_main, ddp=args.ddp
        )

        (rmse_c, mae_c, wape_c, smape_c), (rmse_f, mae_f, wape_f, smape_f) = evaluate(
            cfg, coarse if not ddp_is_on(args) else coarse.module,
            fine if not ddp_is_on(args) else fine.module,
            loaders, scaler, device, fine_idx, coarse_idx, None, M_or_parent, use_multi, ddp=args.ddp
        )

        if is_main:
            print(f"[Epoch {ep:03d}] Train  | coarse:{tr_c:.5f}  fine:{tr_f:.5f}")
            print(f"            Val    | COARSE RMSE={rmse_c:.4f} MAE={mae_c:.4f} WAPE={wape_c:.4f} sMAPE={smape_c:.4f}")
            print(f"                    FINE(*) RMSE={rmse_f:.4f} MAE={mae_f:.4f} WAPE={wape_f:.4f} sMAPE={smape_f:.4f}  (only nodes {fine_idx})")

            monitor = cfg["train"].get("monitor", "fine_rmse").lower()
            if monitor == "fine_rmse":
                cur_metric = rmse_f
            else:
                cur_metric = rmse_c 

            is_best = (best_metric is None) or (cur_metric < best_metric - cfg["train"].get("min_delta", 0.0))
            if is_best:
                best_metric = cur_metric
                save_ckpt(ckpt_path,
                        coarse if not ddp_is_on(args) else coarse.module,
                        fine   if not ddp_is_on(args) else fine.module,
                        opt_c, opt_f, ep, best_metric, coarse_idx)
                print(f"[save] best updated @ epoch {ep} -> {best_metric:.4f} -> {ckpt_path}")

            if stopper.step(cur_metric):
                print(f"[early-stop] no improvement for {stopper.patience} epochs. best={best_metric:.4f}")
                break


    if is_main:
        if os.path.exists(ckpt_path):
            load_ckpt(ckpt_path,
                    coarse if not ddp_is_on(args) else coarse.module,
                    fine   if not ddp_is_on(args) else fine.module,
                    None, None, map_location=device)

        coarse_model = coarse if not ddp_is_on(args) else coarse.module
        fine_model = fine if not ddp_is_on(args) else fine.module
        
        print("[Visualization] Enabling attention recording...")
        coarse_model.enable_attn_recording(flag=True, clear=True)
        fine_model.enable_attn_recording(flag=True, clear=True)

        print("[Test] evaluating best checkpoint...")
        
        horizons = [3, 6, 12]
        print(f"[Test] Evaluating for horizons: {horizons}")
        
        for horizon in horizons:
            print(f"\n=== Horizon {horizon} steps ===")
            out = evaluate_by_horizon(
                cfg, coarse if not ddp_is_on(args) else coarse.module,
                fine   if not ddp_is_on(args) else fine.module,
                (train_loader, val_loader, test_loader), scaler, device,
                fine_idx, coarse_idx, None, M_or_parent, use_multi, 
                horizon=horizon, ddp=args.ddp, return_details=True
            )
            (rmse_c, mae_c, wape_c, smape_c), (rmse_f, mae_f, wape_f, smape_f), fine_pernode = out

            print(f"[Test-{horizon}] COARSE  RMSE={rmse_c:.4f} MAE={mae_c:.4f} WAPE={wape_c:.4f} sMAPE={smape_c:.4f}")
            print(f"[Test-{horizon}] FINE(*) RMSE={rmse_f:.4f} MAE={mae_f:.4f} WAPE={wape_f:.4f} sMAPE={smape_f:.4f}  (nodes {fine_idx})")
            if horizon == 12:  
                print(f"[Test-{horizon}][FINE per-node] " + " | ".join(
                    [f"id={d['id']}: RMSE={d['rmse']:.3f}, MAE={d['mae']:.3f}" for d in fine_pernode]
                ))

        steps = list(range(1, 13))
        rmse_list, mae_list, mape_list = [], [], []

        print("\n[Plot] Evaluating per-step metrics for fine-grained nodes...")
        for step in steps:
            (_, _, _, _), (rmse_f, mae_f, wape_f, smape_f), _ = evaluate_single_step(
                cfg,
                coarse if not ddp_is_on(args) else coarse.module,
                fine   if not ddp_is_on(args) else fine.module,
                (train_loader, val_loader, test_loader),
                scaler, device,
                fine_idx, coarse_idx, None, M_or_parent, use_multi,
                step=step, ddp=args.ddp, return_details=True
            )
            rmse_list.append(rmse_f)
            mae_list.append(mae_f)
            mape_list.append(wape_f)  # WAPE≈MAPE


        plt.figure(figsize=(8,5))
        plt.plot(steps, rmse_list, 'o-', label='RMSE')
        plt.plot(steps, mae_list, 's-', label='MAE')
        plt.plot(steps, mape_list, '^-', label='MAPE')
        plt.xlabel('Prediction Step')
        plt.ylabel('Metric Value')
        plt.title('Fine-grained Forecast Performance over 12 Steps')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        fig_path = os.path.join(save_dir, "fine_stepwise_metrics.png")
        plt.savefig(fig_path, dpi=300)
        print(f"[Plot] Saved fine-step metrics curve to: {fig_path}")
        print("\n[Plot] Evaluating per-node step-wise metrics for fine-grained nodes...")
        metrics = evaluate_stepwise_pernode_fine(
            cfg,
            coarse if not ddp_is_on(args) else coarse.module,
            fine   if not ddp_is_on(args) else fine.module,
            (train_loader, val_loader, test_loader),
            scaler, device,
            fine_idx, coarse_idx, M_or_parent, use_multi,
            ddp=args.ddp
        )
        plot_stepwise_pernode(metrics, save_dir, title_prefix="Fine Node")
    ddp_cleanup()

if __name__ == "__main__":
    main()