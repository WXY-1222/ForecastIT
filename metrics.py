import numpy as np


def MSE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mse = np.square(y_pred - y_true)
        mse = np.nan_to_num(mse * mask)
        mse = np.mean(mse)
        return mse


def RMSE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        return rmse


def MAE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        return mae


def MAPE(y_true, y_pred, null_val=0):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype("float32")
        mask /= np.mean(mask)
        mape = np.abs(np.divide((y_pred - y_true).astype("float32"), y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def RMSE_MAE_MAPE(y_true, y_pred):
    return (
        RMSE(y_true, y_pred),
        MAE(y_true, y_pred),
        MAPE(y_true, y_pred),
    )


def MSE_RMSE_MAE_MAPE(y_true, y_pred):
    return (
        MSE(y_true, y_pred),
        RMSE(y_true, y_pred),
        MAE(y_true, y_pred),
        MAPE(y_true, y_pred),
    )


def WAPE(y_true, y_pred, eps=1e-8):
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.abs(y_pred - y_true).sum()
        den = np.abs(y_true).sum()
        wape = num / max(den, eps)
        return float(wape * 100.0)

def SMAPE(y_true, y_pred, eps=1e-8):
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = (np.abs(y_true) + np.abs(y_pred)) + eps
        smape = 2.0 * np.abs(y_pred - y_true) / denom
        smape = np.nan_to_num(smape)
        return float(smape.mean() * 100.0)

def MAPE_AT(y_true, y_pred, threshold=10.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.abs(y_true) >= float(threshold)
        if not np.any(mask):
            return np.nan
        mape = np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])
        mape = np.nan_to_num(mape)
        return float(mape.mean() * 100.0)

def MAPE_AT_10(y_true, y_pred):
    return MAPE_AT(y_true, y_pred, threshold=10.0)


def RMSE_MAE_WAPE_SMAPE_MAPE10(y_true, y_pred):
    return (
        RMSE(y_true, y_pred),
        MAE(y_true, y_pred),
        WAPE(y_true, y_pred),
        SMAPE(y_true, y_pred),
        MAPE_AT_10(y_true, y_pred),
    )