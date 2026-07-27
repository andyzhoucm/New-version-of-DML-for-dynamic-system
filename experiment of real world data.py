import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import os
import time
import datetime
import warnings
import random  
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.func as func
from torchdiffeq import odeint 
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")
import torch._functorch.config
torch._functorch.config.donated_buffer = False

# ==========================================
# 0. 全局超参数与数据集配置
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = '/root/code/data/2018'
PATIENT_IDS = ['559', '563', '570', '575', '588', '591']

# 全局核心消融开关
USE_MSE = True            
USE_DML = False               
FREEZE_PKPD = False      

DT_MINUTES = 5          
HIST_WINDOW = 24        
PRED_WINDOW = 24        
GAP_THRESHOLD = 15      
C_DIM = 16              
MEM_DIM = 8             
MAX_HIST = 47           

BATCH_SIZE = 64         
NUM_WORKERS = 16 
OUTER_EPOCHS = 8       
INNER_N_EPOCHS = 6      
INNER_P_EPOCHS = 1      

print(f"🚀 Global 跨病人启动 | 包含 {len(PATIENT_IDS)} 名患者 | MSE: {USE_MSE} | DML: {USE_DML} | 冻结PKPD: {FREEZE_PKPD} | Device: {DEVICE}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# 1. 数据解析 (带历史截取的特征工程)
# ==========================================
def parse_and_align_data(filepath):
    if not os.path.exists(filepath): return None
    with open(filepath, 'r') as f: 
        soup = BeautifulSoup(f.read(), 'xml')
        
    def extract_node(node_name, val_key='value', ts_key='ts'):
        records = [{'ts': pd.to_datetime(item.get(ts_key), format='%d-%m-%Y %H:%M:%S'), 'val': float(item.get(val_key))}
                   for event in soup.find_all(node_name) for item in event.find_all('event') if item.get(ts_key) and item.get(val_key)]
        return pd.DataFrame(records).sort_values('ts') if records else pd.DataFrame(columns=['ts', 'val'])

    def extract_temp_basal():
        records = []
        for event in soup.find_all('temp_basal'):
            for item in event.find_all('event'):
                ts_b, ts_e, val = item.get('ts_begin'), item.get('ts_end'), item.get('value')
                if ts_b and ts_e and val:
                    records.append({'ts_begin': pd.to_datetime(ts_b, format='%d-%m-%Y %H:%M:%S'),
                                    'ts_end': pd.to_datetime(ts_e, format='%d-%m-%Y %H:%M:%S'),
                                    'val': float(val)})
        return pd.DataFrame(records)

    df_cgm = extract_node('glucose_level')
    df_basal = extract_node('basal')
    df_bolus = extract_node('bolus', ts_key='ts_begin', val_key='dose')
    df_meal = extract_node('meal', val_key='carbs')
    df_hr = extract_node('basis_heart_rate')
    df_temp_skin = extract_node('basis_skin_temperature')
    df_temp_basal = extract_temp_basal()

    if df_cgm.empty: return None

    start_time = df_cgm['ts'].min().floor('5min')
    end_time = df_cgm['ts'].max().ceil('5min')
    grid_1min = pd.date_range(start_time, end_time, freq='1min')
    df_1min = pd.DataFrame(index=grid_1min)

    df_1min['basal_U_min'] = 0.0
    if not df_basal.empty:
        basal_df = df_basal.set_index('ts')
        df_1min['basal_U_min'] = basal_df['val'].reindex(grid_1min).ffill().bfill() / 60.0
    if not df_temp_basal.empty:
        for _, row in df_temp_basal.iterrows():
            mask = (df_1min.index >= row['ts_begin']) & (df_1min.index <= row['ts_end'])
            df_1min.loc[mask, 'basal_U_min'] = float(row['val']) / 60.0

    df_1min['bolus_U_min'] = 0.0
    if not df_bolus.empty:
        bolus_df = df_bolus.copy()
        bolus_df['ts_bin'] = bolus_df['ts'].dt.floor('1min') 
        bolus_grouped = bolus_df.groupby('ts_bin')['val'].sum()
        common_idx = df_1min.index.intersection(bolus_grouped.index)
        df_1min.loc[common_idx, 'bolus_U_min'] = bolus_grouped.loc[common_idx].values
    
    df_1min['U_t'] = df_1min['basal_U_min'] + df_1min['bolus_U_min']
    df_1min['D_t'] = 0.0
    
    if not df_meal.empty:
        df_meal['ts_bin'] = df_meal['ts'].dt.floor('1min')
        mgp = df_meal.groupby('ts_bin')['val'].sum()
        common_idx_m = df_1min.index.intersection(mgp.index)
        df_1min.loc[common_idx_m, 'D_t'] = mgp.loc[common_idx_m].values

    grid_5min = pd.date_range(start_time, end_time, freq='5min')
    df_5min = pd.DataFrame(index=grid_5min)

    df_cgm.set_index('ts', inplace=True)
    df_5min['Y_cgm'] = df_cgm['val'].reindex(grid_5min, method='nearest', tolerance=pd.Timedelta('2min')).interpolate(method='linear', limit=3) 

    df_1min['ts_5min'] = df_1min.index.floor('5min')
    df_agg = df_1min.groupby('ts_5min')[['U_t', 'D_t']].sum()
    
    df_5min['U_ins'] = df_agg['U_t'].reindex(grid_5min).fillna(0.0)
    df_5min['D_carbs'] = df_agg['D_t'].reindex(grid_5min).fillna(0.0)

    if not df_hr.empty:
        hr_df = df_hr.set_index('ts')
        df_5min['HR'] = hr_df['val'].reindex(grid_5min).ffill().bfill().fillna(70.0)
    else: df_5min['HR'] = 70.0
        
    if not df_temp_skin.empty:
        st_df = df_temp_skin.set_index('ts')
        df_5min['SkinTemp'] = st_df['val'].reindex(grid_5min).ffill().bfill().fillna(90.0)
    else: df_5min['SkinTemp'] = 90.0

    df_5min = df_5min.dropna(subset=['Y_cgm'])
    return df_5min.reset_index().rename(columns={'index': 'ts'})

def extract_windows_stride(df_sim, stride):
    valid_mask = df_sim['Y_cgm'].notna()
    df_valid = df_sim[valid_mask].copy()
    
    break_points = [df_valid.index[0]] + df_valid.index[df_valid['ts'].diff() > pd.Timedelta(minutes=GAP_THRESHOLD)].tolist() + [df_valid.index[-1] + 1]

    X_Y, X_U, X_D, X_Z_snap, X_T_seq = [], [], [], [], []
    for i in range(len(break_points) - 1):
        chunk = df_sim.loc[break_points[i] : break_points[i+1]-1].dropna()
        if len(chunk) < HIST_WINDOW + PRED_WINDOW: continue
            
        vals_Y = chunk['Y_cgm'].values.reshape(-1, 1) / 100.0
        vals_U = chunk['U_ins'].values.reshape(-1, 1) * 20.0  
        vals_D = chunk['D_carbs'].values.reshape(-1, 1)
        vals_Z = chunk[['HR', 'SkinTemp']].values / 100.0
        
        hour = chunk['ts'].dt.hour.values + chunk['ts'].dt.minute.values / 60.0
        time_features = np.hstack([np.sin(2 * np.pi * hour / 24.0).reshape(-1, 1), np.cos(2 * np.pi * hour / 24.0).reshape(-1, 1)])
        
        safe_history_features = np.hstack([vals_Y, vals_U, vals_D, vals_Z]) 
        
        for start_idx in range(HIST_WINDOW, len(chunk) - PRED_WINDOW + 1, stride):
            end_idx = start_idx + PRED_WINDOW
            X_Y.append(vals_Y[start_idx : end_idx])
            X_Z_snap.append(safe_history_features[start_idx - HIST_WINDOW : start_idx])
            X_T_seq.append(time_features[start_idx : end_idx])
            
            hist_start = start_idx - MAX_HIST
            if hist_start < 0:
                pad_len = -hist_start
                pad_U = np.zeros((pad_len, 1))
                pad_D = np.zeros((pad_len, 1))
                X_U.append(np.vstack([pad_U, vals_U[0 : end_idx]]))
                X_D.append(np.vstack([pad_D, vals_D[0 : end_idx]]))
            else:
                X_U.append(vals_U[hist_start : end_idx])
                X_D.append(vals_D[hist_start : end_idx])
            
    if len(X_Y) == 0: return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    return np.array(X_Y), np.array(X_U), np.array(X_D), np.array(X_Z_snap), np.array(X_T_seq)

def build_patient_dataset(patient_ids, split_type, stride):
    Y_all, U_all, D_all, Z_all, T_all = [], [], [], [], []
    for pid in patient_ids:
        filepath = os.path.join(DATA_DIR, split_type, f"{pid}-ws-{split_type}ing.xml")
        if os.path.exists(filepath):
            df = parse_and_align_data(filepath)
            if df is not None:
                Y, U, D, Z, T = extract_windows_stride(df, stride=stride)
                if len(Y) > 0:
                    Y_all.append(Y); U_all.append(U); D_all.append(D)
                    Z_all.append(Z); T_all.append(T)
    if len(Y_all) == 0: return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    return np.vstack(Y_all), np.vstack(U_all), np.vstack(D_all), np.vstack(Z_all), np.vstack(T_all)

# ==========================================
# 2. 评估模块 
# ==========================================
def evaluate_final_test_rmse(phys_C, phys_I, encoder_g, enc_carb, enc_ins, res_nn, hidden_dyn_nn, pkpd_carb, pkpd_ins, dl_te, t_eval_t, label="Global", subset_ratio=1.0):
    for m in [phys_C, phys_I, encoder_g, enc_carb, enc_ins, res_nn, hidden_dyn_nn, pkpd_carb, pkpd_ins]:
        m.eval()
        for p in m.parameters(): p.requires_grad = False
        
    sum_sq_err_30, sum_sq_err_60, sum_sq_err_120, total_samples = 0.0, 0.0, 0.0, 0
    
    causal_loss_C_total = 0.0
    causal_loss_I_total = 0.0
    causal_batches_C = 0
    causal_batches_I = 0
    
    temp_phi = 10.0 
    
    total_batches = len(dl_te)
    max_batches = max(1, int(total_batches * subset_ratio)) if subset_ratio < 1.0 else total_batches

    with torch.no_grad():
        for batch_idx, (batch_Y, batch_U_raw, batch_D_raw, batch_Z_snap, batch_T_seq) in enumerate(dl_te):
            if batch_idx >= max_batches:
                break
                
            batch_Y, batch_U_raw, batch_D_raw = batch_Y.to(DEVICE), batch_U_raw.to(DEVICE), batch_D_raw.to(DEVICE)
            batch_Z_snap, batch_T_seq = batch_Z_snap.to(DEVICE), batch_T_seq.to(DEVICE)
            
            T_pts = batch_Y.shape[1]
            B = batch_Y.size(0)
            
            batch_U = pkpd_ins(batch_U_raw, target_len=T_pts)
            batch_D = pkpd_carb(batch_D_raw, target_len=T_pts)
            
            pred_S = run_latent_ode(phys_C, phys_I, encoder_g, enc_carb, enc_ins, res_nn, hidden_dyn_nn, 
                                    batch_Y[:, 0, :], batch_U, batch_D, batch_T_seq, batch_Z_snap, t_eval_t, use_nn=True)
            pred_Y = pred_S[..., :1]
            
            total_samples += B
            sum_sq_err_30 += F.mse_loss(pred_Y[:, 5, :], batch_Y[:, 5, :], reduction='sum').item()
            sum_sq_err_60 += F.mse_loss(pred_Y[:, 11, :], batch_Y[:, 11, :], reduction='sum').item()
            sum_sq_err_120 += F.mse_loss(pred_Y[:, 23, :], batch_Y[:, 23, :], reduction='sum').item()
            
            scales = [0.5, 1.0, 1.5]
            V = len(scales)
            
            batch_Y_rep = batch_Y.repeat(V, 1, 1)
            batch_U_rep = batch_U.repeat(V, 1, 1)
            batch_D_rep = batch_D.repeat(V, 1, 1)
            batch_T_seq_rep = batch_T_seq.repeat(V, 1, 1)
            batch_Z_snap_rep = batch_Z_snap.repeat(V, 1, 1)
            
            mask_C = batch_D.sum(dim=(1, 2)) > 1e-4  
            if mask_C.sum() > 0:
                D_cf = torch.cat([batch_D * s for s in scales], dim=0)
                pred_S_cf_C = run_latent_ode(phys_C, phys_I, encoder_g, enc_carb, enc_ins, res_nn, hidden_dyn_nn, 
                                             batch_Y_rep[:, 0, :], batch_U_rep, D_cf, batch_T_seq_rep, batch_Z_snap_rep, t_eval_t, use_nn=True)
                scores_C = pred_S_cf_C[..., 0].mean(dim=1)
                preds_C = scores_C.view(V, B).transpose(0, 1) 
                
                valid_preds_C = preds_C[mask_C]
                ranks_C = torch.argsort(torch.argsort(valid_preds_C, dim=1), dim=1).float() + 1.0
                target_C = torch.full((valid_preds_C.size(0),), 2, dtype=torch.long, device=DEVICE)
                
                loss_C = F.cross_entropy(ranks_C * temp_phi, target_C, reduction='sum')
                causal_loss_C_total += loss_C.item()
                causal_batches_C += valid_preds_C.size(0)
            
            mask_I = batch_U.sum(dim=(1, 2)) > 1e-4
            if mask_I.sum() > 0:
                U_cf = torch.cat([batch_U * s for s in scales], dim=0)
                pred_S_cf_I = run_latent_ode(phys_C, phys_I, encoder_g, enc_carb, enc_ins, res_nn, hidden_dyn_nn, 
                                             batch_Y_rep[:, 0, :], U_cf, batch_D_rep, batch_T_seq_rep, batch_Z_snap_rep, t_eval_t, use_nn=True)
                scores_I = pred_S_cf_I[..., 0].mean(dim=1) 
                preds_I = scores_I.view(V, B).transpose(0, 1) 
                
                valid_preds_I = preds_I[mask_I]
                ranks_I = torch.argsort(torch.argsort(valid_preds_I, dim=1), dim=1).float() + 1.0
                target_I = torch.full((valid_preds_I.size(0),), 0, dtype=torch.long, device=DEVICE)
                
                loss_I = F.cross_entropy(ranks_I * temp_phi, target_I, reduction='sum')
                causal_loss_I_total += loss_I.item()
                causal_batches_I += valid_preds_I.size(0)

    if total_samples == 0: return
    
    print(f"  | {label:15s} | 样本数: {total_samples:4d} | 30min: {np.sqrt(sum_sq_err_30/total_samples)*100:.2f} | 60min: {np.sqrt(sum_sq_err_60/total_samples)*100:.2f} | 120min: {np.sqrt(sum_sq_err_120/total_samples)*100:.2f} |")
    
    if causal_batches_C > 0 and causal_batches_I > 0:
        print(f"  | >> Causal Loss (C): {causal_loss_C_total/causal_batches_C:.4f} | Causal Loss (I): {causal_loss_I_total/causal_batches_I:.4f} <<")

# ==========================================
# 3. 核心架构设计 
# ==========================================
class LearnablePKPDLayer(nn.Module):
    def __init__(self, kernel_size=36, init_tau=6.0, min_tau=2.0, max_tau=24.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.min_tau = min_tau
        self.max_tau = max_tau
        
        sig_val = (init_tau - min_tau) / (max_tau - min_tau)
        sig_val = max(1e-4, min(1.0 - 1e-4, sig_val))
        init_val = np.log(sig_val / (1.0 - sig_val))
        self.raw_tau = nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
        
        self.register_buffer('steps', torch.arange(kernel_size - 1, -1, -1, dtype=torch.float32))

    def forward(self, x, target_len=None):
        if target_len is None: target_len = x.shape[1]
        x_in = x.transpose(1, 2)
        
        tau = self.min_tau + (self.max_tau - self.min_tau) * torch.sigmoid(self.raw_tau)
        kernel_unnorm = (self.steps / tau) * torch.exp(1.0 - (self.steps / tau))
        kernel = (kernel_unnorm / (kernel_unnorm.sum() + 1e-8)).view(1, 1, -1)
        
        x_padded = F.pad(x_in, (self.kernel_size - 1, 0))
        out = F.conv1d(x_padded, kernel, bias=None)
        
        return out[:, :, -target_len:].transpose(1, 2)
    
class AdvancedSmoothResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2) 
        self.norm2 = nn.LayerNorm(dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU() 
        nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.01)
    def forward(self, x):
        return x + 0.1 * self.drop(self.fc2(self.act(self.norm2(self.fc1(self.act(self.norm1(x)))))))

class FNetBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm1 = nn.LayerNorm(hidden_dim); self.norm2 = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        x_fft = torch.fft.fft2(x, dim=(1, 2)).real
        x = self.norm1(x + x_fft)
        return self.norm2(x + self.ffn(x))

class FourierNuisanceEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_blocks):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.fnet_blocks = nn.Sequential(*[FNetBlock(hidden_dim) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(hidden_dim, out_dim)
    def forward(self, snap_seq):
        x = self.fnet_blocks(self.proj(snap_seq))
        return self.fc_out(x.mean(dim=1))

class CarbsEffectNN(nn.Module):
    def __init__(self, hidden_dim=8, mem_dim=MEM_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + mem_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1) 
        )
        nn.init.normal_(self.net[-1].weight, mean=0, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, Y, D_C, C_and_T): 
        out_fac = self.net(torch.cat([Y, D_C, C_and_T], dim=-1))
        out_base = self.net(torch.cat([Y, torch.zeros_like(D_C), C_and_T], dim=-1))
        return out_fac - out_base

class InsulinEffectNN(nn.Module):
    def __init__(self, hidden_dim=8, mem_dim=MEM_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + mem_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1) 
        )
        nn.init.normal_(self.net[-1].weight, mean=0, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, Y, D_I, C_and_T): 
        out_fac = self.net(torch.cat([Y, D_I, C_and_T], dim=-1))
        out_base = self.net(torch.cat([Y, torch.zeros_like(D_I), C_and_T], dim=-1))
        return out_fac - out_base

class ResidualNN(nn.Module):
    def __init__(self, in_dim=C_DIM+2, out_dim=1, hidden_dim=256, num_blocks=4):
        super().__init__()
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.res_blocks = nn.Sequential(*[AdvancedSmoothResBlock(hidden_dim) for _ in range(num_blocks)])
        self.proj_out = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.proj_out.weight); nn.init.zeros_(self.proj_out.bias)
    def forward(self, h_and_t): return self.proj_out(self.res_blocks(F.silu(self.proj_in(h_and_t))))

# 🌟 核心修改 1: 将隐藏状态网络的输入扩展，重新允许观察 u_t 和 d_t
class HiddenDynamicsNN(nn.Module):
    # 去掉 z_dim
    def __init__(self, h_dim=C_DIM, y_dim=1, hidden_dim=128, num_blocks=4):
        super().__init__()
        self.proj_in = nn.Linear(h_dim + y_dim, hidden_dim) # 不再拼接 z_dim
        self.res_blocks = nn.Sequential(*[AdvancedSmoothResBlock(hidden_dim) for _ in range(num_blocks)])
        self.proj_out = nn.Linear(hidden_dim, h_dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        
    def forward(self, h, Y): # 删掉 u_t, d_t 输入
        return -0.1 * h + self.proj_out(self.res_blocks(F.silu(self.proj_in(torch.cat([h, Y], dim=-1)))))

class NuisanceNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=512, num_blocks=4, dropout=0.2):
        super().__init__()
        self.proj_in = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.res_blocks = nn.Sequential(*[AdvancedSmoothResBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.proj_out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim))
        nn.init.zeros_(self.proj_out[-1].weight); nn.init.zeros_(self.proj_out[-1].bias)
    def forward(self, Z_dyn): return self.proj_out(self.res_blocks(self.proj_in(Z_dyn)))

class BatchedLinearInterpolator(nn.Module):
    def __init__(self, t_eval, x_seq, dev):
        super().__init__()
        self.t_eval = t_eval
        self.x_seq = x_seq
        dt = (self.t_eval[1] - self.t_eval[0]).item() if len(self.t_eval) > 1 else 1.0
        self.dt = dt if dt > 0 else 1e-8
        
    def forward(self, t):
        idx0 = torch.clamp((t / self.dt).long(), 0, self.x_seq.shape[1] - 2)
        t0, t1 = self.t_eval[idx0], self.t_eval[idx0+1]
        return self.x_seq[:, idx0] + (t - t0) / (t1 - t0 + 1e-8) * (self.x_seq[:, idx0+1] - self.x_seq[:, idx0])

class HybridAdditiveODEFunc(nn.Module):
    def __init__(self, phys_C, phys_I, res_nn, hidden_dyn_nn, U_intp, D_intp, T_intp, h0_carb, h0_ins, use_nn=True):
        super().__init__()
        self.phys_C = phys_C; self.phys_I = phys_I
        self.res_nn = res_nn; self.hidden_dyn = hidden_dyn_nn  
        self.U_interp = U_intp; self.D_interp = D_intp; self.T_interp = T_intp
        self.h0_carb = h0_carb; self.h0_ins = h0_ins; self.use_nn = use_nn
        
    def forward(self, t, S):
        Y, h_g = S[..., 0:1], S[..., 1:]
        u_t, d_t, time_t = self.U_interp(t), self.D_interp(t), self.T_interp(t)
        
        C_and_T_carb = torch.cat([self.h0_carb, time_t], dim=-1)
        C_and_T_ins = torch.cat([self.h0_ins, time_t], dim=-1)
        
        dy_phys = self.phys_C(Y, d_t, C_and_T_carb) + self.phys_I(Y, u_t, C_and_T_ins)
        
        # 🌟 彻底蒙上黑盒的眼睛，它现在只能看到 h_g 和 Y
        dh_g_dt = self.hidden_dyn(h_g, Y) 
        
        dy_dt = dy_phys + self.res_nn(torch.cat([h_g, time_t], dim=-1)) if self.use_nn else dy_phys
        return torch.cat([dy_dt, dh_g_dt], dim=-1)

def run_latent_ode(phys_C, phys_I, enc_g, enc_carb, enc_ins, res_nn, hidden_dyn_nn, Y0, U_seq, D_seq, T_seq, Z_snap, t_eval, use_nn=True):
    h0_g = enc_g(Z_snap) 
    h0_carb = enc_carb(Z_snap[..., [0, 2]]) 
    h0_ins = enc_ins(Z_snap[..., [0, 1]])
    
    ode_func = HybridAdditiveODEFunc(phys_C, phys_I, res_nn, hidden_dyn_nn, 
                             BatchedLinearInterpolator(t_eval, U_seq, DEVICE), 
                             BatchedLinearInterpolator(t_eval, D_seq, DEVICE), 
                             BatchedLinearInterpolator(t_eval, T_seq, DEVICE), 
                             h0_carb, h0_ins, use_nn=use_nn)
    return odeint(ode_func, torch.cat([Y0, h0_g], dim=-1), t_eval, method='rk4', options={'step_size': DT_MINUTES/60.0}).transpose(0, 1)

# ==========================================
# 4. DMLEngine 计算引擎
# ==========================================
class DMLEngine:
    def __init__(self, dt): self.dt = dt
    def compute_integrals(self, phys_nn, Y_batch, single_D_batch, C_batch):
        was_training = phys_nn.training
        phys_nn.eval()
        N, T, Dim = Y_batch.shape
        params = dict(phys_nn.named_parameters())
        
        def f_fn(p, y, d_single, c_and_t): return func.functional_call(phys_nn, p, (y, d_single, c_and_t))
        c_dim_actual = C_batch.shape[-1]
        
        f_vals = f_fn(params, Y_batch.reshape(-1, Dim), single_D_batch.reshape(-1, Dim), C_batch.reshape(-1, c_dim_actual)).reshape(N, T, Dim)
        F_int = (f_vals[:, :-1, :] + f_vals[:, 1:, :]) / 2.0 * self.dt
        
        jac_dict = func.vmap(func.jacrev(f_fn, argnums=0), in_dims=(None, 0, 0, 0), randomness='different')(params, Y_batch.reshape(-1, Dim), single_D_batch.reshape(-1, Dim), C_batch.reshape(-1, c_dim_actual))
        J_all = torch.cat([jac.reshape(N, T, Dim, -1) for jac in jac_dict.values()], dim=-1)
        J_int = (J_all[:, :-1, :, :] + J_all[:, 1:, :, :]) / 2.0 * self.dt
        
        if was_training: phys_nn.train()
        return F_int, J_int, J_all.shape[-1]

# ==========================================
# 5. 核心训练管线 
# ==========================================
def train_3step_alternating_loop(dl_f1, dl_f2, dl_tr_all, dl_te_all, phys_C, phys_I, 
                                 enc_g, enc_carb, enc_ins, res_nn, hidden_dyn_nn, pkpd_carb, pkpd_ins,
                                 n_q1_C, n_H1_C, n_q2_C, n_H2_C, n_q1_I, n_H1_I, n_q2_I, n_H2_I, 
                                 dml_engine, T_pts, Dim, P_C_total, P_I_total):
    
    opt_f_params = list(phys_C.parameters()) + list(phys_I.parameters()) + list(enc_carb.parameters()) + list(enc_ins.parameters())
    if not FREEZE_PKPD:
        opt_f_params += list(pkpd_carb.parameters()) + list(pkpd_ins.parameters())
    opt_f = torch.optim.Adam(opt_f_params, lr=1e-3, weight_decay=1e-4)
    
    opt_r = torch.optim.Adam(list(res_nn.parameters()) + list(enc_g.parameters()) + list(hidden_dyn_nn.parameters()), lr=3e-3, weight_decay=1e-4)
    
    if USE_DML:
        opt_n = torch.optim.Adam(
            list(n_q1_C.parameters()) + list(n_H1_C.parameters()) + list(n_q2_C.parameters()) + list(n_H2_C.parameters()) +
            list(n_q1_I.parameters()) + list(n_H1_I.parameters()) + list(n_q2_I.parameters()) + list(n_H2_I.parameters()), 
            lr=3e-3
        )
    
    t_eval_t = torch.tensor(np.arange(0, T_pts * (DT_MINUTES/60.0), (DT_MINUTES/60.0)), dtype=torch.float32, device=DEVICE)
    
    def cache_dynamic_targets(dl):
        cached_C, cached_I = [], []
        with torch.no_grad():
            for batch_Y, batch_U_raw, batch_D_raw, batch_Z_snap, batch_T_seq in dl:
                batch_Y, batch_U_raw, batch_D_raw, batch_Z_snap, batch_T_seq = [x.to(DEVICE) for x in (batch_Y, batch_U_raw, batch_D_raw, batch_Z_snap, batch_T_seq)]
                
                batch_U = pkpd_ins(batch_U_raw, target_len=T_pts)
                batch_D = pkpd_carb(batch_D_raw, target_len=T_pts)
                
                h0_carb = enc_carb(batch_Z_snap[..., [0, 2]]).unsqueeze(1).expand(-1, T_pts, -1)
                h0_ins = enc_ins(batch_Z_snap[..., [0, 1]]).unsqueeze(1).expand(-1, T_pts, -1)
                
                C_and_T_carb = torch.cat([h0_carb, batch_T_seq], dim=-1)
                C_and_T_ins = torch.cat([h0_ins, batch_T_seq], dim=-1)
                
                pred_S = run_latent_ode(phys_C, phys_I, enc_g, enc_carb, enc_ins, res_nn, hidden_dyn_nn, batch_Y[:, 0, :], batch_U, batch_D, batch_T_seq, batch_Z_snap, t_eval_t, use_nn=True)
                h_g_seq = pred_S[..., 1:] 
                
                F_C, J_C, _ = dml_engine.compute_integrals(phys_C, batch_Y, batch_D, C_and_T_carb)
                F_I, J_I, _ = dml_engine.compute_integrals(phys_I, batch_Y, batch_U, C_and_T_ins)
                
                Z_C = torch.cat([h_g_seq[:, :-1, :], h0_ins[:, :-1, :], batch_Y[:, :-1, :], batch_U[:, :-1, :], batch_T_seq[:, :-1, :]], dim=-1).reshape(-1, C_DIM + MEM_DIM + 4)
                Z_I = torch.cat([h_g_seq[:, :-1, :], h0_carb[:, :-1, :], batch_Y[:, :-1, :], batch_D[:, :-1, :], batch_T_seq[:, :-1, :]], dim=-1).reshape(-1, C_DIM + MEM_DIM + 4)
                
                R_C = (batch_Y[:, 1:, :] - batch_Y[:, :-1, :] - F_C).reshape(-1, Dim)
                R_I = (batch_Y[:, 1:, :] - batch_Y[:, :-1, :] - F_I).reshape(-1, Dim)
                
                cached_C.append((Z_C.cpu(), R_C.cpu(), J_C.reshape(-1, Dim * P_C_total).cpu()))
                cached_I.append((Z_I.cpu(), R_I.cpu(), J_I.reshape(-1, Dim * P_I_total).cpu()))
        return cached_C, cached_I

    for outer_ep in range(OUTER_EPOCHS):
        print(f"\n--- Outer Epoch [{outer_ep+1:02d}/{OUTER_EPOCHS}] ---")
        
        with torch.no_grad():
            tau_c = pkpd_carb.min_tau + (pkpd_carb.max_tau - pkpd_carb.min_tau) * torch.sigmoid(pkpd_carb.raw_tau).item()
            tau_i = pkpd_ins.min_tau + (pkpd_ins.max_tau - pkpd_ins.min_tau) * torch.sigmoid(pkpd_ins.raw_tau).item()
        print(f"    [Tau Tracker] 碳水 Tau: {tau_c:.3f} | 胰岛素 Tau: {tau_i:.3f}")

        target_causal_ratio = 1 
        
        # 1. N-Step
        if USE_DML:
            for m in [phys_C, phys_I, enc_g, enc_carb, enc_ins, res_nn, hidden_dyn_nn, pkpd_carb, pkpd_ins]:
                m.eval()
                for p in m.parameters(): p.requires_grad = False
                
            for m in [n_q1_C, n_H1_C, n_q2_C, n_H2_C, n_q1_I, n_H1_I, n_q2_I, n_H2_I]:
                m.train()
                for p in m.parameters(): p.requires_grad = True
                
            c_f1, i_f1 = cache_dynamic_targets(dl_f1)
            c_f2, i_f2 = cache_dynamic_targets(dl_f2)
            
            for _ in range(INNER_N_EPOCHS):   
                for (Z_c, R_c, J_c), (Z_i, R_i, J_i) in zip(c_f1, i_f1):
                    Z_c, R_c, J_c = Z_c.to(DEVICE), R_c.to(DEVICE), J_c.to(DEVICE)
                    Z_i, R_i, J_i = Z_i.to(DEVICE), R_i.to(DEVICE), J_i.to(DEVICE)
                    
                    opt_n.zero_grad()
                    loss_f1 = F.mse_loss(n_q1_C(Z_c), R_c) + F.mse_loss(n_H1_C(Z_c), J_c)
                    loss_f1 += F.mse_loss(n_q1_I(Z_i), R_i) + F.mse_loss(n_H1_I(Z_i), J_i)
                    loss_f1.backward()
                    opt_n.step()
                    
                for (Z_c, R_c, J_c), (Z_i, R_i, J_i) in zip(c_f2, i_f2):
                    Z_c, R_c, J_c = Z_c.to(DEVICE), R_c.to(DEVICE), J_c.to(DEVICE)
                    Z_i, R_i, J_i = Z_i.to(DEVICE), R_i.to(DEVICE), J_i.to(DEVICE)
                    
                    opt_n.zero_grad()
                    loss_f2 = F.mse_loss(n_q2_C(Z_c), R_c) + F.mse_loss(n_H2_C(Z_c), J_c)
                    loss_f2 += F.mse_loss(n_q2_I(Z_i), R_i) + F.mse_loss(n_H2_I(Z_i), J_i)
                    loss_f2.backward()
                    opt_n.step()
                
        # 2. M-Step
        m_step_train_modules = [phys_C, phys_I, enc_carb, enc_ins]
        if not FREEZE_PKPD:
            m_step_train_modules += [pkpd_carb, pkpd_ins]
            
        for m in m_step_train_modules:
            m.train()
            for p in m.parameters(): p.requires_grad = True
            
        for m in [enc_g, res_nn, hidden_dyn_nn]:
            m.eval()
            for p in m.parameters(): p.requires_grad = False
            
        if USE_DML:
            for m in [n_q1_C, n_H1_C, n_q2_C, n_H2_C, n_q1_I, n_H1_I, n_q2_I, n_H2_I]:
                m.eval()
                for p in m.parameters(): p.requires_grad = False
                
        ep_f_mse, ep_score_C, ep_score_I, batches_f, ep_dot_product = 0.0, 0.0, 0.0, 0, 0
        for fold_name, dl, q_C_oos, H_C_oos, q_I_oos, H_I_oos in [("Fold 1", dl_f1, n_q2_C, n_H2_C, n_q2_I, n_H2_I), ("Fold 2", dl_f2, n_q1_C, n_H1_C, n_q1_I, n_H1_I)]:
            for batch_Y, batch_U_raw, batch_D_raw, batch_Z_snap, batch_T_seq in tqdm(dl, desc=f"M-Step {fold_name}", leave=False):
                batch_Y, batch_U_raw, batch_D_raw, batch_Z_snap, batch_T_seq = [x.to(DEVICE) for x in (batch_Y, batch_U_raw, batch_D_raw, batch_Z_snap, batch_T_seq)]
                
                batch_U = pkpd_ins(batch_U_raw, target_len=T_pts)
                batch_D = pkpd_carb(batch_D_raw, target_len=T_pts)
                
                opt_f.zero_grad()
                pred_S_cl = run_latent_ode(phys_C, phys_I, enc_g, enc_carb, enc_ins, res_nn, hidden_dyn_nn, batch_Y[:, 0, :], batch_U, batch_D, batch_T_seq, batch_Z_snap, t_eval_t, use_nn=True)
                loss_mse = F.mse_loss(pred_S_cl[..., 0:1], batch_Y)
                
                if USE_MSE: loss_mse.backward()
                else: (0.0 * sum([p.sum() for p in phys_C.parameters()] + [p.sum() for p in phys_I.parameters()])).backward()
                
                if USE_DML:
                    with torch.no_grad():
                        h0_carb = enc_carb(batch_Z_snap[..., [0, 2]]).unsqueeze(1).expand(-1, T_pts, -1)
                        h0_ins = enc_ins(batch_Z_snap[..., [0, 1]]).unsqueeze(1).expand(-1, T_pts, -1)
                        
                        F_C, J_C, _ = dml_engine.compute_integrals(phys_C, batch_Y, batch_D, torch.cat([h0_carb, batch_T_seq], dim=-1))
                        F_I, J_I, _ = dml_engine.compute_integrals(phys_I, batch_Y, batch_U, torch.cat([h0_ins, batch_T_seq], dim=-1))
                        
                        Z_C = torch.cat([pred_S_cl[:, :-1, 1:], h0_ins[:, :-1, :], batch_Y[:, :-1, :], batch_U[:, :-1, :], batch_T_seq[:, :-1, :]], dim=-1).reshape(-1, C_DIM + MEM_DIM + 4)
                        Z_I = torch.cat([pred_S_cl[:, :-1, 1:], h0_carb[:, :-1, :], batch_Y[:, :-1, :], batch_D[:, :-1, :], batch_T_seq[:, :-1, :]], dim=-1).reshape(-1, C_DIM + MEM_DIM + 4)
                        
                    # 计算 Psi_C 和 Psi_I (保持你的原代码不变)
                    Psi_C = torch.einsum('ntdp,ntd->p', (J_C - H_C_oos(Z_C).reshape(-1, T_pts-1, Dim, P_C_total)), (batch_Y[:, 1:, :] - batch_Y[:, :-1, :] - F_C - q_C_oos(Z_C).reshape(-1, T_pts-1, Dim))) / (batch_Y.size(0) * (T_pts-1))
                    Psi_I = torch.einsum('ntdp,ntd->p', (J_I - H_I_oos(Z_I).reshape(-1, T_pts-1, Dim, P_I_total)), (batch_Y[:, 1:, :] - batch_Y[:, :-1, :] - F_I - q_I_oos(Z_I).reshape(-1, T_pts-1, Dim))) / (batch_Y.size(0) * (T_pts-1))
                    
                    if target_causal_ratio > 0:
                        # ==========================================
                        # 1. 碳水网络 (Carbs) 的因果梯度投影
                        # ==========================================
                        scaled_Psi_C = -Psi_C * target_causal_ratio
                        pointer = 0
                        for param in phys_C.parameters():
                            num_p = param.numel()
                            g_dml = scaled_Psi_C[pointer : pointer + num_p].view_as(param)
                            
                            if param.grad is not None:
                                g_mse = param.grad.clone()
                                
                                # 🌟 Causal PCGrad 核心逻辑 🌟
                                # 检查 MSE 梯度是否在和 DML 梯度作对
                                dot_product = torch.sum(g_mse * g_dml) 
                                ep_dot_product += (dot_product / (torch.norm(g_mse) * torch.norm(g_dml) + 1e-8)).item()
                                if dot_product < 0:
                                    # 如果在作对，抹除 MSE 在 DML 逆方向上的分量
                                    g_mse = g_mse - (dot_product / (torch.sum(g_dml * g_dml) + 1e-8)) * g_dml
                                
                                # 范数对齐：保证 DML 有足够的驱动力
                                norm_mse = torch.norm(g_mse) + 1e-8
                                norm_dml = torch.norm(g_dml) + 1e-8
                                g_dml_aligned = g_dml * (norm_mse / norm_dml)
                                
                                # 最终注入：此时的 g_mse 已经绝对服从 g_dml 的方向
                                param.grad = g_mse + g_dml_aligned 
                                
                            pointer += num_p

                        # ==========================================
                        # 2. 胰岛素网络 (Insulin) 的因果梯度投影
                        # ==========================================
                        scaled_Psi_I = -Psi_I * target_causal_ratio 
                        pointer = 0
                        for param in phys_I.parameters():
                            num_p = param.numel()
                            g_dml = scaled_Psi_I[pointer : pointer + num_p].view_as(param)
                            
                            if param.grad is not None:
                                g_mse = param.grad.clone()
                                
                                # 🌟 Causal PCGrad 核心逻辑 🌟
                                dot_product = torch.sum(g_mse * g_dml) 
                                ep_dot_product += (dot_product / (torch.norm(g_mse) * torch.norm(g_dml) + 1e-8)).item()
                                if dot_product < 0:
                                    g_mse = g_mse - (dot_product / (torch.sum(g_dml * g_dml) + 1e-8)) * g_dml
                                
                                norm_mse = torch.norm(g_mse) + 1e-8
                                norm_dml = torch.norm(g_dml) + 1e-8
                                g_dml_aligned = g_dml * (norm_mse / norm_dml)
                                
                                param.grad = g_mse + g_dml_aligned
                                
                            pointer += num_p
                            
                    ep_score_C += torch.mean(torch.abs(Psi_C)).item() 
                    ep_score_I += torch.mean(torch.abs(Psi_I)).item()

                torch.nn.utils.clip_grad_norm_(phys_C.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(phys_I.parameters(), max_norm=1.0)
                opt_f.step()
                ep_f_mse += loss_mse.item(); batches_f += 1
                
        print(f"    [M-Step] MSE: {ep_f_mse/batches_f:.4f} | True Grad (C): {ep_score_C/batches_f:.2e} | True Grad (I): {ep_score_I/batches_f:.2e} | Causal Dot Product: {ep_dot_product/batches_f:.2e}")

        # 3. R-Step
        for m in [enc_g, res_nn, hidden_dyn_nn]:
            m.train()
            for p in m.parameters(): p.requires_grad = True
            
        for m in [phys_C, phys_I, enc_carb, enc_ins, pkpd_carb, pkpd_ins]:
            m.eval()
            for p in m.parameters(): p.requires_grad = False
            
        if USE_DML:
            for m in [n_q1_C, n_H1_C, n_q2_C, n_H2_C, n_q1_I, n_H1_I, n_q2_I, n_H2_I]:
                m.eval()
                for p in m.parameters(): p.requires_grad = False
                
        ep_r_mse, batches_r = 0.0, 0
        for dl in [dl_f1, dl_f2]:
            for batch_Y, batch_U_raw, batch_D_raw, batch_Z_snap, batch_T_seq in tqdm(dl, desc="R-Step", leave=False):
                batch_Y, batch_U_raw, batch_D_raw, batch_Z_snap, batch_T_seq = [x.to(DEVICE) for x in (batch_Y, batch_U_raw, batch_D_raw, batch_Z_snap, batch_T_seq)]
                
                batch_U = pkpd_ins(batch_U_raw, target_len=T_pts)
                batch_D = pkpd_carb(batch_D_raw, target_len=T_pts)
                
                opt_r.zero_grad()
                pred_S = run_latent_ode(phys_C, phys_I, enc_g, enc_carb, enc_ins, res_nn, hidden_dyn_nn, batch_Y[:, 0, :], batch_U, batch_D, batch_T_seq, batch_Z_snap, t_eval_t, use_nn=True)
                loss_r = F.mse_loss(pred_S[..., 0:1], batch_Y)
                loss_r.backward()
                opt_r.step()
                ep_r_mse += loss_r.item(); batches_r += 1
        print(f"    [R-Step] Avg MSE: {ep_r_mse/batches_r:.4f}")

        # evaluate_final_test_rmse(phys_C, phys_I, enc_g, enc_carb, enc_ins, res_nn, hidden_dyn_nn, 
        #                          pkpd_carb, pkpd_ins, dl_te_all, t_eval_t, label=f"Epoch {outer_ep+1} 20% TEST", subset_ratio=0.2)

    print("\n================================================")
    print(">>> 训练结束，开始执行全量测试集终极评估 (100%) <<<")
    print("================================================")
    evaluate_final_test_rmse(phys_C, phys_I, enc_g, enc_carb, enc_ins, res_nn, hidden_dyn_nn, 
                             pkpd_carb, pkpd_ins, dl_te_all, t_eval_t, label="FINAL GLOBAL TEST", subset_ratio=1.0)

    return t_eval_t


def plot_rolling_forecast(phys_C, phys_I, enc_g, enc_carb, enc_ins, res_nn, hidden_dyn_nn, pkpd_carb, pkpd_ins, Y_te, U_te, D_te, Z_snap_te, T_seq_te, t_eval_t, dt_minutes, t_pts, device, patient_id, save_dir):
    if Y_te.shape[0] == 0: return
    
    for m in [phys_C, phys_I, enc_g, enc_carb, enc_ins, res_nn, hidden_dyn_nn, pkpd_carb, pkpd_ins]:
        m.eval()
        for p in m.parameters(): p.requires_grad = False
    
    num_windows_to_plot = min(3000, Y_te.shape[0])
    plot_step = 12 
    
    gt_continuous = [Y_te[i, 0, 0] * 100 for i in range(num_windows_to_plot)]
    gt_continuous.extend(Y_te[num_windows_to_plot-1, 1:, 0] * 100)
    
    with torch.no_grad():
        U_smooth_all = pkpd_ins(torch.tensor(U_te, dtype=torch.float32, device=device), target_len=t_pts).cpu().numpy()
        D_smooth_all = pkpd_carb(torch.tensor(D_te, dtype=torch.float32, device=device), target_len=t_pts).cpu().numpy()
        
    U_continuous = [U_smooth_all[i, 0, 0] for i in range(num_windows_to_plot)]
    U_continuous.extend(U_smooth_all[num_windows_to_plot-1, 1:, 0])
    
    D_continuous = [D_smooth_all[i, 0, 0] for i in range(num_windows_to_plot)]
    D_continuous.extend(D_smooth_all[num_windows_to_plot-1, 1:, 0])
    
    t_abs = np.arange(len(gt_continuous)) * (dt_minutes / 60.0)

    fig, axes = plt.subplots(3, 1, figsize=(24, 20), sharex=True, gridspec_kw={'height_ratios': [2.5, 1, 1.5]})
    ax_cgm, ax_ud, ax_effects = axes[0], axes[1], axes[2]
    
    ax_cgm.plot(t_abs, gt_continuous, 'k-', lw=3, label='Ground Truth (CGM)', alpha=0.6)
    ax_cgm.set_title(f'Patient {patient_id} - Pure 2D Blackbox Causal ODE Forecast', fontweight='bold', fontsize=18)
    ax_cgm.set_ylabel('Glucose (mg/dL)', fontsize=14)
    ax_cgm.axhspan(70, 180, color='green', alpha=0.1, label='Target Range')

    ax_ud.plot(t_abs, U_continuous, color='blue', lw=2, label='Insulin (1D-CNN Smoothed)', alpha=0.7)
    ax_ud_twin = ax_ud.twinx()
    ax_ud_twin.plot(t_abs, D_continuous, color='orange', lw=2, label='Carbs (1D-CNN Smoothed)', alpha=0.7)
    ax_ud.set_ylabel('Insulin Dose', fontsize=14, color='blue')
    ax_ud_twin.set_ylabel('Carbs (g)', fontsize=14, color='orange')
    ax_ud.set_title('Intervention Signals', fontsize=14)
    
    ax_effects.set_ylabel('Causal Pull (mg/dL per 5min)', fontsize=14)
    ax_effects.set_xlabel('Absolute Time (Hours)', fontsize=14)
    ax_effects.set_title(f'Decoupled Causal Effects (Truth Recovered by DML) - Pat {patient_id}', fontsize=14)

    with torch.no_grad():
        for i in range(0, num_windows_to_plot, plot_step):
            Y0 = torch.tensor(Y_te[i:i+1, 0, :], dtype=torch.float32, device=device)
            U_batch = torch.tensor(U_te[i:i+1], dtype=torch.float32, device=device)
            D_batch = torch.tensor(D_te[i:i+1], dtype=torch.float32, device=device)
            Z_snap_batch = torch.tensor(Z_snap_te[i:i+1], dtype=torch.float32, device=device)
            T_seq_batch = torch.tensor(T_seq_te[i:i+1], dtype=torch.float32, device=device)
            
            U_batch = pkpd_ins(U_batch, target_len=t_pts)
            D_batch = pkpd_carb(D_batch, target_len=t_pts)
            
            pred_S_hybrid = run_latent_ode(phys_C, phys_I, enc_g, enc_carb, enc_ins, res_nn, hidden_dyn_nn, Y0, U_batch, D_batch, T_seq_batch, Z_snap_batch, t_eval_t, use_nn=True)
            pred_hybrid = pred_S_hybrid[..., :1].cpu().numpy()
            
            t_pred_axis = t_abs[i : i + t_pts]
            ax_cgm.plot(t_pred_axis, pred_hybrid[0, :, 0] * 100, '-', color='red', lw=2.5, alpha=0.85, label='Hybrid NODE' if i==0 else "")
            
            h0_carb = enc_carb(Z_snap_batch[..., [0, 2]]).unsqueeze(1).expand(-1, t_pts, -1)
            h0_ins = enc_ins(Z_snap_batch[..., [0, 1]]).unsqueeze(1).expand(-1, t_pts, -1)
            
            pull_C = phys_C(pred_S_hybrid[..., :1].to(DEVICE), D_batch, torch.cat([h0_carb, T_seq_batch], dim=-1)).cpu().numpy()[0, :, 0] * 100
            pull_I = phys_I(pred_S_hybrid[..., :1].to(DEVICE), U_batch, torch.cat([h0_ins, T_seq_batch], dim=-1)).cpu().numpy()[0, :, 0] * 100
            ax_effects.plot(t_pred_axis, pull_C, '-', color='orange', lw=2, alpha=0.7, label='Carbs Pull' if i==0 else "")
            ax_effects.plot(t_pred_axis, pull_I, '-', color='blue', lw=2, alpha=0.7, label='Insulin Pull' if i==0 else "")

    ax_cgm.legend(loc='upper right', fontsize=12); ax_ud.legend(loc='upper left', fontsize=12); ax_ud_twin.legend(loc='upper right', fontsize=12); ax_effects.legend(loc='upper right', fontsize=12)
    rolling_save_path = os.path.join(save_dir, f'debug_mechanistic_P{patient_id}.png')
    plt.savefig(rolling_save_path, dpi=300, bbox_inches='tight'); plt.close()

def main():
    set_seed(42)
    
    print(">>> 1. 挂载 Global 数据并执行 [跨病人] 2-Fold 分割...")
    
    mid_idx = len(PATIENT_IDS) // 2
    f1_pids = PATIENT_IDS[:mid_idx]
    f2_pids = PATIENT_IDS[mid_idx:]
    print(f"    * Fold 1 包含病人: {f1_pids}")
    print(f"    * Fold 2 包含病人: {f2_pids}")
    
    Y_f1, U_f1, D_f1, Z_f1, T_f1 = build_patient_dataset(f1_pids, 'train', stride=20)
    Y_f2, U_f2, D_f2, Z_f2, T_f2 = build_patient_dataset(f2_pids, 'train', stride=20)
    
    if len(Y_f1) == 0 or len(Y_f2) == 0:
        print("错误：无法加载所有病人数据，请检查数据集路径。")
        return

    ds_f1 = TensorDataset(*(torch.tensor(x, dtype=torch.float32) for x in [Y_f1, U_f1, D_f1, Z_f1, T_f1]))
    dl_f1 = DataLoader(ds_f1, batch_size=BATCH_SIZE, shuffle=True)
    
    ds_f2 = TensorDataset(*(torch.tensor(x, dtype=torch.float32) for x in [Y_f2, U_f2, D_f2, Z_f2, T_f2]))
    dl_f2 = DataLoader(ds_f2, batch_size=BATCH_SIZE, shuffle=True)

    Y_tr_all = np.vstack([Y_f1, Y_f2])
    U_tr_all = np.vstack([U_f1, U_f2])
    D_tr_all = np.vstack([D_f1, D_f2])
    Z_tr_all = np.vstack([Z_f1, Z_f2])
    T_tr_all = np.vstack([T_f1, T_f2])
    dl_tr_all = DataLoader(TensorDataset(*(torch.tensor(x, dtype=torch.float32) for x in [Y_tr_all, U_tr_all, D_tr_all, Z_tr_all, T_tr_all])), batch_size=BATCH_SIZE, shuffle=False)
    
    Y_te_all, U_te_all, D_te_all, Z_te_all, T_te_all = build_patient_dataset(PATIENT_IDS, 'test', stride=1)
    dl_te_all = DataLoader(TensorDataset(*(torch.tensor(x, dtype=torch.float32) for x in [Y_te_all, U_te_all, D_te_all, Z_te_all, T_te_all])), batch_size=BATCH_SIZE, shuffle=False)

    T_pts = Y_tr_all.shape[1]
    Dim = Y_tr_all.shape[2]
    
    print("\n>>> 2. 初始化网络...")
    encoder_g = FourierNuisanceEncoder(in_dim=5, hidden_dim=128, out_dim=C_DIM, num_blocks=4).to(DEVICE)
    encoder_carb = FourierNuisanceEncoder(in_dim=2, hidden_dim=64, out_dim=MEM_DIM, num_blocks=2).to(DEVICE)
    encoder_ins = FourierNuisanceEncoder(in_dim=2, hidden_dim=64, out_dim=MEM_DIM, num_blocks=2).to(DEVICE)
    
    pkpd_carb = LearnablePKPDLayer(kernel_size=36, init_tau=4.0, min_tau=2.0, max_tau=8.0).to(DEVICE)
    pkpd_ins = LearnablePKPDLayer(kernel_size=48, init_tau=12.0, min_tau=8.0, max_tau=24.0).to(DEVICE)
    
    phys_C = CarbsEffectNN(hidden_dim=8, mem_dim=MEM_DIM).to(DEVICE)
    phys_I = InsulinEffectNN(hidden_dim=8, mem_dim=MEM_DIM).to(DEVICE)
    
    res_nn = ResidualNN(in_dim=C_DIM+2, out_dim=1, hidden_dim=256, num_blocks=6).to(DEVICE)
    hidden_dyn_nn = HiddenDynamicsNN(h_dim=C_DIM, y_dim=1, hidden_dim=128, num_blocks=4).to(DEVICE)
    
    dml_engine = DMLEngine(dt=DT_MINUTES / 60.0) 
    _, _, P_C_total = dml_engine.compute_integrals(phys_C, torch.zeros(2, T_pts, 1).to(DEVICE), torch.zeros(2, T_pts, 1).to(DEVICE), torch.zeros(2, T_pts, MEM_DIM + 2).to(DEVICE))
    _, _, P_I_total = dml_engine.compute_integrals(phys_I, torch.zeros(2, T_pts, 1).to(DEVICE), torch.zeros(2, T_pts, 1).to(DEVICE), torch.zeros(2, T_pts, MEM_DIM + 2).to(DEVICE))

    nuisance_in_dim = C_DIM + MEM_DIM + 4 
    n_q1_C, n_H1_C = NuisanceNN(nuisance_in_dim, Dim).to(DEVICE), NuisanceNN(nuisance_in_dim, Dim * P_C_total).to(DEVICE)
    n_q2_C, n_H2_C = NuisanceNN(nuisance_in_dim, Dim).to(DEVICE), NuisanceNN(nuisance_in_dim, Dim * P_C_total).to(DEVICE)
    n_q1_I, n_H1_I = NuisanceNN(nuisance_in_dim, Dim).to(DEVICE), NuisanceNN(nuisance_in_dim, Dim * P_I_total).to(DEVICE)
    n_q2_I, n_H2_I = NuisanceNN(nuisance_in_dim, Dim).to(DEVICE), NuisanceNN(nuisance_in_dim, Dim * P_I_total).to(DEVICE)

    t_eval_t = train_3step_alternating_loop(dl_f1, dl_f2, dl_tr_all, dl_te_all, phys_C, phys_I, encoder_g, encoder_carb, encoder_ins, res_nn, hidden_dyn_nn, pkpd_carb, pkpd_ins, n_q1_C, n_H1_C, n_q2_C, n_H2_C, n_q1_I, n_H1_I, n_q2_I, n_H2_I, dml_engine, T_pts, Dim, P_C_total, P_I_total)
    
    save_dir = os.path.join(os.getcwd(), 'image', f'global_run_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n  --> 所有病人的评估图表将保存在: {save_dir}")
    
    for pid in PATIENT_IDS:
        Y_te_p, U_te_p, D_te_p, Z_te_p, T_te_p = build_patient_dataset([pid], 'test', stride=1)
        if len(Y_te_p) > 0:
            print(f"      正在绘制病人 {pid} 的预测验证图...")
            plot_rolling_forecast(phys_C, phys_I, encoder_g, encoder_carb, encoder_ins, res_nn, hidden_dyn_nn, pkpd_carb, pkpd_ins, Y_te_p, U_te_p, D_te_p, Z_te_p, T_te_p, t_eval_t, DT_MINUTES, T_pts, DEVICE, pid, save_dir)
            
    print("\n✅ Global 跨病人实验结束。")

if __name__ == "__main__":
    main()

"""
random seed 43
(my_env) root@cpod-1sc7ubm9xiqn:~/code# python /root/code/success.py
🚀 Global 跨病人启动 | 包含 6 名患者 | MSE: True | DML: True | 冻结PKPD: False | Device: cuda
>>> 1. 挂载 Global 数据并执行 [跨病人] 2-Fold 分割...
    * Fold 1 包含病人: ['559', '563', '570']
    * Fold 2 包含病人: ['575', '588', '591']

>>> 2. 初始化网络...

--- Outer Epoch [01/8] ---
    [Tau Tracker] 碳水 Tau: 4.000 | 胰岛素 Tau: 12.000
    [M-Step] MSE: 0.1538 | True Grad (C): 7.29e-04 | True Grad (I): 2.77e-04 | Causal Dot Product: 7.60e-01                                           
    [R-Step] Avg MSE: 0.1493                                                                                                                          

--- Outer Epoch [02/8] ---
    [Tau Tracker] 碳水 Tau: 4.052 | 胰岛素 Tau: 12.159
    [M-Step] MSE: 0.1287 | True Grad (C): 1.40e-04 | True Grad (I): 6.50e-04 | Causal Dot Product: 7.53e-01                                           
    [R-Step] Avg MSE: 0.1230                                                                                                                          

--- Outer Epoch [03/8] ---
    [Tau Tracker] 碳水 Tau: 4.092 | 胰岛素 Tau: 12.278
    [M-Step] MSE: 0.1199 | True Grad (C): 5.81e-05 | True Grad (I): 4.50e-04 | Causal Dot Product: 4.94e-01                                           
    [R-Step] Avg MSE: 0.1264                                                                                                                          

--- Outer Epoch [04/8] ---
    [Tau Tracker] 碳水 Tau: 4.098 | 胰岛素 Tau: 12.355
    [M-Step] MSE: 0.1238 | True Grad (C): 2.36e-05 | True Grad (I): 2.00e-04 | Causal Dot Product: 4.42e-01                                           
    [R-Step] Avg MSE: 0.1234                                                                                                                          

--- Outer Epoch [05/8] ---
    [Tau Tracker] 碳水 Tau: 4.107 | 胰岛素 Tau: 12.355
    [M-Step] MSE: 0.1256 | True Grad (C): 3.42e-05 | True Grad (I): 1.75e-04 | Causal Dot Product: -4.98e-01                                          
    [R-Step] Avg MSE: 0.1293                                                                                                                          

--- Outer Epoch [06/8] ---
    [Tau Tracker] 碳水 Tau: 4.100 | 胰岛素 Tau: 12.348
    [M-Step] MSE: 0.1300 | True Grad (C): 6.61e-06 | True Grad (I): 5.84e-05 | Causal Dot Product: 1.78e+00                                           
    [R-Step] Avg MSE: 0.1303                                                                                                                          

--- Outer Epoch [07/8] ---
    [Tau Tracker] 碳水 Tau: 4.100 | 胰岛素 Tau: 12.352
    [M-Step] MSE: 0.1220 | True Grad (C): 3.42e-05 | True Grad (I): 9.05e-05 | Causal Dot Product: -3.41e-01                                          
    [R-Step] Avg MSE: 0.1263                                                                                                                          

--- Outer Epoch [08/8] ---
    [Tau Tracker] 碳水 Tau: 4.100 | 胰岛素 Tau: 12.360
    [M-Step] MSE: 0.1287 | True Grad (C): 5.54e-05 | True Grad (I): 8.66e-05 | Causal Dot Product: -1.67e+00                                          
    [R-Step] Avg MSE: 0.1254                                                                                                                          

================================================
>>> 训练结束，开始执行全量测试集终极评估 (100%) <<<
================================================
  | FINAL GLOBAL TEST | 样本数: 14562 | 30min: 19.04 | 60min: 31.87 | 120min: 46.90 |
  | >> Causal Loss (C): 0.1356 | Causal Loss (I): 0.0461 <<

  --> 所有病人的评估图表将保存在: /root/code/image/global_run_20260724_114118
      正在绘制病人 559 的预测验证图...
      正在绘制病人 563 的预测验证图...
      正在绘制病人 570 的预测验证图...
      正在绘制病人 575 的预测验证图...
      正在绘制病人 588 的预测验证图...
      正在绘制病人 591 的预测验证图...

✅ Global 跨病人实验结束。
(my_env) root@cpod-1sc7ubm9xiqn:~/code# python /root/code/success.py
🚀 Global 跨病人启动 | 包含 6 名患者 | MSE: True | DML: False | 冻结PKPD: False | Device: cuda
>>> 1. 挂载 Global 数据并执行 [跨病人] 2-Fold 分割...
    * Fold 1 包含病人: ['559', '563', '570']
    * Fold 2 包含病人: ['575', '588', '591']

>>> 2. 初始化网络...

--- Outer Epoch [01/8] ---
    [Tau Tracker] 碳水 Tau: 4.000 | 胰岛素 Tau: 12.000
    [M-Step] MSE: 0.1535 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1457                                                                                                                          

--- Outer Epoch [02/8] ---
    [Tau Tracker] 碳水 Tau: 4.051 | 胰岛素 Tau: 12.158
    [M-Step] MSE: 0.1355 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1260                                                                                                                          

--- Outer Epoch [03/8] ---
    [Tau Tracker] 碳水 Tau: 4.118 | 胰岛素 Tau: 12.276
    [M-Step] MSE: 0.1194 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1216                                                                                                                          

--- Outer Epoch [04/8] ---
    [Tau Tracker] 碳水 Tau: 4.164 | 胰岛素 Tau: 12.361
    [M-Step] MSE: 0.1165 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1247                                                                                                                          

--- Outer Epoch [05/8] ---
    [Tau Tracker] 碳水 Tau: 4.203 | 胰岛素 Tau: 12.405
    [M-Step] MSE: 0.1213 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1227                                                                                                                          

--- Outer Epoch [06/8] ---
    [Tau Tracker] 碳水 Tau: 4.210 | 胰岛素 Tau: 12.397
    [M-Step] MSE: 0.1222 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1191                                                                                                                          

--- Outer Epoch [07/8] ---
    [Tau Tracker] 碳水 Tau: 4.211 | 胰岛素 Tau: 12.407
    [M-Step] MSE: 0.1154 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1177                                                                                                                          

--- Outer Epoch [08/8] ---
    [Tau Tracker] 碳水 Tau: 4.211 | 胰岛素 Tau: 12.419
    [M-Step] MSE: 0.1181 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1201                                                                                                                          

================================================
>>> 训练结束，开始执行全量测试集终极评估 (100%) <<<
================================================
  | FINAL GLOBAL TEST | 样本数: 14562 | 30min: 19.39 | 60min: 32.47 | 120min: 49.36 |
  | >> Causal Loss (C): 0.0000 | Causal Loss (I): 15.9060 <<

  --> 所有病人的评估图表将保存在: /root/code/image/global_run_20260724_120648
      正在绘制病人 559 的预测验证图...
      正在绘制病人 563 的预测验证图...
      正在绘制病人 570 的预测验证图...
      正在绘制病人 575 的预测验证图...
      正在绘制病人 588 的预测验证图...
      正在绘制病人 591 的预测验证图...

✅ Global 跨病人实验结束。

random seed 42

(my_env) root@cpod-1sc7ubm9xiqn:~/code# python /root/code/success.py
🚀 Global 跨病人启动 | 包含 6 名患者 | MSE: True | DML: False | 冻结PKPD: False | Device: cuda
>>> 1. 挂载 Global 数据并执行 [跨病人] 2-Fold 分割...
    * Fold 1 包含病人: ['559', '563', '570']
    * Fold 2 包含病人: ['575', '588', '591']

>>> 2. 初始化网络...

--- Outer Epoch [01/8] ---
    [Tau Tracker] 碳水 Tau: 4.000 | 胰岛素 Tau: 12.000
    [M-Step] MSE: 0.1564 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1442                                                                                                                          

--- Outer Epoch [02/8] ---
    [Tau Tracker] 碳水 Tau: 4.067 | 胰岛素 Tau: 12.159
    [M-Step] MSE: 0.1294 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1275                                                                                                                          

--- Outer Epoch [03/8] ---
    [Tau Tracker] 碳水 Tau: 4.099 | 胰岛素 Tau: 12.217
    [M-Step] MSE: 0.1214 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1211                                                                                                                          

--- Outer Epoch [04/8] ---
    [Tau Tracker] 碳水 Tau: 4.100 | 胰岛素 Tau: 12.219
    [M-Step] MSE: 0.1180 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1204                                                                                                                          

--- Outer Epoch [05/8] ---
    [Tau Tracker] 碳水 Tau: 4.101 | 胰岛素 Tau: 12.220
    [M-Step] MSE: 0.1229 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1203                                                                                                                          

--- Outer Epoch [06/8] ---
    [Tau Tracker] 碳水 Tau: 4.107 | 胰岛素 Tau: 12.221
    [M-Step] MSE: 0.1330 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1249                                                                                                                          

--- Outer Epoch [07/8] ---
    [Tau Tracker] 碳水 Tau: 4.090 | 胰岛素 Tau: 12.250
    [M-Step] MSE: 0.1226 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1203                                                                                                                          

--- Outer Epoch [08/8] ---
    [Tau Tracker] 碳水 Tau: 4.090 | 胰岛素 Tau: 12.256
    [M-Step] MSE: 0.1180 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1189                                                                                                                          

================================================
>>> 训练结束，开始执行全量测试集终极评估 (100%) <<<
================================================
  | FINAL GLOBAL TEST | 样本数: 14562 | 30min: 19.16 | 60min: 31.95 | 120min: 46.06 |
  | >> Causal Loss (C): 0.0046 | Causal Loss (I): 6.9209 <<

  --> 所有病人的评估图表将保存在: /root/code/image/global_run_20260724_125243
      正在绘制病人 559 的预测验证图...
      正在绘制病人 563 的预测验证图...
      正在绘制病人 570 的预测验证图...
      正在绘制病人 575 的预测验证图...
      正在绘制病人 588 的预测验证图...
      正在绘制病人 591 的预测验证图...

✅ Global 跨病人实验结束。

(my_env) root@cpod-1sc7ubm9xiqn:~/code# python /root/code/success.py
🚀 Global 跨病人启动 | 包含 6 名患者 | MSE: True | DML: True | 冻结PKPD: False | Device: cuda
>>> 1. 挂载 Global 数据并执行 [跨病人] 2-Fold 分割...
    * Fold 1 包含病人: ['559', '563', '570']
    * Fold 2 包含病人: ['575', '588', '591']

>>> 2. 初始化网络...

--- Outer Epoch [01/8] ---
    [Tau Tracker] 碳水 Tau: 4.000 | 胰岛素 Tau: 12.000
    [M-Step] MSE: 0.1551 | True Grad (C): 3.19e-04 | True Grad (I): 7.61e-04 | Causal Dot Product: 8.77e-01                                           
    [R-Step] Avg MSE: 0.1474                                                                                                                          

--- Outer Epoch [02/8] ---
    [Tau Tracker] 碳水 Tau: 4.064 | 胰岛素 Tau: 12.164
    [M-Step] MSE: 0.1275 | True Grad (C): 1.10e-04 | True Grad (I): 1.09e-04 | Causal Dot Product: 4.47e-01                                           
    [R-Step] Avg MSE: 0.2198                                                                                                                          

--- Outer Epoch [03/8] ---
    [Tau Tracker] 碳水 Tau: 4.075 | 胰岛素 Tau: 12.167
    [M-Step] MSE: 0.1397 | True Grad (C): 8.94e-06 | True Grad (I): 6.71e-05 | Causal Dot Product: 2.10e+00                                           
    [R-Step] Avg MSE: 0.1377                                                                                                                          

--- Outer Epoch [04/8] ---
    [Tau Tracker] 碳水 Tau: 4.075 | 胰岛素 Tau: 12.176
    [M-Step] MSE: 0.1335 | True Grad (C): 2.34e-05 | True Grad (I): 3.20e-05 | Causal Dot Product: 1.07e+00                                           
    [R-Step] Avg MSE: 0.1294                                                                                                                          

--- Outer Epoch [05/8] ---
    [Tau Tracker] 碳水 Tau: 4.075 | 胰岛素 Tau: 12.176
    [M-Step] MSE: 0.1242 | True Grad (C): 1.37e-05 | True Grad (I): 3.75e-05 | Causal Dot Product: 2.42e+00                                           
    [R-Step] Avg MSE: 0.1254                                                                                                                          

--- Outer Epoch [06/8] ---
    [Tau Tracker] 碳水 Tau: 4.075 | 胰岛素 Tau: 12.176
    [M-Step] MSE: 0.1227 | True Grad (C): 1.58e-05 | True Grad (I): 8.80e-05 | Causal Dot Product: -3.56e-01                                          
    [R-Step] Avg MSE: 0.1218                                                                                                                          

--- Outer Epoch [07/8] ---
    [Tau Tracker] 碳水 Tau: 4.075 | 胰岛素 Tau: 12.176
    [M-Step] MSE: 0.1187 | True Grad (C): 1.12e-05 | True Grad (I): 4.37e-05 | Causal Dot Product: 2.28e+00                                           
    [R-Step] Avg MSE: 0.1191                                                                                                                          

--- Outer Epoch [08/8] ---
    [Tau Tracker] 碳水 Tau: 4.075 | 胰岛素 Tau: 12.176
    [M-Step] MSE: 0.1190 | True Grad (C): 9.01e-06 | True Grad (I): 1.44e-05 | Causal Dot Product: 1.49e+00                                           
    [R-Step] Avg MSE: 0.1185                                                                                                                          

================================================
>>> 训练结束，开始执行全量测试集终极评估 (100%) <<<
================================================
  | FINAL GLOBAL TEST | 样本数: 14562 | 30min: 19.40 | 60min: 32.67 | 120min: 48.78 |
  | >> Causal Loss (C): 0.9786 | Causal Loss (I): 1.0212 <<

  --> 所有病人的评估图表将保存在: /root/code/image/global_run_20260724_133810
      正在绘制病人 559 的预测验证图...
      正在绘制病人 563 的预测验证图...
      正在绘制病人 570 的预测验证图...
      正在绘制病人 575 的预测验证图...
      正在绘制病人 588 的预测验证图...
      正在绘制病人 591 的预测验证图...

✅ Global 跨病人实验结束。

random seed 44

(my_env) root@cpod-1sc7ubm9xiqn:~/code# python /root/code/success.py
🚀 Global 跨病人启动 | 包含 6 名患者 | MSE: True | DML: True | 冻结PKPD: False | Device: cuda
>>> 1. 挂载 Global 数据并执行 [跨病人] 2-Fold 分割...
    * Fold 1 包含病人: ['559', '563', '570']
    * Fold 2 包含病人: ['575', '588', '591']

>>> 2. 初始化网络...

--- Outer Epoch [01/8] ---
    [Tau Tracker] 碳水 Tau: 4.000 | 胰岛素 Tau: 12.000
    [M-Step] MSE: 0.1551 | True Grad (C): 3.19e-04 | True Grad (I): 7.61e-04 | Causal Dot Product: 8.77e-01                                           
    [R-Step] Avg MSE: 0.1474                                                                                                                          

--- Outer Epoch [02/8] ---
    [Tau Tracker] 碳水 Tau: 4.064 | 胰岛素 Tau: 12.164
    [M-Step] MSE: 0.1275 | True Grad (C): 1.10e-04 | True Grad (I): 1.09e-04 | Causal Dot Product: 4.47e-01                                           
    [R-Step] Avg MSE: 0.2198                                                                                                                          

--- Outer Epoch [03/8] ---
    [Tau Tracker] 碳水 Tau: 4.075 | 胰岛素 Tau: 12.167
    [M-Step] MSE: 0.1397 | True Grad (C): 8.94e-06 | True Grad (I): 6.71e-05 | Causal Dot Product: 2.10e+00                                           
    [R-Step] Avg MSE: 0.1377                                                                                                                          

--- Outer Epoch [04/8] ---
    [Tau Tracker] 碳水 Tau: 4.075 | 胰岛素 Tau: 12.176
    [M-Step] MSE: 0.1335 | True Grad (C): 2.34e-05 | True Grad (I): 3.20e-05 | Causal Dot Product: 1.07e+00                                           
    [R-Step] Avg MSE: 0.1294                                                                                                                          

--- Outer Epoch [05/8] ---
    [Tau Tracker] 碳水 Tau: 4.075 | 胰岛素 Tau: 12.176
    [M-Step] MSE: 0.1242 | True Grad (C): 1.37e-05 | True Grad (I): 3.75e-05 | Causal Dot Product: 2.42e+00                                           
    [R-Step] Avg MSE: 0.1254                                                                                                                          

--- Outer Epoch [06/8] ---
    [Tau Tracker] 碳水 Tau: 4.075 | 胰岛素 Tau: 12.176
    [M-Step] MSE: 0.1227 | True Grad (C): 1.58e-05 | True Grad (I): 8.80e-05 | Causal Dot Product: -3.56e-01                                          
    [R-Step] Avg MSE: 0.1218                                                                                                                          

--- Outer Epoch [07/8] ---
    [Tau Tracker] 碳水 Tau: 4.075 | 胰岛素 Tau: 12.176
    [M-Step] MSE: 0.1187 | True Grad (C): 1.12e-05 | True Grad (I): 4.37e-05 | Causal Dot Product: 2.28e+00                                           
    [R-Step] Avg MSE: 0.1191                                                                                                                          

--- Outer Epoch [08/8] ---
    [Tau Tracker] 碳水 Tau: 4.075 | 胰岛素 Tau: 12.176
    [M-Step] MSE: 0.1190 | True Grad (C): 9.01e-06 | True Grad (I): 1.44e-05 | Causal Dot Product: 1.49e+00                                           
    [R-Step] Avg MSE: 0.1185                                                                                                                          

================================================
>>> 训练结束，开始执行全量测试集终极评估 (100%) <<<
================================================
  | FINAL GLOBAL TEST | 样本数: 14562 | 30min: 19.40 | 60min: 32.67 | 120min: 48.78 |
  | >> Causal Loss (C): 0.9786 | Causal Loss (I): 1.0212 <<

  --> 所有病人的评估图表将保存在: /root/code/image/global_run_20260724_140614
      正在绘制病人 559 的预测验证图...
      正在绘制病人 563 的预测验证图...
      正在绘制病人 570 的预测验证图...
      正在绘制病人 575 的预测验证图...
      正在绘制病人 588 的预测验证图...
      正在绘制病人 591 的预测验证图...

✅ Global 跨病人实验结束。

(my_env) root@cpod-1sc7ubm9xiqn:~/code# python /root/code/success.py
🚀 Global 跨病人启动 | 包含 6 名患者 | MSE: True | DML: False | 冻结PKPD: False | Device: cuda
>>> 1. 挂载 Global 数据并执行 [跨病人] 2-Fold 分割...
    * Fold 1 包含病人: ['559', '563', '570']
    * Fold 2 包含病人: ['575', '588', '591']

>>> 2. 初始化网络...

--- Outer Epoch [01/8] ---
    [Tau Tracker] 碳水 Tau: 4.000 | 胰岛素 Tau: 12.000
    [M-Step] MSE: 0.1564 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1442                                                                                                                          

--- Outer Epoch [02/8] ---
    [Tau Tracker] 碳水 Tau: 4.067 | 胰岛素 Tau: 12.159
    [M-Step] MSE: 0.1294 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1275                                                                                                                          

--- Outer Epoch [03/8] ---
    [Tau Tracker] 碳水 Tau: 4.099 | 胰岛素 Tau: 12.217
    [M-Step] MSE: 0.1214 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1211                                                                                                                          

--- Outer Epoch [04/8] ---
    [Tau Tracker] 碳水 Tau: 4.100 | 胰岛素 Tau: 12.219
    [M-Step] MSE: 0.1180 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1204                                                                                                                          

--- Outer Epoch [05/8] ---
    [Tau Tracker] 碳水 Tau: 4.101 | 胰岛素 Tau: 12.220
    [M-Step] MSE: 0.1229 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1203                                                                                                                          

--- Outer Epoch [06/8] ---
    [Tau Tracker] 碳水 Tau: 4.107 | 胰岛素 Tau: 12.221
    [M-Step] MSE: 0.1330 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1249                                                                                                                          

--- Outer Epoch [07/8] ---
    [Tau Tracker] 碳水 Tau: 4.090 | 胰岛素 Tau: 12.250
    [M-Step] MSE: 0.1226 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1203                                                                                                                          

--- Outer Epoch [08/8] ---
    [Tau Tracker] 碳水 Tau: 4.090 | 胰岛素 Tau: 12.256
    [M-Step] MSE: 0.1180 | True Grad (C): 0.00e+00 | True Grad (I): 0.00e+00 | Causal Dot Product: 0.00e+00                                           
    [R-Step] Avg MSE: 0.1189                                                                                                                          

================================================
>>> 训练结束，开始执行全量测试集终极评估 (100%) <<<
================================================
  | FINAL GLOBAL TEST | 样本数: 14562 | 30min: 19.16 | 60min: 31.95 | 120min: 46.06 |
  | >> Causal Loss (C): 0.0046 | Causal Loss (I): 6.9209 <<

  --> 所有病人的评估图表将保存在: /root/code/image/global_run_20260724_143612
      正在绘制病人 559 的预测验证图...
      正在绘制病人 563 的预测验证图...
      正在绘制病人 570 的预测验证图...
      正在绘制病人 575 的预测验证图...
      正在绘制病人 588 的预测验证图...
      正在绘制病人 591 的预测验证图...

✅ Global 跨病人实验结束。

"""

