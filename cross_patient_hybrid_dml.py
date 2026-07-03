"""
多病人联合训练版本 (Global Modeling Baseline) - 附加个体 RMSE 评估
核心改动：
1. 废除了单病人时间切分的 2-Fold，改为直接按病人 ID 进行 2-Fold 切分 (Patient-wise Split)。
2. 预测阶段：不仅计算所有病人的 Global RMSE，还会挨个打印每个病人的独立 RMSE 并画图。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import os
import time
import warnings
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.func as func
from torchdiffeq import odeint_adjoint as odeint
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")

# ==========================================
# 0. 全局超参数与数据集配置
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = '/root/code/data/2018'
# 从用户提供的图片中提取的 6 个病人的 ID
PATIENT_IDS = ['559', '563', '570', '575', '588', '591']

DT_MINUTES = 5          
HIST_WINDOW = 24        
PRED_WINDOW = 24        
GAP_THRESHOLD = 15      
C_DIM = 16              

RMSE_EVAL_MINUTES = 30  
BATCH_SIZE = 128 
NUM_WORKERS = 16 
OUTER_EPOCHS = 5        
INNER_N_EPOCHS = 3      
INNER_P_EPOCHS = 3      

print(f"🚀 多病人联合训练启动 | Compute Device: {DEVICE}")

# ==========================================
# 1. 数据解析与多病人聚合提取
# ==========================================
def parse_and_align_data(filepath):
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
        df_1min.loc[common_idx, 'bolus_U_min'] = bolus_grouped[common_idx].values
    
    df_1min['u_sc'] = df_1min['basal_U_min'] + df_1min['bolus_U_min']

    ka = 0.018 
    u_sc_array = df_1min['u_sc'].values
    n_steps = len(u_sc_array)
    S1, S2, U_t = np.zeros(n_steps), np.zeros(n_steps), np.zeros(n_steps)
    initial_basal = df_1min['basal_U_min'].iloc[0] if len(df_1min) > 0 else 0
    S1[0] = S2[0] = initial_basal / ka
    U_t[0] = ka * S2[0]
    for i in range(1, n_steps):
        S1[i] = S1[i-1] + (u_sc_array[i-1] - ka * S1[i-1]) * 1.0
        S2[i] = S2[i-1] + (ka * S1[i-1] - ka * S2[i-1]) * 1.0
        U_t[i] = ka * S2[i]
    df_1min['U_t'] = U_t

    A_G, V_G, t_max_G = 0.9, 99 * 1.6, 50.0
    df_1min['D_t'] = 0.0
    if not df_meal.empty:
        for _, row in df_meal.iterrows():
            meal_ts = row['ts'].floor('1min')
            carbs = row['val']
            mask = df_1min.index >= meal_ts
            delta_t = (df_1min.index[mask] - meal_ts).total_seconds() / 60.0
            Ra_t = (1000.0 * carbs * A_G / (t_max_G**2)) * delta_t * np.exp(-delta_t / t_max_G)
            df_1min.loc[mask, 'D_t'] += (Ra_t / V_G)

    grid_5min = pd.date_range(start_time, end_time, freq='5min')
    df_5min = pd.DataFrame(index=grid_5min)
    df_cgm.set_index('ts', inplace=True)
    df_5min['Y_cgm'] = df_cgm['val'].reindex(grid_5min, method='nearest', tolerance=pd.Timedelta('2min'))
    df_5min['U_ins'] = df_1min['U_t'].reindex(grid_5min)
    df_5min['D_carbs'] = df_1min['D_t'].reindex(grid_5min)

    if not df_hr.empty:
        hr_df = df_hr.set_index('ts')
        df_5min['HR'] = hr_df['val'].reindex(grid_5min).ffill().fillna(70.0)
    else: df_5min['HR'] = 70.0
        
    if not df_temp_skin.empty:
        st_df = df_temp_skin.set_index('ts')
        df_5min['SkinTemp'] = st_df['val'].reindex(grid_5min).ffill().fillna(90.0)
    else: df_5min['SkinTemp'] = 90.0

    return df_5min.reset_index().rename(columns={'index': 'ts'})

def extract_windows_stride(df_sim, stride):
    valid_mask = df_sim['Y_cgm'].notna()
    df_valid = df_sim[valid_mask].copy()
    
    break_points = [df_valid.index[0]] + df_valid.index[df_valid['ts'].diff() > pd.Timedelta(minutes=GAP_THRESHOLD)].tolist() + [df_valid.index[-1] + 1]

    X_Y, X_U, X_D, X_Z_snap = [], [], [], []
    for i in range(len(break_points) - 1):
        chunk = df_sim.loc[break_points[i] : break_points[i+1]-1].dropna()
        if len(chunk) < HIST_WINDOW + PRED_WINDOW: continue
            
        vals_Y = chunk['Y_cgm'].values.reshape(-1, 1) / 100.0
        vals_U, vals_D = chunk['U_ins'].values.reshape(-1, 1), chunk['D_carbs'].values.reshape(-1, 1)
        vals_Z = chunk[['HR', 'SkinTemp']].values / 100.0
        safe_history_features = np.hstack([vals_Y, vals_U, vals_D, vals_Z]) 
        
        for start_idx in range(HIST_WINDOW, len(chunk) - PRED_WINDOW + 1, stride):
            end_idx = start_idx + PRED_WINDOW
            X_Y.append(vals_Y[start_idx : end_idx])
            X_U.append(vals_U[start_idx : end_idx])
            X_D.append(vals_D[start_idx : end_idx])
            X_Z_snap.append(safe_history_features[start_idx - HIST_WINDOW : start_idx])
            
    if len(X_Y) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return np.array(X_Y), np.array(X_U), np.array(X_D), np.array(X_Z_snap)

def build_patient_dataset(patient_ids, split_type, stride):
    Y_all, U_all, D_all, Z_all = [], [], [], []
    for pid in patient_ids:
        filepath = os.path.join(DATA_DIR, split_type, f"{pid}-ws-{split_type}ing.xml")
        if os.path.exists(filepath):
            df = parse_and_align_data(filepath)
            if df is not None:
                Y, U, D, Z = extract_windows_stride(df, stride=stride)
                if len(Y) > 0:
                    Y_all.append(Y); U_all.append(U); D_all.append(D); Z_all.append(Z)
        else:
            print(f"Warning: 找不到文件 {filepath}")
            
    return np.vstack(Y_all), np.vstack(U_all), np.vstack(D_all), np.vstack(Z_all)

# ==========================================
# 【新增改动】评估模块重构，支持紧凑格式打印
# ==========================================
def evaluate_final_test_rmse(hybrid_model, dl_te, t_eval_t, label="Global"):
    """计算并在单行紧凑地打印指定的测试集误差"""
    hybrid_model.eval()
    sum_sq_err_30, sum_sq_err_60, sum_sq_err_120 = 0.0, 0.0, 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_Y, batch_U, batch_D, batch_Z_snap in dl_te:
            batch_Y, batch_U, batch_D = batch_Y.to(DEVICE), batch_U.to(DEVICE), batch_D.to(DEVICE)
            batch_Z_snap = batch_Z_snap.to(DEVICE)
            
            hybrid_model.set_batch_data(batch_U, batch_D, batch_Z_snap, t_eval_t)
            pred_Y = odeint(hybrid_model, batch_Y[:, 0, :], t_eval_t, method='rk4', options={'step_size': DT_MINUTES/60.0}).transpose(0, 1)
            
            B = batch_Y.size(0)
            total_samples += B
            
            sum_sq_err_30 += F.mse_loss(pred_Y[:, 5, :], batch_Y[:, 5, :], reduction='sum').item()
            sum_sq_err_60 += F.mse_loss(pred_Y[:, 11, :], batch_Y[:, 11, :], reduction='sum').item()
            sum_sq_err_120 += F.mse_loss(pred_Y[:, 23, :], batch_Y[:, 23, :], reduction='sum').item()

    if total_samples == 0:
        print(f"Warning: {label} 测试集为空！")
        return

    rmse_30 = np.sqrt(sum_sq_err_30 / total_samples) * 100.0
    rmse_60 = np.sqrt(sum_sq_err_60 / total_samples) * 100.0
    rmse_120 = np.sqrt(sum_sq_err_120 / total_samples) * 100.0

    print(f"  | {label:13s} | 样本数: {total_samples:4d} | 30min: {rmse_30:5.2f} | 60min: {rmse_60:5.2f} | 120min: {rmse_120:5.2f} |")


# ==========================================
# 2. 神经网络架构 
# ==========================================
class ODEResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.Tanh()
        nn.init.normal_(self.fc1.weight, mean=0, std=0.01)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.01)
    def forward(self, x): return x + 0.1 * self.fc2(self.act(self.fc1(self.act(x))))

class NuisanceEncoder(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=64, out_dim=C_DIM):
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)
    def forward(self, snap_seq):
        _, h_n = self.gru(snap_seq) 
        return self.fc(h_n[-1]) 

class MemoryPhysicsNN(nn.Module):
    def __init__(self, y_dim=1, ud_dim=2, hidden_dim=128):
        super().__init__()
        self.proj_in = nn.Linear(y_dim + ud_dim, hidden_dim)
        self.res_blocks = nn.Sequential(
            ODEResBlock(hidden_dim), ODEResBlock(hidden_dim),
            ODEResBlock(hidden_dim), ODEResBlock(hidden_dim)
        )
        self.proj_out = nn.Linear(hidden_dim, y_dim)
        self.act = nn.Tanh()
        nn.init.normal_(self.proj_in.weight, mean=0, std=0.01)
        nn.init.normal_(self.proj_out.weight, mean=0, std=0.01)
    def forward(self, Y, U, D):
        x = torch.cat([Y, U, D], dim=-1)
        return self.proj_out(self.res_blocks(self.act(self.proj_in(x))))

class ResidualNN(nn.Module):
    def __init__(self, y_dim=1, c_dim=C_DIM, out_dim=1, hidden_dim=128):
        super().__init__()
        self.proj_in = nn.Linear(y_dim + c_dim, hidden_dim)
        self.res_blocks = nn.Sequential(ODEResBlock(hidden_dim), ODEResBlock(hidden_dim))
        self.proj_out = nn.Linear(hidden_dim, out_dim)
        self.act = nn.Tanh()
    def forward(self, Y, C_hist):
        x = torch.cat([Y, C_hist], dim=-1)
        return self.proj_out(self.res_blocks(self.act(self.proj_in(x))))

class NuisanceNN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), 
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, C_hist): return self.net(C_hist)

class BatchedLinearInterpolator(nn.Module):
    def __init__(self, t_eval, x_seq, dev):
        super().__init__()
        self.register_buffer('t_eval', torch.tensor(t_eval, dtype=torch.float32, device=dev))
        self.register_buffer('x_seq', torch.tensor(x_seq, dtype=torch.float32, device=dev))
    def forward(self, t):
        idx = torch.clamp(torch.searchsorted(self.t_eval, t), 1, len(self.t_eval)-1)
        t0, t1 = self.t_eval[idx-1], self.t_eval[idx]
        return self.x_seq[:, idx-1] + (t - t0) / (t1 - t0 + 1e-8) * (self.x_seq[:, idx] - self.x_seq[:, idx-1])

class HybridODEFunc(nn.Module):
    def __init__(self, phys_nn, encoder, res_nn, dev, use_nn=True):
        super().__init__()
        self.phys = phys_nn; self.encoder = encoder; self.res_nn = res_nn
        self.dev, self.use_nn = dev, use_nn
        self.U_interp, self.D_interp, self.C_hist_static = None, None, None
    def set_batch_data(self, U_seq, D_seq, Z_snap_batch, t_eval):
        self.U_interp = BatchedLinearInterpolator(t_eval, U_seq, self.dev)
        self.D_interp = BatchedLinearInterpolator(t_eval, D_seq, self.dev)
        self.C_hist_static = self.encoder(Z_snap_batch)
    def forward(self, t, Y):
        dy_phys = self.phys(Y, self.U_interp(t), self.D_interp(t))
        if self.use_nn: return dy_phys + self.res_nn(Y, self.C_hist_static)
        return dy_phys


# ==========================================
# 3. Double Machine Learning 连续时间积分引擎
# ==========================================
class DMLEngine:
    def __init__(self, dt): self.dt = dt
    def compute_integrals(self, phys_nn, Y_batch, U_batch, D_batch):
        N, T, Dim = Y_batch.shape
        params = dict(phys_nn.named_parameters())
        Y_flat, U_flat, D_flat = Y_batch.reshape(-1, Dim), U_batch.reshape(-1, Dim), D_batch.reshape(-1, Dim)
        def f_fn(p, y, u, d): return func.functional_call(phys_nn, p, (y, u, d))
        
        f_vals = f_fn(params, Y_flat, U_flat, D_flat).reshape(N, T, Dim)
        F_int = (f_vals[:, :-1, :] + f_vals[:, 1:, :]) / 2.0 * self.dt
        
        jac_dict = func.vmap(func.jacrev(f_fn, argnums=0), in_dims=(None, 0, 0, 0))(params, Y_flat, U_flat, D_flat)
        J_all = torch.cat([jac.reshape(N, T, Dim, -1) for jac in jac_dict.values()], dim=-1)
        J_int = (J_all[:, :-1, :, :] + J_all[:, 1:, :, :]) / 2.0 * self.dt
        return F_int, J_int, J_all.shape[-1]

    def compute_F_only(self, phys_nn, Y_batch, U_batch, D_batch):
        N, T, Dim = Y_batch.shape
        f_vals = phys_nn(Y_batch.reshape(-1, Dim), U_batch.reshape(-1, Dim), D_batch.reshape(-1, Dim)).reshape(N, T, Dim)
        return (f_vals[:, :-1, :] + f_vals[:, 1:, :]) / 2.0 * self.dt


# ==========================================
# 4. 核心训练管线 (2-Fold 严格交叉拟合)
# ==========================================
def train_2fold_alternating_loop(dl_f1, dl_f2, hybrid_model, n_q1, n_H1, n_q2, n_H2, dml_engine, T_pts, Dim, P_total):
    print(f"\n>>> 启动跨病人 2-Fold 闭环交替训练")
    opt_f = torch.optim.Adam(hybrid_model.phys.parameters(), lr=1e-3)
    opt_r = torch.optim.Adam(list(hybrid_model.res_nn.parameters()) + list(hybrid_model.encoder.parameters()), lr=3e-3)
    opt_q1 = torch.optim.Adam(n_q1.parameters(), lr=2e-3); opt_H1 = torch.optim.Adam(n_H1.parameters(), lr=2e-3)
    opt_q2 = torch.optim.Adam(n_q2.parameters(), lr=2e-3); opt_H2 = torch.optim.Adam(n_H2.parameters(), lr=2e-3)
    
    t_eval_t = torch.tensor(np.arange(0, T_pts * (DT_MINUTES/60.0), (DT_MINUTES/60.0)), dtype=torch.float32, device=DEVICE)
    history = {'train_f_mse': [], 'train_r_mse': [], 'score': []}
    
    def cache_targets_for_fold(dl):
        cached_data = []
        with torch.no_grad():
            for batch_Y, batch_U, batch_D, batch_Z_snap in tqdm(dl, desc="Caching Targets"):
                batch_Y, batch_U, batch_D = batch_Y.to(DEVICE), batch_U.to(DEVICE), batch_D.to(DEVICE)
                batch_Z_snap = batch_Z_snap.to(DEVICE)
                Delta_Y = batch_Y[:, 1:, :] - batch_Y[:, :-1, :]
                
                F_int_targ, J_int_targ, _ = dml_engine.compute_integrals(hybrid_model.phys, batch_Y, batch_U, batch_D)
                Target_R_flat = (Delta_Y - F_int_targ).reshape(-1, Dim)
                Target_J_flat = J_int_targ.reshape(-1, Dim * P_total)
                C_batch = hybrid_model.encoder(batch_Z_snap)
                C_expanded = C_batch.unsqueeze(1).expand(-1, T_pts-1, -1).reshape(-1, C_DIM)
                
                cached_data.append((C_expanded.cpu(), Target_R_flat.cpu(), Target_J_flat.cpu()))
                torch.cuda.empty_cache()
        return cached_data

    def train_nuisance_inner(cached_data, n_q, n_H, opt_q, opt_H, fold_name):
        n_q.train(); n_H.train()
        for inner_n in range(INNER_N_EPOCHS):
            loss_q_sum, loss_H_sum, b = 0.0, 0.0, 0
            for C_exp, T_R, T_J in cached_data:
                C_exp, T_R, T_J = C_exp.to(DEVICE), T_R.to(DEVICE), T_J.to(DEVICE)
                opt_q.zero_grad(); loss_q = F.mse_loss(n_q(C_exp), T_R); loss_q.backward(); opt_q.step(); loss_q_sum += loss_q.item()
                opt_H.zero_grad(); loss_H = F.mse_loss(n_H(C_exp), T_J); loss_H.backward(); opt_H.step(); loss_H_sum += loss_H.item()
                b += 1
            print(f"    [{fold_name}] Inner E-step {inner_n+1}/{INNER_N_EPOCHS} | Loss Q: {loss_q_sum/b:.5f} | Loss H: {loss_H_sum/b:.5f}")

    def train_causal_f_inner(dl, n_q_oos, n_H_oos, fold_name):
        hybrid_model.train()
        ep_f_mse, ep_score, batches_f = 0.0, 0.0, 0
        
        for inner_p in range(INNER_P_EPOCHS):
            for batch_Y, batch_U, batch_D, batch_Z_snap in tqdm(dl, desc="Caching Targets"):
                batch_Y, batch_U, batch_D = batch_Y.to(DEVICE), batch_U.to(DEVICE), batch_D.to(DEVICE)
                batch_Z_snap = batch_Z_snap.to(DEVICE)
                Delta_Y = batch_Y[:, 1:, :] - batch_Y[:, :-1, :]
                
                with torch.no_grad():
                    _, J_int_targ, _ = dml_engine.compute_integrals(hybrid_model.phys, batch_Y, batch_U, batch_D)
                    C_batch = hybrid_model.encoder(batch_Z_snap)
                    C_expanded = C_batch.unsqueeze(1).expand(-1, T_pts-1, -1).reshape(-1, C_DIM)
                    
                    q_oos = n_q_oos(C_expanded).reshape(-1, T_pts-1, Dim)
                    H_oos = n_H_oos(C_expanded).reshape(-1, T_pts-1, Dim, P_total)
                
                opt_f.zero_grad()
                hybrid_model.set_batch_data(batch_U, batch_D, batch_Z_snap, t_eval_t)
                pred_Y_closed_loop = odeint(hybrid_model, batch_Y[:, 0, :], t_eval_t, method='rk4', options={'step_size': DT_MINUTES/60.0}).transpose(0, 1)
                loss_mse = F.mse_loss(pred_Y_closed_loop, batch_Y)
                
                F_int_graph = dml_engine.compute_F_only(hybrid_model.phys, batch_Y, batch_U, batch_D)
                Psi = torch.einsum('ntdp,ntd->p', (J_int_targ - H_oos), (Delta_Y - F_int_graph - q_oos)) / (batch_Y.size(0) * (T_pts-1))
                loss_score = torch.log(torch.sum(Psi ** 2) + 1e-12)
                
                warmup_epochs = 3
                if outer_ep < warmup_epochs: lambda_weight = 0.0
                else: lambda_weight = min(1.0, (loss_mse.item() / (abs(loss_score.item()) + 1e-8)) * 1e-2)
                
                loss_f = loss_mse + lambda_weight * loss_score; loss_f.backward(); opt_f.step()
                
                if inner_p == INNER_P_EPOCHS - 1:
                    ep_f_mse += loss_mse.item(); ep_score += loss_score.item(); batches_f += 1
            print(f"    [{fold_name}] Inner M-step {inner_p+1}/{INNER_P_EPOCHS} | NODE MSE: {loss_mse.item():.4f}")
        return ep_f_mse, ep_score, batches_f

    for outer_ep in range(OUTER_EPOCHS):
        print(f"\n--- Outer Epoch [{outer_ep+1:02d}/{OUTER_EPOCHS}] ---")
        for param in hybrid_model.phys.parameters(): param.requires_grad = False
        for param in hybrid_model.res_nn.parameters(): param.requires_grad = False
        for param in hybrid_model.encoder.parameters(): param.requires_grad = False
        
        print("  [Step A.1] 缓存积分 Target (No-grad)...")
        cached_f1 = cache_targets_for_fold(dl_f1); cached_f2 = cache_targets_for_fold(dl_f2)
        
        print("  [Step A.1] 独立优化 Nuisance 网络...")
        train_nuisance_inner(cached_f1, n_q1, n_H1, opt_q1, opt_H1, "Fold 1 (Patients)")
        train_nuisance_inner(cached_f2, n_q2, n_H2, opt_q2, opt_H2, "Fold 2 (Patients)")
                
        for param in hybrid_model.phys.parameters(): param.requires_grad = True
        print("  [Step A.2] Cross-fitting 优化因果主干 f_theta...")
        
        mse1, sc1, b1 = train_causal_f_inner(dl_f1, n_q2, n_H2, "Eval Fold 1 (OOS=2)")
        mse2, sc2, b2 = train_causal_f_inner(dl_f2, n_q1, n_H1, "Eval Fold 2 (OOS=1)")
        ep_f_mse = (mse1 + mse2) / (b1 + b2); ep_score = (sc1 + sc2) / (b1 + b2)
        print(f"  --> [Summary f] Avg MSE: {ep_f_mse:.4f} | Avg Score: {ep_score:.3f}")

        for param in hybrid_model.phys.parameters(): param.requires_grad = False
        for param in hybrid_model.res_nn.parameters(): param.requires_grad = True
        for param in hybrid_model.encoder.parameters(): param.requires_grad = True
        
        print("  [Step B] 训练黑盒残差 r_phi...")
        ep_r_mse, batches_r = 0.0, 0
        for dl in [dl_f1, dl_f2]:
            for batch_Y, batch_U, batch_D, batch_Z_snap in tqdm(dl, desc="Residual Tr."):
                batch_Y, batch_U, batch_D = batch_Y.to(DEVICE), batch_U.to(DEVICE), batch_D.to(DEVICE)
                batch_Z_snap = batch_Z_snap.to(DEVICE)
                
                opt_r.zero_grad()
                hybrid_model.set_batch_data(batch_U, batch_D, batch_Z_snap, t_eval_t)
                pred_Y_closed_loop = odeint(hybrid_model, batch_Y[:, 0, :], t_eval_t, method='rk4', options={'step_size': DT_MINUTES/60.0}).transpose(0, 1)
                
                loss_r = F.mse_loss(pred_Y_closed_loop, batch_Y); loss_r.backward()
                torch.nn.utils.clip_grad_norm_(hybrid_model.res_nn.parameters(), max_norm=1.0)
                opt_r.step()
                ep_r_mse += loss_r.item(); batches_r += 1
            
        print(f"  --> [Summary r] Avg MSE: {ep_r_mse/batches_r:.4f}")
        history['train_f_mse'].append(ep_f_mse); history['train_r_mse'].append(ep_r_mse/batches_r); history['score'].append(ep_score)

    return history, t_eval_t


# ==========================================
# 5. 可视化评估模块 
# ==========================================
def plot_rolling_forecast(hybrid_model, Y_te, U_te, D_te, Z_snap_te, t_eval_t, dt_minutes, t_pts, device, patient_id):
    if Y_te.shape[0] == 0: return
    hybrid_model.eval()
    img_dir = os.path.join(os.getcwd(), 'image')
    if not os.path.exists(img_dir): os.makedirs(img_dir)
    
    num_windows_to_plot = min(3000, Y_te.shape[0])
    plot_step = 12 
    
    gt_continuous = [Y_te[i, 0, 0] * 100 for i in range(num_windows_to_plot)]
    gt_continuous.extend(Y_te[num_windows_to_plot-1, 1:, 0] * 100)
    t_abs = np.arange(len(gt_continuous)) * (dt_minutes / 60.0)

    plt.figure(figsize=(24, 8)) 
    plt.plot(t_abs, gt_continuous, 'k-', lw=3, label='Ground Truth (CGM)', alpha=0.6)

    with torch.no_grad():
        for i in range(0, num_windows_to_plot, plot_step):
            Y0 = torch.tensor(Y_te[i:i+1, 0, :], dtype=torch.float32, device=device)
            U_batch = torch.tensor(U_te[i:i+1], dtype=torch.float32, device=device)
            D_batch = torch.tensor(D_te[i:i+1], dtype=torch.float32, device=device)
            Z_snap_batch = torch.tensor(Z_snap_te[i:i+1], dtype=torch.float32, device=device)
            
            hybrid_model.set_batch_data(U_batch, D_batch, Z_snap_batch, t_eval_t)
            
            hybrid_model.use_nn = False
            pred_phys = odeint(hybrid_model, Y0, t_eval_t, method='rk4', options={'step_size': dt_minutes/60.0}).transpose(0, 1).cpu().numpy()
            
            hybrid_model.use_nn = True
            pred_hybrid = odeint(hybrid_model, Y0, t_eval_t, method='rk4', options={'step_size': dt_minutes/60.0}).transpose(0, 1).cpu().numpy()
            
            t_pred_axis = t_abs[i : i + t_pts]
            label_p = 'Physics Causal Skeleton' if i == 0 else ""
            label_h = 'Full Hybrid NODE' if i == 0 else ""
            
            plt.plot(t_pred_axis, pred_phys[0, :, 0] * 100, '--', color='gray', lw=1.5, alpha=0.4, label=label_p)
            plt.plot(t_pred_axis, pred_hybrid[0, :, 0] * 100, '-', color='red', lw=2.5, alpha=0.85, label=label_h)
            plt.scatter(t_pred_axis[0], Y0.item() * 100, color='black', s=30, zorder=5)

    plt.title(f'Patient {patient_id} - Open-loop PK/PD Deep Causal NODE Forecast', fontweight='bold', fontsize=18)
    plt.xlabel('Absolute Time (Hours)', fontsize=14); plt.ylabel('Glucose (mg/dL)', fontsize=14)
    plt.axhspan(70, 180, color='green', alpha=0.1, label='Target Range (70-180 mg/dL)')
    plt.axhline(70, color='red', linestyle=':', alpha=0.5); plt.axhline(180, color='orange', linestyle=':', alpha=0.5)
    plt.xlim(0, t_abs[-1])
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9); plt.grid(True, linestyle='--', alpha=0.4)
    
    rolling_save_path = os.path.join(img_dir, f'rolling_forecast_P{patient_id}_{time.strftime("%H%M%S")}.png')
    plt.savefig(rolling_save_path, dpi=300, bbox_inches='tight')


# ==========================================
# 6. 实验入口 (Main execution)
# ==========================================
def main():
    mid_idx = len(PATIENT_IDS) // 2
    fold1_pids = PATIENT_IDS[:mid_idx]  
    fold2_pids = PATIENT_IDS[mid_idx:]  
    
    print(f">>> [1/4] 构建多病人训练集...\nFold 1 病人: {fold1_pids}\nFold 2 病人: {fold2_pids}")
    
    Y_f1, U_f1, D_f1, Z_f1 = build_patient_dataset(fold1_pids, 'train', stride=20)
    Y_f2, U_f2, D_f2, Z_f2 = build_patient_dataset(fold2_pids, 'train', stride=20)
    
    ds_f1 = TensorDataset(*(torch.tensor(x, dtype=torch.float32) for x in [Y_f1, U_f1, D_f1, Z_f1]))
    dl_f1 = DataLoader(ds_f1, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    ds_f2 = TensorDataset(*(torch.tensor(x, dtype=torch.float32) for x in [Y_f2, U_f2, D_f2, Z_f2]))
    dl_f2 = DataLoader(ds_f2, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    
    print("\n>>> [1.5/4] 构建独立的各病人测试集...")
    test_data_dict = {}
    for pid in PATIENT_IDS:
        y, u, d, z = build_patient_dataset([pid], 'test', stride=1)
        if len(y) > 0:
            test_data_dict[pid] = (y, u, d, z)
            
    Y_te_all = np.vstack([v[0] for v in test_data_dict.values()])
    U_te_all = np.vstack([v[1] for v in test_data_dict.values()])
    D_te_all = np.vstack([v[2] for v in test_data_dict.values()])
    Z_te_all = np.vstack([v[3] for v in test_data_dict.values()])
    
    ds_te_all = TensorDataset(*(torch.tensor(x, dtype=torch.float32) for x in [Y_te_all, U_te_all, D_te_all, Z_te_all]))
    dl_te_all = DataLoader(ds_te_all, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    N_tr = Y_f1.shape[0] + Y_f2.shape[0]
    T_pts = Y_f1.shape[1]
    Dim = Y_f1.shape[2]
    print(f"数据汇总: Train {N_tr} 样本, Test 全局 {Y_te_all.shape[0]} 样本")

    print("\n>>> [2/4] 初始化正交隔离架构...")
    encoder = NuisanceEncoder(in_dim=5, hidden_dim=64, out_dim=C_DIM).to(DEVICE)
    phys_nn = MemoryPhysicsNN(y_dim=1, ud_dim=2, hidden_dim=128).to(DEVICE)
    res_nn = ResidualNN(y_dim=1, c_dim=C_DIM, out_dim=1, hidden_dim=128).to(DEVICE)
    hybrid_model = HybridODEFunc(phys_nn, encoder, res_nn, dev=DEVICE).to(DEVICE)
    
    dml_engine = DMLEngine(dt=DT_MINUTES / 60.0) 
    _, _, P_total = dml_engine.compute_integrals(phys_nn, torch.zeros(2, T_pts, 1).to(DEVICE), torch.zeros(2, T_pts, 1).to(DEVICE), torch.zeros(2, T_pts, 1).to(DEVICE))

    nuisance_q1 = NuisanceNN(in_dim=C_DIM, out_dim=Dim, hidden_dim=128).to(DEVICE)
    nuisance_H1 = NuisanceNN(in_dim=C_DIM, out_dim=Dim * P_total, hidden_dim=128).to(DEVICE)
    nuisance_q2 = NuisanceNN(in_dim=C_DIM, out_dim=Dim, hidden_dim=128).to(DEVICE)
    nuisance_H2 = NuisanceNN(in_dim=C_DIM, out_dim=Dim * P_total, hidden_dim=128).to(DEVICE)

    print("\n>>> [3/4] 开始执行跨病人 2-Fold 交替训练...")
    history, t_eval_t = train_2fold_alternating_loop(
        dl_f1, dl_f2, hybrid_model, 
        nuisance_q1, nuisance_H1, nuisance_q2, nuisance_H2, 
        dml_engine, T_pts, Dim, P_total
    )
    
    # ==========================================
    # 【核心打印重构区域】
    # ==========================================
    print("\n" + "="*70)
    print(" 🚀 最终测试集泛化评估 (Global vs. Patient-level RMSE)")
    print("="*70)
    
    # 1. 打印全局误差 (Global Population)
    t_eval_t = torch.tensor(np.arange(0, T_pts * (DT_MINUTES/60.0), (DT_MINUTES/60.0)), dtype=torch.float32, device=DEVICE)
    evaluate_final_test_rmse(hybrid_model, dl_te_all, t_eval_t, label="GLOBAL ALL")
    print("-" * 70)
    
    # 2. 循环遍历并打印每一个病人的单独误差
    for pid, (Y, U, D, Z) in test_data_dict.items():
        # 为单个病人构建 DataLoader
        ds_te_single = TensorDataset(*(torch.tensor(x, dtype=torch.float32) for x in [Y, U, D, Z]))
        dl_te_single = DataLoader(ds_te_single, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        
        # 紧凑打印
        evaluate_final_test_rmse(hybrid_model, dl_te_single, t_eval_t, label=f"Patient {pid}")
        
        # 绘制该病人的独立预测图
        plot_rolling_forecast(hybrid_model, Y, U, D, Z, t_eval_t, DT_MINUTES, T_pts, DEVICE, pid)
    
    print("="*70 + "\n")
    print("✅ 实验结束。所有病人的独立图表已保存在 /image 目录下。")

if __name__ == "__main__":
    main()