import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import scipy.stats as stats
from scipy.interpolate import UnivariateSpline, CubicSpline
from scipy.integrate import simpson, solve_ivp
import time
import warnings
import gc
warnings.filterwarnings("ignore")

# ==========================================
# 0. 底噪提取器 (利用高频二阶差分无监督榨取)
# ==========================================
def estimate_noise_variance(Y_obs):
    diff2 = Y_obs[:, 2:, :] - 2 * Y_obs[:, 1:-1, :] + Y_obs[:, :-2, :]
    return np.mean(diff2**2) / 6.0

# ==========================================
# 1. 物理引擎 (原汁原味的代数法则，拒绝任何绝对值或截断)
# ==========================================
class SIRPhysicsEngine:
    def f_phys(self, Y, theta):
        S, I = Y[:, 0], Y[:, 1]
        beta, gamma = theta[0], theta[1]
        dS = -beta * S * I
        dI = beta * S * I - gamma * I
        return np.column_stack([dS, dI])

    def grad_theta_f(self, Y, theta):
        N = Y.shape[0]
        grad_theta = np.zeros((N, 2, 2))
        S, I = Y[:, 0], Y[:, 1]
        grad_theta[:, 0, 0] = -S * I
        grad_theta[:, 0, 1] = 0.0
        grad_theta[:, 1, 0] = S * I
        grad_theta[:, 1, 1] = -I
        return grad_theta

# ==========================================
# 2. 数据生成 (高异质性 + 无截断 + 极高精度 ODE)
# ==========================================
def generate_sir_data(N=100, T=40, dt=0.5, seed=None):
    if seed is not None: np.random.seed(seed)
    true_theta = np.array([2.5, 1.0]) 
    dim = 2
    
    Y_list, Y_true_list, Z_list = [], [], []
    engine_gt = SIRPhysicsEngine() 

    for i in range(N):
        # 🌟 异质性 1：极其丰富的初始状态，制造错峰爆发，撑大雅可比矩阵满秩性
        I0 = np.random.uniform(0.005, 0.5)
        S0 = 1.0 - I0
        y0 = np.array([S0, I0]) 
        
        # 🌟 异质性 2：环境管控/免疫力波动的基线和相位随机
        alpha_i = np.random.uniform(0.5, 1.5) 
        phase_i = np.random.uniform(0, 2 * np.pi)
        
        # ⚠️ 必须全人群统一频率和混淆强度，否则将成为无法用 Z 剔除的内生性幽灵
        freq_z = np.random.uniform(0.05, 0.5) 
        nuis_strength = 0.2
        
        def Z_func(t):
            return alpha_i + 0.5 * np.sin(2 * np.pi * freq_z * (t / (T * dt)) + phase_i)
            
        def ode_system(t, y):
            phys = engine_gt.f_phys(y.reshape(1, -1), true_theta).flatten()
            z_val = Z_func(t)
            nuis = np.array([0.0, nuis_strength * z_val]) # 只有感染过程受到隐性干预
            return phys + nuis

        t_span = (0, T * dt)
        t_eval = np.arange(0, T * dt + 1e-8, dt) 
        
        # ⚠️ 必须用极高精度，消除 Python 自身的多项式截断误差
        sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, method='LSODA', rtol=1e-8, atol=1e-8)
        y_fine = sol.y.T 
        z_obs = np.array([Z_func(t) for t in t_eval])
        
        # ⚠️ 绝对不要用 np.clip！允许微弱负数噪音存在，维持统计学 0 均值
        y_obs = y_fine + np.random.normal(0, 0.005, (T+1, dim)) 
        
        Y_list.append(y_obs)
        Y_true_list.append(y_fine)
        Z_list.append(z_obs)

    return np.array(Y_list), np.array(Y_true_list), np.array(Z_list), true_theta

# ==========================================
# 3. Topology 1 DML: 全局平滑 + 代数一致性 + 积分特征匹配
# ==========================================
class UltimateTopology1_DML:
    def __init__(self, dt, sigma_sq_hat, penalty_factor=1.0, micro_steps=10):
        self.dt = dt
        self.micro_steps = micro_steps
        self.dt_micro = dt / micro_steps
        self.sigma_sq_hat = sigma_sq_hat
        self.penalty_factor = penalty_factor
        self.engine = SIRPhysicsEngine()
        self.Y_dense_save = None

    def _compute_high_precision_integrals(self, Y_dense, theta, N_subj, T_pts):
        Dim, P_dim = 2, 2
        f_dense = self.engine.f_phys(Y_dense.reshape(-1, Dim), theta).reshape(N_subj, -1, Dim)
        grad_dense = self.engine.grad_theta_f(Y_dense.reshape(-1, Dim), theta).reshape(N_subj, -1, Dim, P_dim)
        
        F_int = np.zeros((N_subj, T_pts, Dim))
        J_int = np.zeros((N_subj, T_pts, Dim, P_dim))
        
        for k in range(T_pts):
            start_idx = k * self.micro_steps
            end_idx = (k + 1) * self.micro_steps + 1
            F_int[:, k, :] = simpson(f_dense[:, start_idx:end_idx, :], dx=self.dt_micro, axis=1)
            J_int[:, k, :, :] = simpson(grad_dense[:, start_idx:end_idx, :, :], dx=self.dt_micro, axis=1)
            
        return F_int.reshape(-1, Dim), J_int.reshape(-1, Dim * P_dim)

    def fit(self, Y, Z, max_iter=20, tol=1e-4):
        N_subj, T_plus_1, Dim = Y.shape
        T_pts = T_plus_1 - 1
        P_dim = 2
        
        T_eval = np.arange(T_plus_1) * self.dt
        T_dense = np.linspace(0, T_pts * self.dt, T_pts * self.micro_steps + 1)
        
        # -----------------------------------------------------
        # 🚀 绝杀补丁 A：精准计算 Z 的数值积分，消除 ML 离散近似偏误！
        # -----------------------------------------------------
        cs_Z = CubicSpline(T_eval, Z, axis=1)
        Z_dense = cs_Z(T_dense)
        Z_int = np.zeros((N_subj, T_pts, 1), dtype=np.float32)
        for k in range(T_pts):
            start_idx = k * self.micro_steps
            end_idx = (k + 1) * self.micro_steps + 1
            Z_int[:, k, 0] = simpson(Z_dense[:, start_idx:end_idx], dx=self.dt_micro, axis=1)
            
        # 抛弃离散端点，直接喂给 ML 模型最纯粹的真实积分面积！
        Features_ML = Z_int.reshape(-1, 1) 
        # -----------------------------------------------------

        Y_dense = np.zeros((N_subj, len(T_dense), Dim), dtype=np.float32)
        
        # 完美 UnivariateSpline 物理去噪
        for i in range(N_subj):
            for d in range(Dim):
                penalty = T_plus_1 * self.sigma_sq_hat * self.penalty_factor
                smoother = UnivariateSpline(T_eval, Y[i, :, d], s=penalty)
                Y_dense[i, :, d] = smoother(T_dense)
                
        self.Y_dense_save = Y_dense.copy() # 供画图使用
        
        # -----------------------------------------------------
        # 🚀 绝杀补丁 B：代数一致性 (用平滑曲线上的点严格相减，隔离原始噪音)
        # -----------------------------------------------------
        Y_dense_macro = Y_dense[:, ::self.micro_steps, :]
        Y_curr_smooth = np.float32(Y_dense_macro[:, :-1, :].reshape(-1, Dim))
        Y_next_smooth = np.float32(Y_dense_macro[:, 1:, :].reshape(-1, Dim))
        Delta_Y = Y_next_smooth - Y_curr_smooth
                
        theta_est = np.array([1.5, 0.5]) # 合理的初始盲猜

        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        subj_ids = np.repeat(np.arange(N_subj), T_pts)
        cv_masks = [(np.isin(subj_ids, tr), np.isin(subj_ids, va)) for tr, va in kf.split(np.unique(subj_ids))]

        final_G_tilde, final_eps_tilde = None, None

        for k in range(max_iter):
            F_int, J_int_flat = self._compute_high_precision_integrals(Y_dense, theta_est, N_subj, T_pts)
            Target_R = Delta_Y - F_int
            Target_J = J_int_flat

            R_hat, J_hat = np.zeros_like(Target_R), np.zeros_like(Target_J)

            # -----------------------------------------------------
            # 🚀 绝杀补丁 C：闭式解回归 (去掉 lsqr 保证稳定性)
            # -----------------------------------------------------
            for tr_mask, va_mask in cv_masks:
                for d in range(Dim):
                    model_R = make_pipeline(SplineTransformer(n_knots=15, degree=3), Ridge(alpha=1e-5))
                    model_R.fit(Features_ML[tr_mask], Target_R[tr_mask, d])
                    R_hat[va_mask, d] = model_R.predict(Features_ML[va_mask])
                
                for d in range(Dim * P_dim):
                    model_J = make_pipeline(SplineTransformer(n_knots=15, degree=3), Ridge(alpha=1e-5))
                    model_J.fit(Features_ML[tr_mask], Target_J[tr_mask, d])
                    J_hat[va_mask, d] = model_J.predict(Features_ML[va_mask])

            eps_tilde = Target_R - R_hat
            G_tilde = J_int_flat.reshape(-1, Dim, P_dim) - J_hat.reshape(-1, Dim, P_dim)

            A_sys = G_tilde.reshape(-1, P_dim)       
            b_sys = eps_tilde.reshape(-1)         
            
            delta_theta = np.linalg.pinv(A_sys.T @ A_sys) @ (A_sys.T @ b_sys)
            theta_est = np.maximum(theta_est + delta_theta, 0.01) # 防止发散
            
            final_G_tilde = G_tilde
            final_eps_tilde = eps_tilde.reshape(-1, Dim) - np.einsum('ijk,k->ij', G_tilde.reshape(-1, Dim, P_dim), delta_theta)
            final_eps_tilde = final_eps_tilde.reshape(N_subj, T_pts, Dim)

            print(f"Iter {k+1}: Theta={theta_est}, Delta={delta_theta}, Norm Delta={np.linalg.norm(delta_theta):.6f}")

            if np.max(np.abs(delta_theta)) < tol:
                break

        # --- 完美三明治方差推断 ---
        G_tilde_subj = final_G_tilde.reshape(N_subj, T_pts, Dim, P_dim)
        J_sum = np.zeros((P_dim, P_dim))
        Sigma_sum = np.zeros((P_dim, P_dim))
        
        for i in range(N_subj):
            G_i_flat = G_tilde_subj[i].reshape(-1, P_dim) 
            J_sum += G_i_flat.T @ G_i_flat
            psi_i = np.zeros(P_dim)
            for t_step in range(T_pts):
                psi_i += G_tilde_subj[i, t_step].T @ final_eps_tilde[i, t_step]
            Sigma_sum += np.outer(psi_i, psi_i)
            
        J_hat = J_sum / N_subj
        Sigma_hat = Sigma_sum / N_subj
        Var_hat = (1.0 / N_subj) * (np.linalg.pinv(J_hat) @ Sigma_hat @ np.linalg.pinv(J_hat))
        SE = np.sqrt(np.diag(Var_hat))
        
        return theta_est, SE

# ==========================================
# 4. 验证与画图流水线 (N=2000, 挑战极低 SE)
# ==========================================
def run_validation(seed, penalty_factor):
    dt = 0.01
    T = 40
    Y, Y_true, Z, true_theta = generate_sir_data(N=100, T=T, dt=dt, seed=seed)
    
    # 无监督提取原始数据底噪
    estimated_sigma_sq = estimate_noise_variance(Y)
    
    # 调用终极代数大一统架构
    dml = UltimateTopology1_DML(dt=dt, sigma_sq_hat=estimated_sigma_sq, penalty_factor=penalty_factor)
    est_theta, se = dml.fit(Y, Z) 
    
    t_stats = (est_theta - true_theta) / se
    
    sample_traj = {
        'T_eval': np.arange(T+1) * dt,
        'I_obs': Y[0, :, 1],
        'I_true': Y_true[0, :, 1],
        'T_dense': np.linspace(0, T * dt, T * 10 + 1),
        'I_smooth': dml.Y_dense_save[0, :, 1]
    }
    print(f"Seed {seed}: Est={est_theta}, SE={se}, T-stats={t_stats}")
    return est_theta, se, t_stats, sample_traj

if __name__ == "__main__":
    N_SIMS = 100  
    TRUE_THETA = [2.5, 1.0]
    PENALTY_FACTOR = 1.0 
    
    print(f"🚀 Running ULTIMATE DML SIR Validation ({N_SIMS} Sims)...")
    start_time = time.time()
    
    results = Parallel(n_jobs=4, verbose=5)(
        delayed(run_validation)(seed=i, penalty_factor=PENALTY_FACTOR) for i in range(N_SIMS)
    )
    
    ests = np.array([r[0] for r in results])    
    ses = np.array([r[1] for r in results])     
    t_stats = np.array([r[2] for r in results]) 
    sample_traj = results[0][3] 

    print(f"\n✅ 浴火重生：Done in {time.time() - start_time:.2f} seconds.")
    print("\n" + "="*60)
    print(f"{'Param':<8} {'True':<6} {'Mean Est':<10} {'Mean SE':<10} {'Emp SD':<10} {'Coverage':<10}")
    print("-" * 60)
    for i, name in enumerate(['Beta', 'Gamma']):
        mean_est = np.mean(ests[:, i])
        mean_se = np.mean(ses[:, i])
        emp_sd = np.std(ests[:, i]) 
        cov_rate = np.mean(np.abs(t_stats[:, i]) < 1.96) 
        print(f"{name:<8} {TRUE_THETA[i]:<6.1f} {mean_est:<10.4f} {mean_se:<10.4f} {emp_sd:<10.4f} {cov_rate:<10.2%}")
    print("="*60)

    # --- 终极无瑕疵可视化 ---
    plt.figure(figsize=(20, 6))
    
    plt.subplot(1, 4, 1)
    plt.plot(sample_traj['T_eval'], sample_traj['I_obs'], 'o', color='gray', alpha=0.4, markersize=4, label='Noisy Obs (I)')
    plt.plot(sample_traj['T_eval'], sample_traj['I_true'], 'k--', lw=2, label='True Physics (I)')
    plt.plot(sample_traj['T_dense'], sample_traj['I_smooth'], 'r-', lw=2.5, alpha=0.8, label='Spline Smooth (I)')
    plt.title("State Reconstruction (SIR Infection)")
    plt.xlabel("Time (Days)")
    plt.ylabel("Infected Proportion")
    plt.legend()
    plt.grid(True, alpha=0.3)

    colors = ['blue', 'orange']
    x_norm = np.linspace(-4, 4, 100)
    for i, name in enumerate(['Beta', 'Gamma']):
        plt.subplot(1, 4, i + 2)
        plt.plot(x_norm, stats.norm.pdf(x_norm), 'k--', lw=2, label='N(0,1)')
        sns.kdeplot(t_stats[:, i], fill=True, color=colors[i], alpha=0.3, label=f'{name}')
        sns.histplot(t_stats[:, i], bins=15, color=colors[i], alpha=0.5, stat='density', edgecolor='black')
        plt.title(f"T-stats for {name} (True={TRUE_THETA[i]})")
        plt.xlabel("(Est - True) / SE")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 4, 4)
    labels = ['Beta', 'Gamma']
    x_pos = np.arange(len(labels))
    width = 0.35
    plt.bar(x_pos - width/2, np.mean(ses, axis=0), width, label='Theoretical SE')
    plt.bar(x_pos + width/2, np.std(ests, axis=0), width, label='Empirical SD')
    plt.xticks(x_pos, labels)
    plt.title("Variance Consistency Check")
    plt.ylabel("Standard Error")
    plt.legend()
    
    plt.tight_layout()
    plt.show()