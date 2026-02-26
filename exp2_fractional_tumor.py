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
warnings.filterwarnings("ignore")

# ==========================================
# 0. 魔法工具：无模型底噪提取器
# ==========================================
def estimate_noise_variance(Y_obs):
    """
    利用二阶差分从连续轨迹中提取测量白噪音的方差。
    假设局部轨迹近似线性，二阶差分的方差近似为 6 * sigma^2
    """
    diff2 = Y_obs[:, 2:, :] - 2 * Y_obs[:, 1:-1, :] + Y_obs[:, :-2, :]
    return np.mean(diff2**2) / 6.0


# ==========================================
# 1. 物理引擎: 分数阶肿瘤生长模型 (Fractional Tumor Growth)
# ==========================================
class TumorPhysicsEngine:
    def f_phys(self, Y, D, theta):
        """
        dY/dt = theta1 * Y^(2/3) - theta2 * D * Y
        """
        Y_val = Y[:, 0]
        D_val = D[:, 0]
        t1, t2 = theta[0], theta[1]
        
        dy = t1 * (Y_val ** (2/3)) - t2 * D_val * Y_val
        return dy.reshape(-1, 1)

    def grad_theta_f(self, Y, D, theta):
        """
        df/dth1 = Y^(2/3)
        df/dth2 = -D * Y
        """
        N = Y.shape[0]
        grad_theta = np.zeros((N, 1, 2))
        Y_val = Y[:, 0]
        D_val = D[:, 0]
        
        grad_theta[:, 0, 0] = Y_val ** (2/3)
        grad_theta[:, 0, 1] = -D_val * Y_val
        return grad_theta

# ==========================================
# 2. 数据生成: 带有免疫力混淆和观测噪音的临床数据
# ==========================================
def generate_tumor_data(N=100, T=40, dt=0.5, seed=None):
    if seed is not None: np.random.seed(seed)
    true_theta = np.array([1.2, 0.8]) # theta1(自然生长) = 1.2, theta2(药效) = 0.8
    
    Y_list, Y_true_list, D_list, Z_list = [], [], [], []
    engine_gt = TumorPhysicsEngine() 

    for i in range(N):
        # 🌟 异质性 1：初始肿瘤大小千差万别
        y0 = np.array([np.random.uniform(1.0, 3.5)]) 
        
        # 🌟 异质性 2：病人的基础免疫力和波动周期完全独立
        z_base = np.random.uniform(0.8, 1.5)
        freq_z = np.random.uniform(0.5, 2.0)
        phase_z = np.random.uniform(0, 2*np.pi)
        
        def Z_func(t):
            return z_base + 0.4 * np.sin(2 * np.pi * freq_z * (t / (T*dt)) + phase_z)
            
        # 🌟 异质性 3：极其真实的个性化给药方案 (且存在深度混淆)
        freq_d = np.random.uniform(0.5, 3.0) 
        phase_d = np.random.uniform(0, 2*np.pi)
        d_base = np.random.uniform(1.0, 2.0) # 【修改这里】之前是 0.5~1.5
        confound_strength = np.random.uniform(0.3, 0.8) 
        
        def D_func(t):
            d_val = d_base + confound_strength * Z_func(t) + 0.6 * np.cos(2 * np.pi * freq_d * (t / (T*dt)) + phase_d)
            # 【核心修改】：彻底删掉 np.maximum！恢复物理轨迹的绝对平滑！
            return d_val
            
        def ode_system(t, y):
            d_val = np.array([[D_func(t)]])
            phys = engine_gt.f_phys(y.reshape(1, -1), d_val, true_theta).flatten()
            
            # 🌟 异质性 4：免疫力对不同病人肿瘤的暗中抑制效率也不同
            nuis_strength = np.random.uniform(0.1, 0.35)
            nuis = np.array([nuis_strength * Z_func(t)])
            return phys + nuis

        t_span = (0, T * dt)
        t_eval = np.arange(0, T * dt + 1e-8, dt) 
        
        sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-5, atol=1e-5)
        y_fine = sol.y.T 
        
        d_obs = np.array([[D_func(t)] for t in t_eval])
        z_obs = np.array([[Z_func(t)] for t in t_eval])
        
        # 观测医学影像底噪 (保持 0.2 挑战算法稳定性)
        y_obs = y_fine + np.random.normal(0, 0.005, (T+1, 1)) 
        
        # ⚠️绝对不能限制上限 10.0，否则算法会爆掉！只保证体积大于 0.05
        y_obs = np.maximum(y_obs, 0.05) 
        
        Y_list.append(y_obs)
        D_list.append(d_obs)
        Z_list.append(z_obs)

    return np.array(Y_list), np.array(D_list), np.array(Z_list), true_theta

# ==========================================
# 3. Topology 1 DML 算法本体
# ==========================================
class TumorIntegralDML:
    def __init__(self, dt, sigma_sq_hat, micro_steps=10):
        self.dt = dt
        self.micro_steps = micro_steps
        self.dt_micro = dt / micro_steps
        self.sigma_sq_hat = sigma_sq_hat  # 接收底噪
        self.engine = TumorPhysicsEngine()

    def fit(self, Y, D, Z, max_iter=15, tol=1e-4):
        N_subj, T_plus_1, Dim = Y.shape
        T_pts = T_plus_1 - 1
        P_dim = 2
        
        Y_curr = np.float32(Y[:, :-1, :].reshape(-1, Dim))
        Y_next = np.float32(Y[:, 1:, :].reshape(-1, Dim))
        Delta_Y = Y_next - Y_curr
        
        Z_curr = np.float32(Z[:, :-1, :].reshape(-1, 1))
        Z_next = np.float32(Z[:, 1:, :].reshape(-1, 1))
        Features_ML = np.hstack([Z_curr, Z_next])
        
        # 🚀【Topology I 核心】：因为有 Y^(2/3)，必须用 UnivariateSpline 彻底干掉噪音！
        T_eval = np.arange(T_plus_1) * self.dt
        T_dense = np.linspace(0, T_pts * self.dt, T_pts * self.micro_steps + 1)
        
        Y_dense = np.zeros((N_subj, len(T_dense), Dim))
        for i in range(N_subj):
            # 🚀 极其关键：利用动态榨取出的底噪方差，制定理论最优惩罚边界！
            optimal_s = T_plus_1 * self.sigma_sq_hat
            smoother = UnivariateSpline(T_eval, Y[i, :, 0], s=optimal_s) 
            Y_dense[i, :, 0] = np.maximum(smoother(T_dense), 0.01)
                
        cs_D = CubicSpline(T_eval, D[:, :, 0], axis=1)
        D_dense = cs_D(T_dense).reshape(N_subj, -1, 1)
                
        theta_est = np.array([0.5, 0.5]) # 瞎猜一个初始值

        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        subj_ids = np.repeat(np.arange(N_subj), T_pts)
        cv_masks = [(np.isin(subj_ids, tr), np.isin(subj_ids, va)) for tr, va in kf.split(np.unique(subj_ids))]

        final_G_tilde = None
        final_eps_tilde = None

        for k in range(max_iter):
            f_dense = self.engine.f_phys(Y_dense.reshape(-1, Dim), D_dense.reshape(-1, Dim), theta_est).reshape(N_subj, -1, Dim)
            grad_dense = self.engine.grad_theta_f(Y_dense.reshape(-1, Dim), D_dense.reshape(-1, Dim), theta_est).reshape(N_subj, -1, Dim, P_dim)
            
            F_int = np.zeros((N_subj, T_pts, Dim))
            J_int = np.zeros((N_subj, T_pts, Dim, P_dim))
            
            for step in range(T_pts):
                start_idx = step * self.micro_steps
                end_idx = (step + 1) * self.micro_steps + 1
                F_int[:, step, :] = simpson(f_dense[:, start_idx:end_idx, :], dx=self.dt_micro, axis=1)
                J_int[:, step, :, :] = simpson(grad_dense[:, start_idx:end_idx, :, :], dx=self.dt_micro, axis=1)
            
            F_int_flat = F_int.reshape(-1, Dim)
            J_int_flat = J_int.reshape(-1, Dim * P_dim)

            Target_R = Delta_Y - F_int_flat
            Target_J = J_int_flat

            R_hat = np.zeros_like(Target_R) 
            J_hat = np.zeros_like(Target_J)

            # 稳如泰山的 DML Nuisance 估计
            for tr_mask, va_mask in cv_masks:
                model_R = make_pipeline(SplineTransformer(n_knots=15, degree=3), Ridge(alpha=1e-4))
                model_R.fit(Features_ML[tr_mask], Target_R[tr_mask, 0])
                R_hat[va_mask, 0] = model_R.predict(Features_ML[va_mask])
                
                for d in range(P_dim):
                    model_J = make_pipeline(SplineTransformer(n_knots=15, degree=3), Ridge(alpha=1e-4))
                    model_J.fit(Features_ML[tr_mask], Target_J[tr_mask, d])
                    J_hat[va_mask, d] = model_J.predict(Features_ML[va_mask])

            eps_tilde = Target_R - R_hat
            G_tilde = J_int_flat.reshape(-1, Dim, P_dim) - J_hat.reshape(-1, Dim, P_dim)

            A_sys = G_tilde.reshape(-1, P_dim)       
            b_sys = eps_tilde.reshape(-1)         
            
            delta_theta = np.linalg.pinv(A_sys.T @ A_sys) @ (A_sys.T @ b_sys)
            theta_est = np.maximum(theta_est + delta_theta, 0.05)
            
            final_G_tilde = G_tilde
            final_eps_tilde = eps_tilde.reshape(-1, Dim) - np.einsum('ijk,k->ij', G_tilde.reshape(-1, Dim, P_dim), delta_theta)
            final_eps_tilde = final_eps_tilde.reshape(N_subj, T_pts, Dim)

            if np.max(np.abs(delta_theta)) < tol:
                break

        # --- 三明治方差推断 ---
        G_tilde_subj = final_G_tilde.reshape(N_subj, T_pts, Dim, P_dim)
        
        J_sum = np.zeros((P_dim, P_dim))
        Sigma_sum = np.zeros((P_dim, P_dim))
        
        for i in range(N_subj):
            G_i_flat = G_tilde_subj[i].reshape(-1, P_dim) 
            J_sum += G_i_flat.T @ G_i_flat
            
            psi_i = np.zeros(P_dim)
            for t_step in range(T_pts):
                psi_i += G_tilde_subj[i, t_step, 0, :].T * final_eps_tilde[i, t_step, 0]
            Sigma_sum += np.outer(psi_i, psi_i)
            
        J_hat = J_sum / N_subj
        Sigma_hat = Sigma_sum / N_subj
        J_inv = np.linalg.pinv(J_hat)
        Var_hat = (1.0 / N_subj) * (J_inv @ Sigma_hat @ J_inv)
        SE = np.sqrt(np.diag(Var_hat))
        
        return theta_est, SE

# ==========================================
# 4. 验证流水线
# ==========================================
def run_validation(seed):
    print(f"Seed {seed}: Starting data generation...")
    dt = 0.01
    Y, D, Z, true_theta = generate_tumor_data(N=2000, T=50, dt=dt, seed=seed)
    print(f"Seed {seed}: Data generated complete. Starting DML fitting...")
    
    # 动态榨取真实底噪！
    estimated_sigma_sq = estimate_noise_variance(Y)

    print(f"Seed {seed}: Estimated Noise Variance = {estimated_sigma_sq:.6f}")
    
    # 传入 DML 引擎
    dml = TumorIntegralDML(dt=dt, sigma_sq_hat=estimated_sigma_sq)
    est_theta, se = dml.fit(Y, D, Z) 
    
    t_stats = (est_theta - true_theta) / se

    print(f"Seed {seed}: Estimated Theta = {est_theta}, SE = {se}, T-stats = {t_stats}")
    return est_theta, se, t_stats

if __name__ == "__main__":
    N_SIMS = 50  
    TRUE_THETA = [1.2, 0.8]
    
    print(f"🚀 Running TOPOLOGY I (Fractional Tumor Growth) Validation...")
    start_time = time.time()
    
    results = Parallel(n_jobs=8, verbose=5)(
        delayed(run_validation)(seed=i) for i in range(N_SIMS)
    )
    
    ests = np.array([r[0] for r in results])    
    ses = np.array([r[1] for r in results])     
    t_stats = np.array([r[2] for r in results]) 
    
    print(f"\nDone in {time.time() - start_time:.2f} seconds.")
    print("\n" + "="*60)
    print(f"{'Param':<8} {'True':<6} {'Mean Est':<10} {'Mean SE':<10} {'Emp SD':<10} {'Coverage':<10}")
    print("-" * 60)
    for i, name in enumerate(['Growth_T1', 'Effect_T2']):
        mean_est = np.mean(ests[:, i])
        mean_se = np.mean(ses[:, i])
        emp_sd = np.std(ests[:, i]) 
        cov_rate = np.mean(np.abs(t_stats[:, i]) < 1.96) 
        print(f"{name:<8} {TRUE_THETA[i]:<6.1f} {mean_est:<10.4f} {mean_se:<10.4f} {emp_sd:<10.4f} {cov_rate:<10.2%}")
    print("="*60)

    # --- 绝美可视化模块 ---
    plt.figure(figsize=(18, 6))
    
    colors = ['blue', 'orange']
    for i, name in enumerate(['Growth_T1', 'Effect_T2']):
        plt.subplot(1, 2, i + 1)
        x = np.linspace(-4, 4, 100)
        plt.plot(x, stats.norm.pdf(x), 'k--', lw=2, label='N(0,1)')
        sns.kdeplot(t_stats[:, i], fill=True, color=colors[i], alpha=0.4, label=f'Est {name}')
        sns.histplot(t_stats[:, i], bins=50, color=colors[i], alpha=0.3, stat='density')
        plt.title(f"Asymptotic Normality: {name}\nCoverage: {np.mean(np.abs(t_stats[:, i]) < 1.96):.1%}")
        plt.xlabel("T-statistic")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.show()


    """
    ============================================================
    Param    True   Mean Est   Mean SE    Emp SD     Coverage  
    ------------------------------------------------------------
    Growth_T1 1.2    1.1998     0.0019     0.0013     98.00%    
    Effect_T2 0.8    0.8001     0.0004     0.0004     98.00%    
    ============================================================
    """