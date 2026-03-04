import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import scipy.stats as stats
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline, LSQUnivariateSpline
from scipy.integrate import simpson, solve_ivp
import time
import warnings
import gc
warnings.filterwarnings("ignore")

# ==========================================
# 1. 物理引擎 (极简原生代数形态，无需泰勒校正)
# ==========================================
class AlgebraicPhysicsEngine:
    def f_phys(self, Y, D, theta):
        """物理公式 f(Y, D, theta): Holling Type II"""
        Y1, Y2, Y3 = Y[:, 0], Y[:, 1], Y[:, 2]
        D1, D2, D3 = D[:, 0], D[:, 1], D[:, 2]
        t1, t2, t3 = theta[0], theta[1], theta[2]
        
        dy1 = D1 - (Y1 * Y2) / (t1 + Y1)
        dy2 = D2 - (Y2 * Y3) / (t2 + Y2)
        dy3 = D3 - (Y3 * Y1) / (t3 + Y3)
        
        return np.column_stack([dy1, dy2, dy3])

    def grad_theta_f(self, Y, D, theta):
        """f 关于 theta 的绝对解析导数 (对角阵)"""
        N = Y.shape[0]
        grad_theta = np.zeros((N, 3, 3))
        Y1, Y2, Y3 = Y[:, 0], Y[:, 1], Y[:, 2]
        t1, t2, t3 = theta[0], theta[1], theta[2]
        
        grad_theta[:, 0, 0] = (Y1 * Y2) / (t1 + Y1)**2
        grad_theta[:, 1, 1] = (Y2 * Y3) / (t2 + Y2)**2
        grad_theta[:, 2, 2] = (Y3 * Y1) / (t3 + Y3)**2
        
        return grad_theta

# ==========================================
# 2. 数据生成
# ==========================================
def generate_coupled_data(N=1000, T=30, dt=0.05, seed=None):
    if seed is not None: np.random.seed(seed)
    true_theta = np.array([0.5, 1.0, 1.5])
    dim = 3
    
    Y_list, D_list, Z_list = [], [], []
    engine_gt = AlgebraicPhysicsEngine() 

    for i in range(N):
        alpha_i = np.random.uniform(1.0, 3.0) 
        phase_i = np.random.uniform(0, 2 * np.pi)
        freq_i = np.random.uniform(0.8, 1.2) 
        
        freq_indep = np.random.uniform(1.5, 2.5, dim) 
        phase_indep = np.random.uniform(0, 2 * np.pi, dim)
        
        def Z_func(t):
            return alpha_i + 0.5 * np.sin(2 * np.pi * freq_i * (t / (T * dt)) + phase_i)
            
        def D_func(t):
            indep = 1.0 * np.sin(2 * np.pi * freq_indep * (t / (T * dt)) + phase_indep)
            return np.abs(Z_func(t) + indep)
            
        def ode_system(t, y):
            d_val = D_func(t).reshape(1, -1)
            phys = engine_gt.f_phys(y.reshape(1, -1), d_val, true_theta).flatten()
            z_val = Z_func(t)
            nuis = np.array([0.5 * np.sin(z_val), 0.5 * np.cos(z_val), -0.5 * np.sin(2 * z_val)])
            return phys + nuis

        t_span = (0, T * dt)
        t_eval = np.arange(0, T * dt + 1e-8, dt) 
        y0 = np.random.uniform(1.0, 3.0, dim)
        
        sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-8)
        y_fine = sol.y.T 
        
        d_obs = np.array([D_func(t) for t in t_eval])
        z_obs = np.array([Z_func(t) for t in t_eval])
        
        # 加入仪器测量底噪 (真实方差为 0.005^2 = 0.000025)
        y_obs = y_fine + np.random.normal(0, 0.005, (T+1, dim)) 
        y_obs = np.maximum(y_obs, 0.01) # 防止越界
        
        Y_list.append(y_obs)
        D_list.append(d_obs)
        Z_list.append(z_obs)

    return np.array(Y_list), np.array(D_list), np.array(Z_list), true_theta

# ==========================================
# 3. Integral DML 实现
# ==========================================

class UltimateIntegralDML:
    def __init__(self, dt, micro_steps=10):
        self.dt = dt
        self.micro_steps = micro_steps
        self.dt_micro = dt / micro_steps
        self.engine = AlgebraicPhysicsEngine()

    def _compute_high_precision_integrals(self, Y_dense, D_dense, theta, N_subj, T_pts):
        Dim = Y_dense.shape[-1]
        
        f_dense = self.engine.f_phys(Y_dense.reshape(-1, Dim), D_dense.reshape(-1, Dim), theta).reshape(N_subj, -1, Dim)
        grad_dense = self.engine.grad_theta_f(Y_dense.reshape(-1, Dim), D_dense.reshape(-1, Dim), theta).reshape(N_subj, -1, Dim, Dim)
        
        F_int = np.zeros((N_subj, T_pts, Dim))
        J_int = np.zeros((N_subj, T_pts, Dim, Dim))
        
        for k in range(T_pts):
            start_idx = k * self.micro_steps
            end_idx = (k + 1) * self.micro_steps + 1
            
            f_interval = f_dense[:, start_idx:end_idx, :]
            grad_interval = grad_dense[:, start_idx:end_idx, :, :]
            
            # 沿着微观网格进行辛普森高阶积分
            F_int[:, k, :] = simpson(f_interval, dx=self.dt_micro, axis=1)
            J_int[:, k, :, :] = simpson(grad_interval, dx=self.dt_micro, axis=1)
            
        del f_dense, grad_dense
        gc.collect()
        
        return F_int.reshape(-1, Dim), J_int.reshape(-1, Dim * Dim)

    def _estimate_initial_theta(self, Delta_Y, Y_dense, D_dense, N_subj, T_pts):
        def objective(th):
            F_int_flat, _ = self._compute_high_precision_integrals(Y_dense, D_dense, th, N_subj, T_pts)
            return (Delta_Y - F_int_flat).flatten()
        res = least_squares(objective, x0=np.array([5.0, 5.0, 5.0]), bounds=(0.01, np.inf))
        return res.x

    def fit(self, Y, D, Z, max_iter=20, tol=1e-10):
        N_subj, T_plus_1, Dim = Y.shape
        T_pts = T_plus_1 - 1
        
        Y_curr = np.float32(Y[:, :-1, :].reshape(-1, Dim))
        Y_next = np.float32(Y[:, 1:, :].reshape(-1, Dim))
        Delta_Y = Y_next - Y_curr
        
        Z_curr = np.float32(Z[:, :-1].reshape(-1, 1))
        Z_next = np.float32(Z[:, 1:].reshape(-1, 1))
        Features_ML = np.hstack([Z_curr, Z_next])
        
        # LSQUnivariateSpline 彻底干掉噪音，利用 CubicSpline 完美插值驱动力
        T_eval = np.arange(T_plus_1) * self.dt
        T_dense = np.linspace(0, T_pts * self.dt, T_pts * self.micro_steps + 1)
        
        Y_dense = np.zeros((N_subj, len(T_dense), Dim), dtype=np.float32)
        
        K_knots = 10  # 硬性规定 10 个内部节点 (保证 K << M，对应于黄理论中的截断)
        # 均匀生成内部节点 (必须严格在起点和终点之内)
        interior_knots = np.linspace(T_eval[1], T_eval[-2], K_knots)
        
        for i in range(N_subj):
            for d in range(Dim):
                smoother = LSQUnivariateSpline(T_eval, Y[i, :, d], t=interior_knots)
                Y_dense[i, :, d] = np.maximum(smoother(T_dense), 0.01)
                
        cs_D = CubicSpline(T_eval, D, axis=1)
        D_dense = np.float32(cs_D(T_dense))
        
        theta_est = self._estimate_initial_theta(Delta_Y, Y_dense, D_dense, N_subj, T_pts)

        kf = KFold(n_splits=2, shuffle=True, random_state=42)
        subj_ids = np.repeat(np.arange(N_subj), T_pts)
        cv_masks = [(np.isin(subj_ids, tr), np.isin(subj_ids, va)) for tr, va in kf.split(np.unique(subj_ids))]

        final_G_tilde = None
        final_eps_tilde = None

        for k in range(max_iter):
            F_int, J_int_flat = self._compute_high_precision_integrals(Y_dense, D_dense, theta_est, N_subj, T_pts)

            Target_R = Delta_Y - F_int
            Target_J = J_int_flat

            R_hat = np.zeros_like(Target_R) 
            J_hat = np.zeros_like(Target_J)

            for tr_mask, va_mask in cv_masks:
                for d in range(Dim):
                    model_R = make_pipeline(SplineTransformer(n_knots=15, degree=3), Ridge(alpha=1e-5))
                    model_R.fit(Features_ML[tr_mask], Target_R[tr_mask, d])
                    R_hat[va_mask, d] = model_R.predict(Features_ML[va_mask])
                
                for d in range(Dim * Dim):
                    model_J = make_pipeline(SplineTransformer(n_knots=15, degree=3), Ridge(alpha=1e-5))
                    model_J.fit(Features_ML[tr_mask], Target_J[tr_mask, d])
                    J_hat[va_mask, d] = model_J.predict(Features_ML[va_mask])

            eps_tilde = Target_R - R_hat
            G_tilde = J_int_flat.reshape(-1, Dim, Dim) - J_hat.reshape(-1, Dim, Dim)

            A_sys = G_tilde.reshape(-1, Dim)      
            b_sys = eps_tilde.reshape(-1)         
            
            ATA = A_sys.T @ A_sys                
            ATb = A_sys.T @ b_sys                
            
            # 计算当前 theta_est 下的经验得分函数 (Empirical Score)
            # 对应公式: \Psi_M = (1 / NT) * \sum \tilde{G}_{i,j}^T \tilde{\epsilon}_{i,j}
            empirical_score = ATb / (N_subj * T_pts)
            
            delta_theta = np.linalg.pinv(ATA) @ ATb
            
            final_G_tilde = G_tilde
            G_flat_final = G_tilde.reshape(-1, Dim, Dim)
            correction = np.einsum('ijk,k->ij', G_flat_final, delta_theta)
            e_flat_corrected = eps_tilde.reshape(-1, Dim) - correction
            final_eps_tilde = e_flat_corrected.reshape(N_subj, T_pts, Dim)
            
            print(f"Iteration {k+1}: Theta = {theta_est}, Empirical Score = {empirical_score}, Delta Theta = {delta_theta}")

            # 当经验得分函数的最大绝对值充分趋近于 0 时终止
            if np.max(np.abs(empirical_score)) < tol:
                break
                
            # 保证参数的物理非负性下界
            theta_est = np.maximum(theta_est + delta_theta, 0.01)

        # --- 三明治方差计算 ---
        G_tilde_subj = final_G_tilde.reshape(N_subj, T_pts, Dim, Dim)   # type: ignore
        eps_tilde_subj = final_eps_tilde 
        
        J_sum = np.zeros((Dim, Dim))
        Sigma_sum = np.zeros((Dim, Dim))
        
        for i in range(N_subj):
            G_i = G_tilde_subj[i]       
            e_i = eps_tilde_subj[i] # type: ignore 
            G_i_flat = G_i.reshape(-1, Dim) 
            J_sum += G_i_flat.T @ G_i_flat
            
            psi_i = np.zeros(Dim)
            for t_step in range(T_pts):
                psi_i += G_i[t_step].T @ e_i[t_step]
            Sigma_sum += np.outer(psi_i, psi_i)
            
        J_hat = J_sum / N_subj
        Sigma_hat = Sigma_sum / N_subj
        
        try:
            J_inv = np.linalg.inv(J_hat)
        except np.linalg.LinAlgError:
            J_inv = np.linalg.pinv(J_hat)
            
        Var_hat = (1.0 / N_subj) * (J_inv @ Sigma_hat @ J_inv)
        SE = np.sqrt(np.diag(Var_hat))
        
        return theta_est, SE
# ==========================================
# 4. 验证实验流水线
# ==========================================
def run_validation(seed):
    dt = 0.01
    Y, D, Z, true_theta = generate_coupled_data(N=1000, T=50, dt=dt, seed=seed)

    
    dml = UltimateIntegralDML(dt=dt)
    est_theta, se = dml.fit(Y, D, Z, max_iter=20, tol=1e-10) 
    
    t_stats = (est_theta - true_theta) / se
    return est_theta, se, t_stats

if __name__ == "__main__":
    N_SIMS = 1000  
    TRUE_THETA = [0.5, 1.0, 1.5]
    
    print(f"🚀 Running Ultimate Coupled DML Validation ({N_SIMS} Sims)...")
    start_time = time.time()
    
    # 注意控制 n_jobs，防止高精网格干爆内存
    results = Parallel(n_jobs=16, verbose=5)(
        delayed(run_validation)(seed=i) for i in range(N_SIMS)
    )
    
    ests = np.array([r[0] for r in results])   # type: ignore 
    ses = np.array([r[1] for r in results]) # type: ignore 
    t_stats = np.array([r[2] for r in results]) # type: ignore 
    
    print(f"\nDone in {time.time() - start_time:.2f} seconds.")
    
    print("\n" + "="*60)
    print(f"{'Param':<8} {'True':<6} {'Mean Est':<10} {'Mean SE':<10} {'Emp SD':<10} {'Coverage':<10}")
    print("-" * 60)
    for i in range(3):
        mean_est = np.mean(ests[:, i])
        mean_se = np.mean(ses[:, i])
        emp_sd = np.std(ests[:, i]) 
        cov_rate = np.mean(np.abs(t_stats[:, i]) < 1.96) 
        
        print(f"Theta_{i+1:<2} {TRUE_THETA[i]:<6.1f} {mean_est:<10.4f} {mean_se:<10.4f} {emp_sd:<10.4f} {cov_rate:<10.2%}")
    print("="*60)

    # --- Visualization ---
    plt.figure(figsize=(20, 5)) 
    x = np.linspace(-4, 4, 100)
    colors = ['blue', 'green', 'orange']
    
    for i in range(3):
        plt.subplot(1, 4, i + 1)
        plt.plot(x, stats.norm.pdf(x), 'k--', lw=2, label='N(0,1)')
        sns.kdeplot(t_stats[:, i], fill=True, color=colors[i], alpha=0.3, label=f'Theta_{i+1}')
        sns.histplot(t_stats[:, i], bins=15, color=colors[i], alpha=0.5, stat='density', edgecolor='black')
        
        plt.title(f"T-stats for Theta_{i+1}\nTrue={TRUE_THETA[i]}")
        plt.xlabel("(Est - True) / SE")
        if i == 0:
            plt.ylabel("Density")
        else:
            plt.ylabel("") 
            
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 4, 4)
    labels = ['Theta_1', 'Theta_2', 'Theta_3']
    x_pos = np.arange(len(labels))
    width = 0.35
    
    mean_ses = np.mean(ses, axis=0)
    emp_sds = np.std(ests, axis=0)
    
    plt.bar(x_pos - width/2, mean_ses, width, label='Theoretical SE (Formula)')
    plt.bar(x_pos + width/2, emp_sds, width, label='Empirical SD (Monte Carlo)')
    
    plt.xticks(x_pos, labels)
    plt.title("Variance Consistency Check")
    plt.ylabel("Standard Error")
    plt.legend()
    
    plt.tight_layout()
    plt.show()


"""
============================================================
Param    True   Mean Est   Mean SE    Emp SD     Coverage  
------------------------------------------------------------
Theta_1  0.5    0.5001     0.0020     0.0021     94.90%    
Theta_2  1.0    1.0003     0.0034     0.0036     93.70%    
Theta_3  1.5    1.4999     0.0049     0.0051     93.20%    
============================================================
"""
