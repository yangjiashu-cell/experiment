import cupy as cp
import pandas as pd
from cupy.linalg import pinv as cp_pinv
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# 数据准备
n_samples = 445
n_features = 1
lambda_values = cp.logspace(-3, 0, 50)
data_path = '/root/autodl-tmp/LaLonde.csv'  # 本地数据路径
data = pd.read_csv(data_path)
X = data['treat'].values.reshape(n_samples, n_features)
Y = data['re78'].values
W = data['re75'].values.reshape(n_samples, n_features)

# 标准化 X 和 W
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_W = StandardScaler()
W_scaled = scaler_W.fit_transform(W)

# 将数据转移到 GPU
X_gpu = cp.array(X_scaled)
W_gpu = cp.array(W_scaled)
Y_gpu = cp.array(Y)

# 计算带宽参数
distances_X = pdist(X_scaled, metric='euclidean')
sigma_X = cp.median(cp.array(distances_X))
distances_W = pdist(W_scaled, metric='euclidean')
sigma_W = cp.median(cp.array(distances_W))

# 检查 sigma_X 是否为 0
if sigma_X == 0:
    print("Warning: sigma_X is 0. Setting to 1.0 for RBF kernel.")
    sigma_X = 1.0

print(f"sigma_X: {float(sigma_X)}, sigma_W: {float(sigma_W)}")

# 计算 Gram 矩阵
def compute_gram_matrices(X, W, sigma_X, sigma_W):
    X_diff = X[:, None] - X[None, :]
    W_diff = W[:, None] - W[None, :]
    K_X = cp.exp(-cp.sum(X_diff**2, axis=2) / (2 * sigma_X**2))
    K_W = cp.exp(-cp.sum(W_diff**2, axis=2) / (2 * sigma_W**2))
    return K_X, K_W

K_X, K_W = compute_gram_matrices(X_gpu, W_gpu, sigma_X, sigma_W)

# 特征函数 phi(Y, t)
def phi(Y, t):
    return cp.exp(1j * t * Y)

# 权重函数 m(X, s)
def m(X, s):
    return cp.exp(1j * s * X)

# 估计 H(W, t)
def estimate_H(t, lambda_val, K_X, K_W, Y):
    phi_Y_t = phi(Y, t)
    matrix = K_W @ K_X @ K_W + lambda_val * n_samples**2 * K_W
    alpha = cp_pinv(matrix) @ (K_W @ K_X @ phi_Y_t)
    return alpha

# 交叉验证选择 lambda（批量处理）
def cross_validate_lambda(t_values, X, Y, W, K_X, K_W):
    kf = KFold(n_splits=5)
    best_lambdas = []
    for t in t_values:
        best_lambda = None
        best_score = cp.inf
        for lambda_val in lambda_values:
            scores = []
            for train_idx, val_idx in kf.split(X.get()):
                X_train = X[train_idx]
                Y_train = Y[train_idx]
                W_train = W[train_idx]
                K_X_train = K_X[train_idx][:, train_idx]
                K_W_train = K_W[train_idx][:, train_idx]
                K_W_val_train = K_W[val_idx][:, train_idx]
                alpha = estimate_H(t, lambda_val, K_X_train, K_W_train, Y_train)
                H_W_val = K_W_val_train @ alpha
                Delta_val = phi(Y_train[val_idx], t) - H_W_val
                score = cp.mean(cp.abs(Delta_val)**2)
                scores.append(score)
            avg_score = cp.mean(cp.array(scores))
            if avg_score < best_score:
                best_score = avg_score
                best_lambda = lambda_val
        best_lambdas.append(best_lambda)
    return best_lambdas

# 计算残差 U
def compute_U(t, alpha, K_W, Y):
    H_W_t = K_W @ alpha
    U = phi(Y, t) - H_W_t
    return U

# 向量化计算 T_n(s, t)
def compute_Tn(s_values, t, U, X):
    m_X_s = m(X, s_values[:, None])
    Tn = (1 / cp.sqrt(n_samples)) * cp.sum(U * m_X_s, axis=1)
    return Tn

# 梯形积分
def trapz(y, x):
    dx = x[1:] - x[:-1]
    return cp.sum((y[:-1] + y[1:]) * dx / 2)

# 计算 Delta_phi_m（批量处理）
def compute_Delta_phi_m(X, Y, W, t_values, s_values):
    best_lambdas = cross_validate_lambda(t_values, X, Y, W, K_X, K_W)
    Delta_values = []
    for t, best_lambda in zip(t_values, best_lambdas):
        alpha = estimate_H(t, best_lambda, K_X, K_W, Y)
        U = compute_U(t, alpha, K_W, Y)
        Tn_values = compute_Tn(s_values, t, U, X)
        Tn_abs_squared = cp.abs(Tn_values)**2
        integral_approx = trapz(Tn_abs_squared, s_values)
        Delta_values.append(integral_approx)
    return cp.max(cp.array(Delta_values))

# 并行 bootstrap
def parallel_bootstrap(idx, X, Y, W, t_values, s_values):
    X_boot = X[idx]
    Y_boot = Y[idx]
    W_boot = W[idx]
    return compute_Delta_phi_m(X_boot, Y_boot, W_boot, t_values, s_values).get()

# Bootstrap p 值估计
def bootstrap_p_value(X, Y, W, t_values, s_values, n_bootstraps):
    Delta_obs = compute_Delta_phi_m(X, Y, W, t_values, s_values)
    indices_list = [cp.random.choice(n_samples, n_samples, replace=True).get() for _ in range(n_bootstraps)]
    
    with Parallel(n_jobs=-1) as parallel:
        bootstrap_Deltas = parallel(
            delayed(parallel_bootstrap)(idx, X, Y, W, t_values, s_values)
            for idx in indices_list
        )
    
    bootstrap_Deltas = cp.array(bootstrap_Deltas)
    p_value = cp.mean(bootstrap_Deltas >= Delta_obs).get()
    return p_value, Delta_obs.get(), bootstrap_Deltas.get()

# 设置 t_values 和 s_values
std_Y = cp.std(Y_gpu).get()
std_X = cp.std(X_gpu).get()
t_values = cp.linspace(-2*std_Y, 2*std_Y, 100)
s_values = cp.linspace(-2*std_X, 2*std_X, 100)

# 运行实验
n_experiments = 10
p_values = []
for i in tqdm(range(n_experiments), desc="Running experiments"):
    cp.random.seed(i)
    p_value, Delta_obs, bootstrap_Deltas = bootstrap_p_value(X_gpu, Y_gpu, W_gpu, t_values, s_values, n_bootstraps=100)
    p_values.append(p_value)
    tqdm.write(f"Experiment {i+1}: p-value = {p_value}")

avg_p_value = cp.mean(cp.array(p_values)).get()
rejection_rate = cp.mean(cp.array([1 if p < 0.05 else 0 for p in p_values])).get()
print(f"Average p-value: {avg_p_value}, Rejection rate: {rejection_rate}")

# 可视化
plt.hist(bootstrap_Deltas, bins=30, density=True, alpha=0.7, label='Bootstrap Delta')
plt.axvline(Delta_obs, color='r', linestyle='--', label=f'Observed Delta = {Delta_obs:.2f}')
plt.title('Bootstrap Distribution of Delta')
plt.xlabel('Delta')
plt.ylabel('Density')
plt.legend()
plt.show()