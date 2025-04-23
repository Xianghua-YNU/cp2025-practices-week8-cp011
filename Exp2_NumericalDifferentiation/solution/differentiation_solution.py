import numpy as np
import matplotlib.pyplot as plt

# 定义测试函数
def f(x):
    return x * (x - 1)

# 定义解析导数
def df_analytical(x):
    return 2 * x - 1

# 前向差分法
def forward_diff(f, x, delta):
    return (f(x + delta) - f(x)) / delta

# 中心差分法
def central_diff(f, x, delta):
    return (f(x + delta) - f(x - delta)) / (2 * delta)

# 计算相对误差
def relative_error(approx, exact):
    return np.abs((approx - exact) / exact)

# 实验参数
x = 1.0
exact_value = df_analytical(x)
delta_values = [10**-i for i in range(2, 15, 2)] + [10**-9, 10**-10, 10**-12, 10**-14]

# 计算结果
forward_results = []
central_results = []
forward_errors = []
central_errors = []

for delta in delta_values:
    # 计算前向差分
    fd = forward_diff(f, x, delta)
    forward_results.append(fd)
    forward_errors.append(relative_error(fd, exact_value))
    
    # 计算中心差分
    cd = central_diff(f, x, delta)
    central_results.append(cd)
    central_errors.append(relative_error(cd, exact_value))

# 打印结果表格
print("步长(δ)\t\t前向差分值\t中心差分值\t解析解\t前向差分相对误差\t中心差分相对误差")
for i, delta in enumerate(delta_values):
    print(f"{delta:.0e}\t{forward_results[i]:.8f}\t{central_results[i]:.8f}\t{exact_value:.1f}\t{forward_errors[i]:.8e}\t{central_errors[i]:.8e}")

# 绘制误差-步长关系图
plt.figure(figsize=(10, 6))
plt.loglog(delta_values, forward_errors, 'o-', label='前向差分')
plt.loglog(delta_values, central_errors, 's-', label='中心差分')
plt.xlabel('步长 δ (对数坐标)', fontsize=12)
plt.ylabel('相对误差 (对数坐标)', fontsize=12)
plt.title('数值微分的误差-步长关系', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, which="both", ls="--")
plt.show()
