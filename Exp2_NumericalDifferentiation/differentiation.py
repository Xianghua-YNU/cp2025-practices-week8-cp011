import numpy as np
import matplotlib.pyplot as plt

# 定义函数 f(x) = x(x-1)
def f(x):
    return x * (x - 1)

# 前向差分法计算导数
def forward_diff(f, x, delta):
    return (f(x + delta) - f(x)) / delta

# 中心差分法计算导数
def central_diff(f, x, delta):
    return (f(x + delta) - f(x - delta)) / (2 * delta)

# 计算解析导数
def analytical_derivative(x):
    return 2 * x - 1

# 计算相对误差
def relative_error(approximate, exact):
    return abs(approximate - exact) / abs(exact)

# 主函数
def main():
    # 定义步长序列
    deltas = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]
    
    # 初始化数据表格
    forward_diff_values = []
    central_diff_values = []
    analytical_values = []
    forward_diff_errors = []
    central_diff_errors = []
    
    # 计算每个步长的数值导数值和相对误差
    for delta in deltas:
        x = 1
        forward_diff_val = forward_diff(f, x, delta)
        central_diff_val = central_diff(f, x, delta)
        analytical_val = analytical_derivative(x)
        
        forward_diff_values.append(forward_diff_val)
        central_diff_values.append(central_diff_val)
        analytical_values.append(analytical_val)
        
        forward_diff_err = relative_error(forward_diff_val, analytical_val)
        central_diff_err = relative_error(central_diff_val, analytical_val)
        
        forward_diff_errors.append(forward_diff_err)
        central_diff_errors.append(central_diff_err)
    
    # 打印数据表格
    print("步长\t\t前向差分值\t中心差分值\t解析值\t\t前向差分相对误差\t中心差分相对误差")
    for i, delta in enumerate(deltas):
        print(f"{delta:.2e}\t{forward_diff_values[i]:.4f}\t{central_diff_values[i]:.4f}\t{analytical_values[i]:.4f}\t{forward_diff_errors[i]:.4e}\t{central_diff_errors[i]:.4e}")
    
    # 绘制误差-步长关系图
    plt.figure(figsize=(10, 6))
    plt.loglog(deltas, forward_diff_errors, 'bo-', label='前向差分相对误差')
    plt.loglog(deltas, central_diff_errors, 'ro-', label='中心差分相对误差')
    plt.xlabel('步长 (delta)')
    plt.ylabel('相对误差')
    plt.title('误差-步长关系图')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

