import numpy as np
import matplotlib.pyplot as plt
from scipy.special import spherical_jn

def bessel_up(x, lmax):
    """向上递推计算球贝塞尔函数
    
    Args:
        x: float, 自变量
        lmax: int, 最大阶数
        
    Returns:
        numpy.ndarray, 从0到lmax阶的球贝塞尔函数值
    """
    # 学生在此实现向上递推算法
    # 提示:
    # 1. 初始化结果数组
    # 2. 计算j_0和j_1的初始值
    # 3. 使用递推公式计算高阶项
    
    j = np.zeros(lmax+1,dtype=np.float64)  
    # 初始化一个长度为lmax+1的NumPy数组j,其中每个数都是64位浮点数，用以储存从j0(x)到jlmax(x) 的所有值
    j[0] = np.sinc(x/np.pi) if x != 0 else 1.0  
    # 球贝塞尔函数的定义j0(x) = sin(x)/x，
    # 如果x≠0，就令j[0]=sin(x)/x，如果x=0，就直接设定 j[0]=1.0

    if lmax >= 1:
        j[1] = (np.sin(x)/x**2-np.cos(x)/x) if x!=0 else 0.0  
        # 球贝塞尔函数的定义j1(x)=(sinx/x^2)-cosx/x
        # 如果x≠0，就令j[1]=(sinx/x^2)-cosx/x，如果x=0，就直接设定 j[1]=0.0
    for l in range(1, lmax):
        j[l+1] = ((2*l+1)/x)*j[l]-j[l-1]
        # 根据向上递推关系式：jl+1(x)=(2l+1/x)jl(x)-jl-1(x)，从j0(x)、j1(x)向上递推到jlmax(x)
        # 返回的是一个长度为lmax​+1的数组j: j=[j0(x),j1(x),j2(x), … ,jlmax(x)]
    return j

def bessel_down(x, lmax, m_start=None):
    """向下递推计算球贝塞尔函数
    
    Args:
        x: float, 自变量
        lmax: int, 最大阶数
        m_start: int, 起始阶数，默认为lmax + 15
        
    Returns:
        numpy.ndarray, 从0到lmax阶的球贝塞尔函数值
    """
    # 学生在此实现向下递推算法
    # 提示:
    # 1. 设置足够高的起始阶数
    # 2. 初始化临时数组并设置初始值
    # 3. 使用递推公式向下计算
    # 4. 使用j_0(x)进行归一化
    
    m_start = m_start or lmax+15  
    # 如果没有在bessel_down(x, lmax, m_start=None)中输入起始阶数，就默认为lmax+15
    j = np.zeros(m_start+1, dtype=np.float64) 
    # 初始化一个长度为lmax+1的NumPy数组j,其中每个数都是64位浮点数，用以储存从j0(x)到jm_start(x)的所有值
    j[m_start] = 0.0   # 设置向下递推需要的两个起始点，其中一个为零，另一个为10^(-30)的小量
    j[m_start-1] = 1e-30  
    
    for l in range(m_start-1, 0, -1):
        j[l-1] = ((2*l+1)/x)*j[l] - j[l+1]
        # 根据向下递推关系式：jl-1(x)=(2l+1/x)jl(x)-jl+1(x)，从高阶向下递推回j0(x)
    j = j[:lmax+1]  # 截取前lmax+1项
    scale = spherical_jn(0, x)/j[0] if j[0] != 0 else 0
    # 归一化
    return j * scale

def plot_comparison(x, lmax):
    """绘制不同方法计算结果的比较图
    
    Args:
        x: float, 自变量
        lmax: int, 最大阶数
    """
    # 学生在此实现绘图功能
    # 提示:
    # 1. 计算三种方法的结果
    # 2. 绘制函数值的半对数图
    # 3. 绘制相对误差的半对数图
    # 4. 添加图例、标签和标题
    l_values = np.arange(lmax+1)
    
    # 计算各方法结果
    j_up = np.abs(bessel_up(x, lmax))
    j_down = np.abs(bessel_down(x, lmax))
    j_scipy = np.abs(spherical_jn(l_values, x))
    
    plt.figure(figsize=(12, 5))
    
    # 函数值比较
    plt.subplot(1, 2, 1)
    plt.semilogy(l_values, j_up, 'o-', label='Upward')
    plt.semilogy(l_values, j_down, 's-', label='Downward')
    plt.semilogy(l_values, j_scipy, 'k--', label='Scipy')
    plt.xlabel('Order l')
    plt.ylabel('|j_l(x)|')
    plt.title(f'Comparison (x={x})')
    plt.legend()
    
    # 相对误差比较
    plt.subplot(1, 2, 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err_up = np.abs((j_up - j_scipy)/j_scipy)
        rel_err_down = np.abs((j_down - j_scipy)/j_scipy)
    rel_err_up[j_scipy == 0] = 0
    rel_err_down[j_scipy == 0] = 0
    
    plt.semilogy(l_values, rel_err_up, 'o-', label='Up Error')
    plt.semilogy(l_values, rel_err_down, 's-', label='Down Error')
    plt.xlabel('Order l')
    plt.ylabel('Relative Error')
    plt.title(f'Relative Error (x={x})')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    # 设置参数
    lmax = 25
    x_values = [0.1, 1.0, 10.0]
    
    # 对每个x值进行计算和绘图
    for x in x_values:
        plot_comparison(x, lmax)
        
        # 打印特定阶数的结果
        l_check = [3, 5, 8]
        print(f"\nx = {x}:")
        print("l\tUp\t\tDown\t\tScipy")
        print("-" * 50)
        for l in l_check:
            j_up = bessel_up(x, l)[l]
            j_down = bessel_down(x, l)[l]
            j_scipy = spherical_jn(l, x)
            print(f"{l}\t{j_up:.6e}\t{j_down:.6e}\t{j_scipy:.6e}")

if __name__ == "__main__":
    main()
