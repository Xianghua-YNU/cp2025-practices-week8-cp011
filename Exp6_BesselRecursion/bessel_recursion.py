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
    # 初始化一个长度为lmax+1的NumPy数组j,其中每个数都是64位浮点数，用来储存从j0(x)到jlmax(x)的所有值
    j[0] = np.sinc(x/np.pi) if x != 0 else 1.0  
    # 球贝塞尔函数的定义式为 j0(x) = sin(x)/x，
    # 如果x≠0，就令j[0]=sin(x)/x，如果x=0，就直接设定 j[0]=1.0

    if lmax >= 1:
        j[1] = (np.sin(x)/x**2-np.cos(x)/x) if x!=0 else 0.0  
        # 球贝塞尔函数的定义j1(x)=(sinx/x^2)-cosx/x
        # 如果x≠0，就令j[1]=(sinx/x^2)-cosx/x，如果x=0，就直接设定 j[1]=0.0
    for l in range(1, lmax):
        j[l+1] = ((2*l+1)/x)*j[l]-j[l-1]
        # 球贝塞尔函数向上递推关系式：jl+1(x)=(2l+1/x)jl(x)-jl-1(x)，从j0(x)、j1(x)向上递推到jlmax(x)
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
        # 球贝塞尔函数向下递推关系式：jl-1(x)=(2l+1/x)jl(x)-jl+1(x)，从高阶向下递推回j0(x)
    j = j[:lmax+1]  # 截取数组j的前lmax+1项
    scale = spherical_jn(0, x)/j[0] if j[0] != 0 else 0
    return j * scale
    '''
    归一化，因为一开始给出的两个值j[m_start]和j[m_start-1]是任意的，
    故得到的是球贝塞尔函数值的比例形状，即：形状是对的，但幅度是错的。因此需要再“缩放”成真实值。
    缩放方法：对所求数组乘一个比例因子scale
    其中scale = j[0]真实值/j[0]所求值，j[0]真实值由SciPy的spherical_jn(0, x)精确求出
    最后加入对j[0]的条件判断，以防止程序出错：
    若j[0]所求值=0，则scale=0，若j[0]所求值≠0，则scale=j[0]真实值/j[0]所求值
    '''

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
    l_values = np.arange(lmax+1) # 创建一个数组[0, 1, ..., lmax]，表示球贝塞尔函数的阶数 
    
    # 计算三种方法得到的结果的绝对值：向上递推法，向下递推法，spherical_jn()方法（“真实值”的参考）
    j_up = np.abs(bessel_up(x, lmax))
    j_down = np.abs(bessel_down(x, lmax))
    j_scipy = np.abs(spherical_jn(l_values, x))
    
    plt.figure(figsize=(12, 5)) # 创建一个新的图像窗口，宽度为12英寸，高度为5英寸
    
    # 函数值比较
    plt.subplot(1, 2, 1) # 在上述图像窗口中创建一个子图：将整个图像分成1行2列，此处为第一个子图
    '''
    画三幅“半对数图”：x轴为线性坐标，y轴为对数坐标
    第一幅图以阶数l_values为横坐标，向上递推法所得结果绝对值j_up为纵坐标，线型为“圆点连线”，图例名称为Upward
    第二幅图以阶数l_values为横坐标，向下递推法所得结果绝对值j_down为纵坐标，线型为“方块点连线”，图例名称为Downward
    第三幅图以阶数l_values为横坐标，spherical_jn()方法所得结果绝对值j_scipy为纵坐标，线型为“黑色虚线”，图例名称为Scipy
    '''
    plt.semilogy(l_values, j_up, 'o-', label='Upward')  
    plt.semilogy(l_values, j_down, 's-', label='Downward') 
    plt.semilogy(l_values, j_scipy, 'k--', label='Scipy') 
    plt.xlabel('Order l')  # 横轴标签：阶数l
    plt.ylabel('|j_l(x)|')  # 纵轴标签：球贝塞尔函数的绝对值
    plt.title(f'Comparison (x={x})')  # 设置图的标题，说明这张图的x值
    plt.legend()  # 显示图例
    
    # 相对误差比较
    plt.subplot(1, 2, 2)  # 在上述图像窗口中创建一个子图：将整个图像分成1行2列，此处为第二个子图
    '''
    为避免除以零和非法操作产生的警告干扰程序输出，暂时忽略
    with块执行完后，自动恢复NumPy原来的错误设置
    '''
    with np.errstate(divide='ignore', invalid='ignore'):  
        rel_err_up = np.abs((j_up - j_scipy)/j_scipy) # 向上递推时的相对误差=|(近似值−参考值)/参考值|
        rel_err_down = np.abs((j_down - j_scipy)/j_scipy) # 向下递推时的相对误差=|(近似值−参考值)/参考值|
    rel_err_up[j_scipy==0] = 0 # 若某个阶数的j_scipy=0，就人为地将相对误差设置为0，避免出现 “0 作分母” 的问题
    rel_err_down[j_scipy==0] = 0
    '''
    画两幅“半对数图”：x轴为线性坐标，y轴为对数坐标
    第一幅图以阶数l_values为横坐标，向上递推法所得结果相对误差rel_err_up为纵坐标，线型为“圆点连线”，图例名称为Up Error
    第二幅图以阶数l_values为横坐标，向下递推法所得结果相对误差rel_err_down为纵坐标，线型为“方块点连线”，图例名称为Down Error
    '''
    plt.semilogy(l_values, rel_err_up, 'o-', label='Up Error') 
    plt.semilogy(l_values, rel_err_down, 's-', label='Down Error')
    plt.xlabel('Order l')  # 横轴标签：阶数l
    plt.ylabel('Relative Error')  # 纵轴标签：球贝塞尔函数的相对误差
    plt.title(f'Relative Error (x={x})')  # 设置图的标题，说明这张图的x值
    plt.legend()  # 显示图例
    
    plt.tight_layout()  # 自动调整子图之间的间距，避免文字重叠或图像遮挡
    plt.show()  # 显示图像

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
