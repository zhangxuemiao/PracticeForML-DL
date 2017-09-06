import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def test_01():
    plt.figure(1) # 实例化作图变量
    plt.title("single variable") # 图像标题
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([0, 5, 0, 10]) # x轴范围是0-5，y轴范围是0-10
    plt.grid(True) # 是否网络化
    xx = np.linspace(0, 5, 10) # 在0-10之间产生10个数据
    plt.plot(xx, 2*xx, 'g-')
    plt.show()


def test_02():
    plt.figure(2) #
    plt.title("single variable")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis([-12, 12, -1, 1])
    plt.grid(True)
    xx = np.linspace(-12, 12, 100)
    plt.plot(xx, np.sin(xx), 'g-', label='$sin(x)$')
    plt.plot(xx, np.cos(xx), 'r--', label='$cos(x)$')
    plt.legend()
    plt.show()


def draw(subplt):
    subplt.axis([-12, 12, -1, 1])
    subplt.grid(True)
    xx = np.linspace(-12, 12, 100)
    subplt.plot(xx, np.sin(xx), 'g-', label="$sin(x)$")
    subplt.plot(xx, np.cos(xx), 'r--', label="$cos(x)$")
    subplt.legend()


def test_03():
    plt.figure(3)
    subplt1 = plt.subplot(2,2,1) #两行两列中第一个
    draw(subplt1)
    subplt2 = plt.subplot(2,2,2) #两行两列中第一个
    draw(subplt2)
    subplt3 = plt.subplot(2,2,3) #两行两列中第一个
    draw(subplt3)
    subplt4 = plt.subplot(2,2,4) #两行两列中第一个
    draw(subplt4)
    plt.show()


def draw_3D_04():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    theta = np.linspace(-4*np.pi, 4*np.pi, 500)
    z = np.linspace(0,2,500)
    r = z
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    ax.plot(x, y, z, label='curve')
    ax.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    xx = np.linspace(0, 5, 10)
    yy = np.linspace(0, 5, 10)
    zz1 = xx
    zz2 = 2*xx
    zz3 = 3*xx
    ax.scatter(xx, yy, zz1, c='red', marker='o')
    ax.scatter(xx, yy, zz2, c='red', marker='^')
    ax.scatter(xx, yy, zz3, c='red', marker='*')
    ax.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = X ** 2 + Y ** 2
    ax.plot_surface(X,Y,Z, rstride=1, cstride=1,cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()


def draw_3D_04_modify():
    plt.figure()
    subplt1 = plt.subplot(2, 2, 1, projection='3d')
    theta = np.linspace(-4*np.pi, 4*np.pi, 500)
    z = np.linspace(0,2,500)
    r = z
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    subplt1.plot(x, y, z, label='curve')
    subplt1.legend()

    subplt2 = plt.subplot(2,2,2, projection='3d')
    xx = np.linspace(0, 5, 10)
    yy = np.linspace(0, 5, 10)
    zz1 = xx
    zz2 = 2*xx
    zz3 = 3*xx
    subplt2.scatter(xx, yy, zz1, c='red', marker='o')
    subplt2.scatter(xx, yy, zz2, c='red', marker='^')
    subplt2.scatter(xx, yy, zz3, c='red', marker='*')
    subplt2.legend()

    subplt3 = plt.subplot(2, 2, 3, projection='3d')
    # ax = fig.gca(projection='3d')
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = X ** 2 + Y ** 2
    subplt3.plot_surface(X,Y,Z, rstride=1, cstride=1,cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()


def test_05():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = [1, 1, 2, 2]
    Y = [3, 4, 4, 3]
    Z = [1, 2, 1, 1]
    ax.scatter(X, Y, Z)
    plt.show()


def test_06():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = [1, 1, 2, 2]
    Y = [3, 4, 4, 3]
    Z = [1, 2, 1, 1]
    ax.plot_trisurf(X, Y, Z)
    plt.show()


if __name__ == '__main__':
    # draw_3D_04()
    draw_3D_04_modify()