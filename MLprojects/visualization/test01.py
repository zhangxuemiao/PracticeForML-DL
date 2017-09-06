import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1, 11)

fig = plt.figure(1)
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
l1, = ax1.plot(x, x*x, 'r')             #这里关键哦
l2, = ax2.plot(x, x*x, 'b')           # 注意

plt.legend([l1, l2], ['first', 'second'], loc = 'upper right')  # 用于显示图例，即对图像上各条曲线的描述，如某条曲线是f(x)=sin(x),另一条是f(x)=cos(x)           #其中，loc表示位置的；

plt.show()