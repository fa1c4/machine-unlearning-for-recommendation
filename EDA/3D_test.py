import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# plt.show()

# fig = plt.figure()
# # ax = fig.gca(projection='3d')
# ax = fig.add_subplot(projection='3d')
#
# # Prepare arrays x, y, z
# theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
# z = np.linspace(-2, 2, 100)
# r = z**2 + 1
# x = r * np.sin(theta)
# y = r * np.sin(np.sin(theta))
#
# ax.plot(x, y, z, label='parametric curve')  #这里传入x, y, z的值
# ax.legend()
# plt.show()

test = range(10)

unlearningtimes_fullretrain = [random.randint(0, 10) for _ in range(10)]
acc_fullretrain = [random.randint(0, 10) for _ in range(10)]
unlearningtimes_5shards = [random.randint(0, 10) for _ in range(10)]
acc_5shards = [random.randint(0, 10) for _ in range(10)]
unlearningtimes_10shards = [random.randint(0, 10) for _ in range(10)]
acc_10shards = [random.randint(0, 10) for _ in range(10)]
unlearningtimes_15shards = [random.randint(0, 10) for _ in range(10)]
acc_15shards = [random.randint(0, 10) for _ in range(10)]
unlearningtimes_5fraction = [random.randint(0, 10) for _ in range(10)]
acc_5fraction = [random.randint(0, 10) for _ in range(10)]
unlearningtimes_10fraction = [random.randint(0, 10) for _ in range(10)]
acc_10fraction = [random.randint(0, 10) for _ in range(10)]
unlearningtimes_15fraction = [random.randint(0, 10) for _ in range(10)]
acc_15fraction = [random.randint(0, 10) for _ in range(10)]

fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

x = range(0, 50, 5)
plt.title('accuracy - unlearning data points - unlearning time')
ax.set_xlabel('unlearning data points')
ax.set_ylabel('unlearning time')
ax.set_zlabel('accuracy')

plt.plot(x, unlearningtimes_fullretrain, acc_fullretrain, label='full retrain', linewidth=1, color='r', marker='o')
plt.plot(x, unlearningtimes_5shards, acc_5shards, label='5 shards', linewidth=1, color='b', marker='x')
plt.plot(x, unlearningtimes_10shards, acc_10shards, label='10 shards', linewidth=1, color='g', marker='x')
plt.plot(x, unlearningtimes_15shards, acc_15shards, label='15 shards', linewidth=1, color='y', marker='x')
plt.plot(x, unlearningtimes_5fraction, acc_5fraction, label='1/5 fraction', linewidth=1, color='grey', marker='*')
plt.plot(x, unlearningtimes_10fraction, acc_10fraction, label='1/10 fraction', linewidth=1, color='purple', marker='*')
plt.plot(x, unlearningtimes_15fraction, acc_15fraction, label='1/15 fraction', linewidth=1, color='orange', marker='*')

plt.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
plt.show()

# fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
#
# datasets = [{"x":[1,2,3], "y":[1,4,9], "z":[0,0,0], "colour": "red"} for _ in range(6)]
#
# for dataset in datasets:
#     ax.plot(dataset["x"], dataset["y"], dataset["z"], color=dataset["colour"])
#
# plt.show()

'''
# Fixing random state for reproducibility
np.random.seed(19680801)


def randrange(n, vmin, vmax):
    """
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    """
    return (vmax - vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# 迭代两次，分别绘制两种不同的散点，一种是（红色，小圆圈，z坐标取值范围(-50, -25)），另一种是（蓝色，小三角，z坐标取值范围(-30, -5)）。
for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zlow, zhigh)
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
'''


'''

#定义坐标轴
fig = plt.figure()
ax1 = plt.axes(projection='3d')
#ax = fig.add_subplot(111,projection='3d')  #这种方法也可以画多个子图


#方法二，利用三维轴方法
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#定义图像和三维格式坐标轴
fig=plt.figure()
ax2 = Axes3D(fig)

import numpy as np
z = np.linspace(0,13,1000)
x = 5*np.sin(z)
y = 5*np.cos(z)
zd = 13*np.random.random(100)
xd = 5*np.sin(zd)
yd = 5*np.cos(zd)
ax1.scatter3D(xd,yd,zd, cmap='Blues')  #绘制散点图
ax1.plot3D(x,y,z,'gray')    #绘制空间曲线
plt.show()

fig = plt.figure()  #定义新的三维坐标轴
ax3 = plt.axes(projection='3d')

#定义三维数据
xx = np.arange(-5,5,0.5)
yy = np.arange(-5,5,0.5)
X, Y = np.meshgrid(xx, yy)
Z = np.sin(X)+np.cos(Y)


#作图
ax3.plot_surface(X,Y,Z,cmap='rainbow')
#ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap='rainbow)   #等高线图，要设置offset，为Z的最小值
plt.show()

'''
