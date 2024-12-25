import numpy as np
from scipy.spatial import Delaunay
import map as m

p0=[1,1]
p1=[2,3]
p2=[3,1]
print(m.find_internal_index(p0,p1,p2))


# 定义三个顶点的整数坐标
points = np.array([[1, 1], [2, 3], [3, 1]])

# 创建Delaunay三角剖分
tri = Delaunay(points)

# 生成网格的x, y坐标
x = np.arange(0, 5)
y = np.arange(0, 4)
X, Y = np.meshgrid(x, y)

# 将坐标转换为一维数组
xy = np.column_stack((X.flatten(), Y.flatten()))

# 查找三角形内部的点的索引
inside = tri.find_simplex(xy) >= 0

# 获取三角形内部点的坐标
inside_points = xy[inside].reshape(-1, 2)

print("三角形内部的整数坐标点（包括边界）：")
print(inside_points)