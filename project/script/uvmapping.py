import trimesh
import xatlas

# 使用 trimesh 加载一个网格，但你可以使用任何库
mesh = trimesh.load("000.glb", force='mesh')

# 参数化网格
# `vmapping` 包含每个新顶点的原始顶点索引（形状为 N，类型为 uint32）
# `indices` 包含新三角形的顶点索引（形状为 Fx3，类型为 uint32）
# `uvs` 包含新顶点的纹理坐标（形状为 Nx2，类型为 float32）
vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)
print(uvs)
# 导出带有 UV 坐标的网格
xatlas.export("output.obj", mesh.vertices[vmapping], indices, uvs)