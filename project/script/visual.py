# import struct
# import json
#
# def read_glb_file(filepath):
#     with open(filepath, 'rb') as f:
#         # 读取头部信息
#         magic, version, length=struct.unpack('<4sII',f.read(12))
#         if magic != b'glTF':
#             raise ValueError('Invalid GLB file')
#
#         # 解析JSON头部
#         json_length, json_type = struct.unpack('<I4s', f.read(8))
#         json_str = f.read(json_length).decode()
#         json_data = json.loads(json_str)
#
#         # 解析二进制数据
#         bin_type = f.read(4)
#         bin_length = struct.unpack('<I', f.read(4))[0]
#         bin_data = f.read(bin_length)
#
#     return json_data, bin_data
#
# def extract_mesh_data(json_data,bin_data):
#     # 从JSON数据中获取顶点坐标、法线和纹理坐标的偏移量
#     accessors = json_data['accessors']
#     print('accessors[0]:\n',accessors[0])
#     print('accessors[1]:\n', accessors[1])
#     print('accessors[3]:\n', accessors[3])
#     attributes = json_data['meshes'][0]['primitives'][0]['attributes']
#     print('attibutes:\n',attributes)
#     # positions_offset = accessors[attributes['POSITION']]['byteOffset']
#     # normals_offset = accessors[attributes['NORMAL']]['byteOffset']
#     # texcoords_offset = accessors[attributes['TEXCOORD_0']]['byteOffset']
#     texcoords_offset = 0
#     # https://blog.51cto.com/u_16213406/7138237
#     # 解析顶点坐标数据
#     # 解析发现数据
#     # 解析纹理坐标数据
#     texcoords_accessor = accessors[attributes['TEXCOORD_0']]
#     # texcoords_type = texcoords_accessor['componentType']
#     texcoords_count = texcoords_accessor['count']
#     # texcoords_stride = texcoords_accessor['byteStride']
#     texcoords_data = struct.unpack_from('{0}f'.format(texcoords_count * 2),bin_data, texcoords_offset)
#
#     return texcoords_data
#
# json_data, bin_data=read_glb_file('001.glb')
# texcoords_data = extract_mesh_data(json_data, bin_data)
# print(type(texcoords_data))
# print(texcoords_data)


# import pygltflib
# import numpy as np

# 加载 .glb 文件
# glb = pygltflib.GLTF2().load('001.glb')
#
# for accessors in glb.accessors:
#     print(accessors)
#     if accessors.type == "TEXCOORD_0":
#         buffer_view = glb.bufferViews[accessors.bufferView]
#         buffer = glb.buffers[buffer_view.buffer]
#         data = np.frombuffer(buffer.data, dtype=np.float32,count=accessors.count*2,offset=buffer_view.byteOffset)
#         print(data)


# 获取 UV 坐标
# uv_coords = []
# for mesh in glb.meshes:
#     for primitive in mesh.primitives:
#         print(primitive)
#         if primitive.attributes.TEXCOORD_0 is not None:
#         # if primitive.attributes.get('TEXCOORD_0') is not None:
#             accessor = glb.accessors[primitive.attributes.TEXCOORD_0]
#             print(accessor)
#             buffer_view = glb.bufferViews[accessor.bufferView]
#             buffer = glb.buffers[buffer_view.buffer]
#             print(buffer)
#             data = buffer.data
#             # 读取 UV 坐标
#             uv_data = data[buffer_view.byteOffset:buffer_view.byteOffset + buffer_view.byteLength]
#             uv_coords.append(uv_data)
#
# # 打印 UV 坐标
# print("UV Coordinates:")
# print(uv_coords)

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


