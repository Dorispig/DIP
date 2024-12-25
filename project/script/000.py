import render
from Myscene import Myscene
import map as m
import os
import pyrender
import trimesh
import numpy as np
from PIL import Image
import imageio
import math
import cv2
from scipy import ndimage

file = '000'
path = file + '.glb'
tri_mesh = render.read_glb_mesh(path)
center = render.cal_trimesh_center(tri_mesh = tri_mesh)

# 创建一个白色的材质，RGBA值为[1, 1, 1, 1]
# material = pyrender.MetallicRoughnessMaterial(
#     baseColorFactor=[1, 1, 1, 1]
# )

# 生成pyrender_mesh
pyrender_mesh = pyrender.Mesh.from_trimesh(tri_mesh)#, material=material
# 有, material=material时是无纹理图像的pyrender_mesh，否则为默认纹理图像

# 初始化pyrender_mesh网格，0.99透明纹理
h, w = 1024, 1024
init_texture = np.zeros((h, w, 4))
# init_texture[...,0] = 0#R
# init_texture[...,1] = 0#G
# init_texture[...,2] = 0#B
init_texture[...,3] = 0.99
pyrender_mesh = render.update_texure(pyrender_mesh, init_texture)



# 创建一个正交相机
oc = pyrender.OrthographicCamera(xmag=0.2, ymag=0.2)
# 创建一个相机节点
camera_node = pyrender.Node(camera=oc)
# 生成对称相机位姿
camera_f_up = np.array([0, 1, 0])
theta = math.pi*0.5
camera_f_position = np.array([math.cos(theta), 0, math.sin(theta)])
camera_f_look_at = center - camera_f_position
camera_f_pose = render.cal_pose(camera_f_up, camera_f_look_at, camera_f_position)

camera_b_pose = -camera_f_pose
camera_b_pose[3, 3] = 1
camera_b_pose[:3, 1] = camera_f_pose[:3, 1]
camera_b_pose[:3, 3] = 2 * center - camera_f_position

# 生成对称灯光
light_type = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=50.0)

light1_position = np.array([1, 1, 1])
light1_up = np.array([0, 1, 0])
light1_direction = center - light1_position
light1_pose = render.cal_pose(light1_up, light1_direction, light1_position)

light2_pose = -light1_pose
light2_pose[3, 3] = 1
light2_pose[:3,3] = 2*center - light1_position
# 生成两个场景参数
s_f = Myscene(tri_mesh, pyrender_mesh, camera_node, camera_f_pose, light_type, light1_pose, light_type, light2_pose)
s_b = Myscene(tri_mesh, pyrender_mesh, camera_node, camera_b_pose, light_type, light1_pose, light_type, light2_pose)

# tex = render.get_texture_image(s_f)
# render.save_image(tex,'tex.png')

print(f'{file}/epoch{1}/Iij.png')
Ii, Ij, Iij = render.I_ij(s_f, s_b)
print('已获取渲染视图')
print(Iij.shape)
render.save_image(Iij,'Iij.png',file,999)# '000Iij.png'
# 获取无纹理掩码图以及投影映射
# proj_mapi, Mi, maski, proj_mapj, Mj, maskj, Mij, maskij = render.M_ij(s_f, Ii, s_b, Ij)
# render.save_image(Mij,'000Mij.png')
# print('已获取掩码图')
# 获取深度图
# Di, Dj, Dij=render.Dij(s_f,s_b)
# print('深度图shape:',Dij.shape)
# render.save_image(Dij,'000Dij.png')
# print('已获取深度图')
# 获取法向图:目前还不对
# Ni, Nj, Nij = render.Nij(s_f, s_b)
# Nij[:, :, 0] = Nij[:, :, 0] * maskij
# Nij[:, :, 1] = Nij[:, :, 1] * maskij
# Nij[:, :, 2] = Nij[:, :, 2] * maskij
# Nij[...,0] = Nij[...,0]*maskij
# Nij[...,1] = Nij[...,1]*maskij
# Nij[...,2] = Nij[...,2]*maskij
# print('法向图shape:',Nij.shape)
# render.save_image(Nij,'000Nij.png')
# print('已获取法向图')
# 获取边缘图
# Ei,Ej,Eij=render.Eij(Ii,Ij)
# # print('边缘图shape:',Eij.shape)
# render.save_image(Eij,'000Eij.png')
# print('已获取边缘图')
# 获取置信度图
# Ci, Cj, Cij = m.Cij(s_f, s_b)
# render.save_image(Cij,'000Cij.png')
# C_star = Ci + Cj
# render.save_image(C_star,'000Cstar.png')
# print('已获取置信度图')

# IIi = Ii
# IIj = Ij
# tex = render.get_texture_image(s_f)
# render.save_image(tex,'tex.png')
# IIi_2d = IIi.reshape(-1, IIi.shape[2])
# np.savetxt('IIi.txt',IIi_2d)

# Ti = m.texture_img(s_f, IIi)
# render.save_image(IIi, '000IIi.png')
# render.save_image(Ti, '000Ti.png')
# Ti, Tj, Tij = m.Tij(s_f, IIi, s_b, IIj)
# render.save_image(Tij,'000Tij.png')


# init_texture = pyrender.Texture(source=Transparent_texture)
# Transparent_material = pyrender.MetallicRoughnessMaterial(baseColorTexture=init_texture)
# pyrender_mesh.primitives[0].material = Transparent_material

# 检查材质是否有纹理
# if material.baseColorTexture is not None:
#     # 获取纹理对象
#     texture = material.baseColorTexture
#
#     # 修改纹理为全红
#     # texture.data = np.full((texture.height, texture.width, 4), 255, dtype=np.uint8)
#     texture.source[:, :, 0:2] = 255  # R, G, B通道设置为255，A通道保持不变
#     # 获取纹理的源数据
#     texture_source = texture.source
#
#     # 如果源数据是NumPy数组，可以直接使用
#     if isinstance(texture_source, np.ndarray):
#         texture_image = texture_source
#     else:
#         # 如果源数据是PIL图像，可以转换为NumPy数组
#         texture_image = np.array(texture_source)
#     print(texture_image.shape)
#     # 显示纹理图像
#     import matplotlib.pyplot as plt
#
#     plt.imshow(texture_image)
#     plt.show()
# else:
#     print("Mesh has no texture.")


scene = render.generate_scene(pyrender_mesh)

# 创建一个正交相机
oc = pyrender.OrthographicCamera(xmag=0.2, ymag=0.2)
# 创建一个相机节点
camera_node = pyrender.Node(camera=oc)
# 生成相机位姿
camera_up = np.array([0, 1, 0])
theta = math.pi*0.5
camera_position = np.array([math.cos(theta), 0, math.sin(theta)])
camera_look_at = center - camera_position
camera_pose = render.cal_pose(camera_up, camera_look_at, camera_position)
scene = render.add_camera(scene, camera_node, camera_pose)

light1_position = np.array([1, 1, 1])
light1 = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=100.0)
light1_up = np.array([0, 1, 0])
light1_direction = center - light1_position
light1_pose = render.cal_pose(light1_up, light1_direction, light1_position)
scene = render.add_light(scene, light_type=light1, light_pose= light1_pose)


# renderer = pyrender.OffscreenRenderer(640, 480)

# 渲染场景
# color, depth = renderer.render(scene)

# 显示结果
# import matplotlib.pyplot as plt
# plt.imshow(color)
# plt.show()




# 用pyrender.Viewer()函数来可视化网格
# viewer_flags={'use_raymond_lighting':True}
# viewer = pyrender.Viewer(scene, viewer_flags=viewer_flags)
pyrender.Viewer(scene)