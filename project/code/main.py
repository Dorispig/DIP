import render
from Myscene import Myscene
import map as m
import pyrender
import numpy as np
import math
import os
from PIL import Image
import trimesh

# 在服务器中跑要用下面代码
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

iter = 3
epsilon = 0.00001
file = '000'
path = file + '.glb'
tri_mesh = render.read_glb_mesh(path)
center = render.cal_trimesh_center(tri_mesh = tri_mesh)
pyrender_mesh = pyrender.Mesh.from_trimesh(tri_mesh)
import copy
# 创建一个新的trimesh对象作为副本
tri_mesh_copy = copy.deepcopy(tri_mesh)
# 从副本创建一个新的pyrender.Mesh对象
init_pyrender_mesh = pyrender.Mesh.from_trimesh(tri_mesh_copy)
# init_pyrender_mesh = pyrender.Mesh.from_trimesh(tri_mesh)

# 初始化pyrender_mesh网格
h, w = 1024, 1024
init_texture = np.zeros((h, w, 4))
init_texture[...,0] = 1
init_texture[...,1] = 0
init_texture[...,2] = 0
init_texture[...,3] = 1

init_texture[...,0] = np.flipud(init_texture[..., 0].T)
init_texture[...,1] = np.flipud(init_texture[..., 1].T)
init_texture[...,2] = np.flipud(init_texture[..., 2].T)

pyrender_mesh = render.update_texure(pyrender_mesh, init_texture)
init_pyrender_mesh = render.update_texure(init_pyrender_mesh, init_texture)

C_star = np.zeros((h, w))
T_star = np.zeros_like(init_texture)
T_star[..., 3] = 255

# 生成两组对称灯光
light12_type = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=20.0)

light1_position = np.array([0, 1, 0])
light1_up = np.array([0, 1, 0])
light1_direction = center - light1_position
light1_pose = render.cal_pose(light1_up, light1_direction, light1_position)

light2_pose = -light1_pose
light2_pose[3, 3] = 1
light2_pose[:3, 3] = 2 * center - light1_position

light34_type = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=10.0)
light3_position = np.array([0, 0, 1])
light3_up = np.array([0, 1, 0])
light3_direction = center - light3_position
light3_pose = render.cal_pose(light3_up, light3_direction, light3_position)

light4_pose = -light3_pose
light4_pose[3, 3] = 1
light4_pose[:3, 3] = 2 * center - light3_position

for epoch in range(iter):
    print(f'epoch{epoch}:')
    # 创建一个正交相机
    oc = pyrender.OrthographicCamera(xmag=0.2, ymag=0.2)
    # 创建一个相机节点
    camera_node = pyrender.Node(camera=oc)
    # 生成对称相机位姿
    camera_f_up = np.array([0, 1, 0])
    theta = math.pi * 0.5 + epoch/iter * 2 * math.pi
    camera_f_position = np.array([math.cos(theta), 0, math.sin(theta)])
    camera_f_look_at = center - camera_f_position
    camera_f_pose = render.cal_pose(camera_f_up, camera_f_look_at, camera_f_position)

    camera_b_pose = -camera_f_pose
    camera_b_pose[3, 3] = 1
    camera_b_pose[:3, 1] = camera_f_pose[:3, 1]
    camera_b_pose[:3, 3] = 2 * center - camera_f_position

    # 生成两个场景参数
    s_f = Myscene(tri_mesh, pyrender_mesh, init_pyrender_mesh, camera_node, camera_f_pose,
                  light12_type, light1_pose, light12_type, light2_pose,
                  light34_type, light3_pose, light34_type, light4_pose)
    s_b = Myscene(tri_mesh, pyrender_mesh, init_pyrender_mesh, camera_node, camera_b_pose,
                  light12_type, light1_pose, light12_type, light2_pose,
                  light34_type, light3_pose, light34_type, light4_pose)

    # 获取视图
    Ii, Ij, Iij = render.I_ij(s_f, s_b)
    render.save_image(Iij, 'Iij.png', file, epoch)
    print('已获取前后视图')
    # 获取无纹理掩码图以及投影映射
    proj_mapi, Mi, maski, alphai, proj_mapj, Mj, maskj, alphaj, Mij, maskij, alphaij = render.M_ij(s_f, Ii, s_b, Ij, T_star, mask_range=16)
    render.save_image(Mij, 'Mij.png', file, epoch)
    render.save_image(alphaij, 'Alphaij.png', file, epoch)
    print('已获取掩码图')
    # 获取深度图
    Di, Dj, Dij=render.Dij(s_f,s_b)
    render.save_image(Dij, 'Dij.png', file, epoch)
    print('已获取深度图')
    # 获取法向图:
    Ni, Nj, Nij = render.Nij(s_f, s_b)
    Nij[:, :, 0] = Nij[:, :, 0] * maskij
    Nij[:, :, 1] = Nij[:, :, 1] * maskij
    Nij[:, :, 2] = Nij[:, :, 2] * maskij
    render.save_image(Nij, 'Nij.png', file, epoch)
    print('已获取法向图')
    # 获取边缘图
    Ei,Ej,Eij=render.Eij(Ii,Ij)
    # print('边缘图shape:',Eij.shape)
    render.save_image(Eij, 'Eij.png', file, epoch)
    print('已获取边缘图')


    # TODO:IIi和IIj是SDXL输出的图像
    IIi = Ii.copy()
    IIj = Ij.copy()


    # 获取置信度图
    Ci, Cj, Cij = m.Cij(s_f, s_b)
    render.save_image(Cij, 'C_ij.png', file, epoch)
    # print(np.max(Cij))
    np.savetxt('000Ci.txt',Ci,'%d')
    print('已获取置信度图')

    Ti, Tj, Tij = m.Tij(s_f, IIi, s_b, IIj)
    render.save_image(Tij, 'T_ij.png', file, epoch)
    render.save_image(Ti + Tj, 'Tij.png', file, epoch)
    # print(np.max(Tij))
    print('已获取Tij')

    # C_star = C_star.astype(np.uint8)
    # T_star = T_star.astype(np.uint8)
    # render.save_image(T_star, 'T_star.png', file, epoch)
    # render.save_image(C_star, 'C_star.png', file, epoch)

    T_star[:, :, 0] = (T_star[:, :, 0] * (C_star[:,:]) + Ti[:, :, 0] * (Ci[:,:]/255)) / ((C_star[:,:]) + (Ci[:,:]/255) + epsilon)
    T_star[:, :, 1] = (T_star[:, :, 1] * (C_star[:,:]) + Ti[:, :, 1] * (Ci[:,:]/255)) / ((C_star[:,:]) + (Ci[:,:]/255) + epsilon)
    T_star[:, :, 2] = (T_star[:, :, 2] * (C_star[:,:]) + Ti[:, :, 2] * (Ci[:,:]/255)) / ((C_star[:,:]) + (Ci[:,:]/255) + epsilon)
    # T_star[:, :, 3] = (T_star[:, :, 3] * C_star + Ti[:, :, 3] * Ci) / (C_star + Ci + epsilon)

    C_star[:,:] = C_star[:,:] + (Ci[:,:]/255) - (C_star[:,:]) * (Ci[:,:]/255)
    # C_star = (C_star*255).astype(np.uint8)
    # T_star = T_star.astype(np.uint8)
    # render.save_image(T_star, 'T_star0.png', file, epoch)
    # render.save_image(C_star, 'C_star0.png', file, epoch)

    T_star[:, :, 0] = (T_star[:, :, 0] * (C_star[:,:] ) + Tj[:, :, 0] * (Cj[:,:]/255)) / ((C_star[:,:] ) + (Cj[:,:]/255) + epsilon)
    T_star[:, :, 1] = (T_star[:, :, 1] * (C_star[:,:] ) + Tj[:, :, 1] * (Cj[:,:]/255)) / ((C_star[:,:] ) + (Cj[:,:]/255) + epsilon)
    T_star[:, :, 2] = (T_star[:, :, 2] * (C_star[:,:]) + Tj[:, :, 2] * (Cj[:,:]/255)) / ((C_star[:,:] ) + (Cj[:,:]/255) + epsilon)
    # T_star[:, :, 3] = (T_star[:, :, 3] * C_star + Tj[:, :, 3] * Cj) / (C_star + Cj + epsilon)


    C_star[:,:] = C_star[:,:] + Cj[:,:]/255 - (C_star[:,:]) * (Cj[:,:]/255)

    # C_star = (C_star*255).astype(np.uint8)
    T_star = T_star.astype(np.uint8)
    render.save_image(T_star, 'T_star1.png', file, epoch)
    render.save_image((C_star*255).astype(np.uint8), 'C_star1.png', file, epoch)

    T_new = T_star.copy()
    T_new[..., 0] = np.flipud(T_new[..., 0].T)
    T_new[..., 1] = np.flipud(T_new[..., 1].T)
    T_new[..., 2] = np.flipud(T_new[..., 2].T)
    pyrender_mesh = render.update_texure(pyrender_mesh, T_new)

scene = render.generate_scene(pyrender_mesh)
render.add_light(scene,light12_type,light1_pose)
render.add_light(scene,light12_type,light2_pose)
render.add_light(scene,light34_type,light3_pose)
render.add_light(scene,light34_type,light4_pose)
# render.add_camera(scene,camera_node,camera1_pose)
pyrender.Viewer(scene)


# 下面保存为可视化文件
material = pyrender_mesh.primitives[0].material
texture = material.baseColorTexture
texture_data = (texture.source).astype(np.uint8)  # 确保纹理数据是 uint8 类型
texture_image = Image.fromarray(texture_data)
texture_image.save('texture.png')
mtl_content = f"""
newmtl texture
Ka 1.000000 1.000000 1.000000
Kd 1.000000 1.000000 1.000000
Ks 0.000000 0.000000 0.000000
Tr 1.000000
illum 2
map_Kd texture.png
"""

# 保存 MTL 文件
with open('texture.mtl', 'w') as f:
    f.write(mtl_content)

# 保存 OBJ 文件
tri_mesh.export('mesh.obj')

# 将 MTL 文件路径添加到 OBJ 文件
with open('mesh.obj', 'r+') as f:
    content = f.read()
    f.seek(0, 0)
    f.write('mtllib texture.mtl\n')
    f.write(content)
# 修改obj文件用的纹理图
with open('mesh.obj', 'r') as file:
    lines = file.readlines()

# 替换usemtl声明
new_lines = []
for line in lines:
    if line.startswith('usemtl'):
        new_lines.append('usemtl texture\n')
    else:
        new_lines.append(line)

# 保存更改回OBJ文件
with open('mesh.obj', 'w') as file:
    file.writelines(new_lines)
