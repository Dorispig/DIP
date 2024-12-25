import render
from Myscene import Myscene
import map as m
import pyrender
import numpy as np
import math
import os

# 在服务器中跑要用下面代码
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

epsilon = 0.00001
file = '000'
path = file + '.glb'
tri_mesh = render.read_glb_mesh(path)
center = render.cal_trimesh_center(tri_mesh = tri_mesh)
pyrender_mesh = pyrender.Mesh.from_trimesh(tri_mesh)
# 初始化pyrender_mesh网格:全白，0.99透明纹理
h, w = 1024, 1024
init_texture = np.zeros((h, w, 4))
init_texture[...,0] = 0
init_texture[...,1] = 0
init_texture[...,2] = 0
init_texture[...,3] = 1
init_texture[...,0] = np.flipud(init_texture[...,0].T)
init_texture[...,1] = np.flipud(init_texture[...,1].T)
init_texture[...,2] = np.flipud(init_texture[...,2].T)

pyrender_mesh = render.update_texure(pyrender_mesh, init_texture)

C_star = np.zeros((h, w))
T_star = np.zeros_like(init_texture)
T_star[..., 3] = 255

# C_star = C_star.astype(np.uint8)
# T_star = T_star.astype(np.uint8)
# T_star = np.ones((h, w, 4))
# 生成对称灯光
light_type = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=50.0)

light1_position = np.array([0, 1, 0])
light1_up = np.array([0, 1, 0])
light1_direction = center - light1_position
light1_pose = render.cal_pose(light1_up, light1_direction, light1_position)

light2_pose = -light1_pose
light2_pose[3, 3] = 1
light2_pose[:3, 3] = 2 * center - light1_position
# 下面两个灯光还未加入
light3_position = np.array([0, 0, 1])
light3_up = np.array([0, 1, 0])
light3_direction = center - light3_position
light3_pose = render.cal_pose(light3_up, light3_direction, light3_position)

light4_pose = -light3_pose
light4_pose[3, 3] = 1
light4_pose[:3, 3] = 2 * center - light3_position
iter = 1
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
    s_f = Myscene(tri_mesh, pyrender_mesh, camera_node, camera_f_pose,
                  light_type, light1_pose, light_type, light2_pose,
                  light_type, light3_pose, light_type, light4_pose)
    s_b = Myscene(tri_mesh, pyrender_mesh, camera_node, camera_b_pose,
                  light_type, light1_pose, light_type, light2_pose,
                  light_type, light3_pose, light_type, light4_pose)

    # 获取视图
    Ii, Ij, Iij = render.I_ij(s_f, s_b)
    render.save_image(Iij, 'Iij.png', file, epoch)
    print('已获取前后视图')
    # 获取无纹理掩码图以及投影映射
    # proj_mapi, Mi, maski, proj_mapj, Mj, maskj, Mij, maskij = render.M_ij(s_f, Ii, s_b, Ij, T_star, mask_range=16)
    # render.save_image(Mij, 'Mij.png', file, epoch)
    # print('已获取掩码图')
    # 获取深度图
    Di, Dj, Dij=render.Dij(s_f,s_b)
    # print('深度图shape:',Dij.shape)
    render.save_image(Dij, 'Dij.png', file, epoch)
    print('已获取深度图')
    # 获取法向图:
    # Ni, Nj, Nij = render.Nij(s_f, s_b)
    # Nij[:, :, 0] = Nij[:, :, 0] * maskij
    # Nij[:, :, 1] = Nij[:, :, 1] * maskij
    # Nij[:, :, 2] = Nij[:, :, 2] * maskij
    # print('法向图shape:', Nij.shape)
    # render.save_image(Nij, 'Nij.png', file, epoch)
    # print('已获取法向图')
    # 获取边缘图
    Ei,Ej,Eij=render.Eij(Ii,Ij)
    # print('边缘图shape:',Eij.shape)
    render.save_image(Eij, 'Eij.png', file, epoch)
    print('已获取边缘图')


    # TODO:IIi和IIj是SDXL输出的图像
    IIi = Ii
    IIj = Ij




    # 获取置信度图
    Ci, Cj, Cij = m.Cij(s_f, s_b)
    render.save_image(Cij, 'C_ij.png', file, epoch)
    print(np.max(Cij))
    np.savetxt('000Ci.txt',Ci,'%d')
    print('已获取置信度图')

    Ti, Tj, Tij = m.Tij(s_f, IIi, s_b, IIj)
    render.save_image(Tij, 'T_ij.png', file, epoch)
    render.save_image(Ti + Tj, 'Tij.png', file, epoch)
    print(np.max(Tij))
    print('已获取Tij')

    # C_star = C_star.astype(np.uint8)
    # T_star = T_star.astype(np.uint8)
    # render.save_image(T_star, 'T_star.png', file, epoch)
    # render.save_image(C_star, 'C_star.png', file, epoch)

    T_star[:, :, 0] = (T_star[:, :, 0] * (C_star[:,:] / 255) + Ti[:, :, 0] * (Ci[:,:]/255)) / ((C_star[:,:] / 255) + (Ci[:,:]/255) + epsilon)
    T_star[:, :, 1] = (T_star[:, :, 1] * (C_star[:,:] / 255) + Ti[:, :, 1] * (Ci[:,:]/255)) / ((C_star[:,:] / 255) + (Ci[:,:]/255) + epsilon)
    T_star[:, :, 2] = (T_star[:, :, 2] * (C_star[:,:] / 255) + Ti[:, :, 2] * (Ci[:,:]/255)) / ((C_star[:,:] / 255) + (Ci[:,:]/255) + epsilon)
    # T_star[:, :, 3] = (T_star[:, :, 3] * C_star + Ti[:, :, 3] * Ci) / (C_star + Ci + epsilon)

    C_star = C_star + Ci - C_star * Ci
    # C_star = C_star.astype(np.uint8)
    # T_star = T_star.astype(np.uint8)
    # render.save_image(T_star, 'T_star0.png', file, epoch)
    # render.save_image(C_star, 'C_star0.png', file, epoch)

    T_star[:, :, 0] = (T_star[:, :, 0] * (C_star[:,:] / 255) + Tj[:, :, 0] * (Cj[:,:]/255)) / ((C_star[:,:] / 255) + (Cj[:,:]/255) + epsilon)
    T_star[:, :, 1] = (T_star[:, :, 1] * (C_star[:,:] / 255) + Tj[:, :, 1] * (Cj[:,:]/255)) / ((C_star[:,:] / 255) + (Cj[:,:]/255) + epsilon)
    T_star[:, :, 2] = (T_star[:, :, 2] * (C_star[:,:] / 255) + Tj[:, :, 2] * (Cj[:,:]/255)) / ((C_star[:,:] / 255) + (Cj[:,:]/255) + epsilon)
    # T_star[:, :, 3] = (T_star[:, :, 3] * C_star + Tj[:, :, 3] * Cj) / (C_star + Cj + epsilon)


    C_star = C_star + Cj - C_star * Cj

    C_star = C_star.astype(np.uint8)
    T_star = T_star.astype(np.uint8)
    render.save_image(T_star, 'T_star1.png', file, epoch)
    render.save_image(C_star, 'C_star1.png', file, epoch)

    T_new = Ti+Tj
    T_new[..., 0] = np.flipud(T_new[..., 0].T)
    T_new[..., 1] = np.flipud(T_new[..., 1].T)
    T_new[..., 2] = np.flipud(T_new[..., 2].T)
    pyrender_mesh = render.update_texure(pyrender_mesh, T_new)

scene = render.generate_scene(pyrender_mesh)
render.add_light(scene,light_type,light1_pose)
render.add_light(scene,light_type,light2_pose)
render.add_light(scene,light_type,light3_pose)
render.add_light(scene,light_type,light4_pose)
# render.add_camera(scene,camera_node,camera1_pose)
pyrender.Viewer(scene)
