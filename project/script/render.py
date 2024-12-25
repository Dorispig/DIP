import os
import pyrender
import trimesh
import numpy as np
import imageio
import math
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from Myscene import Myscene
import map

H, W = 1024, 1024

def read_glb_mesh(path):
    tri_mesh = trimesh.load(path, force='mesh')
    return tri_mesh

def get_texture_image(s: Myscene):
    pyrender_mesh = s.pyrender_mesh
    material = pyrender_mesh.primitives[0].material
    texture = material.baseColorTexture
    # 获取纹理的源数据
    texture_source = texture.source

    # 如果源数据是NumPy数组，可以直接使用
    if isinstance(texture_source, np.ndarray):
        texture_image = texture_source
    else:
        # 如果源数据是PIL图像，可以转换为NumPy数组
        texture_image = np.array(texture_source)
    return texture_image

def update_texure(pyrender_mesh, tex_image):
    material = pyrender_mesh.primitives[0].material
    # material.alphaMode='BLEND'
    # 获取纹理对象
    texture = material.baseColorTexture

    # 修改纹理
    texture.source = tex_image  # R, G, B通道设置为255，A通道保持不变
    # # 获取纹理的源数据
    # texture_source = texture.source
    #
    # # 如果源数据是NumPy数组，可以直接使用
    # if isinstance(texture_source, np.ndarray):
    #     texture_image = texture_source
    # else:
    #     # 如果源数据是PIL图像，可以转换为NumPy数组
    #     texture_image = np.array(texture_source)
    return pyrender_mesh
def save_image(image, image_path, file, epoch):
    path = f'{file}/epoch{epoch}'
           # /{image_path}
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        imageio.imwrite(f'{file}/epoch{epoch}/{image_path}', image)
    else:
        imageio.imwrite(f'{file}/epoch{epoch}/{image_path}', image)

def cal_trimesh_center(tri_mesh):
    vertices = np.array(tri_mesh.vertices)
    min_values = np.min(vertices, axis=0)
    max_values = np.max(vertices, axis=0)
    center = (min_values + max_values) / 2
    return center

def generate_scene(pyrender_mesh):
    scene = pyrender.Scene()
    scene.add(pyrender_mesh)
    return scene

def normalize(v):
    return v / np.linalg.norm(v)

def cal_pose(up, look_at, position=[0,0,0]):
    # 标准化上方向和朝向向量
    up = normalize(up)
    look_at=normalize(look_at)
    # 计算局部Z轴
    z_axis = look_at
    # 计算局部X轴
    x_axis = np.cross(look_at,up)
    if np.linalg.norm(x_axis) == 0:
        raise ValueError("Up direction and look-at direction are parallel.")
    x_axis = normalize(x_axis)
    # 计算局部Y轴
    y_axis = np.cross(x_axis, z_axis)
    y_axis = normalize(y_axis)

    # 构建旋转矩阵
    R = np.column_stack((x_axis, y_axis, -z_axis))
    # 生成位姿
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = position
    return pose

def add_light(scene, light_type, light_pose):
    """
        给场景添加灯光
        Parameters
        ----------
        scene:场景
        light_type:(List)灯光种类
        light_pose:(List)灯光位姿

        Returns
        -------

        """
    scene.add(light_type, pose=light_pose)
    return scene

def add_camera(scene, camera_node, camera_pose):
    """
            给场景添加相机
        Parameters
        ----------
        scene:场景
        camera_node:相机节点
        camera_pose:相机位姿

        Returns
        -------

        """
    scene.add_node(camera_node)
    scene.set_pose(camera_node, pose=camera_pose)
    return scene

def view_image(s: Myscene):
    # scene = s.scene
    scene = generate_scene(s.pyrender_mesh)
    add_light(scene, s.light1_type, s.light1_pose)
    add_light(scene, s.light2_type, s.light2_pose)
    add_light(scene, s.light3_type, s.light3_pose)
    add_light(scene, s.light4_type, s.light4_pose)
    add_camera(scene, s.camera_node, s.camera_pose)
    # 创建离屏渲染器
    renderer = pyrender.OffscreenRenderer(H, W)
    # 渲染场景
    color, _ = renderer.render(scene)
    color = (color * 1).astype(np.uint8)
    return color

def I_ij(s1: Myscene, s2: Myscene):
    Ii = view_image(s1)
    Ij = view_image(s2)
    Iij = np.hstack((Ii,Ij))
    return Ii, Ij, Iij

def mask_expansion(mask_image, mask_range=16):
    """
    掩码膨胀，0为掩码。
    Parameters
    ----------
    mask_image
    mask_range:膨胀大小

    Returns
    -------

    """
    # 将掩码图像转换为布尔类型，以便进行膨胀操作
    mask = mask_image.astype(bool)
    mask = mask.astype(np.uint8)*255

    kernel = np.ones((mask_range, mask_range), np.uint8)
    mask_dilate = cv2.dilate(mask, kernel, iterations=1)

    mask_dilate_uint8 = (mask_dilate > 0).astype(np.uint8) * 255

    return mask_dilate_uint8

def tex_Mask(s: Myscene, I, T_star, mask_range=16):
    h, w = I.shape[:2]

    # 先转换成灰度图
    gray_image = 255 - np.dot(I[..., :3], [0.2989, 0.5870, 0.1140])
    # 这里的阈值设置为1，意味着灰度值小于1的像素将被设置为0（黑色），大于等于1的像素将被设置为1（白色）
    mask = gray_image > 1

    tex_h, tex_w = T_star.shape[:2]
    # 获取投影映射
    proj_map = map.projection_map2(s, I)
    alpha = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            u = int(proj_map[i][j][0] * tex_h)
            v = int(proj_map[i][j][1] * tex_w)
            # print(texture_image[u][v][3])
            # alpha[i][j] = int(T_star[u][v][3]/255)
            #if np.sum(T_star[tex_h - v][u][1:3]) == 0:
            if np.sum(T_star[u][v][1:3]) == 0:
                alpha[i][j] = 1
    # # 纹理掩码，无纹理区域设置为1
    tex_mask = mask * alpha
    # tex_mask = mask
    # 将黑色区域（有纹理或背景）膨胀
    # expanse_tex_mask = 255 - mask_expansion(1 - tex_mask, mask_range)
    # 将白色区域（无纹理区域）膨胀
    expanse_tex_mask = mask_expansion(tex_mask, mask_range)
    # expanse_tex_mask = tex_mask.astype(np.uint8)*255
    return proj_map, expanse_tex_mask, mask

def M_ij(si: Myscene, Ii, sj: Myscene, Ij, T_star, mask_range=16):
    proj_mapi, Mi, maski = tex_Mask(si, Ii, T_star, mask_range)
    proj_mapj, Mj, maskj = tex_Mask(sj, Ij, T_star, mask_range)
    Mij = np.hstack((Mi,Mj))
    maskij = np.hstack((maski,maskj))
    return proj_mapi, Mi, maski, proj_mapj, Mj, maskj, Mij, maskij

def Depth(s: Myscene):
    scene = generate_scene(s.pyrender_mesh)
    add_light(scene, s.light1_type, s.light1_pose)
    add_light(scene, s.light2_type, s.light2_pose)
    add_light(scene, s.light3_type, s.light3_pose)
    add_light(scene, s.light4_type, s.light4_pose)
    add_camera(scene, s.camera_node, s.camera_pose)
    # 创建离屏渲染器
    renderer = pyrender.OffscreenRenderer(H, W)
    # 渲染场景
    depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
    # 映射到0-255
    depth = np.exp(-depth)
    d_min = np.min(depth)
    d_max = np.max(depth)
    depth[:, :] = 255 - 255 / (d_min - d_max) * (depth[:, :] - d_max)
    return (depth * 255).astype(np.uint8)

def Dij(s1: Myscene, s2: Myscene):
    Di = Depth(s1)
    Dj = Depth(s2)
    Dij = np.hstack((Di,Dj))
    return Di, Dj, Dij

def Edge(I):
    edges = cv2.Canny(I, 100, 200)
    return edges

def Eij(Ii, Ij):
    Ei = Edge(Ii)
    Ej = Edge(Ij)
    Eij = np.hstack((Ei,Ej))
    return Ei, Ej, Eij

def Normal_image(s: Myscene):
    pyrender_mesh = s.pyrender_mesh
    tri_mesh = s.tri_mesh
    vertices = tri_mesh.vertices
    faces = tri_mesh.faces
    normals = tri_mesh.vertex_normals  # 顶点法向量（已归一化）
    # normals[:, :] = 255 / 2 * (normals[:, :] + 1)  # -1,1-->0,255
    R = 1 / 2 * (normals[:, 0] + 1)
    G = 1 / 2 * (normals[:, 1] + 1)
    B = 1 / 2 * (normals[:, 2] + 1)
    normal_color = np.stack([R, G, B, np.ones_like(R)], axis=-1)
    # print(normal_color)
    normal_tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=normal_color)
    normal_pyrender_mesh = pyrender.Mesh.from_trimesh(normal_tri_mesh)
    s_normal = s
    s_normal.pyrender_mesh = normal_pyrender_mesh
    Ni = view_image(s_normal)
    s.pyrender_mesh = pyrender_mesh
    return Ni

def Nij(s1: Myscene, s2: Myscene):
    Ni = Normal_image(s1)
    Nj = Normal_image(s2)
    Nij = np.hstack((Ni,Nj))
    return Ni, Nj, Nij




