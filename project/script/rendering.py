import os
import pyrender
import trimesh
import numpy as np
import imageio
import math
import cv2
from scipy import ndimage

# def load_pyrender_mesh(path, material):
#     # 读取mesh，返回用material渲染之后的pyrender.Mesh
#     mesh = trimesh.load(path,force='mesh')
#     return pyrender.Mesh.from_trimesh(mesh, material=material)

def generate_scene(pyrender_mesh, light_type, light_pose):
    """
        生成给定灯光下的场景
    Parameters
    ----------
    pyrender_mesh:渲染了的网格
    light_type:(List)灯光种类
    light_pose:(List)灯光位姿

    Returns
    -------

    """
    # 创建一个场景并添加网格
    scene = pyrender.Scene()
    scene.add(pyrender_mesh)
    for i in range(len(light_type)):
        scene.add(light_type[i], pose=light_pose[i])
    return scene

def scene_add_camera(scene, camera_node, camera_pose):
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

def view_scene(scene):
    """
    可视化场景
    Parameters
    ----------
    scene:场景

    Returns
    -------

    """
    # 用pyrender.Viewer()函数来可视化网格
    viewer_flags={'use_raymond_lighting':True}
    # viewer = pyrender.Viewer(scene, viewer_flags=viewer_flags)
    viewer = pyrender.Viewer(scene)

def save_view_fig(scene, path):
    """
    保存场景当前视图
    Parameters
    ----------
    scene:当前场景
    path：保存路径

    Returns
    -------

    """
    # 创建离屏渲染器
    renderer = pyrender.OffscreenRenderer(512, 512)
    # 渲染场景
    color, depth = renderer.render(scene)
    # 保存渲染结果
    imageio.imwrite(path, (color * 255).astype(np.uint8))

def generate_symmetric_view_scenes(pyrender_mesh, center, path_front, path_back, camera_node, camera_pose, light_type, light_pose):
    """
        生成对称视图并保存
    Parameters
    ----------
    pyrender_mesh:渲染了的网格
    center:模型中心
    path_front:前视图保存路径
    path_back:后视图保存路径
    camera_node:相机节点
    camera_pose:相机位姿
    light_type:灯光类型
    light_pose:灯光位姿

    Returns
    -------

    """
    scene_front_init = generate_scene(pyrender_mesh, light_type, light_pose)
    scene_front = scene_add_camera(scene_front_init, camera_node, camera_pose)
    save_view_fig(scene_front,path_front)

    scene_back_init = generate_scene(pyrender_mesh, light_type, light_pose)
    position_front_camera = camera_pose[:3, 3]
    position_back_camera = 2*center - position_front_camera
    camera_pose_symmetric = camera_pose
    camera_pose_symmetric[:3, 3] = position_back_camera
    camera_pose_symmetric[:3, 0] = -camera_pose[:3, 0]
    camera_pose_symmetric[:3, 2] = -camera_pose[:3, 2]
    scene_back = scene_add_camera(scene_back_init, camera_node, camera_pose)
    save_view_fig(scene_back, path_back)

def save_2image_hstack(path1,path2,path3):
    """
        将前两张图像横向拼接并保存
    Parameters
    ----------
    path1
    path2
    path3

    Returns
    -------

    """
    # 读取两张PNG图片
    image1 = imageio.v3.imread(path1)  # 替换为第一张图片的路径
    image2 = imageio.v3.imread(path2)  # 替换为第二张图片的路径

    # 检查两张图片的高度是否相同
    if image1.shape[0] != image2.shape[0]:
        raise ValueError("Images must have the same height.")

    # 水平连接两张图片
    combined_image = np.hstack((image1, image2))

    # 保存连接后的图片
    path3_folder = os.path.dirname(path3)
    if not os.path.exists(path3_folder):
        # 如果文件夹不存在，则创建文件夹
        os.makedirs(path3_folder)
    imageio.imwrite(path3, combined_image)

def generate_and_save_mask(rbg_path,mask_path, mask_range=4):
    # 读取图片
    image = imageio.v3.imread(rbg_path)  # 替换为图片的路径

    # 将图片转换为灰度图
    gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    # 应用阈值来生成掩码
    # 这里的阈值设置为1，意味着灰度值小于1的像素将被设置为0（黑色），大于等于1的像素将被设置为1（白色）
    mask = gray_image > 1
    # 掩码膨胀
    mask_expanse = 255-mask_expansion(1-mask, mask_range)

    # 保存掩码图片
    mask_path_folder = os.path.dirname(mask_path)
    if not os.path.exists(mask_path_folder):
        # 如果文件夹不存在，则创建文件夹
        os.makedirs(mask_path_folder)
    imageio.imwrite(mask_path, mask_expanse)

def mask_expansion(mask_image,mask_range=16):
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

file = "007"
tri_mesh = trimesh.load(file+".glb", force='mesh')
# print("uv:\n",tri_mesh.)
vertices = np.array(tri_mesh.vertices)
min_values=np.min(vertices,axis=0)
max_values=np.max(vertices,axis=0)
# center =np.mean(vertices, axis=0)
center = (min_values+max_values)/2
print(center)
# 创建一个白色的材质，RGBA值为[1, 1, 1, 1]
material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[1, 1, 1, 1])
pyrender_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=material)

# 创建一个正交相机
oc = pyrender.OrthographicCamera(xmag=10, ymag=10)
# 创建一个相机节点
camera_node = pyrender.Node(camera=oc)
# camera_pose：前三行的前三列分别表示相机坐标系的x,y,z坐标轴在世界坐标系下的方向，最后一列表示相机位置，最后一行始终是0001，
# 相机朝向：相机坐标系的z坐标轴在世界坐标系下的方向的反方向
# 000
# camera_up=[0,1,0]
# theta=math.pi*0.5
# camera_position = [math.cos(theta), 0, math.sin(theta)]

#001
# camera_up=[0,0,1]
# theta=math.pi*0.3
# # camera_position = [2*math.cos(theta),0,2*math.sin(theta)]
# camera_position = [math.cos(theta),math.sin(theta),center[2]]

# 002
# camera_up=[0,0,1]
# theta=math.pi*0.5
# camera_position = [100*math.cos(theta), 100*math.sin(theta), 0]

# 004,002
# camera_up=[0,1,0]
# theta=math.pi*0.5
# camera_position = [100*math.cos(theta), 10, 100*math.sin(theta)]

# 005
# camera_up=[0,1,0]
# theta=math.pi*0.5
# camera_position = [10*math.cos(theta), 10, 10*math.sin(theta)]

# 006
# camera_up=[0,1,0]
# theta=math.pi*1
# camera_position = [10*math.cos(theta), 10, 10*math.sin(theta)]

# 007
camera_up=[0,1,0]
theta=math.pi*1
camera_position = [20*math.cos(theta), 7, 20*math.sin(theta)]


camera_pose = np.eye(4)
camera_pose[:3, 3] = camera_position
camera_direction = center-camera_position
camera_direction=camera_direction/np.linalg.norm(camera_direction)#归一化
camera_pose[:3, 2] = -camera_direction # 这里要和相机方向相反
camera_right = np.cross(camera_direction, camera_up)#生成第三个轴
camera_pose[:3, 0] = camera_right / np.linalg.norm(camera_right)# 归一化

camera_pose[:3, 1] = camera_up
print("camera_pose:\n",camera_pose)

light_type=[]
light_pose=[]

position_light1 = [10, 10, 10]
light1 = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=100.0)
light1_pose = np.eye(4)
light1_up=[0,1,0]
light1_pose[:3, 3] = position_light1
light1_direction=center-position_light1
light1_pose[:3,2]=-light1_direction
light1_pose[:3,1]=light1_up
light1_right=np.cross(light1_direction,light1_up)
light1_pose[:3,0]=light1_right/np.linalg.norm(light1_right)
print("light1_pose\n",light1_pose)
light_type.append(light1)
light_pose.append(light1_pose)

# position_light2=position_light1
position_light2 = 2*center-position_light1
light2 = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=100.0)
light2_pose = np.eye(4)
light2_pose[:3, 0] = -light1_pose[:3, 0]
light2_pose[:3, 2] = -light1_pose[:3, 2]
light2_pose[:3, 3] = position_light2
print("light2_pose\n",light2_pose)
light_type.append(light2)
light_pose.append(light2_pose)

scene = generate_scene(pyrender_mesh, light_type, light_pose)
scene = scene_add_camera(scene, camera_node, camera_pose)
# save_view_fig(scene,'rendered_image.png')
view_scene(scene)
path_front = file+"/"+"view_front.png"
path_back = file+"/"+"view_back.png"
generate_symmetric_view_scenes(pyrender_mesh, center, path_front, path_back, camera_node, camera_pose, light_type, light_pose)
path_stack = file+"/"+"view_symmetric.png"
save_2image_hstack(path_front, path_back, path_stack)
path_mask = file+"/"+"view_mask.png"
generate_and_save_mask(path_stack, path_mask)
