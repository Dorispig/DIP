from Myscene import Myscene
import numpy as np
from scipy.spatial import Delaunay
import pyrender
import trimesh
import render

H, W=1024, 1024
def find_internal_index(p0, p1, p2):
    points = np.array([p0,p1,p2])
    x_min = np.min(points[:,0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    # 创建Delaunay三角剖分
    tri = Delaunay(points)

    # 生成网格的x, y坐标
    x = np.arange(x_min, x_max+1)
    y = np.arange(y_min, y_max+1)
    X, Y = np.meshgrid(x, y)

    # 将坐标转换为一维数组
    xy = np.column_stack((X.flatten(), Y.flatten()))

    # 查找三角形内部的点的索引
    inside = tri.find_simplex(xy) >= 0

    # 获取三角形内部点的坐标
    inside_points = xy[inside].reshape(-1, 2)
    return inside_points

def texture_linear_interpolation(face, tex_coord, tex_img):
    # 对纹理图像线性插值，填充三角形内部的纹理（这里三角形三点上均存在纹理）
    tex_h, tex_w = tex_img.shape[:2]
    for i in range(face.shape[0]):
        idx_p0, idx_p1, idx_p2 = face[i][:3]
        idx_p0, idx_p1, idx_p2 = int(idx_p0), int(idx_p1), int(idx_p2)
        u0, v0 = int(tex_coord[idx_p0][0] * tex_h), int(tex_coord[idx_p0][1] * tex_w)
        u1, v1 = int(tex_coord[idx_p1][0] * tex_h), int(tex_coord[idx_p1][1] * tex_w)
        u2, v2 = int(tex_coord[idx_p2][0] * tex_h), int(tex_coord[idx_p2][1] * tex_w)
        inside_idx = find_internal_index([u0, v0], [u1, v1], [u2, v2])
        for j in range(inside_idx.shape[0]):
            u, v = inside_idx[j][:]
            uv0 = [u0, v0, 1]
            uv1 = [u1, v1, 1]
            uv2 = [u2, v2, 1]
            uv = [u, v, 1]
            UV = np.column_stack((uv0, uv1, uv2))
            w = np.linalg.inv(UV) @ uv
            tex_img[u][v] = tex_img[u0][v0] * w[0] + tex_img[u1][v1] * w[1] + tex_img[u2][v2] * w[2]
    return tex_img

# 需要修改成从网格上的点投影到相机平面
def projection_map2(s: Myscene, view_image):
    # 将视图与纹理坐标对应
    h, w = view_image.shape[:2]
    proj_map = np.zeros((h, w, 2))
    tri_mesh = s.tri_mesh
    pyrender_mesh = s.pyrender_mesh
    camera_pose = s.camera_pose
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    camera_direction = -t
    positions = pyrender_mesh.primitives[0].positions
    tex_coord = pyrender_mesh.primitives[0].texcoord_0
    face = pyrender_mesh.primitives[0].indices
    for u in range(0, h):
        for v in range(0, w):
            x_normalized = (u / (H) - 0.5) * 2 * 0.2
            y_normalized = (v / (W) - 0.5) * 2 * 0.2
            v_camera = (x_normalized, y_normalized, 0)
            v_world = R @ v_camera + t
            ray_origin = v_world[:3]  # 从相机平面发出,沿着z轴反方向
            # 使用 trimesh 的射线与网格交点计算函数
            locations, index_ray, index_tri = tri_mesh.ray.intersects_location(ray_origins=[ray_origin],
                                                                               ray_directions=[-camera_direction])

            if len(locations) > 0:
                p = locations[0]
                f = face[index_tri[0]]
                p0 = positions[f[0]]
                p1 = positions[f[1]]
                p2 = positions[f[2]]
                P = np.column_stack((p0, p1, p2))
                w = np.linalg.solve(P, p)
                # w = np.linalg.inv(P) @ p
                uv0 = tex_coord[f[0]]
                uv1 = tex_coord[f[1]]
                uv2 = tex_coord[f[2]]
                UV = np.column_stack((uv0, uv1, uv2))
                uv = UV @ w
                proj_map[u][v] = uv
                # _, indices = tree.query(locations[0])
                # # print(f"射线与网格的交点为：{locations[0]}")  # 交点坐标
                # project_points.append(tri_mesh.faces[index_tri[0]])
                # 计算交点深度
            # else:
            #    print("射线与网格没有交点")

    return proj_map

def projection_map(s: Myscene, view_image):
    # 将三维点与视图对应
    h, w = view_image.shape[:2]
    tri_mesh = s.tri_mesh
    pyrender_mesh = s.pyrender_mesh
    camera_pose = s.camera_pose
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    camera_direction = t
    positions = pyrender_mesh.primitives[0].positions
    face = pyrender_mesh.primitives[0].indices
    n = positions.shape[0]
    proj_map = np.full((n, 2), -1.0)# proj_map[i]表示第i个顶点在相机视图上的坐标
    for i in range(n):
        p = positions[i]
        ray_origin = p
        locations, index_ray, index_tri = tri_mesh.ray.intersects_location(ray_origins=[ray_origin],
                                                                           ray_directions=[camera_direction])
        if len(locations) == 1 or len(locations) == 0:
            # print("不算起始点")
            p_world = p
            p_camera = R.T @ (p_world - t)
            p_u = (0.5 + p_camera[0] / 0.4) * W
            p_v = (0.5 - p_camera[1] / 0.4) * H
            p_uv=[p_v, p_u]
            # p_uv = p_camera / 0.2
            # p_uv = p_uv.astype(np.int)
            proj_map[i][:] = p_uv[:2]
    # np.savetxt('proj_map2.txt', proj_map, fmt='%f')
    return proj_map
def confidence_map(s: Myscene):
    camera_pose = s.camera_pose
    camera_direction = -camera_pose[:3,2]
    pyrender_mesh = s.pyrender_mesh
    tex_image = render.get_texture_image(s)
    tex_h, tex_w = tex_image.shape[:2]
    points = pyrender_mesh.primitives[0].positions
    face = pyrender_mesh.primitives[0].indices
    tex_coord = pyrender_mesh.primitives[0].texcoord_0
    normals = pyrender_mesh.primitives[0].normals
    confidence_data = normals @ (-camera_direction)
    Ci = np.zeros((tex_h, tex_w))
    for i in range(points.shape[0]):
        u = int(tex_coord[i][0] * tex_h)
        v = int(tex_coord[i][1] * tex_w)
        Ci[u][v] = max(0., confidence_data[i])
    # 下面对面循环做插值
    Ci = texture_linear_interpolation(face,tex_coord,Ci)

    Ci = (Ci*255).astype(np.uint8)
    return Ci

def Cij(si: Myscene,sj: Myscene):
    Ci = confidence_map(si)
    Cj = confidence_map(sj)
    Cij = np.hstack((Ci, Cj))
    return Ci, Cj, Cij



def texture_img(s: Myscene, II):
    proj_map = projection_map(s, II)
    h, w = II.shape[:2]
    pyrender_mesh = s.pyrender_mesh
    tex_image = render.get_texture_image(s)
    tex_h, tex_w = tex_image.shape[:2]
    T = np.zeros_like(tex_image)
    # T[...,3]=(tex_image[...,3]*255).astype(np.uint8)
    T[...,3] = 255
    tex_coord = pyrender_mesh.primitives[0].texcoord_0
    face = pyrender_mesh.primitives[0].indices
    for i in range(proj_map.shape[0]):
        if proj_map[i][0] != -1:
            IIu, IIv = int(proj_map[i][0]), int(proj_map[i][1] )#* h* w
            # print('IIu, IIv',[IIu, IIv])
            tex_u, tex_v = int(tex_coord[i][0] * tex_h), int(tex_coord[i][1] * tex_w) #
            # print('tex_u, tex_v',[tex_u, tex_v])
            # print(f'{II[IIu][IIv][:3]},{II[IIu - 1][IIv][:3]},{ II[IIu][IIv - 1][:3]},{II[IIu - 1][IIv - 1][:3]}')
            # T[tex_u][tex_v][:3] = II[IIu][IIv][:3]
            T[tex_u][tex_v][0] = min(II[IIu+1][IIv+1][0], II[IIu + 1][IIv-1][0], II[IIu-1][IIv + 1][0],II[IIu - 1][IIv - 1][0])
            T[tex_u][tex_v][1] = min(II[IIu+1][IIv+1][1], II[IIu + 1][IIv-1][1], II[IIu-1][IIv + 1][1],II[IIu - 1][IIv - 1][1])
            T[tex_u][tex_v][2] = min(II[IIu+1][IIv+1][2], II[IIu + 1][IIv-1][2], II[IIu-1][IIv + 1][2],II[IIu - 1][IIv - 1][2])
            # T[tex_u][tex_v][3] = 255# 将透明度设置为255
    # print(T)
    # 对面插值
    T = texture_linear_interpolation(face, tex_coord, T)
    # T = (T*255).astype(np.uint8)
    return T

def Tij(si: Myscene, IIi, sj: Myscene, IIj):
    Ti = texture_img(si, IIi)
    Tj = texture_img(sj, IIj)
    Tij = np.hstack((Ti, Tj))
    return Ti, Tj, Tij

