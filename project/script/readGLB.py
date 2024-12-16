import pyrender
import trimesh
import numpy as np
import imageio

def readGLB_trimesh(path):
    """
    读取glb模型为网格
    Parameters
    ----------
    path

    Returns
    -------

    """
    tri_mesh = trimesh.load(path,force='mesh')# force='mesh'要加，不然拿不出位置信息
    return tri_mesh

def readGLB_pyrender_mesh(path,material):
    """
    读取渲染后的glb模型
    Parameters
    ----------
    path
    material

    Returns
    -------

    """
    trimesh=readGLB_trimesh(path)
    pyrender_mesh = pyrender.Mesh.from_trimesh(trimesh, material=material)
    return pyrender_mesh

file = "000"
tri_mesh = readGLB_trimesh(file+".glb")
vertices = np.array(tri_mesh.vertices)# n*3的点数组

# 创建一个白色的材质，RGBA值为[1, 1, 1, 1]
material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[1, 1, 1, 1])
pyrender_mesh = readGLB_pyrender_mesh(file+".glb",material)


"""
TODO:建立映射：projection mapping 和 UV mapping
"""


