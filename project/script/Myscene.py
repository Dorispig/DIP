import render
import pyrender
class Myscene:
    def __init__(self,tri_mesh= None,pyrender_mesh= None,camera_node= None,camera_pose= None,
                 light1_type= None,light1_pose= None,light2_type= None,light2_pose= None,
                 light3_type= None,light3_pose= None,light4_type= None,light4_pose= None):
        self.tri_mesh = tri_mesh
        self.pyrender_mesh = pyrender_mesh
        self.camera_node = camera_node
        self.camera_pose = camera_pose
        self.light1_type = light1_type
        self.light1_pose = light1_pose
        self.light2_type = light2_type
        self.light2_pose = light2_pose
        self.light3_type = light3_type
        self.light3_pose = light3_pose
        self.light4_type = light4_type
        self.light4_pose = light4_pose
        # self.scene = pyrender.Scene()
        # self.scene.add(pyrender_mesh)
        # render.add_light(self.scene, self.light1_type, self.light1_pose)
        # render.add_light(self.scene, self.light2_type, self.light2_pose)
        # render.add_camera(self.scene, self.camera_node, self.camera_pose)