import os
import numpy as np
import torch
# from Code.Models.options import load_options
# from Code.renderer import Camera, Scene, TransferFunction
from pyquaternion import Quaternion

class Arcball():
    '''
    arcball camera model:
        camera matrix initialized with AABB center as origin, on pos z_axis, dist away, looking at origin.
        user rotation will rotate the matrix
        user zoom will translate the matrix
    '''
    def __init__(
        self,
        scene_aabb:np.ndarray,
        coi:np.ndarray,
        dist,
        fov,
    ) -> None:
        self.fov = fov
        self.coi = coi.astype(dtype=np.float32)
        self.coi_translate = np.eye(4, dtype=np.float32)
        self.coi_translate[:3, 3] = -coi
        self.aabb = scene_aabb
        self.coaabb = scene_aabb.reshape(2, 3).mean(0)
        self.dist = dist
        self.aabb = scene_aabb.astype(dtype=np.float32)
        self.vMat = np.eye(4, dtype=np.float32)
        # print(coi, scene_aabb, dist, fov)
        # self.vMat[:3, 3] = -coi
        # self.vMat[2, 3] -= dist
        self.translation = np.eye(4, dtype=np.float32)
        self.translation[:3, 3] = [0.,0.,-dist]
        self.rotation = np.eye(4, dtype=np.float32)
        self.vMat = self.translation @ self.rotation @ self.coi_translate
        self.c2w = np.linalg.inv(self.vMat)
        self.mouse_start = np.zeros(2, dtype=np.float32)
        self.mouse_curr = np.zeros(2, dtype=np.float32)
        self.camera_dirs = None
        # self.zoom_unit = scene_aabb
    
    def position(self) -> np.ndarray:
        return self.c2w[:3, 3]
    
    def get_coi(self) -> np.ndarray:
        return self.coi
    
    def get_cam_dir(self) -> np.ndarray:
        return normalize_vec(self.coi - self.position())
    
    def rotate(self) -> None:
        '''
        calculate arcball rotation by p_start and p_curr
        '''
        # print(self.vMat,"\n")
        # print(self.c2w)
        ball_start = screen_to_arcball(self.mouse_start)
        ball_curr = screen_to_arcball(self.mouse_curr)
        rot_radians = vec_angle(ball_start, ball_curr)
        rot_axis = normalize_vec(np.cross(ball_start, ball_curr))
        # print(ball_start, ball_curr)
        rot = axis_rotate(rot_radians, rot_axis)
        q = Quaternion(axis=rot_axis, radians=rot_radians)
        # print(np.rad2deg(rot_radians), rot_axis, q.rotation_matrix)
        # rotate the camera axes and position in place
        self.rotation[:3,:3] = q.rotation_matrix @ self.rotation[:3,:3]
        cam_pos_wrt_origin = self.position() - self.coi
        # print(cam_pos_wrt_origin.shape)
        # self.vMat[:3, 3] = (rot @ cam_pos_wrt_origin[...,None]).squeeze() + self.coi
        self.update_vMat()
        self.update_c2w()
        # print("end")
        # print(self.vMat,"\n")
        # print(self.c2w)
        
    def zoom(self, delta) -> None:
        '''
        use absolution zoom amount when far away,
        but when close to COI, approach COI with proportional distance
        '''
        cam_dir = self.get_cam_dir()
        abs_translation = self.aabb.max()/10
        proportion_translation = self.translation[2, 3]/20
        trans = np.minimum(np.abs(abs_translation), np.abs(proportion_translation))
        # print(abs_translation, proportion_translation, trans)
        trans = -trans if delta < 0 else trans
        self.translation[2, 3] -= trans
        self.update_vMat()
        self.update_c2w()
        # print(self.translation, "\n")
        # print(self.vMat,"\n")
        # print(self.c2w, "\n")
    
    def pan(self, deltaxy):
        unit_motion = self.translation[2, 3] * 0.5
        delta_xy_cam = np.array([*deltaxy, 0., 0.], dtype=np.float32)*unit_motion
        delta_xy_world = self.c2w @ delta_xy_cam
        self.coi += delta_xy_world[:3]
        
        self.update_coi(self.coi)
        self.update_vMat()
        self.update_c2w()
    
    def update_dist(self, d) -> None:
        self.translation = np.eye(4, dtype=np.float32)
        self.translation[:3, 3] = [0.,0.,-d]
        self.vMat = self.translation @ self.rotation @ self.coi_translate
        self.c2w = np.linalg.inv(self.vMat)
        
    def update_coi(self, coi) -> None:
        self.coi = coi.astype(dtype=np.float32)
        self.coi_translate = np.eye(4, dtype=np.float32)
        self.coi_translate[:3, 3] = -coi
        self.vMat = self.translation @ self.rotation @ self.coi_translate
        self.c2w = np.linalg.inv(self.vMat)
        
    def update_vMat(self) -> None:
        self.vMat = (self.translation  @ self.rotation @ self.coi_translate).astype(np.float32)
    
    def update_c2w(self) -> None:
        self.c2w = np.linalg.inv(self.vMat).astype(np.float32)
        # print(self.c2w)
    
    def get_c2w(self) -> np.ndarray:
        return self.c2w
    
    def resize(self, width, height) -> None:
        self.camera_dirs = generate_camera_dirs(width, height, self.fov)
    
    def generate_dirs(self, width, height) -> np.ndarray:
        # print(self.get_c2w())
        return generate_dirs(width, height, self.get_c2w(), self.fov, self.camera_dirs)
    
    def reset_view_xy(self, coi, dist):
        self.translation = np.eye(4, dtype=np.float32)
        self.dist = dist
        self.translation[:3, 3] = [0.,0.,-self.dist]
        self.rotation = np.eye(4, dtype=np.float32)
        self.update_coi(coi)
        self.update_vMat()
        self.update_c2w()

def axis_rotate(radians: np.ndarray, axis: np.ndarray):
    '''
    https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    '''
    axis = normalize_vec(axis)
    c = np.cos(radians)
    s = np.sin(radians)
    x,y,z = axis
    return np.array([
        [x*x*(1-c)+c,      x*y*(1-c)-z*s,    x*z*(1-c)+y*s, 0],
        [y*x*(1-c)+z*s,    y*y*(1-c)+c,      y*z*(1-c)-x*s, 0],
        [z*x*(1-c)-y*s,    z*y*(1-c)+x*s,    z*z*(1-c)+c, 0],
        [0,0,0,1]
    ])

def screen_to_arcball(p:np.ndarray):
    dist = np.dot(p, p)
    if dist < 1.:
        return np.array([*p, np.sqrt(1.-dist)])
    else:
        return np.array([*normalize_vec(p), 0.])
    
def normalize_vec(v: np.ndarray):
    if v is None:
        print("None")
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.all(norm == np.zeros_like(norm)):
        return np.zeros_like(v)
    else:
        return v/norm

def vec_angle(v0: np.ndarray, v1: np.ndarray):
    return np.arccos(np.clip(np.dot(v0, v1)/(np.linalg.norm(v0)*np.linalg.norm(v1)), -1., 1.))

def generate_camera_dirs(width, height, fov):
    x, y = np.meshgrid(
        np.arange(width),
        np.arange(height),
        indexing="xy",
    )
    x = x.flatten()
    y = y.flatten()
    
    # move x, y to pixel center, rescale to [-1, 1] and invert y
    x = (2*(x+0.5)/width - 1) *  np.tan(np.deg2rad(fov/2))*(width/height)
    y = (1 - 2*(y+0.5)/height) * np.tan(np.deg2rad(fov/2))
    z = -np.ones(x.shape, dtype=np.float32)
    camera_dirs = np.stack([x,y,z],-1).astype(dtype=np.float32) # (height*width, 3)
    return camera_dirs

def generate_dirs(width, height, c2w, fov, camera_dirs = None):
    '''
    generate viewing ray directions of number width x height.
    instance vars need:
        self.fov
    '''
    if(camera_dirs is None):
        camera_dirs = generate_camera_dirs(width, height, fov)
    # map camera space dirs to world space
    directions = (c2w[:3,:3] @ camera_dirs.T).T
    directions = normalize_vec(directions)
    # print(directions.max(0), directions.min(0), np.linalg.norm(directions, axis=-1).max())
    
    return directions.reshape(height, width, 3)

def torch_float_to_numpy_uint8(tensor: torch.Tensor):
    return (tensor*255).cpu().numpy().astype(np.uint8)