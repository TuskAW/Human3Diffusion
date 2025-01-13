import blenderproc as bproc

import argparse, sys, os, math
import bpy
from mathutils import Vector, Matrix
import sys
import time
import numpy as np
from blenderproc.python.types.MeshObjectUtility import MeshObject, convert_to_meshes
import PIL.Image as Image
import pickle as pkl

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--view', type=int, default=132)
parser.add_argument(
    "--subject",
    type=str,
    default='0001',
    required=True,
)
parser.add_argument('--resolution', type=int, default=512)

parser.add_argument('--reset_object_euler', action='store_true')   
 
parser.add_argument('--radius', type=float, default=1.5)

parser.add_argument('--save_folder', type=str, default=1.5)                    

args = parser.parse_args()

def create_camera_to_world_matrix(elevation, azimuth, radius=1.0):
    elevation = np.radians(elevation)
    azimuth = np.radians(azimuth)
    x = np.cos(elevation) * np.sin(azimuth) * radius
    y = np.sin(elevation) * radius
    z = np.cos(elevation) * np.cos(azimuth) * radius

    camera_pos = np.array([x, y, z])
    target = np.array([0, 0, 0])
    up = np.array([0, 1, 0])

    forward = target - camera_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    new_up = np.cross(right, forward)
    new_up /= np.linalg.norm(new_up)
    cam2world = np.eye(4)
    cam2world[:3, :3] = np.array([right, new_up, -forward]).T
    cam2world[:3, 3] = camera_pos
    return cam2world

def convert_opengl_to_blender(camera_matrix):
    if isinstance(camera_matrix, np.ndarray):
        flip_yz = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        camera_matrix_blender = np.dot(flip_yz, camera_matrix)
    return camera_matrix_blender

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def normalize_scene_human():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    offset[2] += 0.01
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset

    norm_human_loc = bpy.data.objects["human"].location
    bpy.ops.object.select_all(action="DESELECT")

    return norm_human_loc, scale

def get_a_camera_location(loc):
    location = Vector([loc[0],loc[1],loc[2]])
    direction = - location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    rotation_euler = rot_quat.to_euler()
    return location, rotation_euler

def normalize_camera(camera_matrix):
    if isinstance(camera_matrix, np.ndarray):
        camera_matrix = camera_matrix.reshape(-1, 4, 4)
        translation = camera_matrix[:, :3, 3]
        translation = translation / (
            np.linalg.norm(translation, axis=1, keepdims=True) + 1e-8
        )
        camera_matrix[:, :3, 3] = translation
    return camera_matrix.reshape(-1, 16)

def get_3x4_RT_matrix_from_blender(cam):
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()
    T_world2bcam = -1*R_world2bcam @ location
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT

def get_calibration_matrix_K_from_blender(mode='simple'):
    scene = bpy.context.scene
    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale
    height = scene.render.resolution_y * scale
    camdata = scene.camera.data

    if mode == 'simple':
        aspect_ratio = width / height
        K = np.zeros((3,3), dtype=np.float32)
        K[0][0] = width / 2 / np.tan(camdata.angle / 2)
        K[1][1] = height / 2. / np.tan(camdata.angle / 2) * aspect_ratio
        K[0][2] = width / 2.
        K[1][2] = height / 2.
        K[2][2] = 1.
        K.transpose()
    
    if mode == 'complete':
        focal = camdata.lens
        sensor_width = camdata.sensor_width
        sensor_height = camdata.sensor_height
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

        if (camdata.sensor_fit == 'VERTICAL'):
            s_u = width / sensor_width / pixel_aspect_ratio 
            s_v = height / sensor_height
        else:
            pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
            s_u = width / sensor_width
            s_v = height * pixel_aspect_ratio / sensor_height

        alpha_u = focal * s_u
        alpha_v = focal * s_v
        u_0 = width / 2
        v_0 = height / 2
        skew = 0

        K = np.array([
            [alpha_u,    skew, u_0],
            [      0, alpha_v, v_0],
            [      0,       0,   1]
        ], dtype=np.float32)
    
    return K

def load_human_obj(human_obj_file, human_tex_file, smpl_z_component):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.import_scene.obj(filepath=human_obj_file, axis_forward='-Z', axis_up='Y')
    obj_object = bpy.context.selected_objects[-1]
    obj_object.name = 'human'
    obj_object.rotation_euler[2] += 1.5708
    obj_object.rotation_euler[2] -= smpl_z_component

    bpy.ops.object.select_all(action='DESELECT')
    obj = bpy.data.objects['human']
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()

    object_material = bpy.data.materials.new('object_material')
    object_material.use_nodes = True
    bsdf = object_material.node_tree.nodes["Principled BSDF"]
    bsdf.inputs['Specular'].default_value = 0
    object_texture = object_material.node_tree.nodes.new('ShaderNodeTexImage')
    object_texture.image = bpy.data.images.load(human_tex_file)
    object_material.node_tree.links.new(bsdf.inputs['Base Color'], object_texture.outputs['Color'])
    obj.active_material = object_material

def reset_scene() -> None:
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

bproc.init()

cam = bpy.context.scene.objects['Camera']
cam.data.sensor_width = 32
desired_fov_deg = 49.1
desired_fov_rad = math.radians(desired_fov_deg)
sensor_width = cam.data.sensor_width
focal_length = sensor_width / (2 * math.tan(desired_fov_rad / 2))
cam.data.lens = focal_length

def get_camera_objects():
    cameras = [obj for obj in bpy.context.scene.objects if obj.type == 'CAMERA']
    return cameras

VIEWS = ["_0", "_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9", "_10", "_11", "_12", "_13", "_14", "_15", "_16", "_17", "_18", "_19", "_20", "_21", "_22", "_23", "_24", "_25", "_26", "_27", "_28", "_29", "_30", "_31", "_32", "_33", "_34", "_35", "_36", "_37", "_38", "_39", "_40", "_41", "_42", "_43", "_44", "_45", "_46", "_47", "_48", "_49", "_50", "_51", "_52", "_53", "_54", "_55", "_56", "_57", "_58", "_59", "_60", "_61", "_62", "_63", "_64", "_65", "_66", "_67", "_68", "_69", "_70", "_71", "_72", "_73", "_74", "_75", "_76", "_77", "_78", "_79", "_80", "_81", "_82", "_83", "_84", "_85", "_86", "_87", "_88", "_89", "_90", "_91", "_92", "_93", "_94", "_95", "_96", "_97", "_98", "_99", "_100", "_101", "_102", "_103", "_104", "_105", "_106", "_107", "_108", "_109", "_110", "_111", "_112", "_113", "_114", "_115", "_116", "_117", "_118", "_119", "_120", "_121", "_122", "_123", "_124", "_125", "_126", "_127", "_128", "_129", "_130", "_131"]
ELEVATION = np.arange(-45, 75, 105/args.view)
AZIMUTH = np.arange(0*5, 360*5, 360*5/args.view)

def save_images(save_folder, scan_subj_path) -> None:
    reset_scene()

    haven_hdri_path = bproc.loader.get_random_world_background_hdr_img_path_from_haven('/mnt/qb/ponsmoll/yxue/Blenderproc/Resources_studio')
    bproc.world.set_world_background_hdr_img(haven_hdri_path)

    subj_name = scan_subj_path.split('/')[-1]
    scan_file = os.path.join(scan_subj_path, '{}.obj'.format(subj_name))
    scan_tex = os.path.join(scan_subj_path, 'material0.jpeg')
    scan_smpl_path = os.path.join(scan_subj_path.replace('model', 'smplx'), 'smplx_param.pkl')
    smpl_param = pkl.load(open(scan_smpl_path, 'rb'))
    smpl_global_orient_z = smpl_param['global_orient'][0, 1]

    os.makedirs(os.path.join(save_folder, subj_name), exist_ok=True)

    load_human_obj(scan_file, scan_tex, smpl_global_orient_z)

    if args.reset_object_euler:
        for obj in scene_root_objects():
            obj.rotation_euler[0] = 0
        bpy.ops.object.select_all(action="DESELECT")

    norm_human_loc, norm_scale = normalize_scene_human() 

    try:
        mesh_objects = convert_to_meshes([obj for obj in scene_meshes()])
        for obj in mesh_objects:
            print("removing invalid normals")
            for mat in obj.get_materials():
                mat.set_principled_shader_value("Normal", [1,1,1])
    except:
        print("don't know why")
    
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    
    radius = args.radius
    
    camera_locations = [np.array([0,-radius,0]),] * args.view 
    for location in camera_locations:
        _location,_rotation = get_a_camera_location(location)
        bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=_location, rotation=_rotation,scale=(1, 1, 1))
        _camera = bpy.context.selected_objects[0]
        _constraint = _camera.constraints.new(type='TRACK_TO')
        _constraint.track_axis = 'TRACK_NEGATIVE_Z'
        _constraint.up_axis = 'UP_Y'
        _camera.parent = cam_empty
        _constraint.target = cam_empty
        _constraint.owner_space = 'LOCAL'

    bpy.context.view_layer.update()

    viewidx = args.view
    
    for j in range(viewidx):
        view = f"{viewidx:03d}"+ VIEWS[j]

        if j < 100:
            elevation = ELEVATION[j]
            azimuth = AZIMUTH[j]
        else:
            elevation = 0
            azimuth = 360/(viewidx-100)*(j-100)
        
        camera_matrix = create_camera_to_world_matrix(elevation, azimuth, args.radius)
        camera_matrix = convert_opengl_to_blender(camera_matrix)

        bproc.camera.set_resolution(args.resolution, args.resolution)
        bproc.camera.add_camera_pose(camera_matrix)
        
        RT_path = os.path.join(save_folder, subj_name, view+"_RT.txt")
        K_path = os.path.join(save_folder, subj_name, view+"_K.txt")
        K = get_calibration_matrix_K_from_blender()
        np.savetxt(RT_path, camera_matrix.reshape(-1, 16))
        np.savetxt(K_path, K)
        
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    data = bproc.renderer.render()

    for j in range(viewidx):
        index = j
        
        view = f"{viewidx:03d}"+ VIEWS[j]
        
        depth_map = data['depth'][index]
        depth_max = np.max(depth_map)
        valid_mask = depth_map!=depth_max
        invalid_mask = depth_map==depth_max
        depth_map[invalid_mask] = 0
        
        depth_map = np.uint16((depth_map / 10) * 65535)

        normal_map = data['normals'][index]*255

        valid_mask = valid_mask.astype(np.int8)*255

        color_map = data['colors'][index]
        color_map = np.concatenate([color_map, valid_mask[:, :, None]], axis=-1)

        Image.fromarray(color_map.astype(np.uint8)).save(
        '{}/{}/rgb_{}.png'.format(save_folder, subj_name, view), "png", quality=100)
        
        Image.fromarray(normal_map.astype(np.uint8)).save(
        '{}/{}/normals_{}.png'.format(save_folder, subj_name, view), "png", quality=100)
        

if __name__ == "__main__":
    start_i = time.time()

    # blenderproc run --blender-install-path /home/yuxuan/project/ render_bproc_thuman2.py --subject 0001
    save_folder = './rendering_data/ImagedreamLGM_thuman2_132view'
    scan_base_path = '/mnt/qb/ponsmoll/datasets/clothing/public_datasets/THuman2.1_Release/model/'
    
    subj = args.subject
    scan_subj_path = os.path.join(scan_base_path, subj)

    save_images(save_folder, scan_subj_path)

    end_i = time.time()