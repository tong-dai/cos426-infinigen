import argparse
import os
import sys
from pathlib import Path
import itertools
import logging
from copy import copy

logging.basicConfig(
    format='[%(asctime)s.%(msecs)03d] [%(name)s] [%(levelname)s] | %(message)s',
    datefmt='%H:%M:%S',
    level=logging.WARNING
)

import bpy
import bmesh
import mathutils

import gin
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pprint import pformat
import imageio

from infinigen.assets.lighting import sky_lighting

from infinigen.core.placement.factory import AssetFactory
from infinigen.core.placement.camera import spawn_camera, set_active_camera
from infinigen.core.nodes import Nodes, NodeWrangler
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed

from infinigen.core import execute_tasks, surface, init

logging.basicConfig(level=logging.INFO)

MIN_PROBABILITY = 0.1

class Branch:
    def __init__(self, start, direction, distance, starting_radius,
                 branch_probability, mean_branch_length, max_segment_angle, 
                 mean_segment_length, max_branch_angle, rotation_normal):
        self.start = start
        self.direction = direction
        self.distance = distance
        self.starting_radius = starting_radius
        self.branch_probability = branch_probability
        self.mean_branch_length = mean_branch_length
        self.max_segment_angle = max_segment_angle
        self.mean_segment_length = mean_segment_length
        self.max_branch_angle = max_branch_angle
        self.rotation_normal = rotation_normal

        self.radius_multiplier = 0.5
        self.branch_probability_multiplier = 0.7
        self.mean_branch_length_multiplier = 0.5
        self.max_segment_angle_multiplier = 1.1
        

def handle_branching(bm, branch, next):
    random_probability = np.random.uniform(0, 1)
    if branch.branch_probability < MIN_PROBABILITY or random_probability > branch.branch_probability:
        return
    
    new_angle = np.random.uniform(-branch.max_segment_angle, branch.max_segment_angle)
    new_length = np.random.uniform(0, branch.mean_branch_length * 2)
    
    mat_rot = mathutils.Matrix.Rotation(new_angle, 4, 'X')
    new_direction = branch.direction @ mat_rot

    new_branch_probability = branch.branch_probability * branch.branch_probability_multiplier
    new_mean_branch_length = branch.mean_branch_length * branch.mean_branch_length_multiplier
    new_max_segment_angle = branch.max_segment_angle * branch.max_segment_angle_multiplier
    new_starting_radius = branch.starting_radius * branch.radius_multiplier

    new_branch = Branch(next, new_direction, new_length, new_starting_radius,
                        new_branch_probability, new_mean_branch_length,
                        new_max_segment_angle, branch.mean_segment_length, branch.max_branch_angle,
                        branch.rotation_normal)

    branch_geometry(bm, new_branch)

def branch_geometry(bm, branch):
    next = branch.start
    last = branch.start
    direction = branch.direction  
    radius = branch.starting_radius

    while np.linalg.norm(branch.start - next) < branch.distance:
        angle = np.random.uniform(-branch.max_segment_angle, branch.max_segment_angle)
        length = np.random.uniform(0, branch.mean_segment_length * 2)

        mat_rot = mathutils.Matrix.Rotation(angle, 4, 'X')
        mat_out = direction @ mat_rot

        next = mat_out
        next = next * length
        next = next + last

        start_vert = bm.verts.new(next)
        end_vert = bm.verts.new(last)
        
        edge = bm.edges.new([start_vert, end_vert])
        
        ret = bmesh.ops.extrude_edge_only(bm, edges=[edge])
        verts = [v for v in ret['geom'] if isinstance(v, bmesh.types.BMVert)]
        for v in verts:
            v.co += mathutils.Vector((radius, radius, radius))

        handle_branching(bm, branch, next)
        last = next

def create_branch_object(distance, starting_radius):
    start = mathutils.Vector((0, 0, 0))
    direction = mathutils.Vector((0, 0, -1))
    dist = distance

    radius = starting_radius
    branch_probability = 0.2
    mean_branch_length = 1.1
    max_segment_angle = np.radians(40)
    mean_segment_length = 0.08
    max_branch_angle = np.radians(65)
    rotation_normal = mathutils.Vector((0, 0, 1))

    bm = bmesh.new()
    main_branch = Branch(start, direction, dist, radius,
                         branch_probability, mean_branch_length, max_segment_angle, 
                         mean_segment_length, max_branch_angle, rotation_normal)
    branch_geometry(bm, main_branch)

    mesh = bpy.data.meshes.new("branch_mesh")
    branch_object = bpy.data.objects.new("Branch", mesh)
    bm.to_mesh(mesh)
    bm.free()
    return branch_object


def my_shader(nw: NodeWrangler, params: dict):
    color1 = (np.random.uniform(0.1, 0.4), np.random.uniform(0.1, 0.3), np.random.uniform(0.1, 1), 1)
    color2 = (np.random.uniform(0.1, 1), np.random.uniform(0.1, 0.2), np.random.uniform(0.1, 0.5), 1)

    air_density = np.random.uniform(1, 10)

    geometry = nw.new_node(Nodes.NewGeometry)
    
    color_ramp = nw.new_node(Nodes.ColorRamp, input_kwargs={'Fac': geometry.outputs[params['geometry_mode']]})
    color_ramp.color_ramp.elements[0].color = color1
    color_ramp.color_ramp.elements[1].color = color2
    
    emission = nw.new_node(Nodes.Emission, input_kwargs={'Color': color_ramp, 'Strength': params['emission_strength']})
    
    sky_texture = nw.new_node(Nodes.SkyTexture, attrs={'sky_type': "NISHITA", 'sun_elevation': params['sun_elevation'], 'air_density': air_density})

    mix_shader = nw.new_node(Nodes.MixShader)
    nw.links.new(mix_shader.inputs[1], emission.outputs[0])
    nw.links.new(mix_shader.inputs[2], sky_texture.outputs[0])
    
    material_output = nw.new_node(Nodes.MaterialOutput,
        input_kwargs={'Surface': mix_shader},
        attrs={'is_active_output': True})

class MyAsset(AssetFactory):

    def __init__(self, factory_seed: int, overrides=None):
        self.name = 'branches'
        super().__init__(factory_seed)
        
        with FixedSeed(factory_seed):
            self.params = self.sample_params()
            if overrides is not None:
                self.params.update(overrides)

    def sample_params(self):
        modes = ['Backfacing', 'Incoming', 'Tangent']
        return {
            'distance': np.random.uniform(7.0, 25.0),  
            'starting_radius': np.random.uniform(0.01, 0.1),
            'emission_strength': np.random.uniform(0, 10),
            'sun_elevation': np.random.uniform(0, np.pi * 2),
            'geometry_mode': modes[np.random.randint(0, 3)],
        }

    def create_asset(self, **_):
        branch_obj = create_branch_object(
            distance=self.params['distance'],
            starting_radius=self.params['starting_radius'],
        )
        bpy.context.collection.objects.link(branch_obj)
        
        surface.add_material(branch_obj, my_shader, input_kwargs=dict(params=self.params))
        return branch_obj
    
def set_eevee_bloom():
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'

    bpy.context.scene.eevee.use_bloom = True
    bpy.context.scene.eevee.bloom_threshold = 0.8
    bpy.context.scene.eevee.bloom_knee = 0.5
    bpy.context.scene.eevee.bloom_radius = 6.5
    bpy.context.scene.eevee.bloom_color = (1, 1, 1)
    bpy.context.scene.eevee.bloom_intensity = 0.6
    bpy.context.scene.eevee.bloom_clamp = 0.5
    

@gin.configurable
def compose_scene(output_folder, scene_seed, overrides=None, **params):
    sky_lighting.add_lighting()

    cam = spawn_camera()
    cam.location = (7, 7, 3.5)
    cam.rotation_euler = np.deg2rad((70, 0, 135))
    set_active_camera(cam)
    set_eevee_bloom()

    factory = MyAsset(factory_seed=np.random.randint(0, 1e7))
    if overrides is not None:
        factory.params.update(overrides)

    factory.spawn_asset(i=np.random.randint(0, 1e7))

def iter_overrides(ranges):
    mid_vals = {k: v[len(v)//2] for k, v in ranges.items()}
    for k, v in ranges.items():
        for vi in v:
            res = copy(mid_vals)
            res[k] = vi
            yield res

def create_param_demo(args, seed):

    modes = ['Backfacing', 'Incoming', 'Tangent']
    override_ranges = {
        'distance': np.linspace(7.0, 25.0, num=3),  
        'starting_radius': np.linspace(0.01, 0.1, num=3),
        'emission_strength': np.linspace(0, 10, num=3),
        'sun_elevation': np.linspace(0, np.pi * 2, num=3),
        'geometry_mode': modes
    }
    for i, overrides in enumerate(iter_overrides(override_ranges)):
        butil.clear_scene()
        print(f'{i=} {overrides=}')
        with FixedSeed(seed):
            compose_scene(args.output_folder, seed, overrides=overrides)
        
        if args.save_blend:
            butil.save_blend(args.output_folder/f'scene_{i}.blend', verbose=True)

        bpy.context.scene.frame_set(i)
        bpy.context.scene.frame_start = i
        bpy.context.scene.frame_end = i
        bpy.ops.render.render(animation=True)

        imgpath = args.output_folder/f'{i:04d}.png'
        img = Image.open(imgpath)
        ImageDraw.Draw(img).text(
            xy=(10, 10), 
            text='\n'.join(f'{k}: {v:.2f}' for k, v in overrides.items()), 
            fill=(76, 252, 85),
            font=ImageFont.load_default(size=50)
        )
        img.save(imgpath)
        

def create_video(args, seed):
    butil.clear_scene()
    with FixedSeed(seed):
        compose_scene(args.output_folder, seed)

    butil.save_blend(args.output_folder/'scene.blend', verbose=True)

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = args.duration_frames
    bpy.ops.render.render(animation=True)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=Path)
    parser.add_argument('--mode', type=str, choices=['param_demo', 'video'])
    parser.add_argument('--duration_frames', type=int, default=1)
    parser.add_argument('--save_blend', action='store_true')
    parser.add_argument('-s', '--seed', default=None, help="The seed used to generate the scene")
    parser.add_argument('-g', '--configs', nargs='+', default=['base'],
                        help='Set of config files for gin (separated by spaces) '
                             'e.g. --gin_config file1 file2 (exclude .gin from path)')
    parser.add_argument('-p', '--overrides', nargs='+', default=[],
                        help='Parameter settings that override config defaults '
                             'e.g. --gin_param module_1.a=2 module_2.b=3')
    parser.add_argument('-d', '--debug', action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.INFO)

    args = init.parse_args_blender(parser)
    logging.getLogger("infinigen").setLevel(args.loglevel)

    seed = init.apply_scene_seed(args.seed)
    init.apply_gin_configs(
        configs=args.configs, 
        overrides=args.overrides,
        configs_folder='infinigen_examples/configs'
    )

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 50

    args.output_folder.mkdir(exist_ok=True, parents=True)
    bpy.context.scene.render.filepath = str(args.output_folder.absolute()) + '/'


    if args.mode == 'param_demo':
        create_param_demo(args, seed)
    elif args.mode == 'video':
        create_video(args, seed)
    else:
        raise ValueError(f'Unrecognized {args.mode=}')
    