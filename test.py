import bpy
import math
import os
import numpy as np

def add_constraint(camera: bpy.types.Object, obj_to_track: bpy.types.Object):
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = obj_to_track
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

def set_camera_prop(camera: bpy.types.Object, focus_obj: bpy.types.Object, focal_length, name="camera_"):
    camera.name = name
    camera.data.lens = focal_length
    camera.data.dof.focus_object = focus_obj

def distribute_points_on_sphere(N_lat, N_lon):  # P (num_points) = N_lat * N_lon
    points = []
    latitudes = np.linspace(-math.pi / 2, math.pi / 2, N_lat)  # Latitude range: -π/2 to π/2
    for phi in latitudes:
        longitudes = np.linspace(0, 2 * math.pi, N_lon)
        for theta in longitudes:
            x = 50 * np.cos(phi) * np.cos(theta)
            y = 50 * np.cos(phi) * np.sin(theta)
            z = 50 * np.sin(phi)
            points.append([x, y, z])
    return np.array(points)

# Load the .blend file
blend_file_path = "mercedes.blend"  # Change this to the actual path
bpy.ops.wm.open_mainfile(filepath=blend_file_path)

scene = bpy.context.scene
focus_object = bpy.data.objects["Mercedes"]  # Replace "Cube" with the object you want to focus on, e.g., "Mercedes"

view_layer = bpy.context.view_layer

# Add a new light source
light_data = bpy.data.lights.new(name="New Light", type='SUN')
light_data.energy = 1000
light_object = bpy.data.objects.new(name="New Light", object_data=light_data)
view_layer.active_layer_collection.collection.objects.link(light_object)
light_object.location = (4.23433, 3.2186, 2.0674)
light_object.select_set(True)
view_layer.objects.active = light_object

# Generate and position cameras
num_cameras = 110
focal_length = 100
output_dir = "./outputs2/"  # Make sure this directory exists
os.makedirs(output_dir, exist_ok=True)

points = distribute_points_on_sphere(11, 10)

for i, p in enumerate(points):
    bpy.ops.object.camera_add(location=p)
    current_camera = bpy.context.object
    set_camera_prop(current_camera, focus_object, focal_length=focal_length, name=f"cam_({i})")
    add_constraint(current_camera, focus_object)

# Render images from each camera's perspective
for obj in scene.objects:
    if obj.type == "CAMERA":
        scene.camera = obj
        output_path = os.path.join(output_dir, f"{obj.name}.png")
        scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        print(f"Rendered and saved: {output_path}")
