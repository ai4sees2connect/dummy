import bpy
import os

# Load your .blend file
blend_file_path = "path/to/your/dmu_model.blend"
bpy.ops.wm.open_mainfile(filepath=blend_file_path)

# Access the scene
scene = bpy.context.scene

# List all objects in the .blend file
print("Available objects in the scene:")
for obj in bpy.data.objects:
    print(f"Object name: {obj.name}, Type: {obj.type}")

# Automatically select the first mesh object (assuming your DMU is a mesh)
model = next((obj for obj in bpy.data.objects if obj.type == 'MESH'), None)

if not model:
    raise ValueError("No mesh object found in the .blend file. Please check your file.")

print(f"Using model: {model.name}")

# Set the object's location if needed
model.location = (0, 50, 50)

# Set up lighting
view_layer = bpy.context.view_layer

light_data = bpy.data.lights.new(name="New Light", type='SUN')
light_data.energy = 1000

light_object = bpy.data.objects.new(name="New Light", object_data=light_data)
view_layer.active_layer_collection.collection.objects.link(light_object)
light_object.location = (4.23433, 53.2186, 52.0674)

# Camera placement and rendering settings
num_cameras = 100
focal_length = 70

def add_constraint(camera: bpy.types.Object, obj_to_track: bpy.types.Object):
    constraint = camera.constraints.new(type='TRACK_TO')
    constraint.target = obj_to_track
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

def set_camera_prop(camera: bpy.types.Object, focus_obj: bpy.types.Object, focal_length=focal_length, name="camera_"):
    camera.name = name
    camera.data.lens = focal_length
    camera.data.dof.focus_object = focus_obj

output_dir = "outputs1/"
os.makedirs(output_dir, exist_ok=True)

# Place cameras around the model
for i in range(0, 10):
    for j in range(0, 10):
        bpy.ops.object.camera_add(location=(10, i * 10, j * 10))
        current_camera = bpy.context.object
        set_camera_prop(current_camera, model, focal_length=focal_length, name=f"cam_({i},{j})")
        add_constraint(current_camera, model)

# Render from each camera
for obj in scene.objects:
    if obj.type == "CAMERA":
        scene.camera = obj
        file = os.path.join(output_dir, f"{obj.name}.png")
        scene.render.filepath = file
        bpy.ops.render.render(write_still=True)
        print(f"Rendered image saved at: {file}")
