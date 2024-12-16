import bpy
import os

# File to save the camera positions, rotations, and intrinsics
output_file = "/home/hussain/Projects/3DVSS/3D_file/uploads_files_2930090_blend/annot.txt"

def save_camera_position():
    # Get the Camera object
    camera = bpy.data.objects.get("Camera")
    
    if not camera:
        print("Error: Camera not found.")
        return
    
    # Ask user for a position name
    position_name = "pos1_actual"
    
    # Read position and rotation
    position = camera.location
    rotation = camera.rotation_euler
    
    # Fetch Camera Intrinsics
    camera_data = camera.data
    focal_length = camera_data.lens  # Focal length in mm
    sensor_width = camera_data.sensor_width  # Sensor width in mm
    sensor_height = camera_data.sensor_height  # Sensor height in mm
    lens_unit = camera_data.lens_unit  # Perspective or orthographic
    lens_type = "Perspective" if camera_data.type == 'PERSP' else "Orthographic"

    # Prepare data to save
    data = (
        f"Position Name: {position_name}\n"
        f"Location: X={position.x:.4f}, Y={position.y:.4f}, Z={position.z:.4f}\n"
        f"Rotation: X={rotation.x:.4f}, Y={rotation.y:.4f}, Z={rotation.z:.4f}\n"
        f"Camera Intrinsics:\n"
        f"  Focal Length: {focal_length:.2f} mm\n"
        f"  Sensor Width: {sensor_width:.2f} mm\n"
        f"  Sensor Height: {sensor_height:.2f} mm\n"
        f"  Lens Type: {lens_type}\n"
        f"{'-'*40}\n"
    )
    
    # Append data to the file
    try:
        with open(output_file, "a") as file:
            file.write(data)
        print(f"Camera position and intrinsics saved to '{output_file}' as '{position_name}'.")
    except Exception as e:
        print(f"Error saving file: {e}")

# Run the function
save_camera_position()

