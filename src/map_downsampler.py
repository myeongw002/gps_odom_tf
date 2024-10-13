import open3d as o3d
import os

# Get the original file path from the user
input_path = input('원본 PCD 파일의 경로를 입력하십시오: ').strip().replace("'", "")

# Check if the file exists
if not os.path.exists(input_path):
    print(f"Error: The file {input_path} does not exist.")
    exit(1)

# Generate the output file path by appending "_smaller" to the filename
file_dir, file_name = os.path.split(input_path)  # Separate the directory and file name
file_base, file_ext = os.path.splitext(file_name)  # Separate the file base and extension
output_path = os.path.join(file_dir, f"{file_base}_smaller{file_ext}")  # Create new filename with _smaller suffix

# Load the point cloud
pcd = o3d.io.read_point_cloud(input_path)

# Downsample the point cloud
voxel_size = 0.1  # Define the voxel size for downsampling
downsampled_pcd = pcd.voxel_down_sample(voxel_size)

# Save the downsampled point cloud
o3d.io.write_point_cloud(output_path, downsampled_pcd)

print(f"Downsampled point cloud saved to: {output_path}")
