import csv
import utm
import numpy as np
from scipy.linalg import svd
import open3d as o3d

# Function to read and convert GPS data (in lat/lon) to UTM, offset from the first point
def convert_and_offset_gps_to_utm(input_csv):
    points = []
    with open(input_csv, 'r') as infile:
        reader = csv.DictReader(infile)
        
        # Read the first row to get the initial offset
        first_row = next(reader)
        initial_lat = float(first_row['.latitude'])
        initial_lon = float(first_row['.longitude'])
        initial_utm_x, initial_utm_y, zone_number, zone_letter = utm.from_latlon(initial_lat, initial_lon)
        
        # Process the first row
        points.append([0, 0, 0])  # First point at (0,0,0)
        
        # Process the remaining rows
        for row in reader:
            lat = float(row['.latitude'])
            lon = float(row['.longitude'])
            utm_x, utm_y, _, _ = utm.from_latlon(lat, lon)
            points.append([utm_x, utm_y, 0])  # Z-coordinate is set to 0
    return points, initial_utm_x, initial_utm_y  # Return points and initial UTM offset

# Function to read odometry data from a text file
def read_odometry(highway_txt):
    points = []
    with open(highway_txt, 'r') as infile:
        for line in infile:
            x, y, _ = map(float, line.strip().split())
            points.append([x, y, 0])  # Z-coordinate is set to 0
    return points

# Function to calculate transformation matrix that converts UTM to Odometry
def apply_rigid_transformation(utm_points, odom_points):
    # Ensure that both point sets have the same number of points
    min_points = min(len(utm_points), len(odom_points))
    utm_points = np.array(utm_points[:min_points])
    odom_points = np.array(odom_points[:min_points])

    # Compute centroids (mean of the points)
    utm_centroid = np.mean(utm_points, axis=0)
    odom_centroid = np.mean(odom_points, axis=0)

    # Subtract centroids from points to center them
    centered_utm = utm_points - utm_centroid
    centered_odom = odom_points - odom_centroid

    # Calculate rotation matrix using Singular Value Decomposition (SVD)
    H = centered_utm.T @ centered_odom
    U, S, Vt = svd(H)
    rotation_matrix = U @ Vt

    # Ensure proper orientation of the rotation matrix
    if np.linalg.det(rotation_matrix) < 0:
        Vt[-1, :] *= -1
        rotation_matrix = U @ Vt

    # Calculate translation vector
    translation = odom_centroid - rotation_matrix @ utm_centroid

    # Create 4x4 homogeneous transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix  # Set rotation part
    transform_matrix[:3, 3] = translation       # Set translation part

    print(f"Translation vector: {translation}")
    print(f"Rotation matrix:\n{rotation_matrix}")
    
    return transform_matrix

# Function to create 3D transform matrix with GPS offset
def create_3d_transform_matrix(rotation_matrix, translation, gps_offset):
    # 확장된 3D 회전 행렬 (z축을 기준으로 회전)
    R_3x3 = np.eye(3)  # 기본적으로 3x3 단위 행렬 생성
    R_3x3[:2, :2] = rotation_matrix  # 2D 회전 행렬을 상위 2x2 부분에 할당
    
    # 평행 이동 벡터 확장 (3차원, z축은 0으로 설정) + GPS offset 추가
    t_3x1 = np.array([translation[0] + gps_offset[0], translation[1] + gps_offset[1], 0])
    
    # 최종 4x4 동차 좌표 변환 행렬
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R_3x3  # 회전 부분
    transform_matrix[:3, 3] = t_3x1  # 평행 이동 부분
    
    return transform_matrix

# Function to save the 4x4 transformation matrix to a file
def save_transform_matrix_to_file(matrix, file_path):
    np.savetxt(file_path, matrix, fmt='%.6f')
    print(f"Transformation matrix saved to {file_path}")

# Function to create Open3D point cloud
def create_point_cloud(points, color):
    # Convert points to numpy array and create point cloud
    points_np = np.array(points)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_np[:, :3])  # Ignore the homogeneous coordinate
    point_cloud.paint_uniform_color(color)  # Set color for the point cloud
    return point_cloud

# Function to visualize UTM, transformed, and odometry points using Open3D
def visualize_utm_odometry(transformed_points, odom_points):
    # Create Open3D point clouds
    utm_transformed_pc = create_point_cloud(transformed_points, [1, 0, 0])  # Red for transformed UTM points
    odom_pc = create_point_cloud(odom_points, [0, 1, 0])  # Green for odometry points

    # Visualize both point clouds together
    o3d.visualization.draw_geometries([utm_transformed_pc, odom_pc])


if __name__ == "__main__":
    input_csv = '/home/teammiracle/ROS/catkin_ws/src/gps_odom_tf/path/gps_path/2024-10-19-10-28-57-ublox_gps-fix.csv'
    highway_txt = '/home/teammiracle/ROS/catkin_ws/src/gps_odom_tf/path/highway_path.txt'  # Update with the correct path

    utm_points, initial_utm_x, initial_utm_y = convert_and_offset_gps_to_utm(input_csv)
    odom_points = read_odometry(highway_txt)

    # Convert UTM points list to numpy array for indexing
    utm_points = np.array(utm_points)

    # Calculate transformation matrix from UTM to odometry
    transform_matrix = apply_rigid_transformation(utm_points, odom_points)

    # GPS offset (첫 번째 UTM 좌표) 추가
    gps_offset = (0,0)

    # 3D 변환 행렬 생성
    transform_matrix_3d = create_3d_transform_matrix(transform_matrix[:2, :2], transform_matrix[:3, 3], gps_offset)
    
    # Apply transformation to UTM points
    utm_homogeneous = np.hstack((utm_points[:, :3], np.ones((utm_points.shape[0], 1))))  # Add homogeneous coordinate
    transformed_points = np.dot(transform_matrix_3d, utm_homogeneous.T).T
    print("Transformed UTM points:", transformed_points)

    # Visualize the transformed UTM and odometry points
    visualize_utm_odometry(transformed_points, odom_points)

    # 저장할 경로 설정
    tf_matrix_path = '/home/teammiracle/ROS/catkin_ws/src/gps_odom_tf/tf_matrix.txt'

    # 3D 변환 행렬 텍스트 파일로 저장
    save_transform_matrix_to_file(transform_matrix_3d, tf_matrix_path)

    print("3D Transformation Matrix with GPS Offset:")
    print(transform_matrix_3d)
