import csv
import numpy as np
import matplotlib.pyplot as plt
import utm
import random
from sklearn.neighbors import NearestNeighbors

def convert_and_offset_gps_to_utm(input_csv):
    points = []
    with open(input_csv, 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            lat = float(row['.latitude'])
            lon = float(row['.longitude'])
            utm_x, utm_y, _, _ = utm.from_latlon(lat, lon)
            points.append([utm_x, utm_y])
    return np.array(points)

def read_odometry(highway_txt):
    points = []
    with open(highway_txt, 'r') as infile:
        for line in infile:
            x, y, _ = map(float, line.strip().split())
            points.append([x, y])
    return np.array(points)

def find_centroid(points):
    return np.mean(points, axis=0)

def sample_to_match_length(shorter, longer):
    if len(shorter) < len(longer):
        sampled_indices = random.sample(range(len(longer)), len(shorter))
        longer_sampled = longer[sampled_indices]
        return longer_sampled
    return longer

def calculate_rmse(points1, points2):
    nbrs = NearestNeighbors(n_neighbors=1).fit(points2)
    distances, indices = nbrs.kneighbors(points1)
    rmse = np.sqrt(np.mean(distances ** 2))
    return rmse

def rotate_points(points, angle, center):
    # Apply 2D rotation matrix around a given center
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return np.dot(points - center, rotation_matrix.T) + center

def find_optimal_rotation(gps_coords, odom_coords, gps_centroid):
    min_rmse = float('inf')
    best_angle = 0
    angles = np.arange(0,360,0.1)  # Test angles from 0 to 360 degrees

    for angle in angles:
        rotated_gps = rotate_points(gps_coords, angle, gps_centroid)
        rmse = calculate_rmse(rotated_gps, odom_coords)
        if rmse < min_rmse:
            min_rmse = rmse
            best_angle = angle

    return best_angle, min_rmse

def create_transformation_matrix_4x4(rotation_angle, translation_vector):
    # 2D rotation matrix extended to 3D (as 3x3 matrix)
    rotation_matrix = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
        [np.sin(rotation_angle), np.cos(rotation_angle), 0],
        [0, 0, 1]
    ])

    # Extend translation vector to 3D (add 0 for z-axis)
    translation_vector_3d = np.array([translation_vector[0], translation_vector[1], 0])

    # 4x4 transformation matrix
    transformation_matrix_4x4 = np.eye(4)
    transformation_matrix_4x4[:3, :3] = rotation_matrix  # Insert rotation
    transformation_matrix_4x4[:3, 3] = translation_vector_3d  # Insert translation
    
    return transformation_matrix_4x4

def save_transformation_matrix(matrix, file_path):
    np.savetxt(file_path, matrix, fmt='%f')
    print(f"Transformation matrix saved to {file_path}")

def plot_coordinates(gps_coords, odom_coords):
    plt.figure(figsize=(8, 8))
    plt.scatter(gps_coords[:, 0], gps_coords[:, 1], c='blue', label='Transformed GPS')
    plt.scatter(odom_coords[:, 0], odom_coords[:, 1], c='red', label='Odometry')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Transformed GPS and Odometry Coordinates')
    plt.grid(True)
    plt.show()

def main():
    random.seed()  # For reproducibility

    # Paths to your data files
    input_csv = '/home/teammiracle/ROS/catkin_ws/src/gps_odom_tf/path/gps_path/2024-10-19-10-28-57-ublox_gps-fix.csv'
    highway_txt = '/home/teammiracle/ROS/catkin_ws/src/gps_odom_tf/path/highway_path.txt'
    matrix_output_path = '/home/teammiracle/ROS/catkin_ws/src/gps_odom_tf/tf_matrix/transform_matrix.txt'

    # Read and preprocess data
    gps_coords = convert_and_offset_gps_to_utm(input_csv)
    odom_coords = read_odometry(highway_txt)

    # Find centroids
    gps_centroid = find_centroid(gps_coords)
    odom_centroid = find_centroid(odom_coords)

    # Translate GPS coordinates to align with odometry centroid
    gps_transformed = gps_coords - gps_centroid + odom_centroid
    
    # Match the length of both sets by random sampling from the longer one
    if len(gps_transformed) > len(odom_coords):
        gps_transformed = sample_to_match_length(odom_coords, gps_transformed)
    else:
        odom_coords = sample_to_match_length(gps_transformed, odom_coords)

    # Find the optimal rotation that minimizes RMSE
    best_angle, min_rmse = find_optimal_rotation(gps_transformed, odom_coords, odom_centroid)
    print(f"Best angle (radians): {best_angle}, Minimum RMSE: {min_rmse}")

    # Rotate the GPS coordinates with the best angle
    rotated_gps = rotate_points(gps_transformed, best_angle, odom_centroid)

    # Calculate translation vector (from gps_centroid to odom_centroid)
    translation_vector = odom_centroid - gps_centroid

    # Create 4x4 transformation matrix with rotation and translation
    transformation_matrix = create_transformation_matrix_4x4(best_angle, translation_vector)
    print(f"4x4 Transformation Matrix:\n{transformation_matrix}")

    # Save the transformation matrix to the specified path
    save_transformation_matrix(transformation_matrix, matrix_output_path)

    # Plot the transformed GPS coordinates and odometry coordinates
    plot_coordinates(rotated_gps, odom_coords)

    
if __name__ == "__main__":
    main()
