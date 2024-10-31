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
    # Convert angle to radians
    angle_rad = np.radians(angle)
    # Apply 2D rotation matrix around a given center
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    return np.dot(points - center, rotation_matrix.T) + center

def find_optimal_rotation(gps_coords, odom_coords, gps_centroid, step=0.01):
    rmse_values = []
    angles = np.arange(0, 360, step)  # Reduced resolution for plotting purposes

    for angle in angles:
        rotated_gps = rotate_points(gps_coords, angle, gps_centroid)
        rmse = calculate_rmse(rotated_gps, odom_coords)
        rmse_values.append((angle, rmse))

    angles, rmses = zip(*rmse_values)
   
    # Find the angle with the minimum RMSE
    min_rmse = min(rmses)
    best_angle = angles[rmses.index(min_rmse)]

    print("RMSE values calculated")
    print(f"Best angle (degrees): {best_angle}, Minimum RMSE: {min_rmse}")
    # Plot RMSE against angles
    
    plt.plot(angles, rmses)
    plt.xlabel('Rotation Angle (degrees)')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Rotation Angle')
    plt.grid(True)
    plt.show()


    return best_angle, min_rmse

def save_to_file(translation_vector, gps_centroid, best_angle, file_path):
    with open(file_path, 'w') as f:
        # Save translation vector on the first line
        f.write("Translation Vector: " + ' '.join(map(str, translation_vector)) + "\n")
        # Save GPS centroid on the second line
        f.write("GPS Centroid: " + ' '.join(map(str, gps_centroid)) + "\n")
        # Save best angle on the third line
        f.write("Best Angle: " + str(best_angle) + "\n")
    print(f"Data saved to {file_path}")

def plot_coordinates(gps_coords, odom_coords, title="Transformed GPS and Odometry Coordinates"):
    plt.figure(figsize=(8, 8))
    plt.scatter(gps_coords[:, 0], gps_coords[:, 1], c='blue', label='Transformed GPS')
    plt.scatter(odom_coords[:, 0], odom_coords[:, 1], c='red', label='Odometry')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()

def main():
    random.seed()  # For reproducibility

    # Paths to your data files
    input_csv = '/home/teammiracle/ROS/catkin_ws/src/gps_odom_tf/path/gps_path/2024-10-19-10-28-57-ublox_gps-fix.csv'
    highway_txt = '/home/teammiracle/ROS/catkin_ws/src/gps_odom_tf/path/highway_path5.txt'
    output_file_path = '/home/teammiracle/ROS/catkin_ws/src/gps_odom_tf/tf_data/transform_data4.txt'

    # Read and preprocess data
    gps_coords = convert_and_offset_gps_to_utm(input_csv)
    odom_coords = read_odometry(highway_txt)

    # Find centroids
    gps_centroid = find_centroid(gps_coords)
    odom_centroid = find_centroid(odom_coords)
    print(f"GPS Centroid: {gps_centroid}, Odometry Centroid: {odom_centroid}")
    # Calculate translation vector
    translation_vector = odom_centroid - gps_centroid
    print(f"Translation Vector: {translation_vector}")
    # Translate GPS coordinates
    gps_translated = gps_coords + translation_vector

    # Match the length of both sets by random sampling from the longer one
    if len(gps_translated) > len(odom_coords):
        gps_translated = sample_to_match_length(odom_coords, gps_translated)
    else:
        odom_coords = sample_to_match_length(gps_translated, odom_coords)

    # Find the optimal rotation that minimizes RMSE
    gps_centroid_translated = find_centroid(gps_translated)
    print("Calculating roatation")
    best_angle, min_rmse = find_optimal_rotation(gps_translated, odom_coords, gps_centroid_translated)
    print(f"Best angle (degrees): {best_angle}, Minimum RMSE: {min_rmse}")

    # Rotate the GPS coordinates with the best angle
    rotated_gps = rotate_points(gps_translated, best_angle, gps_centroid_translated)

    # Plot the transformed GPS coordinates and odometry coordinates
    plot_coordinates(rotated_gps, odom_coords, "Rotated GPS and Odometry Coordinates")

    # Save the translation vector, GPS centroid, and best angle to one file
    save_to_file(translation_vector, gps_centroid_translated, best_angle, output_file_path)
    print("Transformation completed successfully!")
    print("Translation Vector: ", translation_vector)
    print("Rotation Angle: ", best_angle)

if __name__ == "__main__":
    main()
