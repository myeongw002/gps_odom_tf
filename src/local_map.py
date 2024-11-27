#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import tf
from sklearn.cluster import DBSCAN
import heapq
from scipy.interpolate import splprep, splev
from visualization_msgs.msg import Marker

def interpolate_path(path, smooth_factor=2.0, num_points=100):
    """
    Perform spline interpolation on a given path.
    
    Parameters:
    - path (np.ndarray): Array of shape (N, 2) representing the path points (x, y).
    - smooth_factor (float): Smoothing factor for the spline.
    - num_points (int): Number of points in the interpolated path.
    
    Returns:
    - np.ndarray: Interpolated path with shape (num_points, 2).
    """
    if len(path) >= 4:  # Ensure there are enough points for cubic spline interpolation
        # Separate x and y coordinates
        x, y = path[:, 0], path[:, 1]
        
        # Perform spline interpolation
        tck, u = splprep([x, y], s=smooth_factor)
        u_fine = np.linspace(0, 1, num_points)
        x_smooth, y_smooth = splev(u_fine, tck)
        
        # Return interpolated path as a (num_points, 2) array
        return np.vstack((x_smooth, y_smooth)).T
    else:
        # Return original path if there are not enough points
        return path


class Node:
    def __init__(self, position, g=0, h=0, parent=None):
        self.position = position
        self.g = g  # Cost from start to current node
        self.h = h  # Heuristic - estimated cost from current node to goal
        self.f = g + h  # Total cost
        self.parent = parent  # Parent node for path tracing

    def __lt__(self, other):
        return self.f < other.f  # For priority queue comparison


class CloudProcessor:
    def __init__(self, path_file):
        rospy.init_node('cloud_processor', anonymous=True)
        
        # Initialize subscribers and plot setup
        self.pointcloud_subscriber = rospy.Subscriber('/velodyne_points', PointCloud2, self.callback)
        self.odometry_subscriber = rospy.Subscriber('/localization', Odometry, self.odometry_callback)
        self.block_pub = rospy.Publisher('/path_blocked', Bool, queue_size=1)
        self.path_marker_pub = rospy.Publisher('/global_a_star_path_marker', Marker, queue_size=1)

        self.latest_points = None  # To store the latest point cloud data

        # ROI and other parameters
        self.x_max = 20.0
        self.x_min = 0.0
        self.y_max = 5.0
        self.y_min = -5.0
        self.roi_y_min = -2.0  # Set ROI y-limits
        self.roi_y_max = 2.0
        self.vehicle_width = 1.0  # Vehicle width in meters
        self.repulsive_radius = 2.0
        self.step_size = 0.5  # Step size for moving towards the goal in local planning

        # Load path
        self.global_path = self.load_path(path_file)
        self.current_pose = None  # Placeholder for current position and yaw
        self.global_a_star_path = None
        self.setup_plot()
    
    def load_path(self, path_file):
        """Load path coordinates from a file, ignoring the z coordinate."""
        path = np.loadtxt(path_file, delimiter=' ')
        path = path[:, :2]  # Select only x, y columns and ignore z
        return path

    def setup_plot(self):
        """Initialize the plot for real-time 2D visualization."""
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        
        # Axes and legend setup
        self.ax.arrow(0, 0, 1, 0, color='green', head_width=0.3, head_length=0.3, linewidth=1.5, label="X-direction (1,0)")
        self.ax.arrow(0, 0, 0, 1, color='red', head_width=0.3, head_length=0.3, linewidth=1.5, label="Y-direction (0,1)")
        self.ax.invert_xaxis()
        plt.legend()
        plt.ion()
        plt.show()

    def is_within_bounds(self, node, clusters, vehicle_width):
        """Check if the node position is safe considering vehicle width."""
        for cluster_points in clusters.values():
            cluster_points = np.array(cluster_points)  # Convert to numpy array
            distances = np.linalg.norm(cluster_points - np.array(node.position), axis=1)
            if np.any(distances < vehicle_width ):
                return False
        return True


    def generate_path_with_astar(self, start, goal, clusters):
        """Generate a path from start to goal using A* algorithm considering vehicle width."""
        open_set = []
        start_node = Node(start, g=0, h=np.linalg.norm(np.array(start) - np.array(goal)))
        heapq.heappush(open_set, start_node)
        closed_set = set()

        while open_set:
            current_node = heapq.heappop(open_set)
            
            # Check if the goal is reached
            if np.linalg.norm(np.array(current_node.position) - np.array(goal)) < self.step_size:
                path = []
                while current_node:
                    path.append(current_node.position)
                    current_node = current_node.parent
                return path[::-1]  # Return reversed path

            closed_set.add(current_node.position)

            # Generate neighbors (4-way or 8-way)
            neighbors = [
                (current_node.position[0] + dx, current_node.position[1] + dy)
                for dx, dy in [(-self.step_size, 0), (self.step_size, 0), (0, -self.step_size), (0, self.step_size)]
            ]

            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue
                
                # Check for collision with clusters
                neighbor_node = Node(neighbor, g=current_node.g + self.step_size, 
                                     h=np.linalg.norm(np.array(neighbor) - np.array(goal)), 
                                     parent=current_node)
                
                if not self.is_within_bounds(neighbor_node, clusters, self.vehicle_width):
                    continue  # Skip nodes within collision range

                # Add the neighbor to open set if it passes the checks
                heapq.heappush(open_set, neighbor_node)
        
        return None  # No path found

    def odometry_callback(self, msg):
        """Odometry callback to update the current pose."""
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        # Convert orientation from quaternion to Euler angles to get yaw
        _, _, yaw = tf.transformations.euler_from_quaternion([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ])
        self.current_pose = (position.x, position.y, yaw)

    def transform_to_local_frame(self, global_path):
        """Transform the global path to the local frame of the robot."""
        if self.current_pose is None:
            return None
        x_r, y_r, yaw = self.current_pose

        transform = np.array([
            [np.cos(-yaw), -np.sin(-yaw), -x_r * np.cos(-yaw) + y_r * np.sin(-yaw)],
            [np.sin(-yaw),  np.cos(-yaw), -x_r * np.sin(-yaw) - y_r * np.cos(-yaw)],
            [0, 0, 1]
        ])
        
        global_path_hom = np.hstack((global_path, np.ones((global_path.shape[0], 1))))
        local_path = transform @ global_path_hom.T
        return local_path[:2].T

    def process_cloud(self, points):
        """Filter points, apply voxel grid, and perform clustering within the ROI."""
        # 필터링, 다운샘플링 등 기존 코드 유지
        points = points[(points[:, 0] >= self.x_min) & (points[:, 0] <= self.x_max)]
        points = points[(points[:, 1] >= self.roi_y_min) & (points[:, 1] <= self.roi_y_max)]

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud = cloud.voxel_down_sample(voxel_size=0.2)
        points = np.asarray(cloud.points)
        points = points[(points[:, 2] >= -1.0) & (points[:, 2] <= 0)]
        points_2d = points[:, :2]

        if points_2d.shape[0] == 0:
            return {}  # 포인트가 없을 경우 빈 딕셔너리 반환

        # DBSCAN을 통한 클러스터링
        dbscan = DBSCAN(eps=0.5, min_samples=3)
        labels = dbscan.fit_predict(points_2d)

        # 클러스터 개수 계산
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print("Number of clusters:", n_clusters)

        # 클러스터별 포인트 그룹화
        clusters = {}
        for point, label in zip(points_2d, labels):
            if label == -1:
                continue  # 잡음 포인트 무시
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(point)
        
        return clusters

    def callback(self, msg):
        """Callback function to handle incoming PointCloud2 data."""
        self.latest_points = np.array([[p[0], p[1], p[2]] for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)])
        

    def calculate_min_distance_to_clusters(self, clusters):
        """
        Calculate the minimum distance and angle from the robot's current position to the nearest cluster.
        
        Parameters:
        - clusters (dict): Dictionary of clusters with each key as a label and value as a list of points in that cluster.
        
        Returns:
        - (float, float): The minimum distance to the nearest cluster and the angle in degrees between the robot's heading and the direction to the cluster.
        """
        min_distance = float('inf')
        min_angle = 0.0

        # Ensure the current pose is available
        if self.current_pose is None:
            return min_distance, min_angle

        x_r, y_r, yaw = self.current_pose  # Extract robot's position and orientation (yaw)

        for cluster_points in clusters.values():
            # Calculate the center of the cluster
            cluster_center = np.mean(cluster_points, axis=0)
            
            # Calculate distance to cluster center
            dx = cluster_center[0] - x_r
            dy = cluster_center[1] - y_r
            distance_to_cluster = np.sqrt(dx**2 + dy**2)

            # Calculate the angle to the cluster center relative to the robot's yaw
            angle_to_cluster = np.degrees(np.arctan2(dy, dx)) - np.degrees(yaw)
            
            # Normalize the angle to [-180, 180] degrees
            angle_to_cluster = (angle_to_cluster + 180) % 360 - 180

            # Check if this cluster is the closest
            if distance_to_cluster < min_distance:
                min_distance = distance_to_cluster
                min_angle = angle_to_cluster

        return min_distance, min_angle



    def find_nearest_index(self, path, position):
        """
        Find the index of the nearest point on a given path to the current position.

        Parameters:
        - path (np.ndarray): Array of shape (N, 2) representing path points (x, y).
        - position (tuple): The current position (x, y) of the robot.

        Returns:
        - int: Index of the nearest point on the path to the robot's position.
        """
        min_dist = float('inf')
        nearest_index = 0
        x_r, y_r = position  # Extract robot's position

        for i, (x_p, y_p) in enumerate(path):
            dist = np.sqrt((x_r - x_p) ** 2 + (y_r - y_p) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest_index = i

        return nearest_index

    def update_plot(self):
        """Update the matplotlib plot with the latest point cloud data and local path."""
        self.ax.clear()  # Clear previous frame data

        # Plot static components
        self.ax.set_ylim(self.x_min, self.x_max)
        self.ax.set_xlim(self.y_min, self.y_max)
        self.ax.arrow(0, 0, 1, 0, color='green', head_width=0.3, head_length=0.3, linewidth=1.5, label="X-direction (1,0)")
        self.ax.arrow(0, 0, 0, 1, color='red', head_width=0.3, head_length=0.3, linewidth=1.5, label="Y-direction (0,1)")
        self.ax.invert_xaxis()

        if self.latest_points is not None:
            points_2d_clusters = self.process_cloud(self.latest_points)
            n_clusters = len(points_2d_clusters)

            # Initialize overall bounding box limits for all clusters
            all_x_min, all_y_min = float('inf'), float('inf')
            all_x_max, all_y_max = float('-inf'), float('-inf')

            # Plot each cluster with a unique color and update global bounding box limits
            for i, (label, cluster_points) in enumerate(points_2d_clusters.items()):
                cluster_points = np.array(cluster_points)
                color = plt.cm.get_cmap('tab10', 10)(i % 10)  # Get color in sequence
                self.ax.scatter(cluster_points[:, 1], cluster_points[:, 0], color=color, s=5)

                # Update global bounding box to include all clusters
                x_min, y_min = cluster_points.min(axis=0)
                x_max, y_max = cluster_points.max(axis=0)
                all_x_min = min(all_x_min, x_min)
                all_y_min = min(all_y_min, y_min)
                all_x_max = max(all_x_max, x_max)
                all_y_max = max(all_y_max, y_max)

            # Plot the global bounding box around all clusters
            bbox_x = [all_y_min, all_y_max, all_y_max, all_y_min, all_y_min]
            bbox_y = [all_x_min, all_x_min, all_x_max, all_x_max, all_x_min]
            self.ax.plot(bbox_x, bbox_y, color='black', linestyle='--', label="Combined Bounding Box")

            # Plot path points within the combined bounding box
            if self.global_path is not None:
                local_path = self.transform_to_local_frame(self.global_path)
                self.ax.scatter(local_path[:, 1], local_path[:, 0], color='blue', s=10, label="Global Path")
                path_in_bbox = local_path[
                    (local_path[:, 0] >= all_x_min-3) & (local_path[:, 0] <= all_x_max+3) &
                    (local_path[:, 1] >= all_y_min) & (local_path[:, 1] <= all_y_max)
                ]
                
                if path_in_bbox.size > 0:
                    start = tuple(path_in_bbox[0])
                    goal = tuple(path_in_bbox[-1])
                    

                    if self.global_a_star_path is None and n_clusters >= 2:
                    # A* 경로 생성
            
                        a_star_path = self.generate_path_with_astar(start, goal, points_2d_clusters)
                        if a_star_path:
                            a_star_path = np.array(a_star_path)
                            self.ax.plot(a_star_path[:, 1], a_star_path[:, 0], color='purple', linewidth=1.5, label="A* Path")

                            # 오도메트리를 고려하여 로컬 A* 경로를 글로벌 좌표로 변환
                            global_a_star_path = []
                            if self.current_pose is not None:
                                x_r, y_r, yaw = self.current_pose
                                for point in a_star_path:
                                    # 로컬 좌표 -> 글로벌 좌표 변환
                                    x_local, y_local = point
                                    x_global = x_local * np.cos(yaw) - y_local * np.sin(yaw) + x_r
                                    y_global = x_local * np.sin(yaw) + y_local * np.cos(yaw) + y_r
                                    global_a_star_path.append([x_global, y_global])

                                # numpy 배열로 변환하여 global_path 업데이트
                                global_a_star_path = np.array(global_a_star_path)
                                self.global_a_star_path = interpolate_path(global_a_star_path)
                                # Create a Marker message for the global A* path
                                marker = Marker()
                                marker.header.frame_id = "map"  # Or use the appropriate frame
                                marker.header.stamp = rospy.Time.now()
                                marker.ns = "global_a_star_path"
                                marker.id = 0
                                marker.type = Marker.LINE_STRIP
                                marker.action = Marker.ADD
                                marker.pose.orientation.w = 1.0

                                # Set marker properties
                                marker.scale.x = 0.1  # Line width
                                marker.color.r = 1.0
                                marker.color.g = 0.0
                                marker.color.b = 1.0
                                marker.color.a = 1.0

                                # Add points to the marker from global_a_star_path
                                marker.points = []
                                for x, y in self.global_a_star_path:
                                    point = Point()
                                    point.x = x
                                    point.y = y
                                    point.z = 0  # Set Z if needed
                                    marker.points.append(point)

                                # Publish the marker
                                self.path_marker_pub.publish(marker)
                            else:
                                print("Odometry data not available for transformation.")

                    elif n_clusters >= 1:
                        # Calculate distance to the nearest cluster
                        min_distance, min_angle = self.calculate_min_distance_to_clusters(points_2d_clusters)
                        if min_distance <= 5 and abs(min_angle) <= 10:
                            self.block_pub.publish(Bool(data=True))
                            rospy.loginfo("Path blocked by the nearest cluster.")
                        rospy.loginfo(f"Distance to the nearest cluster: {min_distance}")
                    
                    else:
                        self.block_pub.publish(Bool(data=False))
                        rospy.loginfo("Path is clear.")
                    
                    if self.global_a_star_path is not None:
                        rospy.loginfo(f"Global A* Path: {self.global_a_star_path}")
                        
            self.ax.legend()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()




    def spin_with_while(self):
        """Main loop with while to continuously process and plot data."""
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.update_plot()
            rate.sleep()

def main():
    path_file = "/home/teammiracle/ROS/catkin_ws/src/gps_odom_tf/path/highway_path5.txt"
    processor = CloudProcessor(path_file)
    processor.spin_with_while()

if __name__ == '__main__':
    main()
