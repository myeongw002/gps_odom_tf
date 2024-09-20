#!/usr/bin/env python3

import rospy
import numpy as np
import tf.transformations as tf_trans

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point

from erp42_serial2.msg import ERP_SET, ERP_STATUS

class Stanley_Controller:
    def __init__(self):
        rospy.init_node('stanley_controller')
        self.orientation = 0.0
        self.speed = 0.0
        self.robot_position = Point()
        self.path = []
        self.slopes = []
        self.gps_path = "/home/team-miracle/ROS/catkin_ws/src/gps_odom_tf/path/path_sample.txt"
        self.max_speed = 3 

        # Subscribers
        rospy.Subscriber('/erp_state', ERP_STATUS, self.state_callback)
        rospy.Subscriber('/localization', Odometry, self.odom_callback)

        # Publisher
        self.set_state_pub = rospy.Publisher('/set_state', ERP_SET, queue_size=1)
        self.set_state = ERP_SET()

        # Load path and calculate slopes
        self.set_paths()
        self.calculate_path_slopes()

    def state_callback(self, data: ERP_STATUS):
        self.speed = data.status_speed

    def odom_callback(self, data: Odometry):
        # Update robot position
        self.robot_position = data.pose.pose.position

        # Get orientation from odometry
        orientation_q = data.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, yaw) = tf_trans.euler_from_quaternion(orientation_list)
        self.orientation = yaw

        # Perform Stanley control
        self.stanley_control()

    def set_paths(self):
        with open(self.gps_path, 'r') as path_file:
            for line in path_file:
                if line.strip() == "":
                    continue
                x_str, y_str, _ = line.strip().split()
                x = float(x_str)
                y = float(y_str)
                self.path.append([x, y])
        rospy.loginfo(f"Loaded {len(self.path)} path points.")

    def calculate_path_slopes(self):
        if len(self.path) < 2:
            rospy.logwarn("Not enough points to calculate slopes.")
            return

        for i in range(len(self.path) - 1):
            x1, y1 = self.path[i]
            x2, y2 = self.path[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            slope = np.arctan2(dy, dx)
            self.slopes.append(slope)
        rospy.loginfo(f"Calculated {len(self.slopes)} path slopes.")

    def find_closest_path_point(self):
        min_dist = float('inf')
        closest_index = 0
        robot_x = self.robot_position.x
        robot_y = self.robot_position.y
        for i, (x, y) in enumerate(self.path):
            dx = robot_x - x
            dy = robot_y - y
            dist = np.hypot(dx, dy)
            if dist < min_dist:
                min_dist = dist
                closest_index = i
        return min_dist, closest_index

    def stanley_control(self):
        # Ensure path and slopes are available
        if not self.path or not self.slopes:
            rospy.logwarn("Path or slopes are not initialized.")
            return

        # Find the closest path point
        min_dist, closest_index = self.find_closest_path_point()

        # Ensure the closest index is within bounds for slopes
        if closest_index >= len(self.slopes):
            rospy.logwarn(f"Closest index {closest_index} is out of bounds for slopes.")
            closest_index = len(self.slopes) - 1  # Use the last valid index
            if closest_index < 0:
                rospy.logerr("No valid slope available for Stanley control.")
                return

        # Calculate heading error
        path_heading = self.slopes[closest_index]
        heading_error = path_heading - self.orientation
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))  # Normalize

        # Calculate cross track error
        robot_x = self.robot_position.x
        robot_y = self.robot_position.y
        path_x = self.path[closest_index][0]
        path_y = self.path[closest_index][1]

        # Calculate cross track error (e_ct)
        dx = robot_x - path_x
        dy = robot_y - path_y
        e_ct = dy * np.cos(path_heading) - dx * np.sin(path_heading)

        # Stanley control law
        k = 0.5  # Control gain
        v = self.speed if self.speed != 0 else 0.1  # Avoid division by zero
        delta = heading_error + np.arctan2(k * e_ct, v)
        deg = int(np.rad2deg(delta) * 71)

        # Publish steering angle
        self.set_state.set_steer = np.clip(deg, -2000, 2000)  # Convert to degrees
        self.set_state.set_speed = self.max_speed
        self.set_state_pub.publish(self.set_state)
        
        rospy.loginfo(f"Steering angle set to {np.rad2deg(delta):.2f} degrees")

if __name__ == "__main__":
    controller = Stanley_Controller()
    rospy.spin()
