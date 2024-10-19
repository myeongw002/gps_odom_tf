#!/usr/bin/env python
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from erp42_serial2.msg import ERP_SET, ERP_STATUS

class PurePursuitController:
    def __init__(self):

        self.path_file = rospy.get_param("~path", '/home/team-miracle/ROS/catkin_ws/src/gps_odom_tf/path/path.txt')
        self.Ld = rospy.get_param("~lookahead_distance", 2.0)  # Lookahead distance
        self.L = rospy.get_param("~L", 1.04)  # Vehicle's wheelbase
        self.max_steering = rospy.get_param("~max_steering", 28.5)
        self.max_steering = np.radians(self.max_steering)

        # Load path data (ref_xs, ref_ys)
        self.ref_xs, self.ref_ys = self.load_path(self.path_file)

    def load_path(self, file_path):
        data = np.loadtxt(file_path, delimiter=' ')
        ref_xs = data[:, 0]
        ref_ys = data[:, 1]
        return ref_xs, ref_ys

    def compute_steering(self, rear_x, rear_y, yaw, v, ld):
        # Find the closest point on the path
        min_dist = float('inf')
        min_index = 0
        n_points = len(self.ref_xs)

        for i in range(n_points):
            dx = rear_x - self.ref_xs[i]
            dy = rear_y - self.ref_ys[i]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                min_index = i

        # Lookahead point calculation
        lookahead_point = None
        for i in range(min_index, n_points):
            dx = self.ref_xs[i] - rear_x
            dy = self.ref_ys[i] - rear_y
            dist = np.sqrt(dx**2 + dy**2)
            if dist >= ld:
                lookahead_point = (self.ref_xs[i], self.ref_ys[i])
                break

        if lookahead_point is None:
            lookahead_point = (self.ref_xs[-1], self.ref_ys[-1])

        lx, ly = lookahead_point

        # Transform the lookahead point to the vehicle's coordinate system
        dx = lx - rear_x
        dy = ly - rear_y
        local_x = np.cos(yaw) * dx + np.sin(yaw) * dy
        local_y = np.sin(yaw) * dx - np.cos(yaw) * dy

        # Calculate steering angle using Pure Pursuit geometry
        steering_angle = np.arctan2(2 * self.L * local_y, self.Ld**2)
        steering_angle = np.clip(steering_angle, -self.max_steering, self.max_steering)

        rospy.loginfo(f"Lookahead Point: ({lx}, {ly})")
        rospy.loginfo(f"Steering Angle (radians): {steering_angle}, (degrees): {np.degrees(steering_angle)}")

        return steering_angle


class PurePursuitControllerNode:
    def __init__(self):
        # ROS node initialization
        rospy.init_node('pure_pursuit_controller')

        # Create a PurePursuitController object
        self.controller = PurePursuitController()

        # Initial state
        self.front_x = 0.0
        self.front_y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.rear_x = 0.0
        self.rear_y = 0.0
        self.LD = 0
        self.max_speed = rospy.get_param("~max_speed", 18)

        # Publisher for control command
        self.steer_pub = rospy.Publisher('/erp42_set', ERP_SET, queue_size=1)

        # Subscriber for odometry
        rospy.Subscriber('/localization', Odometry, self.odometry_callback)
        rospy.Subscriber('/erp42_state', ERP_STATUS, self.status_callback)

        # Control rate
        self.rate = rospy.Rate(20)  # 20 Hz

    def odometry_callback(self, msg: Odometry):
        # Update vehicle state from odometry message
        self.front_x = msg.pose.pose.position.x
        self.front_y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

        # Compute rear axle position
        self.rear_x = self.front_x - self.controller.L * np.cos(self.yaw)
        self.rear_y = self.front_y - self.controller.L * np.sin(self.yaw)

    def status_callback(self, msg: ERP_STATUS):
        self.v = msg.status_speed / 10
        self.LD = self.v * 2

    def main(self):
        while not rospy.is_shutdown():
            # Compute the steering command using Pure Pursuit control
            steer = self.controller.compute_steering(self.rear_x, self.rear_y, self.yaw, self.v, self.LD)
            steer = -1 * np.rad2deg(steer)
            rospy.loginfo(f"Steering angle: {steer}")

            set_msg = ERP_SET()
            set_steer = int(steer * 71)
            set_msg.set_steer = np.clip(set_steer, -2000, 2000)
            set_msg.set_speed = self.max_speed
            self.steer_pub.publish(set_msg)

            self.rate.sleep()


if __name__ == '__main__':
    try:
        pp_controller = PurePursuitControllerNode()
        pp_controller.main()

    except rospy.ROSInterruptException:
        pass

