#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from tf.transformations import euler_from_quaternion
from erp42_serial2.msg import ERP_SET, ERP_STATUS
from std_msgs.msg import Bool

class StanleyController:
    def __init__(self):
        self.path_file = rospy.get_param("~path", '/home/team-miracle/ROS/catkin_ws/src/gps_odom_tf/path/path.txt')
        self.K = rospy.get_param("~K", 0.1)
        self.L = rospy.get_param("~L", 1.04)
        self.max_steering = float(rospy.get_param("~max_steering", 28.5))
        self.max_steering = np.radians(self.max_steering)
        # Load initial path data
        self.ref_xs, self.ref_ys, self.ref_yaws = self.load_path(self.path_file)
        
        # Set placeholders for extracted path
        self.extracted_xs, self.extracted_ys, self.extracted_yaws = None, None, None
        self.use_extracted_path = False  # Flag to switch between paths
        self.avoid_obstacle_done = False
        
    def load_path(self, file_path):
        # Load path from file and calculate yaws
        data = np.loadtxt(file_path, delimiter=' ')
        ref_xs = data[:, 0]
        ref_ys = data[:, 1]
        ref_yaws = np.arctan2(np.gradient(ref_ys), np.gradient(ref_xs))
        return ref_xs, ref_ys, ref_yaws

    def set_extracted_path(self, xs, ys):
        # Set extracted path points and calculate yaws
        self.extracted_xs = np.array(xs)
        self.extracted_ys = np.array(ys)
        self.extracted_yaws = np.arctan2(np.gradient(self.extracted_ys), np.gradient(self.extracted_xs))
        self.use_extracted_path = True  # Use extracted path

    def reset_to_original_path(self):
        # Switch back to original path
        self.use_extracted_path = False
        self.avoid_obstacle_done =  True
        rospy.loginfo("Switched back to the original path.")

    def get_current_path(self):
        # Return active path based on the flag
        if self.use_extracted_path and self.extracted_xs is not None:
            rospy.loginfo("Using extracted path")
            return self.extracted_xs, self.extracted_ys, self.extracted_yaws
        else:
            return self.ref_xs, self.ref_ys, self.ref_yaws

    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    def compute_steering(self, x, y, yaw, v):
        # Get the active path
        ref_xs, ref_ys, ref_yaws = self.get_current_path()

        # Find closest point on the path
        min_dist = float('inf')
        min_index = 0
        n_points = len(ref_xs)
        front_x = x + self.L * np.cos(yaw)
        front_y = y + self.L * np.sin(yaw)

        for i in range(n_points):
            dx = front_x - ref_xs[i]
            dy = front_y - ref_ys[i]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                min_index = i

        # Check if the extracted path has ended
        if self.use_extracted_path and min_index == n_points - 2 and min_dist < 0.5:  # Threshold for "end of path"
            self.reset_to_original_path()

        # Calculate cross-track error and heading error
        ref_x = ref_xs[min_index]
        ref_y = ref_ys[min_index]
        ref_yaw = ref_yaws[min_index]
        dx = ref_x - front_x
        dy = ref_y - front_y
        perp_vec = [np.cos(ref_yaw + np.pi/2), np.sin(ref_yaw + np.pi/2)]
        cte = np.dot([dx, dy], perp_vec)
        yaw_error = self.normalize_angle(ref_yaw - yaw)
        cte_term = np.arctan2(self.K * cte, v)

        steer = yaw_error + cte_term
        steer = np.clip(steer, -self.max_steering, self.max_steering)
        rospy.loginfo(f"Goal Point: {min_index}/{n_points}, CTE: {cte}, Yaw Error: {yaw_error}")
        rospy.loginfo(f"Steering Angle (radians): {steer}, (degrees): {np.degrees(steer)}")
        return steer


class StanleyControllerNode:
    def __init__(self):
        rospy.init_node('stanley_controller')
        
        # Create StanleyController object
        self.controller = StanleyController()
        
        # Publisher for control command
        self.steer_pub = rospy.Publisher('/erp42_set', ERP_SET, queue_size=1)
        
        # Subscribers
        rospy.Subscriber('/global_a_star_path_marker', Marker, self.extracted_path_callback)
        rospy.Subscriber('/localization', Odometry, self.odometry_callback)
        rospy.Subscriber('/erp42_status', ERP_STATUS, self.status_callback)
        rospy.Subscriber("path_blocked", Bool, self.blocked_callback)
        # Initial state variables
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.max_speed = rospy.get_param("~max_speed", 18)
        self.blocked=False
        # Control rate
        self.rate = rospy.Rate(20)  # 20 Hz

    def extracted_path_callback(self, msg):
        # Handle marker message to update the extracted path
        if msg.type == Marker.LINE_STRIP:
            xs, ys = [], []
            for point in msg.points:
                xs.append(point.x)
                ys.append(point.y)
            self.controller.set_extracted_path(xs, ys)
            rospy.loginfo("Extracted path set as the active path.")

    def odometry_callback(self, msg: Odometry):
        # Update vehicle state from odometry message
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

    def status_callback(self, msg: ERP_STATUS):
        # Update vehicle speed from ERP status message
        self.v = msg.status_speed / 10  # Scale as needed

    def blocked_callback(self, msg):
        self.blocked = msg.data

    def main(self):
        while not rospy.is_shutdown():
            # Compute steering command using Stanley controller
            
            if self.controller.avoid_obstacle_done == True and self.blocked:
                steer = 0
                speed = 0
            
            else: 
                steer = self.controller.compute_steering(self.x, self.y, self.yaw, self.v)
                steer = -1 * np.rad2deg(steer)  # Convert to degrees and adjust direction
                speed = self.max_speed    
                

            # Create control message
            set_msg = ERP_SET()
            set_msg.set_steer = np.clip(int(steer * 71), -2000, 2000)  # Convert and clip
            set_msg.set_speed = speed  # Set maximum speed

            # Publish the control message
            self.steer_pub.publish(set_msg)
            self.rate.sleep()


if __name__ == '__main__':
    try:
        stanley = StanleyControllerNode()
        stanley.main()
    except rospy.ROSInterruptException:
        pass
