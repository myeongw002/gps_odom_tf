#!/usr/bin/env python
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion  # 쿼터니언 변환을 위한 import
from erp42_serial2.msg import ERP_SET,ERP_STATUS


class StanleyController:
    def __init__(self):

        self.path_file = rospy.get_param("~path", '/home/team-miracle/ROS/catkin_ws/src/gps_odom_tf/path/path_sample.txt')
        self.K = rospy.get_param("~K", 0.5)
        self.L = rospy.get_param("~L", 1.14)
        self.max_steering = rospy.get_param("~K", 27)
        self.max_steering = max_steering=np.radians(self.max_steering)


        # Load path data (ref_xs, ref_ys, ref_yaws)
        self.ref_xs, self.ref_ys, self.ref_yaws = self.load_path(self.path_file)

    def load_path(self, file_path):
        # 경로 파일에서 첫 번째와 두 번째 열을 읽어옴
        data = np.loadtxt(file_path, delimiter=' ')
        ref_xs = data[:, 0]
        ref_ys = data[:, 1]
        
        # ref_yaws는 경로의 기울기를 사용하여 계산
        ref_yaws = np.arctan2(np.gradient(ref_ys), np.gradient(ref_xs))
        
        return ref_xs, ref_ys, ref_yaws

    def normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2.0 * np.pi
        while angle < -np.pi:
            angle += 2.0 * np.pi
        return angle

    def compute_steering(self, x, y, yaw, v):
        # 기준 경로에서 가장 가까운 점 찾기
        min_dist = float('inf')
        min_index = 0
        n_points = len(self.ref_xs)

        front_x = x + self.L * np.cos(yaw)
        front_y = y + self.L * np.sin(yaw)

        for i in range(n_points):
            dx = front_x - self.ref_xs[i]
            dy = front_y - self.ref_ys[i]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                min_index = i

        # 횡방향 오차 계산
        ref_x = self.ref_xs[min_index]
        ref_y = self.ref_ys[min_index]
        ref_yaw = self.ref_yaws[min_index]

        dx = ref_x - front_x
        dy = ref_y - front_y
        perp_vec = [np.cos(ref_yaw + np.pi/2), np.sin(ref_yaw + np.pi/2)]
        cte = np.dot([dx, dy], perp_vec)

        # 헤딩 오차 계산 및 스티어링 각도 결정
        yaw_error = self.normalize_angle(ref_yaw - yaw)

        # CTE에 대한 민감도를 조정하기 위해 k 값을 조정
        cte_term = np.arctan2(self.K * cte, v)

        steer = yaw_error + cte_term
        steer = np.clip(steer, -self.max_steering, self.max_steering)
        
        # 디버깅을 위한 로그 추가 (경로 인덱스 포함)
        print("----------------")
        rospy.loginfo(f"Path Index: {min_index}")
        rospy.loginfo(f"x: {x}, y: {y}, yaw: {np.degrees(yaw)}, v: {v}")
        rospy.loginfo(f"ref_x: {ref_x}, ref_y: {ref_y}, ref_yaw: {np.degrees(ref_yaw)}")
        rospy.loginfo(f"CTE: {cte}, Yaw Error: {np.degrees(yaw_error)}, CTE Term: {np.degrees(cte_term)}")
        rospy.loginfo(f"Steering Angle (radians): {steer}, (degrees): {np.degrees(steer)}")

        return steer



class StanleyControllerNode:
    def __init__(self):
        # ROS node initialization
        rospy.init_node('stanley_controller')
        
        
        # Create a StanleyController object
        self.controller = StanleyController()
        
        # Initial state
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 2.0
        self.max_speed = 3
        # Publisher for control command
        self.steer_pub = rospy.Publisher('/steering_cmd', ERP_SET, queue_size=1)
        
        # Subscriber for odometry
        rospy.Subscriber('/localization', Odometry, self.odometry_callback)
        rospy.Subscriber('/erp_state', ERP_STATUS, self.status_callback)
        
        # Control rate
        self.control_rate = rospy.Rate(10)  # 10 Hz
        
        # Start the control loop
        self.control_loop()

    def odometry_callback(self, msg:Odometry):
        # Update vehicle state from odometry message
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        # tf.transformations의 euler_from_quaternion을 사용하여 쿼터니언을 오일러 각도로 변환
        _, _, self.yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])


    def status_callback(self, msg:ERP_STATUS):
        self.v = msg.status_speed


    def control_loop(self):
        while not rospy.is_shutdown():
            # Compute the steering command using Stanley control
            steer = self.controller.compute_steering(self.x, self.y, self.yaw, self.v)
            steer = -1 * np.rad2deg(steer)
            rospy.loginfo(f"Steerinf angle: {steer}")
            set_msg = ERP_SET()
            set_steer = int(steer * 71)
            set_msg.set_steer = np.clip(set_steer,-2000,2000)
            set_msg.set_speed = self.max_speed
            # Publish the steering command
            self.steer_pub.publish(set_msg)
            
            # Wait for the next iteration
            self.control_rate.sleep()

if __name__ == '__main__':
    try:
        StanleyControllerNode()
    except rospy.ROSInterruptException:
        pass
