#!/usr/bin/env python3
import rospy
import rospkg
import os
from nav_msgs.msg import Odometry
import open3d as o3d
import numpy as np
from math import sqrt, pow
import time

class Pathmaker():  # Convert latitude, longitude to x, y coordinates
    def __init__(self, cen_x, cen_y, path_file):
        self.path_src = rospy.Subscriber('/localization', Odometry, self.callback, queue_size=1)  # Subscribing to Odometry messages with queue_size=1
        
        # Ensure directory exists
        directory = os.path.dirname(path_file)
        if not os.path.exists(directory):
            os.makedirs(directory)  # Create the directory if it does not exist

        self.path = open(path_file, 'w')  # Opening the text file where the path will be saved
        self.x, self.y = 0, 0
        self.x_prev, self.y_prev = 0, 0  # Initialize previous coordinates
        self.cen_x, self.cen_y = cen_x, cen_y

    def distance(self, x, y, x_prev, y_prev):  # Calculate Euclidean distance
        return sqrt(pow((x - x_prev), 2) + pow((y - y_prev), 2))

    def callback(self, msg):
        self.x, self.y = msg.pose.pose.position.x, msg.pose.pose.position.y

        if self.distance(self.x, self.y, self.x_prev, self.y_prev) > 1:
            self.x_prev = self.x
            self.y_prev = self.y

            if self.x < self.cen_x and self.y > self.cen_y:
                n = 1
            elif self.x >= self.cen_x and self.y > self.cen_y:
                n = 2
            elif self.x >= self.cen_x and self.y <= self.cen_y:
                n = 3
            elif self.x < self.cen_x and self.y <= self.cen_y:
                n = 4

            rospy.loginfo(f"{n} 사분면") 
            self.path.writelines([str(self.x), ' ', str(self.y), ' ', str(n), '\n'])


if __name__ == '__main__':
    rospy.init_node('Pathmaker_node', anonymous=True)
    rospy.wait_for_message('/localization', Odometry)

    # Request user input for path where the path data will be saved
    path_file = input('경로 데이터를 저장할 파일의 절대 경로를 입력하십시오 (예: /home/user/path.txt): ').strip().replace("'", "")
    rospy.loginfo(f"Path file: {path_file}")

    # Request user input for PCD file path to calculate the map's centroid
    package_path = input('PCD 파일의 절대 경로를 입력하십시오 (예: /home/user/PCD/scans.pcd): ').strip().replace("'", "")
    rospy.loginfo(f"PCD file: {package_path}")

    # Check if PCD file exists
    if not os.path.exists(package_path):
        rospy.logerr(f"PCD file does not exist at {package_path}")
        exit(1)

    # Read point cloud and calculate centroid of the map
    pcd = o3d.io.read_point_cloud(package_path)  # PCD file path input by user
    points = np.asarray(pcd.points)
    
    if points.size == 0:
        rospy.logerr("The PCD file is empty or invalid.")
        exit(1)

    centroid = np.mean(points, axis=0)
    rospy.loginfo(f"Map Centroid: {centroid}")

    # Create Pathmaker instance
    pm = Pathmaker(centroid[0], centroid[1], path_file)
    
    # Wait for user to signal the end of rosbag playback
    key = input('rosbag 재생이 끝나면 e를 입력하십시오: ')
    if key == "e":
        pm.path.close()
        del pm

    rospy.loginfo("변환 완료")
