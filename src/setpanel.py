#!/usr/bin/env python3
import rospy
import rospkg
import os
from geometry_msgs.msg import PoseWithCovarianceStamped
import open3d as o3d
import numpy as np
from math import sqrt, pow
import time

class Pathmaker(): # 위도, 경도를 x, y 좌표로 변환
    def __init__(self):
        self.path_src = rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.callback, queue_size=1) # Subscribing to NavSatFix messages with queue_size=1
        self.path = open('/home/user/catkin_ws/src/erp42_serial2-main/path/pahel.txt', 'w') # Opening the text file
        self.x, self.y, self.x_prev, self.y_prev = 0, 0, 0, 0
        self.count = 4
        key = input('구간 이름 설정')
        self.pname = key

    def callback(self, msg):
        if self.count == 0:
            self.count = 4
            self.path.writelines(['\n'])
            key = input('구간 이름 설정')
            self.pname = key

        self.x, self.y = msg.pose.pose.position.x, msg.pose.pose.position.y

        rospy.loginfo("count = {}".format(self.count)) 
        self.path.writelines([str(self.x), ' ', str(self.y), ' ', '/'])
        self.count -= 1


        
if __name__ == '__main__':
    rospy.init_node('Pathmaker_node', anonymous=True)
    rospy.wait_for_message('/initialpose', PoseWithCovarianceStamped)

    pm = Pathmaker()

    rospy.spin()
