#!/usr/bin/env python3
import rospy
import utm
import numpy as np
from sensor_msgs.msg import NavSatFix
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# 텍스트 파일에서 translation_vector, gps_centroid, best_angle을 불러오는 함수
def load_transform_params(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # 첫 번째 줄에서 translation vector 추출
        translation_vector = np.array([float(x) for x in lines[0].strip().split()[2:]])
        # 두 번째 줄에서 gps_centroid 추출
        gps_centroid = np.array([float(x) for x in lines[1].strip().split()[2:]])
        # 세 번째 줄에서 best_angle 추출
        best_angle = float(lines[2].strip().split()[2])
    return translation_vector, gps_centroid, best_angle

# 회전 변환 함수
def rotate_points(points, angle, center):
    # Convert angle to radians
    angle_rad = np.radians(angle)
    # Apply 2D rotation matrix around a given center
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    return np.dot(points - center, rotation_matrix.T) + center

class GPSOdometryMarker:
    def __init__(self, transform_params_path):
        # ROS 노드 초기화
        rospy.init_node('gps_to_odometry_marker', anonymous=False)
        
        # GPS 데이터를 받는 subscriber 설정 (NavSatFix 메시지 타입 사용)
        rospy.Subscriber("/ublox_gps/fix", NavSatFix, self.gps_callback)
        
        # Marker 데이터를 RViz로 publish하기 위한 publisher 설정
        self.marker_pub = rospy.Publisher("/gps_marker", Marker, queue_size=11)
        
        # Marker ID를 설정 (각기 다른 마커를 구분하기 위한 ID)
        self.marker_id = 0

        # 텍스트 파일에서 translation vector, gps_centroid, best_angle 불러오기
        self.translation_vector, self.gps_centroid, self.best_angle = load_transform_params(transform_params_path)

    def gps_callback(self, msg):
        # GPS 좌표 (위도, 경도, 고도)를 UTM 좌표로 변환
        utm_x, utm_y, zone_number, zone_letter = utm.from_latlon(msg.latitude, msg.longitude)
        gps_utm = np.array([utm_x, utm_y])
        rospy.loginfo(f"GPS UTM: {gps_utm}")

        # Translation vector를 적용하여 좌표 이동
        translated_coords = gps_utm + self.translation_vector
        rospy.loginfo(f"Translated Coords: {translated_coords}")
        
        # 회전 변환을 적용하여 odometry 좌표로 변환
        odometry_coords = rotate_points(translated_coords, self.best_angle, self.gps_centroid)
        rospy.loginfo(f"Odometry Coords: {odometry_coords}")

        # Marker 메시지를 생성하여 변환된 좌표를 RViz에서 표시
        marker = Marker()
        marker.header.frame_id = "map"  # 오도메트리 좌표계를 기준으로
        marker.header.stamp = rospy.Time.now()
        marker.ns = "gps_marker"
        marker.id = self.marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # 마커의 크기 설정
        marker.scale.x = 1.5
        marker.scale.y = 1.5
        marker.scale.z = 1.5
        
        # 마커의 색상 설정 (파란색)
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        
        # 변환된 좌표를 마커의 위치로 설정
        marker.pose.position.x = odometry_coords[0]
        marker.pose.position.y = odometry_coords[1]
        marker.pose.position.z = 0  # GPS 고도를 그대로 사용

        # 마커를 RViz에 표시
        self.marker_pub.publish(marker)
        
        # 마커 ID를 증가시켜 다음 마커를 구분
        self.marker_id  = 1
        
if __name__ == "__main__":
    try:
        # 변환 파라미터 파일 경로 설정
        transform_params_path = rospy.get_param("~transform_params_path", "/home/teammiracle/ROS/catkin_ws/src/gps_odom_tf/tf_data/transform_data_highway.txt")
        
        # GPS Odometry Marker 노드 실행
        gps_marker_node = GPSOdometryMarker(transform_params_path)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
