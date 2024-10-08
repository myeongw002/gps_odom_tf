#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
import pyproj
import tf_conversions
import tf2_ros
import geometry_msgs.msg
import open3d as o3d
import signal
import sys

# UTM 변환기를 생성합니다.
wgs84_to_utm = pyproj.Transformer.from_crs("epsg:4326", "epsg:32652")  # 한국 서쪽 지역에 맞춘 EPSG 코드

gps_positions = []
odom_positions = []

def gps_callback(data):
    global gps_positions
    # GPS 데이터를 UTM 좌표로 변환
    utm_x, utm_y = wgs84_to_utm.transform(data.latitude, data.longitude)
    gps_positions.append([utm_x, utm_y, 0.0])  # 3D 포인트로 변환 (z = 0)
    rospy.loginfo(f"GPS 데이터 수신: ({utm_x}, {utm_y})")

def odom_callback(data):
    global odom_positions
    # Odometry 데이터의 x, y 위치를 저장
    x = data.pose.pose.position.x
    y = data.pose.pose.position.y
    odom_positions.append([x, y, 0.0])  # 3D 포인트로 변환 (z = 0)
    rospy.loginfo(f"Odometry 데이터 수신: ({x}, {y})")

def calculate_initial_transform():
    if len(gps_positions) < 2:
        rospy.logwarn("초기 변환을 계산하기 위한 GPS 데이터가 충분하지 않습니다.")
        return np.eye(4)

    # 첫 번째와 두 번째 GPS 위치를 사용하여 초기 변환 계산
    first_position = np.array(gps_positions[0])
    second_position = np.array(gps_positions[1])
    
    # Translation 설정: 첫 번째 위치
    translation = first_position
    
    # Rotation 설정: 첫 번째와 두 번째 위치 사이의 방향 계산
    direction = second_position - first_position
    yaw = np.arctan2(direction[1], direction[0])
    
    # Yaw 회전을 쿼터니언으로 변환하고 변환 행렬 생성
    rotation_matrix = tf_conversions.transformations.euler_matrix(0, 0, yaw)
    rotation_matrix[:3, 3] = translation
    
    return rotation_matrix

def calculate_tf():
    if len(gps_positions) < 10 or len(odom_positions) < 10:
        rospy.logwarn("GPS 또는 Odometry 데이터가 충분하지 않습니다.")
        return None

    # 데이터 개수를 일치시키기 위해 랜덤 샘플링
    min_length = min(len(gps_positions), len(odom_positions))
    sampled_indices = np.random.choice(len(odom_positions), min_length, replace=False)
    
    sampled_odom_positions = np.array(odom_positions)[sampled_indices]
    
    # Open3D PointCloud 객체 생성
    gps_pcd = o3d.geometry.PointCloud()
    odom_pcd = o3d.geometry.PointCloud()
    
    gps_pcd.points = o3d.utility.Vector3dVector(gps_positions)
    odom_pcd.points = o3d.utility.Vector3dVector(odom_positions)
    gps_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    odom_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # 초기 변환 설정
    init_transform = calculate_initial_transform()
    rospy.loginfo(f"Init transform: \n{init_transform}")
    
    # ICP 알고리즘 적용
    threshold = 0.05  # 포인트 매칭을 위한 거리 임계값 (예: 0.5미터)
    icp_result = o3d.pipelines.registration.registration_icp(
        odom_pcd, gps_pcd, threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000)
    )
    
    
    R = icp_result.transformation[:3, :3]
    t = icp_result.transformation[:3, 3]
    
    # ICP 결과를 로그로 출력
    #rospy.loginfo(icp_result)
    rospy.loginfo(f"ICP 결과 변환 행렬 (Rotation):\n{R}")
    rospy.loginfo(f"ICP 결과 변환 벡터 (Translation):\n{t}")
    rospy.loginfo(f"ICP 피트니스: {icp_result.fitness}")
    rospy.loginfo(f"RMSE: {icp_result.inlier_rmse}")
    # ICP 결과를 적용한 변환된 odometry 포인트 클라우드
    odom_pcd.transform(icp_result.transformation)
    
    # 시각화를 위해 각각의 점군에 색상 추가
    gps_pcd.paint_uniform_color([1, 0, 0])  # 빨간색 (GPS)
    odom_pcd.paint_uniform_color([0, 1, 0])  # 초록색 (Odometry)
    
    # 두 점군 시각화
    o3d.visualization.draw_geometries([gps_pcd, odom_pcd], window_name="ICP 결과 시각화", width=800, height=600)
    
    return (R, t), icp_result.fitness

def signal_handler(sig, frame):
    rospy.loginfo("프로그램이 종료됩니다. ICP 계산을 시작합니다...")
    result = calculate_tf()
    sys.exit(0)

def main():
    rospy.init_node('gps_odom_tf_icp')

    rospy.Subscriber('/ublox_gps/fix', NavSatFix, gps_callback)
    rospy.Subscriber('/Odometry', Odometry, odom_callback)

    # 프로그램 종료 시 신호 처리
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    rospy.loginfo("GPS와 Odometry 데이터 수집 중...")
    rospy.spin()

if __name__ == '__main__':
    main()

