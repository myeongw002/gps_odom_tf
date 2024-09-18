#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
import pyproj
import tf_conversions
import open3d as o3d
import signal
import sys
import time

# UTM 변환기를 생성합니다.
wgs84_to_utm = pyproj.Transformer.from_crs("epsg:4326", "epsg:32652")  # 한국 서쪽 지역에 맞춘 EPSG 코드

# 전역 변수들
gps_positions = []
odom_positions = []
start_time = None
time_interval = 20  # 20초간 데이터를 저장

# GPS 콜백 함수
def gps_callback(data):
    global gps_positions
    # GPS 데이터를 UTM 좌표로 변환
    utm_x, utm_y = wgs84_to_utm.transform(data.latitude, data.longitude)
    gps_positions.append([utm_x, utm_y, 0.0])  # 3D 포인트로 변환 (z = 0)
    rospy.loginfo(f"GPS 데이터 수신: ({utm_x}, {utm_y})")

# Odometry 콜백 함수
def odom_callback(data):
    global odom_positions
    # Odometry 데이터의 x, y 위치를 저장
    x = data.pose.pose.position.x
    y = data.pose.pose.position.y
    odom_positions.append([x, y, 0.0])  # 3D 포인트로 변환 (z = 0)
    rospy.loginfo(f"Odometry 데이터 수신: ({x}, {y})")

# 두 프레임간 변환 관계를 계산하는 함수 (변위 벡터 사용)
# 두 프레임간 변환 관계를 계산하는 함수 (변위 벡터 사용)
def calculate_transform():
    if len(gps_positions) < 2 or len(odom_positions) < 2:
        rospy.logwarn("두 프레임의 변환을 계산하기 위한 데이터가 충분하지 않습니다.")
        return None

    # 첫 번째와 두 번째 GPS, Odometry 위치를 사용하여 변위 벡터 계산
    gps_displacement = np.array(gps_positions[-1]) - np.array(gps_positions[0])
    odom_displacement = np.array(odom_positions[-1]) - np.array(odom_positions[0])

    # 변위 벡터의 크기 계산 (노름)
    gps_displacement_norm = np.linalg.norm(gps_displacement)
    odom_displacement_norm = np.linalg.norm(odom_displacement)

    # 변위 벡터 크기 차이 계산
    displacement_norm_diff = abs(gps_displacement_norm - odom_displacement_norm)

    # 변위 벡터로부터 회전각 계산 (yaw)
    gps_angle = np.arctan2(gps_displacement[1], gps_displacement[0])
    odom_angle = np.arctan2(odom_displacement[1], odom_displacement[0])
    yaw_diff = gps_angle - odom_angle

    # 초기 위치를 기반으로 변환 행렬 생성
    translation = np.array(gps_positions[0])
    rotation_matrix = tf_conversions.transformations.euler_matrix(0, 0, yaw_diff)
    rotation_matrix[:3, 3] = translation
    
    # 로그 출력: 변위 벡터의 크기 및 크기 차이
    rospy.loginfo(f"GPS 변위 벡터 크기: {gps_displacement_norm}")
    rospy.loginfo(f"Odometry 변위 벡터 크기: {odom_displacement_norm}")
    rospy.loginfo(f"변위 벡터 크기 차이: {displacement_norm_diff}")
    
    rospy.loginfo(f"Translation: {translation}")
    rospy.loginfo(f"Yaw difference: {np.degrees(yaw_diff)} degrees")
    
    return rotation_matrix


# ICP 알고리즘을 적용하여 두 포인트 클라우드간의 변환 계산 및 평가
def perform_icp_and_visualize():
    if len(gps_positions) < 10 or len(odom_positions) < 10:
        rospy.logwarn("GPS 또는 Odometry 데이터가 충분하지 않습니다.")
        return None

    # Open3D PointCloud 객체 생성
    gps_pcd = o3d.geometry.PointCloud()
    odom_pcd = o3d.geometry.PointCloud()
    
    gps_pcd.points = o3d.utility.Vector3dVector(gps_positions)
    odom_pcd.points = o3d.utility.Vector3dVector(odom_positions)
    
    # 초기 변환 설정
    init_transform = calculate_transform()
    rospy.loginfo(f"초기 변환 행렬: \n{init_transform}")
    
    # ICP 평가
    threshold = 0.05  # 포인트 매칭을 위한 거리 임계값
    evaluation = o3d.pipelines.registration.evaluate_registration(
        odom_pcd, gps_pcd, threshold, init_transform)

    rospy.loginfo(f"ICP 평가 피트니스: {evaluation.fitness}")
    rospy.loginfo(f"ICP 평가 RMSE: {evaluation.inlier_rmse}")
    
    # ICP 결과 시각화
    odom_pcd.transform(init_transform)
    
    gps_pcd.paint_uniform_color([1, 0, 0])  # 빨간색 (GPS)
    odom_pcd.paint_uniform_color([0, 1, 0])  # 초록색 (Odometry)
    
    # 좌표축 시각화 추가
    gps_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[gps_positions[0][0], gps_positions[0][1], 0])
    odom_axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
    odom_axes.transform(init_transform)
    # 두 프레임의 포인트 클라우드와 좌표축을 시각화
    #o3d.visualization.draw_geometries([gps_pcd, odom_pcd, gps_axes, odom_axes], window_name="ICP 결과 및 좌표축 시각화", width=800, height=600)
    o3d.visualization.draw_geometries([gps_axes, odom_axes])
    o3d.visualization.draw_geometries([gps_pcd, odom_pcd])
    

# 프로그램 종료 시 호출될 함수
def signal_handler(sig, frame):
    rospy.loginfo("프로그램 종료 중... ICP 계산 및 평가를 시작합니다.")
    perform_icp_and_visualize()
    sys.exit(0)

# 메인 함수
def main():
    global start_time
    rospy.init_node('gps_odom_tf_icp')

    rospy.Subscriber('/ublox_gps/fix', NavSatFix, gps_callback)
    rospy.Subscriber('/Odometry', Odometry, odom_callback)

    # 프로그램 종료 시 신호 처리
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    rospy.loginfo("GPS와 Odometry 데이터 수집 중...")
    start_time = time.time()

    # 20초 경과 후 변위 벡터 기반 변환 계산
    while not rospy.is_shutdown():
        if time.time() - start_time >= time_interval:
            rospy.loginfo("20초 경과, 변위 벡터 기반 변환 계산 중...")
            calculate_transform()
            rospy.loginfo("변환 계산 완료.")
            break

        rospy.sleep(0.1)

    rospy.spin()

if __name__ == '__main__':
    main()

