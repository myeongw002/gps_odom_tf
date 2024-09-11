import open3d as o3d
import numpy as np
import os
import glob

# Transformation matrix 생성 함수 (3x4 -> 4x4)
def create_transformation_matrix(pose_data):
    # pose_data는 3x4 행렬로 변환 가능한 형태 (ex: 12개의 요소를 가진 리스트)
    transformation_matrix = np.array(pose_data).reshape(3, 4)
    
    # 마지막 행 [0, 0, 0, 1]을 추가하여 4x4 행렬 생성
    bottom_row = np.array([[0, 0, 0, 1]])
    transformation_matrix = np.vstack((transformation_matrix, bottom_row))
    
    return transformation_matrix

# 포인트 클라우드 데이터를 처리하는 함수
def process_pcd_files(data_folder, pose_file, output_filename):
    scans_folder = os.path.join(data_folder, "Scans")

    # PCD 파일 목록 불러오기
    pcd_files = sorted(glob.glob(os.path.join(scans_folder, "*.pcd")))

    # pose 파일 읽기
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            # 각 줄을 파싱하여 pose를 리스트 형태로 저장 (필요한 부분만 추출)
            pose_data = list(map(float, line.split()))
            poses.append(pose_data[:12])  # 12개의 요소만 사용 (3x4 행렬)

    combined_pcd = o3d.geometry.PointCloud()

    for i, pcd_file in enumerate(pcd_files):
        print(f"Processing {pcd_file} with pose from {pose_file}")

        # PCD 파일 불러오기
        pcd = o3d.io.read_point_cloud(pcd_file)

        # Pose 불러오기 및 변환 행렬 생성
        pose = poses[i]
        transformation_matrix = create_transformation_matrix(pose)

        # 변환 적용
        pcd.transform(transformation_matrix)

        # 변환된 포인트 클라우드를 결합
        combined_pcd += pcd

    # 결합된 포인트 클라우드 시각화
    o3d.visualization.draw_geometries([combined_pcd])

    # 결합된 포인트 클라우드 저장
    output_file = os.path.join(data_folder, output_filename)
    o3d.io.write_point_cloud(output_file, combined_pcd)
    print(f"Combined point cloud saved to {output_file}")

    return combined_pcd

# 두 결과 비교 시각화
def compare_point_clouds(pc1, pc2):
    pc1.paint_uniform_color([1, 0, 0])  # 첫 번째 결과를 빨간색으로 표시
    pc2.paint_uniform_color([0, 1, 0])  # 두 번째 결과를 녹색으로 표시

    o3d.visualization.draw_geometries([pc1, pc2], window_name="Comparison of Optimized and Odom Poses")

# ICP 및 평가 수행 함수
def icp_and_evaluate(source, target):
    threshold = 0.02  # ICP에서 허용하는 최대 거리 (조정 가능)
    trans_init = np.eye(4)  # 초기 변환 행렬 (단위 행렬)

    
    # 정합 후 평가 (evaluate_registration 사용)
    print("Evaluating registration using evaluate_registration...")
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    
    print("Evaluation after registration:")
    print(f"Fitness: {evaluation.fitness}")
    print(f"Inlier RMSE: {evaluation.inlier_rmse}")


# 메인 함수
if __name__ == "__main__":
    data_folder = input("데이터 폴더의 경로를 입력하세요: ").strip().replace("'", "")

    # 최적화된 포즈 사용 (optimized_poses.txt)
    optimized_pcd = process_pcd_files(data_folder, os.path.join(data_folder, "optimized_poses.txt"), "combined_map_optimized.pcd")

    # 최적화되지 않은 포즈 사용 (odom_poses.txt)
    odom_pcd = process_pcd_files(data_folder, os.path.join(data_folder, "odom_poses.txt"), "combined_map_odom.pcd")

    # 두 결과 비교 시각화
    compare_point_clouds(optimized_pcd, odom_pcd)

    # ICP 평가 및 결과 평가
    icp_and_evaluate(odom_pcd, optimized_pcd)

