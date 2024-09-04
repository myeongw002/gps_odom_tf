import numpy as np
import math

def rotation_matrix_to_quaternion(R):
    """
    회전 행렬을 쿼터니언으로 변환
    :param R: 3x3 회전 행렬
    :return: 쿼터니언 (qx, qy, qz, qw)
    """
    assert R.shape == (3, 3), "입력 행렬은 3x3 크기여야 합니다."
    
    qw = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    
    return qx, qy, qz, qw

def quaternion_to_euler(qx, qy, qz, qw):
    """
    쿼터니언을 Euler 각도로 변환 (roll, pitch, yaw)
    :param qx, qy, qz, qw: 쿼터니언 값
    :return: roll, pitch, yaw (각도는 라디안 단위)
    """
    # Roll (x축 회전)
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))

    # Pitch (y축 회전)
    pitch = math.asin(2 * (qw * qy - qz * qx))

    # Yaw (z축 회전)
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

    return roll, pitch, yaw

# 예제 회전 행렬
R = np.array([
    [ 0.99973469, -0.02303384,  0       ],
    [ 0.02303384,  0.99973469,  0        ],
    [ 0,          0,          1,        ],
    ])

# 회전 행렬을 쿼터니언으로 변환
qx, qy, qz, qw = rotation_matrix_to_quaternion(R)
print(f"쿼터니언: qx={qx}, qy={qy}, qz={qz}, qw={qw}")

# 쿼터니언을 Euler 각도로 변환
roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)
print(f"Euler 각도: roll={roll}, pitch={pitch}, yaw={yaw}")

