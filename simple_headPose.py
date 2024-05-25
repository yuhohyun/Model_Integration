import dlib
import cv2
import math
import numpy as np

# face detector와 landmark predictor 정의
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:/Project/Model_Integration/videos/shape_predictor_68_face_landmarks.dat')

# 3D 모델 포인트
model_points = np.array([
                            (0.0, 0.0, 0.0),             # 코 끝
                            (0.0, -330.0, -65.0),        # 턱
                            (-225.0, 170.0, -135.0),     # 왼쪽 눈의 왼쪽 코너
                            (225.0, 170.0, -135.0),      # 오른쪽 눈의 오른쪽 코너
                            (-150.0, -150.0, -125.0),    # 왼쪽 입 코너
                            (150.0, -150.0, -125.0)      # 오른쪽 입 코너
                        ])

# 비디오 파일 열기
cap = cv2.VideoCapture('C:\\Project\\Model_Integration\\videos\\sample_gaze.MOV')

# 비디오가 열렸는지 확인
if not cap.isOpened():
    print("비디오 파일을 열 수 없습니다.")
    exit()

while True:
    # 비디오에서 프레임을 하나씩 읽음
    ret, frame = cap.read()
    if not ret:
        break  # 프레임을 더 이상 읽을 수 없으면 루프 종료

    # 얼굴 감지
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        # 얼굴 주변 영역 확장
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        expand_ratio = 1.5  # 얼굴을 1.5배 확장
        new_x = max(int(x - (expand_ratio - 1) / 2 * w), 0)
        new_y = max(int(y - (expand_ratio - 1) / 2 * h), 0)
        new_w = min(int(w * expand_ratio), frame.shape[1] - new_x)
        new_h = min(int(h * expand_ratio), frame.shape[0] - new_y)

        # 얼굴 영역 추출
        face_roi = frame[new_y:new_y+new_h, new_x:new_x+new_w]

        # 이미지 크기 및 카메라 내부 파라미터
        size = face_roi.shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                   [0, focal_length, center[1]],
                                   [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4,1))  # 렌즈 왜곡은 없다고 가정

        # 얼굴 영역에 대해 얼굴 특징점 감지
        shape = predictor(face_roi, face)

        # 얼굴 중심점 계산
        center_x = (shape.part(30).x + shape.part(8).x) // 2
        center_y = (shape.part(30).y + shape.part(8).y) // 2

        # 2D 이미지 포인트 추출
        image_points = np.array([
            (shape.part(30).x, shape.part(30).y),   # 코 끝
            (shape.part(8).x, shape.part(8).y),     # 턱
            (shape.part(36).x, shape.part(36).y),   # 왼쪽 눈의 왼쪽 코너
            (shape.part(45).x, shape.part(45).y),   # 오른쪽 눈의 오른쪽 코너
            (shape.part(48).x, shape.part(48).y),   # 왼쪽 입 코너
            (shape.part(54).x, shape.part(54).y)    # 오른쪽 입 코너
        ], dtype="double")

        # 머리 포즈 추정
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # 머리 포즈를 이용하여 회전 행렬을 추출하고 회전 행렬을 각 축의 회전 각도로 분해
        (rotation_matrix, jacobian) = cv2.Rodrigues(rotation_vector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_matrix)

        # 축에 대한 회전각도를 계산
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))

        cv2.putText(frame, "Pitch: " + "{:7.2f}".format(pitch), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), thickness=2)
        cv2.putText(frame, "Yaw: " + "{:7.2f}".format(yaw), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), thickness=2)
        cv2.putText(frame, "Roll: " + "{:7.2f}".format(roll), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 255), thickness=2)

        # 머리 위치와 방향을 나타내는 3D 축을 그림.
        nose_end_point2D, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        p1 = (int(center_x), int(center_y))  # 얼굴 중심점을 시작점으로 변경
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(frame, p1, p2, (255, 0, 0), 2)

        # 머리 방향이 정면인지 아닌지를 판단하여 터미널에 출력
        tolerance = 20  # 허용 오차 각도
        if abs(pitch) < tolerance and abs(yaw) < tolerance and abs(roll) < tolerance:
            direction = "정면"
        else:
            direction = "비정면"

        print(f"머리 방향: {direction}\n상하 회전 각도: {pitch:.2f}, 좌우 회전 각도: {yaw:.2f}, 좌우 기울기 각도: {roll:.2f}\n")

    # 결과를 화면에 표시.
    cv2.imshow("Output", frame)

    # ESC 키를 누르면 루프를 종료.
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 비디오 캡처 객체를 해제하고 창을 닫음.
cap.release()
cv2.destroyAllWindows()
