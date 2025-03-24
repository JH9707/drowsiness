import cv2
import numpy as np
import pandas as pd
import pickle
import mediapipe as mp
from picamera2 import Picamera2
import pygame
import tensorflow as tf

# 모델 및 상수 초기화
model = pickle.load(open('model.pkl', 'rb'))

# 얼굴 특징 좌표 컬럼명 설정
cols = []
for pos in ['nose_', 'forehead_', 'left_eye_', 'mouth_left_', 'chin_', 'right_eye_', 'mouth_right_']:
    for dim in ('x', 'y'):
        cols.append(pos + dim)

def extract_features(img, face_mesh):
    """
    이미지에서 얼굴 랜드마크를 추출하여 지정된 포인트의 x, y 좌표를 반환합니다.
    """
    # 관심 landmark 인덱스 설정
    NOSE = 1
    FOREHEAD = 10
    LEFT_EYE = 33
    MOUTH_LEFT = 61
    CHIN = 199
    RIGHT_EYE = 263
    MOUTH_RIGHT = 291

    # 랜드마크 그리기용 스펙 (옵션)
    landmark_spec = mp.solutions.drawing_utils.DrawingSpec(
        color=(0, 255, 0), thickness=1, circle_radius=0.01)
    
    result = face_mesh.process(img)
    face_features = []
    
    # 얼굴 랜드마크가 검출되면 그리기
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                img, face_landmarks, mp.solutions.face_mesh.FACEMESH_CONTOURS)

    # 지정된 인덱스의 landmark 좌표 추출
    if result.multi_face_landmarks is not None:
        for face_landmarks in result.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [FOREHEAD, NOSE, MOUTH_LEFT, MOUTH_RIGHT, CHIN, LEFT_EYE, RIGHT_EYE]:
                    face_features.append(lm.x)
                    face_features.append(lm.y)

    return face_features

def normalize(poses_df):
    """
    얼굴 특징 좌표를 코(nose)를 기준으로 정규화하고,
    좌측 눈과 우측 입술 사이의 거리를 기준으로 스케일링합니다.
    """
    normalized_df = poses_df.copy()
    
    for dim in ['x', 'y']:
        for feature in ['forehead_' + dim, 'nose_' + dim, 'mouth_left_' + dim, 
                        'mouth_right_' + dim, 'left_eye_' + dim, 'chin_' + dim, 'right_eye_' + dim]:
            normalized_df[feature] = poses_df[feature] - poses_df['nose_' + dim]
        
        diff = normalized_df['mouth_right_' + dim] - normalized_df['left_eye_' + dim]
        for feature in ['forehead_' + dim, 'nose_' + dim, 'mouth_left_' + dim, 
                        'mouth_right_' + dim, 'left_eye_' + dim, 'chin_' + dim, 'right_eye_' + dim]:
            normalized_df[feature] = normalized_df[feature] / diff
    
    return normalized_df

def draw_axes(img, pitch, yaw, roll, tx, ty, size=50):
    """
    예측된 pitch, yaw, roll 값을 사용해 이미지 위에 좌표축(3축)을 그립니다.
    tx, ty는 축의 기준 좌표(예: 코의 위치)입니다.
    """
    yaw = -yaw
    # 회전 벡터를 회전 행렬로 변환
    rotation_matrix = cv2.Rodrigues(np.array([pitch, yaw, roll]))[0].astype(np.float64)
    
    # 기본 축 정의 (x, y, z 및 원점)
    axes_points = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ], dtype=np.float64)
    
    axes_points = rotation_matrix @ axes_points
    axes_points = (axes_points[:2, :] * size).astype(int)
    axes_points[0, :] += tx
    axes_points[1, :] += ty
    
    new_img = img.copy()
    # x축 (파란색)
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 0].ravel()), (255, 0, 0), 3)
    # y축 (녹색)
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 1].ravel()), (0, 255, 0), 3)
    # z축 (빨간색)
    cv2.line(new_img, tuple(axes_points[:, 3].ravel()), tuple(axes_points[:, 2].ravel()), (0, 0, 255), 3)
    return new_img

def preprocess_eye(eye_img):
    """
    눈 이미지 전처리: 그레이스케일 변환, 크기 조정, 정규화 후 모델 입력 형태로 reshape.
    """
    eye_img = cv2.cvtColor(eye_img, cv2.COLOR_RGB2GRAY)
    eye_img = cv2.resize(eye_img, (24, 24))
    eye_img = eye_img / 255.0
    return eye_img.reshape(1, 24, 24, 1)

# 초기화
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('power_alarm.wav')  

face_mesh = mp.solutions.face_mesh.FaceMesh(
    min_detection_confidence=0.01, min_tracking_confidence=0.8)

# TFLite 졸음 감지 모델 초기화
drowsiness_model = tf.lite.Interpreter(model_path="model1.tflite")
drowsiness_model.allocate_tensors()
input_details = drowsiness_model.get_input_details()
output_details = drowsiness_model.get_output_details()

# 눈의 랜드마크 인덱스 (좌, 우)
LEFT_EYE = [33, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144]
RIGHT_EYE = [362, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380]

# 졸음 감지 관련 변수 초기화
closed_eye_frames = 0
threshold_eye_frames = 15
drowsy_text = "Awake"

# 얼굴 정면 감지 관련 변수
face_frames = 0
threshold_face_frames = 50
notfronttext = ''

alarm_playing = False

# PiCamera2 초기화 및 설정
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()

# 메인 루프: 영상 캡처 및 처리
while True:
    # 이미지 캡처 및 전처리 (회전, 좌우 반전, 색상 변환)
    img = picam2.capture_array()
    img = cv2.rotate(img, cv2.ROTATE_180)
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = img.shape
    text = ''
    origin_img = img.copy()

    # 얼굴 랜드마크 검출
    results = face_mesh.process(img)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 좌측 및 우측 눈의 좌표 계산
            left_eye = np.array([[int(pt.x * img_w), int(pt.y * img_h)]
                                  for pt in [face_landmarks.landmark[i] for i in LEFT_EYE]])
            right_eye = np.array([[int(pt.x * img_w), int(pt.y * img_h)]
                                   for pt in [face_landmarks.landmark[i] for i in RIGHT_EYE]])

            # 눈 영역 bounding box 계산
            l_x, l_y, l_w, l_h = cv2.boundingRect(left_eye)
            r_x, r_y, r_w, r_h = cv2.boundingRect(right_eye)

            left_eye_img = img[l_y:l_y + l_h, l_x:l_x + l_w]
            right_eye_img = img[r_y:r_y + r_h, r_x:r_x + r_w]

            if left_eye_img.size > 0 and right_eye_img.size > 0:
                # 좌측 눈 예측
                left_eye_input = preprocess_eye(left_eye_img).astype(np.float32)
                drowsiness_model.set_tensor(input_details[0]['index'], left_eye_input)
                drowsiness_model.invoke()
                left_eye_pred = drowsiness_model.get_tensor(output_details[0]['index'])

                # 우측 눈 예측
                right_eye_input = preprocess_eye(right_eye_img).astype(np.float32)
                drowsiness_model.set_tensor(input_details[0]['index'], right_eye_input)
                drowsiness_model.invoke()
                right_eye_pred = drowsiness_model.get_tensor(output_details[0]['index'])

                # 졸음 상태 판단 (예측값 임계치 0.99 기준)
                if left_eye_pred < 0.99 and right_eye_pred < 0.99:
                    closed_eye_frames += 1
                    if closed_eye_frames >= threshold_eye_frames:
                        drowsy_text = "Drowsy"
                else:
                    closed_eye_frames = 0
                    drowsy_text = "Awake"
                
                # 눈 영역 경계선 그리기
                cv2.polylines(img, [left_eye], True, (0, 255, 0), 1)
                cv2.polylines(img, [right_eye], True, (0, 255, 0), 1)

    # 얼굴 특징 추출 및 포즈 예측
    face_features = extract_features(img, face_mesh)
    if not face_features:
        print("Warning: No face features detected in this frame.")
    else:
        print("Extracted Features:", face_features)

    if len(face_features):
        # 추출한 특징을 데이터프레임으로 변환 후 정규화
        face_features_df = pd.DataFrame([face_features], columns=cols)
        face_features_normalized = normalize(face_features_df)
        
        # pickle 모델을 통해 pitch, yaw, roll 예측
        pitch_pred, yaw_pred, roll_pred = model.predict(face_features_normalized).ravel()

        # 코의 좌표를 기준으로 좌표축 그리기
        nose_x = face_features_df['nose_x'].values * img_w
        nose_y = face_features_df['nose_y'].values * img_h
        img = draw_axes(img, pitch_pred, yaw_pred, roll_pred, nose_x, nose_y)

        # 얼굴 방향 판단 (각도 기준)
        if pitch_pred > 0.3:
            text = 'Top'
            if yaw_pred > 0.3:
                text = 'Top Left'
            elif yaw_pred < -0.3:
                text = 'Top Right'
        elif pitch_pred < -0.3:
            text = 'Bottom'
            if yaw_pred > 0.3:
                text = 'Bottom Left'
            elif yaw_pred < -0.3:
                text = 'Bottom Right'
        elif yaw_pred > 0.3:
            text = 'Left'
        elif yaw_pred < -0.3:
            text = 'Right'
        else:
            text = 'Forward'

        # 정면이 아닐 경우 프레임 카운트 증가 및 경고 메시지 표시
        if text != 'Forward':
            face_frames += 1
            if face_frames >= threshold_face_frames:      
                notfronttext = 'Look Forward!!'
        else:
            face_frames = 0

    # 알람 재생 제어: 얼굴이 정면이 아니거나 졸음 상태이면 알람 재생
    if face_frames <= threshold_face_frames and drowsy_text == 'Awake':
        if alarm_playing:
            alarm_playing = False
            pygame.mixer.stop()
    else:
        if not alarm_playing:
            alarm_sound.play()
            alarm_playing = True 

    # 결과 텍스트를 원본 이미지에 출력
    cv2.putText(origin_img, text, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0) if notfronttext == 'LookForward!!' else (0, 255, 0), 2)
    cv2.putText(origin_img, drowsy_text, (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if drowsy_text == "Drowsy" else (0, 255, 0), 2)
    
    # 결과 영상 출력
    cv2.imshow('img', img)
    cv2.imshow('origin', origin_img)

    # 'q' 키 입력 시 종료
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):
        break

# 종료 작업: 카메라 및 윈도우 정리
picam2.stop()
picam2.close()
cv2.destroyAllWindows()
