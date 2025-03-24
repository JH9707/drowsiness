# 전방주시 및 졸음 상태 인식 프로젝트
- Mediapipe를 통한 얼굴 인식 및 헤드포즈 추정
- 학습데이터를 통한 졸음 상태 추정
## 프로젝트 개요
이 프로젝트는

- Face Orientation (헤드 포즈 추정)

- Drowsiness Detection (졸음 감지)

두 가지를 동시에 수행합니다.
## 주요 기능

- 얼굴 랜드마크 추출: Mediapipe의 FaceMesh를 이용해 얼굴의 주요 포인트(코, 이마, 눈, 입, 턱 등)를 추출합니다.

- 특징 정규화: 추출한 좌표를 코를 기준으로 정규화하여 상대 좌표로 변환합니다.

- 포즈 예측: 정규화된 데이터를 pickle로 저장된 머신러닝 모델로부터 pitch, yaw, roll을 예측하고, 이를 기반으로 얼굴 방향(Forward, Top, Left 등)을 판단합니다.

- 축 그리기: 예측된 회전값을 바탕으로 이미지 위에 좌표축(파란색, 녹색, 빨간색)을 그려 헤드 포즈를 시각화합니다.

- 졸음 감지: 좌측/우측 눈 이미지를 전처리 후, TensorFlow Lite 모델을 통해 졸음 여부를 예측합니다.

- 알람 기능: 얼굴이 정면을 바라보지 않거나 졸음 상태로 판단되면 pygame을 이용해 알람을 재생합니다.

- 실시간 영상 처리: PiCamera2를 이용하여 실시간으로 영상을 받아 처리하고 결과를 화면에 출력합니다.

## 주요 의존성
- OpenCV: 이미지 처리 및 영상 스트림 처리

- Numpy: 수치 계산

- Pandas: 데이터프레임을 이용한 특징 처리

- Pickle: 학습된 모델 불러오기

- MediaPipe: 얼굴 랜드마크 추출

- PiCamera2: 라즈베리파이 카메라 인터페이스

- Pygame: 알람 사운드 재생

- TensorFlow: 졸음 감지를 위한 TFLite 모델
## 하드웨어 및 환경
- Raspberry Pi 5 / Raspberry Pi OS
- IR 카메라

## 코드 구조
### 라이브러리 임포트 및 모델 로드
- 필요한 라이브러리를 불러오고, pickle 모델과 TensorFlow Lite 모델, 알람 사운드 등을 초기화합니다.

### 데이터 전처리 및 특징 추출 함수

- extract_features(img, face_mesh): 얼굴 랜드마크에서 지정된 포인트들의 (x, y) 좌표를 추출합니다.

- normalize(poses_df): 추출된 좌표를 코의 좌표를 기준으로 정규화합니다.

- preprocess_eye(eye_img): 눈 영역 이미지를 전처리하여 TFLite 모델 입력 형식으로 변환합니다.

### 시각화 함수

- draw_axes(img, pitch, yaw, roll, tx, ty, size=50): 예측된 회전값을 사용해 이미지에 좌표축을 그립니다.

### 실시간 영상 처리 루프

- PiCamera2로부터 영상을 받아 회전, 좌우 반전 후 색상 변환을 수행합니다.

- 얼굴 및 눈 영역을 검출하고, 각 영역에 대해 전처리 및 예측을 진행합니다.

- 얼굴 포즈와 졸음 상태에 따라 텍스트를 화면에 출력하고, 필요 시 알람을 재생합니다.

## 사용 방법
### 의존성 설치
- `pip install opencv-python numpy pandas mediapipe picamera2 pygame tensorflow` (필요에 따라 다른 패키지도 설치)
### 모델파일 준비
- `model.pkl` : 얼굴 포즈 예측용 모델
- `model1.tflite` : 졸음 감지용 TensorFLow Lite 모델
- `power_alarm.wav` : 알람 사운드 파일
### 실행
- `python main.py`
- 실행 후, PiCamera2를 통해 실시간 영상이 표시되며 얼굴 포즈 및 졸음 상태에 따라 알람이 재생됩니다.

