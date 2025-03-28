# 필요한 모듈들을 불러옵니다.
import os                       # 운영체제 관련 기능을 사용하기 위한 모듈
import sys                      # 시스템 관련 기능 (예: 프로그램 종료)을 사용하기 위한 모듈
import cv2                      # OpenCV: 영상 및 이미지 처리를 위한 라이브러리
from argparse import ArgumentParser, SUPPRESS  # 명령줄 인자 처리를 위한 라이브러리
from pathlib import Path        # 파일 경로를 객체로 다루기 위한 라이브러리

# OTX API 관련 코드는 현재 사용하지 않으므로 제거하였습니다.
# 기존 OTX API 관련 코드:
# from otx.api.usecases.exportable_code.demo.demo_package import (
#     AsyncExecutor,
#     ChainExecutor,
#     ModelContainer,
#     SyncExecutor,
#     create_visualizer,
# )
# os.environ["FEATURE_FLAGS_OTX_ACTION_TASKS"] = "1"

# 명령줄 인자를 처리하는 함수입니다.
def build_argparser():
    parser = ArgumentParser(add_help=False)  # 기본 도움말 옵션 없이 ArgumentParser 객체 생성
    args = parser.add_argument_group("Options")  # "Options" 그룹 생성하여 옵션들을 모음
    args.add_argument("-i", "--input", required=True,
                      help="입력 비디오 파일 경로 또는 카메라 ID (예: 0)")  # 입력값 지정 (필수)
    # 모델과 디바이스 관련 옵션은 OTX API와 관련된 코드이므로 생략합니다.
    return parser

# 얼굴과 눈을 감지하여 조건에 맞으면 "Face Detected" 메시지를 화면에 출력하는 함수입니다.
def detect_face_and_eyes(frame, face_cascade, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 컬러 프레임을 그레이스케일로 변환
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 얼굴 감지 (스케일 팩터 1.3, 최소 이웃수 5)
    for (x, y, w, h) in faces:  # 감지된 각 얼굴에 대해
        roi_gray = gray[y:y+h, x:x+w]  # 얼굴 영역(ROI)을 추출
        eyes = eye_cascade.detectMultiScale(roi_gray)  # 얼굴 영역 내에서 눈 감지
        if len(eyes) >= 2:  # 두 개 이상의 눈이 감지되면
            cv2.putText(frame, "Face Detected", (x, y - 10),  # "Face Detected" 메시지 출력
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 얼굴 영역에 초록색 사각형 그리기
    return frame  # 처리된 프레임 반환

# 메인 함수: 명령줄 인자 처리, 영상 입력 및 얼굴 감지 실행을 담당합니다.
def main():
    args = build_argparser().parse_args()  # 명령줄 인자 파싱

    # 얼굴 및 눈 감지를 위한 Haar Cascade 분류기를 로드합니다.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    # 입력값이 숫자 문자열이면 카메라 ID로 처리, 아니면 비디오 파일 경로로 처리합니다.
    input_source = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(0)  # 영상 캡쳐 객체 생성

    # 영상이 열려있는 동안 프레임을 반복해서 읽어옵니다.
    while cap.isOpened():
        ret, frame = cap.read()  # 프레임 읽기
        if not ret:  # 더 이상 읽을 프레임이 없으면 종료
            break
        # 얼굴과 눈 감지 함수를 호출하여 프레임 처리
        frame = detect_face_and_eyes(frame, face_cascade, eye_cascade)
        cv2.imshow("Face Detection", frame)  # 결과 프레임을 창에 표시
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 종료
            break

    cap.release()             # 영상 캡쳐 객체 해제
    cv2.destroyAllWindows()   # 모든 OpenCV 창 닫기
    return 0

# 스크립트를 직접 실행할 경우 main() 함수를 호출합니다.
if __name__ == "__main__":
    sys.exit(main() or 0)
