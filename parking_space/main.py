import cv2
import matplotlib.pyplot as plt
import numpy as np
from util import get_parking_spots_bboxes, empty_or_not

# 두 이미지의 평균 차이를 계산하는 함수
def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

# 마스크 이미지와 비디오 파일 경로 설정
mask = './mask_1920_1080.png'
video_path = "./data/parking_1920_1080.mp4"

# 마스크 이미지를 그레이스케일로 읽기
mask = cv2.imread(mask, 0)
# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 마스크 이미지에서 연결된 컴포넌트와 통계 정보 추출
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

# 주차 공간의 바운딩 박스 좌표를 가져옴
spots = get_parking_spots_bboxes(connected_components)

# 주차 공간 상태와 차이값을 저장할 리스트 초기화
spots_status = [None for j in spots]
diffs = [None for j in spots]

# 이전 프레임을 저장할 변수 초기화
previous_frame = None
# 현재 프레임 번호 초기화
frame_nmr = 0
# 프레임 간격 설정
step = 30

while True:
    # 비디오에서 프레임 읽기
    ret, frame = cap.read()

    # 프레임을 읽지 못하면 루프 종료
    if not ret:
        break

    # 매 'step' 프레임마다 주차 공간 상태를 업데이트
    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_index, spot in enumerate(spots):
            x1, y1, w, h = spot
            # 현재 프레임에서 주차 공간 영역을 잘라냄
            spot_crop = frame[y1:y1+h, x1:x1+w, :]
            # 이전 프레임과의 차이 계산
            diffs[spot_index] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

        # print([diffs[j] for j in np.argsort(diffs)][::-1])
        # plt.figure()
        # plt.hist([diffs[j] / np.amax(diffs) for j in np.argsort(diffs)][::-1], bins=20)
        # if frame_nmr == 300:
        #     plt.show()

    if frame_nmr % step == 0:
        if previous_frame is None:
            # 이전 프레임이 없으면 모든 주차 공간을 검사
            arr_ = range(len(spots))
        else:
            # 차이가 큰 주차 공간만 선택
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]

        for spot_index in arr_:
            spot = spots[spot_index]
            x1, y1, w, h = spot
            # 현재 프레임에서 주차 공간 영역을 잘라냄
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            # 주차 공간이 비어있는지 확인
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_index] = spot_status

    if frame_nmr % step == 0:
        # 이전 프레임을 현재 프레임으로 업데이트
        previous_frame = frame.copy()

    for spot_index, spot in enumerate(spots):
        spot_status = spots_status[spot_index]
        x1, y1, w, h = spots[spot_index]
        # 주차 공간 상태에 따라 사각형 색상 설정 (초록색: 비어있음, 빨간색: 차 있음)
        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    # 화면에 주차 공간 정보 표시
    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))), (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 프레임을 윈도우에 표시
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow("Frame", frame)

    # 프레임 번호 증가
    frame_nmr += 1

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 비디오 캡처 객체와 모든 윈도우 해제
cap.release()
cv2.destroyAllWindows()