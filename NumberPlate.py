import cv2
import numpy as np
import pytesseract
import time
from PIL import Image

# --Read Input Image-- (이미지 불러오기)

src = cv2.imread("9999.jpg", cv2.IMREAD_COLOR) # 이미지 불러오기

'''
dst = src.copy()           #이미지영역을 반으로 자르기(번호판 인식률 속도를 높이기 위함)
dst = src[480:960, 50:670]

cv2.imshow("half img", dst)
cv2.waitKey(0)
'''

prevtime = time.time() # 걸린 시간 체크하는 함수

# 변수 선언
height, width, channel = src.shape # 이미지에 대한 값을 가질 변수

numcheck = 0 # 반복문에서 번호판 문자열 검사할 변수
charsok = 0 # 반복문에서 번호판 글자를 제대로 읽었는지 검사할 변수
add_w_padding, add_h_padding = 0, 0 # 추가할 padding값을 가질 변수
w_padding_max, h_padding_max = 0, 0 # 일정한 padding값을 가지게되었을때 반복문을 제어할 변수

# --Convert Image to Grayscale-- (이미지 흑백변환)

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) # 이미지 흑백변환

# --Maximize Contrast(Optional)-- (흑백대비 최대화)

structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

# --Adaptive Thresholding-- (가우시안블러(이미지 노이즈 제거) 및 쓰레시 홀딩)

img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0) # GaussianBlur 적용

img_thresh = cv2.adaptiveThreshold( # adaptiveThreshold 적용
    img_blurred,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=9
)

# --Find Contours-- (윤곽선 찾기)

contours, hierarchy = cv2.findContours( # opencv의 findContours를 이용하여 contours에 저장
    img_thresh,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE
)

temp_result = np.zeros((height, width, channel), dtype=np.uint8) # numpy.zeros를 이용하여 윤곽선 범위 저장

cv2.drawContours(temp_result, contours, -1, (255, 255, 255)) # 윤곽선 그리기


# --Prepare Data-- (데이터 비교하기, 글자영역으로 추정되는 rectangle 그리기)

temp_result = np.zeros((height, width, channel), dtype=np.uint8) # drawContours를 이용해 그린 윤곽선에 다시 numpy.zeros를 이용해 다시 윤곽선 범위 저장 (안하면 윤곽선 좀 남아있음)


contours_dict = [] # contour 정보를 모두 저장받을 리스트 변수

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour) # 위치 높낮이 데이터 정보 저장
    cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=2) # 윤곽선을 감싸는 사각형 구하기

    # insert to dict
    contours_dict.append({ # contour 정보를 모두 저장
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2),
        'cy': y + (h / 2)
    })

# --Select Candidates by Char Size-- (글자 같은 영역 찾기)

MIN_AREA = 80 # 윤곽선의 가운데 렉트 최소 넓이 80
MIN_WIDTH, MIN_HEIGHT = 2, 8 # 바운드 렉트의 최소 너비와 높이는 2, 8
MIN_RATIO, MAX_RATIO = 0.25, 1.0 # 바운드 렉트의 비율 가로 대비 세로 비율 최솟값 0.25, 최댓값 1.0

possible_contours = [] # 글자로 예상되는 contour들을 저장받을 리스트 변수

cnt = 0 # count 변수
for d in contours_dict: # contours_dict에 저장된 것을 조건에 맞다면 possible_contours에 append
    area = d['w'] * d['h']
    ratio = d['w'] / d['h']

    if area > MIN_AREA \
            and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
            and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt
        cnt += 1
        possible_contours.append(d)

# visualize possible contours
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for d in possible_contours:
    #     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                  thickness=2) # 글자로 예상되는 영역만 rectangle 그리기

# --Select Candidates by Arrangement of Contours-- (글자의 연속성(번호판으로 예상되는 영역) 찾기)

MAX_DIAG_MULTIPLYER = 4.7  # 5 contour와 contour의 사이의 길이 (값계속 바꿔가면서 테스트 해야함)
MAX_ANGLE_DIFF = 13  # 12.0 첫번째 contour와 두번째 contour의 직각 삼각형의 앵글 세타각도
MAX_AREA_DIFF = 0.5  # 0.5  면적의 차이
MAX_WIDTH_DIFF = 0.8 # 0.8 contour 간의 가로길이 차이
MAX_HEIGHT_DIFF = 0.2 # 0.2 contour 간의 세로길이 차이
MIN_N_MATCHED = 4  # 3 글자영역으로 예측된 것의 최소 갯수 (ex 3개이상이면 번호판일 것)


def find_chars(contour_list): # 재귀함수로 번호판 후보군을 계속 찾음
    matched_result_idx = [] # 최종 결과값의 인덱스를 저장

    for d1 in contour_list: # 컨투어(d1, d2)를 서로 비교
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']])) # d1과 d2거리를 계산
            if dx == 0: # dx의 절댓값이 0이라면 (d1과 d2의 x값을 갖고 있다면)
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx)) # 아크탄젠트 값을 구함 (라디안)
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h']) # 면적의 비율
            width_diff = abs(d1['w'] - d2['w']) / d1['w'] # 너비의 비율
            height_diff = abs(d1['h'] - d2['h']) / d1['h'] # 높이의 비율

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                    and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                    and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx']) # 설정한 파라미터 기준에 맞는 값들의 인덱스만 append

        # append this contour
        matched_contours_idx.append(d1['idx']) # d1을 빼먹고 넣었으므로 d1도 넣어줌

        if len(matched_contours_idx) < MIN_N_MATCHED: # 예상한 번호판의 최소 갯수가 맞지 않다면 continue
            continue

        matched_result_idx.append(matched_contours_idx) # 최종후보군으로 넣음 append

        unmatched_contour_idx = [] # 최종 후보군이 아닌 것들도 아닌 것들끼리 한번 더 비교
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx: # matched_contour_idx가 아닌 것들
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx) # numpy.take를 이용해서 unmathced_contour에 저장

        # recursive
        recursive_contour_list = find_chars(unmatched_contour) # 다시 돌려봄

        for idx in recursive_contour_list:
            matched_result_idx.append(idx) # 최종 결과값을 mathced_result_idx에 다시 저장

        break

    return matched_result_idx


result_idx = find_chars(possible_contours)

matched_result = [] # 예상되는 번호판 contour정보를 담을 리스트 변수
for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))

# visualize possible contours (번호판 contour 그리기)
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for r in matched_result: # 번호판으로 예상되는 역역을 그림
    for d in r:
        #         cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                      thickness=2)


# --Rotate Plate Images-- (이미지 회전)

plate_imgs = [] # 번호판 이미지를 담을 리스트 변수
plate_infos = [] # 번호판 정보를 담을 리스트 변수

longest_idx, longest_text = -1, 0 # idx값 초기화
plate_chars = [] # 번호판 리스트 변수

while charsok == 0: # 번호판 글자로 예상되는 값이 나올 때까지 반복
    PLATE_WIDTH_PADDING = 1.267 + add_w_padding  # 가로 패딩 값 예제 디폴트는 1.3
    PLATE_HEIGHT_PADDING = 1.51 + add_h_padding  # 세로 패딩 값 예제 디폴트는 1.5
    MIN_PLATE_RATIO = 3 #3 최소 번호판 비율
    MAX_PLATE_RATIO = 10 #10 최대 번호판 비율

    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']

        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy'] #번호판의 간격을 삼각형을 기준으로 세타 값을 구함
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )

        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus)) # 라디안 값을 구해서 각도로 바꿈

        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0) # 로테이션 이미지 구하기

        img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height)) # 이미지 변형

        img_cropped = cv2.getRectSubPix( # 회전된 이미지에서 원하는 부분만 자름
            img_rotated,
            patchSize=(int(plate_width), int(plate_height)),
            center=(int(plate_cx), int(plate_cy))
        )

        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[
            0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO: # 번호판 비율이 맞지 않다면 continue
            continue

        plate_imgs.append(img_cropped) # plate_imgs에 append

        plate_infos.append({ # plate_infos에 append
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })

    cv2.imwrite('08.jpg', img_cropped) #사진 돌려서 각도 맞추기(Rotate)

    # --Another Thresholding to Find Chars--

    for i, plate_img in enumerate(plate_imgs):
        if numcheck > 3: # 예상되는 번호판 영역에서 문자열을 검사해 숫자 3개가 넘는다면(번호판일 확률이 높다면)
            break

        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
        _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU) # 쓰레시홀딩

        # find contours again (same as above)
        contours, hierarchy = cv2.findContours(plate_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # contour 다시 찾기

        plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
        plate_max_x, plate_max_y = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour) # for문을 돌려 boundingRect를 다시 구함

            area = w * h # 면적
            ratio = w / h # 비율

            if area > MIN_AREA \
                    and w > MIN_WIDTH and h > MIN_HEIGHT \
                    and MIN_RATIO < ratio < MAX_RATIO: # 설정한 기준(파라미터)에 맞는지 다시 확인
                if x < plate_min_x: # x, y의 최댓값,최소값을 구함
                    plate_min_x = x
                if y < plate_min_y:
                    plate_min_y = y
                if x + w > plate_max_x:
                    plate_max_x = x + w
                if y + h > plate_max_y:
                    plate_max_y = y + h

        img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x] # 이미지를 번호판 부분만 잘라내기

        img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0) # GaussianBlur(노이즈 제거)
        _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU) # 쓰레시홀딩 한번 더
        img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, # 이미지에 패딩(여백)을 줌
                                        value=(0, 0, 0)) # 검은색

        cv2.imwrite('00.jpg', img_result)
        chars = pytesseract.image_to_string(Image.open('00.jpg'), config='--psm 7 --oem 0', lang='kor') # 저장한 이미지를 불러 pytesseract로 읽음
        nowtime = time.time()
        sec = nowtime - prevtime
        print("걸린시간 %0.5f" % sec)
        print("이미지 불러 온 후 글자 : " + chars)

        result_chars = ''  # 번호판 인식 문자 정보를 담을 변수
        has_digit = False
        for c in chars:  # 판독해서 특수문자를 제외한 한글 문자와 숫자 넣기
            if ord('가') <= ord(c) <= ord('힣') or c.isdigit():  # 숫자나 한글이 포함되어 있는지
                if c.isdigit():
                    has_digit = True  # 숫자가 하나라도 있는지
                result_chars += c
        plate_chars.append(result_chars) # 결과 result_chars를 plate_chars에 append

        for n in range(len(result_chars)):  # 번호판 형식이 맞는지 다시한번 검사 및 문자열 자르기
            if len(result_chars) < 7:  # 번호판 길이가 7자리(번호판의 최소 길이는 7자리)보다 짧다면
                break
            elif result_chars[0].isdigit() == False:  # 첫문자가 문자라면(숫자가 아니라면) 자르기
                result_chars = result_chars[1:result_chars.__len__()]
            elif result_chars[len(result_chars) - 1].isdigit() == False:  # 마지막 문자가 한글데이터라면(숫자가 아니라면) 자르기
                result_chars = result_chars[0:(result_chars.__len__() - 1)]

        for j in range(len(result_chars)):  # 번호판의 배열이 나오는지를 검사 ex) 12가3456(7자리번호판) or 123가4567(8자리번호판)
            if len(result_chars) < 7:  # 결과길이가 7자리(번호판의 최소 길이는 7자리)보다 짧다면
                break
            elif (j == 2 and result_chars[j].isdigit() == True) and result_chars[j+1].isdigit() == True:  # 번호판의 3번째와 4번째가 동시에 숫자라면(글자가 아니라면)
                break
            elif (j != 2 and j != 3) and result_chars[j].isdigit() == False:  # 번호판의 3,4번째(글자영역)가 아닌데 문자라면
                break
            elif (j == 2 and result_chars[j].isdigit() == False) and result_chars[j+1].isdigit() == False:  # 번호판의 3,4번째자리가 둘 다 문자라면
                break
            if 6 <= j and result_chars[j].isdigit() == True:  # 6번째까지 숫자자리에 문자가 없고 7번째 영역이 숫자라면 번호판일 것
                charsok = 1  # 반복문을 멈춤
                break

        if has_digit and len(result_chars) > longest_text:  # 조건을 만족하면
            longest_idx = i  # 가장 긴 값을 인덱스로 줌

        for numch, in result_chars:  # 문자열 검사를 통해 숫자가 3개 이상이라면 번호판일 확률이 높으므로 이 plate_imgs는 번호판일 것임 그러므로 패딩값을 조절하면 되기에 이미지는 고정할 것
            if numch.isdigit() == True:
                numcheck += 1

# --Result-- (결과값)

    info = plate_infos[longest_idx]  # 번호판 좌표 정보 담기
    chars = plate_chars[longest_idx]  # 번호판 문자열 정보 담기

    # 가로 패딩값을 0.1씩 늘림 -> 가로를 초기화 후 세로 패딩값을 0.1씩 늘림 -> 가로 세로 패딩값을 0.1씩 늘림 모두 0.6이 되면 프로그램 종료
    if add_w_padding <= 0.6 and w_padding_max == 0:  # w패딩이 0.5보다 작다면 (가로 패딩만 먼저 늘려보기)
        add_w_padding += 0.1  # w패딩을 0.1씩 증가

    elif w_padding_max == 1 and add_h_padding <= 0.6 and h_padding_max == 0:  # w패딩이 0.5를 찍고 h패딩이 0.5보다 작다면
        add_w_padding = 0  # w패딩을 다시 Default값으로 (세로 패딩만 늘려보기)
        add_h_padding += 0.1  # h패딩을 0.1씩 증가

    if add_w_padding == 0.6:  # 0.6까지 늘어났다면
        w_padding_max = 1
    if add_h_padding == 0.6:  # 0.6까지 늘어났다면
        h_padding_max = 1
        add_w_padding = 0
        add_h_padding = 0

    if w_padding_max == 1 and h_padding_max == 1:  # 너비높이 0.1씩 증가시키기
        add_w_padding += 0.1
        add_h_padding += 0.1
        if add_w_padding == 0.6 and add_h_padding == 0.6:  # 패딩값을 너비 높이 다 0.6씩 늘렸다면(번호판을 못 찾았다면)
            break
    # 초기화
    numcheck = 0
    plate_imgs = []
    plate_chars = []

print("최종 값 : " + chars)

img_out = src.copy()

cv2.rectangle(img_out, pt1=(info['x'], info['y']), pt2=(info['x'] + info['w'], info['y'] + info['h']),
              color=(255, 0, 0), thickness=2)  # 원본 이미지에 번호판 영역 그리기

cv2.imwrite('010.jpg', img_out)  # 원본 이미지에서 번호판 영역 그린 이미지