import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import librosa
#!pip install tensorflow
from tensorflow.keras.models import load_model

model = load_model("DFCC_pksong0517.keras")   # 만들어둔 model 가져오기

# test_label 오름차순 정렬을 위한 숫자 부분 추출 함수
def extract_number_from_test_label(test_label):
    match = re.search(r'\d+', test_label)
    if match:
        return match.group(0)
    return None

# test_labels 불러오기
test_label = pd.read_table('test_label.txt', sep='- -', header=None, names=['voice_name','label'])
test_label['voice_number'] = test_label['voice_name'].apply(extract_number_from_test_label)
test_label.sort_values('voice_number', inplace=True)
le = LabelEncoder()
test_labels = le.fit_transform(test_label['label'])

# test_wav 파일 가져오기
def test_dataset():
    folder = "test"
    dataset = []
    for file in sorted(os.listdir(folder)):
        if 'wav' in file:
            abs_file_path = os.path.join(folder,file)
            data, sr = librosa.load(abs_file_path, sr = 16000)   # data = 진폭값, sr = sample_rate = 16,000(초당 샘플 갯수)
            dataset.append([data, file])

    print("test_Dataset 생성 완료")
    return pd.DataFrame(dataset,columns=['data', 'file'])

test_wav = test_dataset()

# train & test 때 사용했던 max_length(=111872)로 test_wav Padding
def set_length_with_padding(data, max_length):
    result = []
    for i in data:
        padded_audio = librosa.util.fix_length(i, size=max_length)
        result.append(padded_audio)
    result = np.array(result)
    return result

test_x = np.array(test_wav.data)
test_x = set_length_with_padding(test_x, 111872)
print("padding 완료")

# MFCC 특징 추출하기
def preprocess_dataset(data):
    mfccs = []
    for i in data:
        extracted_features = librosa.feature.mfcc(y=i,sr=16000,n_mfcc=40)
        mfccs.append(extracted_features)

    return mfccs

test_mfccs = preprocess_dataset(test_x)
test_mfccs = np.array(test_mfccs)
test_mfccs = test_mfccs.reshape(-1, test_mfccs.shape[1], test_mfccs.shape[2], 1)

# 테스트 레이블 전처리 함수
def extract_number_from_test_label(name):
    """파일명에서 숫자 추출 (예: KDF_E_1004 → 1004)"""
    return int(name.split('_')[-1].split('.')[0])

# 테스트 레이블 불러오기
test_label = pd.read_table(
    'label/test_label.txt',
    sep='- -',
    engine='python',
    header=None,
    names=['voice_name','label']
)

# 파일명 전처리 (YSG 등 프리픽스 제거)
test_label['voice_name'] = test_label['voice_name'].str[4:].str.strip()
test_label['voice_number'] = test_label['voice_name'].apply(extract_number_from_test_label)
test_label.sort_values('voice_number', inplace=True)

# 레이블 인코딩
le = LabelEncoder()
test_labels = le.fit_transform(test_label['label'])

# 테스트 데이터 경로 설정
test_data_path = 'test'

# 데이터 저장 리스트 초기화
test_file_names = []
test_x = []
test_y = []

# 음성 데이터 처리 파이프라인
for idx, row in test_label.iterrows():
    file_name = row['voice_name']
    label_encoded = test_labels[idx]  # 인코딩된 라벨 사용[6]

    # 실제 오디오 파일 경로 생성
    wav_path = os.path.join(test_data_path, file_name)

    # 파일 존재 여부 확인 (검색 결과[5][9] 반영)
    if not os.path.exists(wav_path):
        print(f"경고: {wav_path} 파일 없음")
        continue

    try:
        # 오디오 처리 (메모리 엔트리[10]의 CNN 입력 요구사항 반영)
        y, sr = librosa.load(wav_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        mfcc_combined = np.concatenate((mfcc, mfcc_delta, mfcc_delta2), axis=0)
        mfcc_mean = np.mean(mfcc_combined, axis=1)

        test_x.append(mfcc_mean)
        test_y.append(label_encoded)
        test_file_names.append(file_name)

    except Exception as e:
        print(f"{file_name} 처리 실패: {str(e)}")

from tensorflow.keras.utils import to_categorical
# 리스트를 numpy 배열로 변환
X_test = np.array(test_x)

# 라벨 인코딩 (0: fake, 1: real)
le = LabelEncoder()
y_test = le.fit_transform(test_y)

# 예: 파일 이름과 데이터, 라벨을 묶어서 정렬
combined = list(zip(test_file_names, test_mfccs, test_labels))
combined.sort(key=lambda x: x[0])  # 파일 이름 기준 정렬

# 다시 분리
sorted_file_names, sorted_mfccs, sorted_labels = zip(*combined)

# numpy 배열로 변환
sorted_mfccs = np.array(sorted_mfccs)
sorted_labels = np.array(sorted_labels)

y_pred = model.predict(sorted_mfccs)
y_pred_probs = y_pred.flatten()

from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(sorted_labels, y_pred_probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
optimal_threshold = thresholds[np.argmax(f1_scores)]

y_pred_classes = (y_pred_probs > optimal_threshold).astype(int)

from sklearn.metrics import accuracy_score
acc = accuracy_score(sorted_labels, y_pred_classes)
print("정렬 후 정확도:", acc)

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# 혼동 행렬(Confusion Matrix) 시각화
cm = confusion_matrix(sorted_labels, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# label을 sort시킴
# 파일 읽기
with open('label/test_label.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 번호 추출 함수
def extract_number(line):
    # 예시: YSG KDF_E_1004.wav - - Real
    m = re.search(r'_(\d+)\.wav', line)
    return int(m.group(1)) if m else float('inf')

# 정렬
sorted_lines = sorted(lines, key=extract_number)

# 저장
with open('label/test_label_sorted.txt', 'w', encoding='utf-8') as f:
    f.writelines(sorted_lines)

# test_label_sorted.txt 기준으로 예측 파일 생성
sorted_test_label = pd.read_table(
    'label/test_label_sorted.txt',
    sep='- -',
    engine='python',
    header=None,
    names=['voice_name','label']
)

#test_result 또한 sort시킴
with open('pksong0517_test_result_sorted.txt', 'w') as f:
    for idx, row in sorted_test_label.iterrows():
        # 프리픽스 제거된 파일명 사용
        file_name = row['voice_name'].split(' ', 1)[1].strip()
        pred_label = 'Real' if y_pred_classes[idx] == 1 else 'Fake'
        f.write(f"{file_name} {pred_label}\n")

#sort된 label을 통해 eval.pl 실행
!perl eval.pl pksong0517_test_result_sorted.txt test_label_sorted.txt
