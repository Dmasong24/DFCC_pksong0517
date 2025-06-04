import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import librosa
from tensorflow.keras.models import load_model

model = load_model("DFCC_pksong0517.keras")   # 만들어둔 model 가져오기

# test_label 오름차순 정렬을 위한 숫자 부분 추출 함수
def extract_number_from_test_label(test_label):
    match = re.search(r'\d+', test_label)
    if match:
        return match.group(0)
    return None

# test_labels 불러오기
test_label = pd.read_table('label/test_label.txt', sep='- -', header=None, names=['voice_name','label'])
test_label['voice_number'] = test_label['voice_name'].apply(extract_number_from_test_label)
test_label.sort_values('voice_number', inplace=True)
le = LabelEncoder()
test_labels = le.fit_transform(test_label['label'])

# test_wav 파일 가져오기
def test_dataset():
    folder = "test"
    dataset = []
    for file in os.listdir(folder):
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

# test_accuracy 측정
test_loss, test_acc = model.evaluate(test_mfccs, test_labels)
print(f"테스트 정확도: {test_acc: 3f}")

# 결과값 저장
with open('pksong0517_test_result.txt', 'w') as f:
    f.write(f"Loss: {test_loss:.3f}\n")
    f.write(f"Accuracy: {test_acc:.3f}\n")