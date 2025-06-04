import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import librosa
from tensorflow.keras.models import load_model

model = load_model("DFCC_pksong0517.keras")   # 만들어둔 model 가져오기

# eval_label 오름차순 정렬을 위한 숫자 부분 추출 함수
def extract_number_from_test_label(test_label):
    match = re.search(r'\d+', test_label)
    if match:
        return match.group(0)
    return None

# eval_labels 불러오기
eval_label = pd.read_table('eval_label.txt', sep='- -', header=None, names=['voice_name','label'])
eval_label['voice_number'] = eval_label['voice_name'].apply(extract_number_from_test_label)
eval_label.sort_values('voice_number', inplace=True)
le = LabelEncoder()
eval_labels = le.fit_transform(eval_label['label'])

# eval_wav 파일 가져오기
def eval_dataset():
    folder = "eval"
    dataset = []
    for file in os.listdir(folder):
        if 'wav' in file:
            abs_file_path = os.path.join(folder,file)
            data, sr = librosa.load(abs_file_path, sr = 16000)   # data = 진폭값, sr = sample_rate = 16,000(초당 샘플 갯수)
            dataset.append([data, file])

    print("eval_Dataset 생성 완료")
    return pd.DataFrame(dataset,columns=['data', 'file'])

eval_wav = eval_dataset()

# train & test 때 사용했던 max_length(=111872)로 eval_wav Padding
def set_length_with_padding(data, max_length):
    result = []
    for i in data:
        padded_audio = librosa.util.fix_length(i, size=max_length)
        result.append(padded_audio)
    result = np.array(result)
    return result

eval_x = np.array(eval_wav.data)
eval_x = set_length_with_padding(eval_x, 111872)
print("padding 완료")

# MFCC 특징 추출하기
def preprocess_dataset(data):
    mfccs = []
    for i in data:
        extracted_features = librosa.feature.mfcc(y=i,sr=16000,n_mfcc=40)
        mfccs.append(extracted_features)

    return mfccs

eval_mfccs = preprocess_dataset(eval_x)
eval_mfccs = np.array(eval_mfccs)
eval_mfccs = eval_mfccs.reshape(-1, eval_mfccs.shape[1], eval_mfccs.shape[2], 1)

# eval_accuracy 측정
eval_loss, eval_acc = model.evaluate(eval_mfccs, eval_labels)
print(f"테스트 정확도: {eval_acc: 3f}")