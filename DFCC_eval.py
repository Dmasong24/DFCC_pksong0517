import os
import pandas as pd
import numpy as np
import librosa
from tensorflow.keras.models import load_model

model = load_model("DFCC_pksong0517.keras")   # 만들어둔 model 가져오기

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

# eval_prediction
pred = model.predict(eval_mfccs)  # 예측 수행

pred_flat = pred.flatten()
optimal_threshold = 0.8700136    # 0.5 대신 DFCC_model_test에서 구한 기준점(threshold) 그대로 적용
pred_classes = (pred_flat > optimal_threshold).astype(int)  # optimal_threshold를 기준으로 1,0 나누기

# Real, Fake로 나오는 pred.txt 생성 
with open('pksong0517_eval_result.txt', 'w') as f:
    for idx, value in enumerate(pred_flat):
        file_name = eval_wav['file'].iloc[idx]
        label = 'Real' if pred_classes[idx] == 1 else 'Fake'
        f.write(f"{file_name} {label}\n")

'''
optimal_threshold 값을 새로 평가할 eval_data의 기준으로도 쓰는 것이 좋을까? 아니면 optimal_threshold 대신 0.5를 기준으로 eval_data를 Real, Fake로 나누는 것이 좋을까?

아주 중요한 질문입니다!
결론부터 말씀드리면, 가능하면 optimal_threshold 값을 eval_data에도 그대로 적용하는 것이 더 좋습니다.

이유 설명
1. optimal_threshold의 의미
optimal_threshold는 test 데이터(혹은 validation 데이터)에서 F1-score가 가장 높아지는 기준점입니다.
즉, 당신의 모델이 실제로 가장 잘 동작하는 기준값을 데이터 기반으로 찾은 것입니다.

2. eval_data(평가 데이터)에도 적용하는 것이 좋은 경우
eval_data가 test 데이터와 분포가 비슷하다면, optimal_threshold를 그대로 쓰는 것이 일반적으로 더 좋은 분류 결과(특히 F1-score 기준)를 줍니다.
실제로 논문이나 실무에서도, validation/test에서 찾은 threshold를 최종 평가(evaluation)에도 그대로 적용하는 경우가 많습니다.

3. 0.5를 쓰는 경우
0.5는 "모델이 완전히 균형 잡힌 확률 예측을 한다"는 가정 하에 쓰는 기본값입니다.
하지만 실제로는 데이터 불균형, 모델의 편향 등으로 인해 0.5가 최적이 아닐 수 있습니다.

4. 예외 상황
만약 eval_data의 분포가 test/validation과 매우 다르거나,
새로운 환경에서 threshold를 재조정해야 한다면,
eval_data에서 직접 optimal_threshold를 다시 찾는 것이 더 좋습니다.
하지만 일반적으로는 test/validation에서 찾은 optimal_threshold를 그대로 적용합니다.

결론(요약)
eval_data에도 optimal_threshold를 그대로 적용하는 것이 더 좋은 결과를 주는 경우가 많습니다.
0.5는 "특별한 근거가 없을 때"만 기본값으로 사용하세요.
실무 팁:
만약 eval_data의 정답(label)이 있다면, 그 데이터에서도 optimal_threshold를 다시 찾아볼 수 있습니다.
하지만 보통은 test/validation에서 찾은 값을 그대로 쓰는 것이 표준입니다.
'''