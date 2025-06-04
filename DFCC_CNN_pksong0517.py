import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import librosa   # 오디오 전처리를 위한 라이브러리 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# train_label 불러오기
train_label = pd.read_table('label/train_label.txt', sep='- -', header=None, names=['voice_name','label'])
le = LabelEncoder()
train_labels = le.fit_transform(train_label['label'])

# test_label 오름차순 정렬을 위한 숫자 부분 추출 함수 
def extract_number_from_test_label(test_label):
    match = re.search(r'\d+', test_label)
    if match:
        return match.group(0)
    return None

# test_label 불러오기
test_label = pd.read_table('label/test_label.txt', sep='- -', header=None, names=['voice_name','label'])   # test_data도 마찬가지로 가져와 one-hot encoding
test_label['voice_number'] = test_label['voice_name'].apply(extract_number_from_test_label)
test_label.sort_values('voice_number', inplace=True)
test_labels = le.transform(test_label['label'])

# wav 파일 불러오기
def train_dataset():
    folder = "train"
    dataset = []
    for file in sorted(os.listdir(folder)):
        if 'wav' in file:
            abs_file_path = os.path.join(folder,file)
            data, sr = librosa.load(abs_file_path, sr = 16000)   # data = 진폭값, sr = sample_rate = 16,000(초당 샘플 갯수)
            dataset.append([data, file])

    print("Train_Dataset 생성 완료")
    return pd.DataFrame(dataset,columns=['data', 'file'])

def test_dataset():
    folder = "test"
    dataset = []
    for file in os.listdir(folder):
        if 'wav' in file:
            abs_file_path = os.path.join(folder,file)
            data, sr = librosa.load(abs_file_path, sr = 16000)
            dataset.append([data])

    print("Test_Dataset 생성 완료")
    return pd.DataFrame(dataset,columns=['data'])

train_wav = train_dataset()
test_wav = test_dataset()

# audio 파일 증강하기(Augmentation)
def add_noise(train):   # 원본 음성에 noise를 추가해주는 함수
    Augmentation_dataset = []
    for i in range(len(train)):
        data = train.data[i]
        label = train.file[i]

        noise_level=0.005
        noise = np.random.randn(len(data))  # 정규 분포를 따르는 노이즈 생성
        augmented_data = data + noise_level * noise  # 데이터에 노이즈 추가
        augmented_data = np.clip(augmented_data, -1, 1)  # 값이 -1과 1 사이로 유지되도록 함

        augmetation_label = label.replace('.wav', '_noise.wav')  # label 이름 수정
        Augmentation_dataset.append([augmented_data, augmetation_label])

    print("noise_dataset 생성 완료")
    return pd.DataFrame(Augmentation_dataset,columns=['data', 'file'])

noise_wav = add_noise(train_wav)
augmented_wav = pd.concat([train_wav, noise_wav], axis=0)
augmented_labels = np.tile(train_labels, 2) # label도 그대로 2배 증폭

# wav 파일 음성 가장 긴 길이를 기준으로 padding 하기
train_x = np.array(augmented_wav.data)
test_x = np.array(test_wav.data)

# 가장 긴 길이 계산
train_max_length = np.max([len(x) for x in train_x])
test_max_length = np.max([len(x) for x in test_x])
max_length = np.max([train_max_length, test_max_length])

def set_length_with_padding(data, max_length):
    result = []
    for i in data:
        padded_audio = librosa.util.fix_length(i, size=max_length)
        result.append(padded_audio)
    result = np.array(result)
    return result

train_x = set_length_with_padding(train_x, max_length)
test_x = set_length_with_padding(test_x, max_length)
print("padding 완료")

# MFCC 특징 추출하기
def preprocess_dataset(data):
    mfccs = []
    for i in data:
        extracted_features = librosa.feature.mfcc(y=i,sr=16000,n_mfcc=40)
        mfccs.append(extracted_features)

    return mfccs

train_mfccs = preprocess_dataset(train_x)
train_mfccs = np.array(train_mfccs)
train_mfccs = train_mfccs.reshape(-1, train_mfccs.shape[1], train_mfccs.shape[2], 1)

# train_data와 val_data 나누기
X_train, X_val, y_train, y_val = train_test_split(train_mfccs, augmented_labels,
                                                  test_size = 0.2,   # val_data 0.2로 8:2로 나눌 것임
                                                  stratify = augmented_labels,   # Train 데이터와 Test 데이터에 Real과 Fake가 골고루 섞이도록 나눔
                                                  random_state = 42)   # 분할하는 데이터를 섞을 때 기준이 되는 값으로, 이 값을 고정시켜줘야 데이터셋이 변경되지 않아 정확한 비교 가능

# CNN 학습
def Build_DFCC_CNN(input_shape=train_mfccs[0].shape):
    model = Sequential()

    # Conv 1
    model.add(Conv2D(16, kernel_size=(2,2), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Conv 2
    model.add(Conv2D(32, kernel_size=(2,2), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

     # Conv 3
    model.add(Conv2D(64, kernel_size=(2,2), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Flatten + DNN
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model

model = Build_DFCC_CNN()

model.compile(loss="binary_crossentropy",   # 이진 분류이므로 binary_crossentropy 사용
              optimizer="adam",             # optimizer로는 최신이며 언제나 안정적인 성능을 뽑아내는 Adam 사용
              metrics=["accuracy"])

callbacks_list = [
    EarlyStopping(
        monitor="val_accuracy",   # model의 val_accuracy를 모니터링 하다가
        patience=2,   # 2번의 epoch 동안 val_accuracy가 증가하지 않으면 훈련을 중지 (epoch 다 돌리지 않음)
    ),
    ModelCheckpoint(
        filepath="DFCC_pksong0517.keras",   # model 파일의 저장 경로
        monitor="val_loss",
        save_best_only=True,   # val_loss가 좋아지지 않으면 모델 저장 파일을 덮어쓰지 않겠다는 뜻입니다. 즉 훈련하는 동안 가장 좋은 모델이 저장됩니다.
    )
]

history = model.fit(X_train, y_train,
                    epochs=50,
                    batch_size=32,   # 일단 국룰이 32로 잡아봄. 추후 성능에 따라 바꿔가며 실험에 볼 필요 있음
                    callbacks=callbacks_list,
                    validation_data=(X_val, y_val))

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()

test_mfccs = preprocess_dataset(test_x)
test_mfccs = np.array(test_mfccs)
test_mfccs = test_mfccs.reshape(-1, test_mfccs.shape[1], test_mfccs.shape[2], 1)

test_loss, test_acc = model.evaluate(test_mfccs, test_labels)
print(f"테스트 정확도: {test_acc: 3f}")   # 0603 음성 길이 통일에 padding + Conv4 적용 결과 정확도 0.9265(구글 코랩 0.932)