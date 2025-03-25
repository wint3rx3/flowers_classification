# 1. 서론
본 연구의 목적은 꽃 이미지 데이터셋을 활용하여 꽃의 품종을 분류하는 모델을 개발하는 것이다. 이 데이터셋은 실제 환경에서 다양한 각도와 조명 조건으로 촬영된 꽃 이미지로 구성되어 있으며, 각 이미지는 '데이지(daisy)', '민들레(dandelion)', '장미(rose)', '해바라기(sunflower)', '튤립(tulip)'이라는 다섯 가지 주요 품종으로 분류된다. 본 연구를 통해 딥러닝 기반 이미지 분류 기술을 적용하여 꽃의 품종을 자동으로 식별할 수 있는 모델을 구축하고, 모델의 정확도와 성능을 평가하고자 한다.

꽃 품종 자동 분류 문제는 다음과 같이 정의할 수 있다. 입력으로 주어진 꽃 이미지를 받아들여, 해당 이미지가 위의 다섯 가지 품종 중 어떤 품종에 해당하는지 예측하는 문제이다. 이는 컴퓨터 비전 및 CNN(합성곱 신경망) 기반 머신러닝 기술을 활용하여 이미지에서 유의미한 특징을 추출하고, 이를 기반으로 품종을 예측하는 분류 모델을 개발하는 과정으로 이루어진다.

수작업을 통한 꽃 품종의 분류는 시간과 비용이 많이 소모되며, 전문가의 개입이 필요하므로 효율성이 낮다. 자동화된 품종 분류 시스템을 구축하면 이런 단점을 극복하여 시간과 비용을 절약하고, 전문적이지 않은 사용자라도 빠르고 정확한 품종 분류를 가능하게 할 수 있다. 또한, 화훼 농업 종사자 및 꽃 관련 산업에서는 품종별 관리와 판매 전략 수립이 중요하므로, 정확하고 신속한 품종 식별이 가능한 자동화 모델의 도입은 산업 효율성 증진에 크게 기여할 것으로 기대된다.

# 2. 데이터셋 설명
이번 모에서 사용한 데이터셋은 Kaggle 사이트에 업로드된 **'Flower Recognition' 데이터셋**이다. 이 데이터셋은 DPhi Data Sprint #25에서 제공한 꽃 이미지 데이터로, 데이지(daisy), 민들레(dandelion), 장미(rose), 해바라기(sunflower), 튤립(tulip) 등 다섯 가지 꽃 품종으로 구성된 이미지를 포함하고 있다. 총 5개 품종의 이미지가 포함된 학습 데이터셋과 모델 성능 평가를 위한 테스트 데이터셋으로 구분되어 있다. 테스트 데이터셋은 총 3035개의 이미지로 이루어져 있다. 모든 이미지는 JPEG 형식으로 제공된다.

```
Flower dataset
├── train
│   ├── daisy
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── dandelion
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── rose
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── sunflower
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── tulip
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
├── test
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── LICENSE.txt
├── Testing_set_flower.csv
└── sample_submission.csv
```

보다시피, train 데이터 셋에는 라벨링이 되어있지만, test 데이터에는 존재하지 않는다. 따라서, test 데이터 셋을 8:2로 train과 validation으로 분리하여 학습을 진행하였고, 최종적으로 test 데이터 셋에서 무작위의 이미지를 입력한 후, 그 결과를 육안으로 직접 확인하는 과정을 진행하였다. 각 클래스에는 일정하지 않은 크기의 꽃 이미지가 607개 씩 존재한다.

# 3. 모델 설명

이번 프로젝트에서 꽃 이미지 데이터셋의 다중 클래스 분석을 위해 레퍼런스로 제공된 VGG16, 그리고 이보다 심화된 CNN 구조를 가진 ResNet50과 EfficientNet, 마지막으로 최신 하이브리드 모델인 Swin Transformer 모델을 사용하였다.

참고자료로 주어진 VGG16은 2014년 이미지넷(ILSVRC) 대회에서 뛰어난 성능을 기록하여 널리 주목받았으며, 옥스퍼드 대학교의 VGG 연구팀에서 개발한 모델이다. 이 모델은 13개의 합성곱 층과 3개의 완전 연결층(Fully Connected Layer)을 포함한 총 16개의 층으로 구성된 합성곱 신경망으로, 모든 합성곱 층에 3x3 크기의 작은 필터를 사용하는 것이 특징이다. 이러한 구조는 깊은 층에서도 효과적인 이미지 특성 추출을 가능하게 한다.

VGG16을 본 연구의 사전 훈련 모델로 선택한 이유는 우선 이미지넷과 같은 대규모 이미지 분류 문제에서 이미 뛰어난 성능을 입증했기 때문에, 꽃 이미지 분류에도 높은 정확도를 기대할 수 있기 때문이다. 또한 VGG16은 간결하면서 직관적인 구조를 갖추고 있어 모델을 이해하고 해석하는 데 용이하며, 이는 모델의 구축부터 훈련, 평가 및 하이퍼파라미터 튜닝까지 제한된 시간 내에 수행해야 하는 프로젝트의 요구사항에 잘 맞는다. 더불어 VGG16은 이미지넷에서 미리 학습된 가중치를 제공하므로 이를 활용한 전이 학습(transfer learning)을 통해 상대적으로 작은 규모의 꽃 이미지 데이터셋에서도 효과적으로 높은 정확도를 달성할 수 있다는 장점이 있다.

ResNet50은 2015년 마이크로소프트 연구진에 의해 개발된 모델로, 기존 CNN 모델의 깊이가 깊어질수록 성능이 오히려 저하되는 문제인 기울기 소실(Vanishing Gradient) 문제를 해결하기 위해 잔차 학습(Residual Learning) 방식을 도입하였다. ResNet은 입력 신호를 출력 신호에 직접 연결하는 스킵 연결(skip connection)을 활용하여, 네트워크가 더 깊은 구조에서도 안정적인 학습을 가능하게 하였다. 그중에서도 ResNet50 모델은 50개의 층으로 구성되어 있으며, 깊은 네트워크 구조에도 불구하고 우수한 성능과 효율적인 학습을 동시에 이루었다.

EfficientNet은 2019년 구글에서 제안한 모델로, CNN의 성능을 높이기 위한 새로운 접근 방식을 제시하였다. 이 모델은 네트워크의 깊이(depth), 너비(width), 해상도(resolution)를 균형 있게 확장하는 복합 스케일링(Compound Scaling) 기법을 활용하여, 효율적이면서도 높은 정확도를 달성하였다. EfficientNet은 특히 파라미터 수와 연산 비용이 적은 상태에서도 우수한 성능을 제공하여 다양한 이미지 분류 태스크에서 널리 사용되고 있다.

마지막으로 Swin Transformer는 2021년 마이크로소프트에서 발표한 최신 모델로, 기존의 CNN과 달리 Transformer의 셀프 어텐션(self-attention)을 컴퓨터 비전에 적용하면서도 효율성을 높인 모델이다. Swin Transformer는 이미지를 작은 패치(patch) 단위로 나누고, 이러한 패치들을 윈도우(window) 단위로 그룹화하여 국소적인 셀프 어텐션(window-based self-attention)을 수행한다. 또한 윈도우를 점진적으로 이동(shifted window)시키는 방식으로 윈도우 간의 연결성을 확보하여 지역적 정보와 전역적 정보를 효율적으로 학습할 수 있게 한다. 이와 같은 설계로 인해 계산량은 줄이면서도 기존의 CNN 및 일반적인 Vision Transformer(ViT)에 비해 이미지 분류를 비롯한 다양한 비전 태스크에서 우수한 성능을 달성하였다.

이번 프로젝트에서는 Image Classification 태스크의 주요 모델을 사용해보며 꽃 이미지 데이터셋의 다중 클래스 분류에 가장 적합한 모델을 탐색하고, 각 모델의 특징을 실습해보는 시간을 가졌다.

# 4. 실험 방법

### 가. 필요한 라이브러리 설치

```python
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
```

### 나. 데이터셋 다운로드 준비

```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d imsparsh/flowers-dataset

import zipfile

zip_path = "flowers-dataset.zip"  # 다운로드된 파일 이름
extract_path = "./flowers-dataset"  # 압축을 풀 경로

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

TRAIN_DIR = '/content/flowers-dataset/train'
TEST_DIR  = '/content/flowers-dataset/test'

classes = sorted(os.listdir(TRAIN_DIR))
print(classes)
```

### 다. EDA

```python
# 각 클래스별 이미지 시각화 및 사이즈 확인
plt.figure(figsize=(15, 4))
for i, class_name in enumerate(classes):
    class_path = os.path.join(TRAIN_DIR, class_name)
    # 해당 클래스 폴더에서 첫 번째 이미지 선택
    image_files = os.listdir(class_path)
    first_image_path = os.path.join(class_path, image_files[0])
    img = load_img(first_image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.title(f"{class_name}\n{img_array.shape}")
    plt.axis('off')
plt.tight_layout()
plt.show()
```

![image](https://github.com/user-attachments/assets/e967d450-653f-484b-8392-536c660f2a2d)


```python
# 각 클래스별 이미지 갯수 확인
counts = []
for class_name in classes:
    class_path = os.path.join(TRAIN_DIR, class_name)
    count = len(os.listdir(class_path))
    counts.append(count)
    print(f"{class_name}: {count} images")
  
# 바 그래프 시각화 (영어 라벨, 각 막대 위에 수치 표시)
plt.figure(figsize=(16, 8))
colors = ['blue', 'orange', 'green', 'red', 'purple']
bars = plt.barh(classes, counts, color=colors)

plt.xlabel('Number of Images', fontsize=14)
plt.ylabel('Flower Breed', fontsize=14)
plt.title('Number of Images per Flower Breed', fontsize=16)

for bar in bars:
    width = bar.get_width()
    plt.text(width + 5, bar.get_y() + bar.get_height()/2, int(width),
             va='center', ha='left', fontsize=12)

plt.show()
```

![image](https://github.com/user-attachments/assets/44f63e99-1282-4eaa-bde3-236112c05634)


데이터 셋에서 손상된 파일을 제거한 후, 인위적으로 클래스 균형을 맞춰주는 작업을 진행하였다.

### 라. 데이터 증강

```python
aug_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

sample_class = classes[0]
sample_class_path = os.path.join(TRAIN_DIR, sample_class)
sample_image_name = os.listdir(sample_class_path)[0]
sample_image_path = os.path.join(sample_class_path, sample_image_name)
sample_img = load_img(sample_image_path, target_size=(224, 224))
sample_img_array = img_to_array(sample_img)

augmented_img_array = aug_gen.random_transform(sample_img_array)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(sample_img)
plt.title("Original")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(augmented_img_array.astype('uint8'))
plt.title("Augmented")
plt.axis('off')
plt.tight_layout()
plt.show()
```

![image](https://github.com/user-attachments/assets/b41485bc-c126-46fc-9048-3c7c0c22c62b)


데이터 증강을 보기 위한 샘플 테스트

### 마. VGG16을 사용한 모델 학습 및 평가

```python
NUM_CLASSES = 5
vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg16_base.trainable = False  # 사전 학습된 가중치는 고정

model = Sequential([
    vgg16_base,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

![image](https://github.com/user-attachments/assets/c40f8bf0-8f39-455c-903a-34805d0d05d4)



```python
BATCH_SIZE = 128
EPOCHS = 5
```

```python
train_datagen = ImageDataGenerator(
    rescale=1/255.0,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20%를 validation으로 분리
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='sparse',  # 정수형 라벨 사용
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    shuffle=False
)
```

```python
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# 모델 평가 (validation 데이터로 평가)
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")
```

Validation Loss: 0.5987
Validation Accuracy: 0.7938

```python
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
```

![image](https://github.com/user-attachments/assets/fd163968-eb50-4bd5-b2cd-70e795d158e3)


### 바. ResNet50과 EfficientNet B0을 사용한 모델 학습 및 평가

Fine-Tuned ResNet Validation Loss: 1.3396
Fine-Tuned ResNet Validation Accuracy: 0.5624

Fine-Tuned EfficientNet Validation Loss: 1.6076
Fine-Tuned EfficientNet Validation Accuracy: 0.2354

![image](https://github.com/user-attachments/assets/ac5bd395-4350-4153-b0b9-49bbe457541a)


### 사. Swin Transformer (CNN + Transformer)를 사용한 모델 학습 및 평가

```python
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

NUM_CLASSES = 5
EPOCHS = 5
LEARNING_RATE = 1e-3

SWIN_MODEL_URL = "https://tfhub.dev/sayakpaul/swin_tiny_patch4_window7_224/1"

inputs = Input(shape=(224, 224, 3))

swin_layer = hub.KerasLayer(
    SWIN_MODEL_URL,
    trainable=False,
    output_shape=[1000]
)

features = Lambda(lambda x: swin_layer(x),
                  output_shape=lambda input_shape: (input_shape[0], 1000))(inputs)
x = Dense(256, activation='relu')(features)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model_swin = Model(inputs=inputs, outputs=outputs)

model_swin.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model_swin.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,              # 학습률을 1/5로 감소
    patience=2,              # 2 epoch 동안 개선 없으면 감소
    min_lr=1e-6
)

history_swin = model_swin.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

loss_swin, accuracy_swin = model_swin.evaluate(validation_generator)
print(f"Swin Validation Loss: {loss_swin:.4f}")
print(f"Swin Validation Accuracy: {accuracy_swin:.4f}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history_swin.history['loss'], label='Train Loss')
plt.plot(history_swin.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Swin Transformer Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_swin.history['accuracy'], label='Train Accuracy')
plt.plot(history_swin.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Swin Transformer Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
```

Swin Validation Loss: 0.2887
Swin Validation Accuracy: 0.9015

![image](https://github.com/user-attachments/assets/542cca6c-432a-4209-b5e5-b23422b289e3)


### 아. test 데이터를 이용한 실전

```python
import random

test_files = [f for f in os.listdir(TEST_DIR)]
selected_files = random.sample(test_files, 5)

plt.figure(figsize=(20, 4))

for i, file in enumerate(selected_files):
    image_path = os.path.join(TEST_DIR, file)
    # 이미지 로드 및 전처리
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0  # 스케일링
    # 배치 차원 추가
    img_batch = np.expand_dims(img_array, axis=0)
    
    # 모델 예측 (모델은 이미 학습된 상태여야 함)
    pred = model.predict(img_batch)
    pred_index = np.argmax(pred, axis=1)[0]
    pred_label = classes[pred_index]
    
    # 이미지 시각화
    plt.subplot(1, 5, i+1)
    plt.imshow(img)
    plt.title(f"Predicted: {pred_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()
```

![image](https://github.com/user-attachments/assets/a7761ac7-168b-4892-949a-34cb8b0ab4e1)


# 5. 결과 및 분석

먼저 Kaggle에서 제공한 꽃 데이터셋을 기반으로 CNN을 활용한 분류 모델을 학습시켰다. 첫 번째로 선행 코드를 참고하여 VGG16 모델을 활용하여 학습한 결과, 약 79%의 검증 정확도와 약 0.48의 loss 값을 보였다. 베이스 모델을 이용하여 5 에포크만으로도 정확도가 비교적 준수하게 나온 점이 특이하게 기술할 만 하다.

성능을 더 향상시키고, 다양한 모델을 시도해 보기 위해 ResNet50과 EfficientNetB0를 새로 도입해보았다. 베이스 모델만 가지고 실험했을 때, ResNet은 5 에포크에서 50%의 검증 정확도를 보였고, EfficientNet은 상당히 처참한 기록을 보였다. 각각 에포크를 30으로 늘리고, 전이학습을 적용하였으며, 학습률도 조정해보았지만, VGG16보다 낮은 정확도와 loss값을 기록하며 기대에 미치지 못한 결과를 보였다. 성능 면에서 그나마 나은 지표를 보인, ResNet50의 추가적인 하이퍼 파라미터 조정이 필요해보인다.

Swin Transformer (Base) 모델은 앞선 실험들 보다 많은 연구가 진행 된 시점에서 발표된 모델인 만큼, 5 에포크만에 90% 이상의 정확도와 0.2의 loss 값을 보였다. 훈련 데이터와 검증 데이터 모두 3번의 에포크에서 좋은 성능이 나왔고,  성능 향상의 결과를 눈으로 확인해 볼 수 있었다. CNN 구조의 특징을 조금 가져온 트랜스포머 기반 모델이기에 선택했지만, YOLOv11의 결과도 궁금해지게 되는 대목이다.

# 6. 결론

본 연구에서는 꽃 이미지 데이터셋을 활용하여 꽃 품종 자동 분류 모델을 개발하기 위해 여러 딥러닝 아키텍처를 비교 분석하였다. 우선, VGG16 모델은 5 에포크 만에 약 79%의 검증 정확도와 0.48의 loss를 기록하며 비교적 우수한 성능을 보였다. 반면, ResNet50과 EfficientNetB0를 베이스 모델로 단독 실험한 결과, ResNet50은 5 에포크에서 약 50%의 정확도를 보였으며, EfficientNetB0는 더욱 낮은 성능을 나타냈다. 전이학습, 에포크 수 확대, 학습률 조정 등 추가적인 튜닝에도 불구하고 ResNet50은 최대 약 60%의 검증 정확도에 머물러 VGG16에 미치지 못하였다.

이와 달리, Swin Transformer (Base) 모델은 CNN의 특징 일부를 계승한 트랜스포머 기반 구조로, 5 에포크 만에 90% 이상의 정확도와 0.2의 loss를 기록하며 눈에 띄는 성능 향상을 보여주었다. 특히, 훈련 및 검증 데이터 모두 3 에포크에서 우수한 성능을 달성한 점은 해당 모델이 꽃 데이터셋에 효과적으로 적응했음을 시사한다.

분석 결과, 상대적으로 규모가 작거나 복잡성이 제한된 데이터셋에서는 깊은 구조의 모델보다는 VGG16과 같이 간결한 아키텍처나, Swin Transformer와 같이 최신 트랜스포머 기반 모델이 더 높은 성능을 발휘할 가능성이 크다는 결론을 내릴 수 있다. 또한, 반복적인 데이터 전처리 과정을 최소화하고, 사전에 처리된 데이터를 재사용함으로써 GPU 자원을 효율적으로 관리하는 것도 모델 학습 시간을 단축하고 실험의 효율성을 높이는 데 중요한 요소로 작용하였다.

향후 연구에서는 본 연구의 인사이트를 바탕으로, 보다 정교한 모델 설계와 철저한 하이퍼파라미터 최적화를 진행하여 꽃 품종 자동 분류의 정확도와 일반화 성능을 더욱 향상시키는 방안을 모색할 것이다. 이러한 결과는 화훼 농업 및 꽃 관련 산업에서 품종 관리와 전략 수립 자동화의 중요한 기반으로 활용될 수 있을 것이다.

# 참고 문헌
https://www.kaggle.com/code/dipanshurautela2001/transferlearningvgg-0-85-accuracy

# 부록

[Google Colab](https://colab.research.google.com/drive/1NKalNIYI1YhzmD5V8RVrh-TS3owZKCbU?usp=sharing)
