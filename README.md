# 🌸 붓꽃 분류 (Iris Classification) 프로젝트

## 1. 프로젝트 개요

본 프로젝트는 가장 고전적인 머신러닝 데이터셋 중 하나인 **붓꽃(Iris) 데이터셋**을 사용하여 **K-최근접 이웃(K-Nearest Neighbors, KNN)** 알고리즘 기반의 분류 모델을 구축하고 평가하는 과정을 담고 있습니다. Scikit-learn 라이브러리의 표준적인 머신러닝 워크플로우를 익히는 것을 목표로 합니다.

## 2. 목표 (Goals)

* Scikit-learn 라이브러리의 표준 사용법을 익힙니다. (모델 로드, 훈련, 예측)
* 데이터 탐색 및 시각화(EDA)를 통해 데이터의 특징을 파악합니다.
* 기본적인 분류 알고리즘(KNN)을 적용하고 성능을 평가합니다.

## 3. 사용 환경 및 라이브러리

* **언어**: Python 3
* **주요 라이브러리**:
    * `pandas`: 데이터 처리
    * `sklearn`: 데이터셋 로드, 모델 구축, 평가 (특히 `load_iris`, `KNeighborsClassifier`, `train_test_split`, `accuracy_score`, `confusion_matrix`)
    * `seaborn`, `matplotlib`: 데이터 시각화

## 4. 데이터셋 및 전처리

### 4.1. 데이터 로드

* **데이터셋**: Scikit-learn에서 기본으로 제공하는 `load_iris` 함수를 사용하여 데이터를 로드했습니다.
* **특성 (Features, X)**: `sepal length (cm)`, `sepal width (cm)`, `petal length (cm)`, `petal width (cm)` 4가지 특성을 사용합니다.
* **타겟 (Target, Y)**: `target` (붓꽃 품종, 0: setosa, 1: versicolor, 2: virginica)을 사용합니다.
* **데이터 정보**: 총 150개의 엔트리(데이터)가 있으며, 결측값(Non-Null)은 없습니다.

### 4.2. 데이터 탐색 및 시각화 (EDA)

* `df.describe()`를 통해 기초 통계를 확인했습니다.
* Seaborn을 사용하여 'petal length (cm)'와 'petal width (cm)' 두 변수 간의 관계를 품종('species')별로 시각화했습니다. 시각화 결과, 세 품종이 명확히 구분됨을 확인했습니다.

### 4.3. 데이터 분할

* `train_test_split`을 사용하여 데이터를 훈련 셋과 테스트 셋으로 분할했습니다.
* **훈련 데이터 (X_train, Y_train)**: 105개 샘플
* **테스트 데이터 (X_test, Y_test)**: 45개 샘플
* `random_state=42`를 설정하여 일관된 분할 결과를 확보했습니다.

## 5. 모델 구축 및 평가 (KNN)

### 5.1. 모델 구축 및 훈련

* **알고리즘**: **K-최근접 이웃 (K-Nearest Neighbors, KNN)** 분류기를 사용했습니다.
* **하이퍼파라미터**: 이웃의 수 $K$를 **5**로 설정했습니다 (`n_neighbors=5`).
* 훈련 데이터(`X_train`, `Y_train`)를 사용하여 모델을 훈련했습니다.

### 5.2. 예측 및 평가

* 훈련된 모델을 테스트 데이터(`X_test`)에 적용하여 `Y_pred`를 예측했습니다.
* **평가 지표**: 정확도(Accuracy)와 혼동 행렬(Confusion Matrix)을 사용했습니다.

| 지표 | 결과 |
| :--- | :--- |
| **모델 정확도 (Accuracy)** | **0.9778** |

### 5.3. 혼동 행렬 (Confusion Matrix)

다음은 혼동 행렬 시각화 결과입니다.
