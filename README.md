# 1장. 마켓과 머신러닝
k-최근접 이웃 사용

## 00. 상황
> 물고기 거래 앱
1. 생선 이름을 자동으로 알려주는 머신러닝을 개발해라.

## 01. 생선 분류 문제
### 도미 데이터 준비
1. 무게와 길이
> 케글 데이터 셋 사용 : https://www.kaggle.com/datasets/vipullrathod/fish-market/code

**이진 분류** : 여러개의 종류 중 하나를 구별해 내는 문제를 분류라고 한다. 특히 2개 중 하나를 고르는 문제를 이진 분류라고 한다.

### pandas 문법
> pd.DataFrame, read_csv <br>
df = pd.DataFrame, df[열 필터링건 조건]

### matplotlib.pyplot 그래프 그려주는 라이브러리
빙어, 도미 데이터
<img src="image.png">

## 02 첫 번째 머신러닝 프로그램
1. k-최근접 이웃 알고리즘
2. 도미 1, 빙어 0 정답 생성

# 2장 훈련 세트와 테스트 세트
### 지도 학습
지도 학습 알고리즘은 학습하기 위한 데이터와 정답이 필요하다.

데이터와 정답을 입력과 타깃이라 한다. 이 둘을 합쳐 훈련 데이터라고 한다.

입력으로 사용된 길이와 무게를 특성이라고 한다.

### 비지도 학습
정답(타깃) 없이 입력 데이터만 사용한다.

## 훈련 세트와 테스트 세트
도미 35 빙어 14개로 훈련 세트와 테스트 세트 구분 
### 샘플링 편향
데이터가 골고루 있지 않아서 발생, 골고루 넣어 줄 수 있는 파이썬 라이브러리 numpy

## 넘파이
배열 라이브러리이다. 고차원의 배열을 손쉽게 만들고 조작 할 수 있는 간편한 도구. shape 이 명령을 사용하면 (샘플 수/행 , 특성 수/열)를 출력한다.

np.arange(49)는 0~48개의 1씩 증가하는 배열 인덱스를 만든다.

train_input[:,0] :는 모든 행, 0은 첫 번째 열을 선택 한다는 뜻이다.
