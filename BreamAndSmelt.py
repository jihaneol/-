from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import os

# Download latest version
path = "C:\\Users\\user\\.cache\\kagglehub\\datasets\\vipullrathod\\fish-market\\versions\\1"
files = os.listdir(path)
data = None
for file in files:
    if file.endswith(".csv"):
        file_path = os.path.join(path, file)
        data = pd.read_csv(file_path)
df = pd.DataFrame(data)

#'Bream' 필터링, [열 필터링 조건]
bream_data = df[df["Species"] == "Bream"]
smelt_data = df[df["Species"] == "Smelt"]


# 사이킷런을 위해 2차원 리스트 생성
fish_data = pd.concat([bream_data[["Length1","Weight"]], smelt_data[["Length1","Weight"]]], axis=0)
fish_target = [1]*35 + [0]*14
fish_data_list =fish_data.values.tolist() 
kn = KNeighborsClassifier()

train_input = fish_data_list[:35]
train_target   = fish_target[:35]
test_input= fish_data_list[35:]
test_target = fish_target[35:]
# 0.0 이 나온다.샘플링 편향 발생..
kn.fit(train_input, train_target)
kn.score(test_input, test_target)


# np는 가로로 출력하게 해준다.
print(fish_data_list)
input_arr = np.array(fish_data_list)
target_arr = np.array(fish_target)
print(input_arr)
print(input_arr.shape)
np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)
print(index)


train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]
test_input= input_arr[index[35:]]
test_target = target_arr[index[35:]]

kn.fit(train_input, train_target)
i = kn.score(test_input, test_target)
print(i)

print(kn.predict(test_input))
print(test_target)
