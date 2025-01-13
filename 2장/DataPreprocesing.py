from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

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
fish_data_list =np.array(fish_data.values.tolist()) 
kn = KNeighborsClassifier()
# 파이썬 리스트 보다 np로 만드는게 좋다. c, c++과 같은 저수준 언어로 개발되어있어서.
# 튜플로 제공해야 한다.
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

train_input, test_input, train_target, test_target = train_test_split(
    fish_data_list, fish_target,stratify=fish_target, random_state=42
)


# 0.0 이 나온다.샘플링 편향 발생..
kn.fit(train_input, train_target)
kn.score(test_input, test_target)

distance, indexes = kn.kneighbors([[25,150]])

plt.scatter(train_input[:,0],train_input[:,1])
plt.scatter(25,150,marker="^")
plt.scatter(train_input[indexes,0],train_input[indexes,1], marker="D")
plt.xlabel("length")
plt.ylabel("weight")
plt.xlim((0,1000))
plt.show()

mean = np.mean(train_input, axis=0)
std = np.std(train_input,axis=0)

train_scaled = (train_input - mean) / std

new = ([25,150]-mean)/std

test_scaled = (test_input - mean) / std

kn.fit(train_scaled, train_target)
kn.score(test_scaled, test_target)

print(kn.predict([new]))