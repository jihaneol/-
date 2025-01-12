from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
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
# 'Weight'와 'Length1' 분리 col
bream_weight = bream_data["Weight"]
bream_length = bream_data["Length1"]
smelt_weight = smelt_data["Weight"]
smelt_length = smelt_data["Length1"]


# 사이킷런을 위해 2차원 리스트 생성
length= pd.concat([bream_length,smelt_length],axis=0 , ignore_index=True)
weight  = pd.concat([bream_weight, smelt_weight], axis=0, ignore_index=True)
fish_data = pd.concat([bream_data[["Length1","Weight"]], smelt_data[["Length1","Weight"]]], axis=0)
fish_target = [1]*35 + [0]*14
fish_data_list =fish_data.values.tolist() 
kn = KNeighborsClassifier()

kn.fit(fish_data_list, fish_target)
print(kn.score(fish_data_list, fish_target))
print(kn.predict([[30,600]]))
print(kn._fit_X)
print(kn._y)
# 참고 데이터를 49개, 기본 5개이다.가장 가까운 데이터를 참고한다.
kn49 = KNeighborsClassifier(n_neighbors=49)

kn49.fit(fish_data_list, fish_target)
kn49.predict(fish_data_list)
print(kn49.score(fish_data_list, fish_target))
print(35/49)

