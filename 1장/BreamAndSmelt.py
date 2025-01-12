import matplotlib.pyplot as plt # 맷플롯립, 산점도, 그래프 그리는 패키지
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
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length') # x 축
plt.ylabel('weight') # y 축
plt.show()