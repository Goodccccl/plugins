import os
import shutil

path = r"F:\Workprojects\TongFu_Bump\data\Score_select_data\error"
save_path = r"F:\Workprojects\TongFu_Bump\data\Manually_select_data\error"
names = os.listdir(path)
# print(names[0].split("_"))
# print(names[0].split("_")[19])

# print(names[7317].split("_"))

num = len(names)
print(num)
# 13, 15, 17, 19为分数
get_index = []

for i in range(num):
    if names[i].split("_")[0] == "NG":
        s1 = float(names[i].split("_")[13])
        s2 = float(names[i].split("_")[15])
        s3 = float(names[i].split("_")[17])
        s4 = float(names[i].split("_")[19])
        if s1 < 0.85 or s2 < 0.85 or s3 < 0.85:
            get_index.append(i)
    if names[i].split("_")[0] == "OK":
        s1 = float(names[i].split("_")[14])
        s2 = float(names[i].split("_")[16])
        s3 = float(names[i].split("_")[18])
        s4 = float(names[i].split("_")[20])
        if s1 < 0.85 or s2 < 0.85 or s3 < 0.85:
            get_index.append(i)
print(len(get_index))

for i in range(len(get_index)):
    img_name = names[get_index[i]]
    shutil.copy(path + "/" + img_name, save_path + "/" + img_name)

