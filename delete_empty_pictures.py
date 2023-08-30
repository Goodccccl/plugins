import os

json_file = r'H:\six-xianchang\label\side-label\abcd\json'
images_file = r'H:\six-xianchang\label\side-label\abcd'

json_list = os.listdir(json_file)
print('json_list:', json_list)
print(len(json_list))

images_list = []
file_list = os.listdir(images_file)
print('file_list:', file_list)
print(len(file_list))
for i in range(len(file_list)):
    name = file_list[i]
    front, ext = os.path.splitext(name)
    if ext == '.bmp':
        name_json = front + '.json'
        images_list.append(name_json)
    # elif ext == '.png':
    #     file = os.path.join(images_file, name)
    #     name2 = front + '.bmp'
    #     images_list.append(name2)
    #     os.rename(file, name2)
    else:
        continue
print(len(file_list))
print('images_list:', images_list)
print(len(images_list))

diff_list = list(set(images_list) - set(json_list))
print('diff_list:', diff_list)
print(len(diff_list))

for i in range(len(diff_list)):
    name = os.path.splitext(diff_list[i])[0] + '.bmp'
    bmp_path = os.path.join(images_file, name)
    os.remove(bmp_path)
print('删除完毕')
