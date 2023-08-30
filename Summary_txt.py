import os

import numpy

if __name__ == '__main__':
    summary_txt_path = r'F:\new_g\all_summary.txt'
    summary2_txt_path = r'F:\new_g\all_summary2.txt'
    txts_path = r'F:\Artificial_neural_Network\yolov5_Trackpad\runs\detect\exp3\labels'
    with open(summary_txt_path, 'w') as w:
        list = os.listdir(txts_path)
        for i in range(len(list)):
            txt_name = list[i]
            txt_path = os.path.join(txts_path, txt_name)
            with open(txt_path, 'r') as f:
                datas = f.readlines()
                num = len(datas)
                data = ''
                for n in range(num):
                    index = int(datas[n][0:1]) + 3
                    datas[n] = str(index) + datas[n][1:]
                    data = data + datas[n]
                tiff_name = txt_name[:-4] + '.tif'
                w.write('TiffFileName:  ' + tiff_name + ';' + '\n' +
                        'DefectList:\n' + data + '\n')
                f.close()
        w.close()

    with open(summary2_txt_path, 'w') as w:
        list = os.listdir(txts_path)
        for i in range(len(list)):
            txt_name = list[i]
            txt_path = os.path.join(txts_path, txt_name)
            with open(txt_path, 'r') as f:
                data = f.readlines()
                num = len(data)
                if num == 1:
                    index = int(data[0][0:1]) + 3
                    w.write(txt_name[4:-4] + "_" + str(index) + '\n')
                else:
                    list_index = numpy.zeros(6)
                    for o in range(num):
                        cls_index = data[o][0:1]
                        if cls_index == '0':
                            list_index[0] += 1
                        if cls_index == '1':
                            list_index[1] += 1
                        if cls_index == '2':
                            list_index[2] += 1
                        if cls_index == '3':
                            list_index[3] += 1
                        if cls_index == '4':
                            list_index[4] += 1
                        if cls_index == '5':
                            list_index[5] += 1
                    max_index = list_index.argmax()
                    index = max_index + 3
                    w.write(txt_name[4:-4] + "_" + str(index) + '\n')
