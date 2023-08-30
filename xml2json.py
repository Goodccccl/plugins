# -*- coding:utf-8 -*-
# 作者：chen
# 将json转换为为xml
import json
import xmltodict
import os

# 初始的文件夹路径
filePath = r"C:\Users\DELL\Desktop\Annotations"
# 存放新文件的空白文件夹
XmlTOJson = r"C:\Users\DELL\Desktop\json"

# 获取文件夹下所有文件名
fileNames = os.listdir(filePath)

for file in fileNames:
    # 如果当前的文件名包含了'xml',
    if file.find("xml") >= 0:
        # 拼接成我们要读取的完整路径
        filePathXml = os.path.join(filePath, file)
        # open 函数 默认是 'r'类型 ，
        FileXml = open(filePathXml)
        xml = FileXml.read()
        xmlparse = xmltodict.parse(xml)
        # parse是的xml解析器
        # json库dumps()是将dict转化成json格式，loads()是将json转化成dict格式。
        # dumps()方法的ident=1，格式化json
        jsonstr = json.dumps(xmlparse, indent=1)
        # 将文件名与文件后缀分离
        file = os.path.splitext(file)
        # type(file) = truple
        strfile = file[0] + '.json'
        # type(strfile) =str
        # 拼接新的文件路径
        JsonPath = os.path.join(XmlTOJson, strfile)
        # 以 w 方式打开新的空白文件
        newFile = open(JsonPath, 'w')
        newFile.write(jsonstr)

        newFile.close()

