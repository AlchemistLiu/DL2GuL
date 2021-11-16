import sys
import os
import glob
import xml.etree.ElementInclude as ET
from xml.dom.minidom import Node, parse

def get_singlefile(path):
    #minidom解析器打开xml文档并将其解析为内存中的一棵树
    DOMTree=parse(path)
    #获取xml文档对象，就是拿到树的根
    bbox_list=DOMTree.documentElement

    #获取bbox_list对象中所有的book节点的list集合
    bboxs=bbox_list.getElementsByTagName('object')

    # 拿坐标
    for bbox in bboxs:
        #根据节点名title/author/pageNumber得到这些节点的集合list
        xmin=bbox.getElementsByTagName('xmin')[0]
        xmin = xmin.childNodes[0].data
        # print (xmin) 383
        ymin=bbox.getElementsByTagName('ymin')[0]
        ymin = ymin.childNodes[0].data
        xmax=bbox.getElementsByTagName('xmax')[0]
        xmax = xmax.childNodes[0].data
        ymax=bbox.getElementsByTagName('ymax')[0]
        ymax = ymax.childNodes[0].data
        cls_name = bbox.getElementsByTagName('name')[0]
        cls_name = cls_name.childNodes[0].data
        # print(xmin, ymin, xmax, ymax, name)
        # 3883 949 4957 2064 tamper
    return (xmin, ymin, xmax, ymax, cls_name)

# print(get_singlefile(r'myDLstudy\tricks_computational-performance\000001.xml'))
# ('3883', '949', '4957', '2064', 'tamper\n')

def get_fname(path):
    a = os.path.basename(path)#带后缀的文件名
    return a.split('.')[0]
# print(get_fname(r'myDLstudy\tricks_computational-performance\000001.xml')) # 000001


def stream_operat():
    # 这一步手动设
    # path = r'F:\dataset\nist\Annotations'
    # path = r'F:\dataset\test'
    path = r'F:\dataset\nist\Annotations'
    # 拿文件名
    with open(r'F:\dataset\test.txt',mode='w') as file_handle:
    # file_handle = open(r'F:\dataset\test.txt',mode='w')
        for filename in os.listdir(path):
            s_list = []
            strname = get_fname(filename)
            s_list.append(strname)
            s_list.append(' ')
            str4 = get_singlefile(filename)
            for i in range(5):
                s_list.append(str4[i])
                if i < 4:
                    s_list.append(' ')
            file_handle.writelines(s_list)
stream_operat()