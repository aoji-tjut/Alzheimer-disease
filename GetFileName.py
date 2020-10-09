import tensorflow as tf
import os
from pprint import pprint

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        # print(root) #当前目录路径
        # print(dirs) #当前路径下所有子目录
        print(files) #当前路径下所有非目录子文件
    return files



if __name__ == '__main__':
    files = file_name("./samples/AD_nii/")
    pprint(files)
