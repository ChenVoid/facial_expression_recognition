import os
import cv2
import numpy as np
import pandas as pd


def load2Image(filepath='./archive/icml_face_data.csv', saveDir='./dataset/archive'):
    '''
    加载数据文件转化为图像数据存储本地
    '''
    data = pd.read_csv(filepath)
    pixels = data[' pixels'].tolist()
    labels = data['emotion'].tolist()
    width, height = 48, 48
    faces = []
    for i in range(len(pixels)):
        one_pixel = pixels[i]
        one_label = str(labels[i])
        face = [int(pixel) for pixel in one_pixel.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (48, 48))
        oneDir = saveDir + one_label + '/'
        if not os.path.exists(oneDir):
            os.makedirs(oneDir)
        one_path = oneDir + str(len(os.listdir(oneDir)) + 1) + '.jpg'
        cv2.imwrite(one_path, face)


# if __name__ == '__main__':
#     load2Image()
