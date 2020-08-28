# Happy Halloween: Dance Battle (Version 1.0)
##工具细节预览

**准备文件**
1. 一张自拍图，脸部清晰（简称**模仿者**），JPG格式
2. 一段可以肉眼辨别出脸部的视频，（简称**模仿对象视频**），MP4格式（*可以是舞蹈视频，或者想要模仿动作的视频，这样比较好笑。*）
3. 一个背景视频，可以是黑白或者彩色RGB视频（简称**背景视频**），MP4格式
4. (*optional*）一段长度类似的音频，MP3格式。（*该音频可不提供，工具可以从提供的任意视频文件提取音频.*）
   
**工具功能细节大纲**
1. 照片脸部识别
2. 视频换脸
3. 视频风格替换
4. 视频拼接
5. 添加音频

## 使用教程


## 功能细节

###1. 导入工具所依赖的python库

```python
import os
import shutil
import cv2
import numpy as np
import paddlehub as hub
from moviepy.editor import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
from PIL import Image
```

###2. 定义工具功能函数
1) 视频分解
```python
def(path_video1, pic_video1, timeF = 1):
    vc = cv2.VideoCapture(r'G:/library/baidu_deeplearning/paddle_hub_project/input/videos/video7.mp4')  # 读入视频文件，命名cv
    n = 1  # 计数
    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False
    #timeF = timeF  # 视频帧计数间隔频率 
    i = 0
    while rval:  # 循环读取视频帧
        rval, frame = vc.read()
        if (n % timeF == 0):  # 每隔timeF帧进行存储操作
            i += 1
            cv2.imwrite(r'G:/library/baidu_deeplearning/paddle_hub_project/input/videos/movie7/{}.jpg'.format(i), frame)  # 存储为图像
        n = n + 1
        cv2.waitKey(1)
    vc.release()

```

2) 人脸识别和换脸


3) 图片对称排版


4) 抠图


5) 两视频组装


6) 视频合并


7) 添加音频
# Paddlehub_project
