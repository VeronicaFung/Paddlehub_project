# Happy Halloween: Dance Battle (Version 1.0)
## 背景
Paddlepaddle深度学习入门课程的创意项目，原本我是打算做个和自己专业相关的模型，后来看到需要基于Paddlehub展开，于是就往嗨了玩>_>... 就有了这个基于几个抠图和视频风格替换的项目教程改编的换脸舞蹈battle，具体的references我放在文档最后一部分啦。另外，我自己喜欢一些古古怪怪的万圣节风格的东西，就把背景示例设为了万圣节主题，Mmmm，至少我自己玩得很开心23333。
简单的说一下PaddleHub，一个确实很实用的集成预训练模型结合Fine-tune API快速完成模型迁移到部署的全流程工作平台，使用感还是不错的，通过简单的代码就可以调用在图像、视频、NLP等领域的模型，同时还有不同的大牛往里边加不同的module，为实现自己的一些小灵感提供了可行且fancy的实现方式。

## 工具功能介绍

1) 视频分解
2) 人脸识别和换脸
3) 图片对称排版
4) 抠图
5) 两视频组装
6) 视频合并
7) 添加音频
功能的实现主要调用到两个Paddluhub中的预训练模型：`face_landmark_localization`和`deeplabv3p_xception65_humanseg`,

## 使用教程

###1. 准备文件
1. 两张正脸照片，脸部清晰，JPG格式
2. 一段可以肉眼辨别出脸部的动作视频，MP4格式（*可以是舞蹈视频，或者想要模仿动作的视频，这样比较好笑。*）
3. 一段背景视频，可以是黑白或者彩色RGB视频，MP4格式
4. (*optional*）一段长度类似的音频，MP3格式。（*该音频可不提供，工具可以从提供的舞蹈视频文件提取音频.*）

###2. 安装工具所依赖的python库

通过pip 安装工具所依赖的库，注意该工具通过python3编写：

```shell
pip install -r requirements.txt
```

###3. python命令行执行说明
首先进入存放本工具的文件夹下，然后通过python3执行文件夹下的`paddle_hub_projext.py`文件，注意需要向工具中提供一系列文件：
```shell
cd path/to/tool
./python ./video_edit_run.py --path_to_files path/to/tool \
                                 --video_path_action path/to/actionvideo \
                                 --timeF_action 10 \
                                 --video_path_bg path/to/actionvideo \
                                 --timeF_bg 10 \
                                 --face_pic_path1 path/to/face_image1 \
                                 --face_pic_path2 path/to/face_image2 \
                                 --music_path path/to/music \
                                 --fps 10 \
                                 --final_video_path path/to/actionvideo \
```
其中：
* `--path_to_files`需提供该工具包中`input`文件夹存储的绝对路径；
* `--video_path_action`中提供舞蹈视频的相对路径（相对于`--path_to_files`中提供的路径）；
* `--video_path_bg`中则提供所需要替换背景视频的相对路径（相对于`--path_to_files`中提供的路径）；
* `--timeF_action`中输入从视频中拆分图片时每间隔多少帧保存一张图片，该处请输入整数，数值输入越小视频将拆分的越细致，最终生成的视频也会越连贯，但运行速度就会特别久（如果心情好，可能我会学学怎么加速）;类似的，`--video_path_bg`输入对背景视频拆分的帧间隔；
* `--face_pic_path1`和`--face_pic_path1`分别输入两张照片，注意照片为正脸，尽可能的清晰。
* `--music_path`中提供所要加载视频中的音频，如果用户要自己提供音频，请将音频重命名为`music.mp3`并放于工具中`input/musics/`文件夹下，否则工具如果检测不到对应文件，会从所提供的actionvideo视频中提取；
* `--final_video_path`中需提供最终视频的保存位置，目前要求指定到工具包下的`output`文件夹下方便为生成的视频添加音频

###4. 玩嗨前测试一下
让我们在PaddlePaddle提供的AIstudio上测试一下功能，将工具包放在`work`文件夹下。同时我在`work`文件夹下准备了一个舞蹈视频、一个背景视频、两张清晰的人脸图以及一段音频：

```shell
cd work
ls -l
```
【文件的截图】

我们将对应的文件路径作为参数传入主代码文件（因为我第一次写python脚本，还不太会将代码分开，再互相调用，所以整个tool的功能都在一个脚本里边）：
```shell
cd path/to/tool
python ./video_edit_run.py --path_to_files /home/aistudio/work/Paddlehub_project/input/ \
                                 --video_path_action videos/actionvideo.mp4 \
                                 --timeF_action 10 \
                                 --video_path_bg videos/bgvideo.mp4 \
                                 --timeF_bg 10 \
                                 --face_pic_path1 images/face_img1.jpg \
                                 --face_pic_path2 images/face_img2.jpg \
                                 --music_path musics/music.mp3 \
                                 --fps 10 \
                                 --final_video_path /home/aistudio/work/Paddlehub_project/output/
```

emmmm，如果将视频拆分成过多的图片，跑起来的还是有点慢的。


## 还存在的bug
1. 代码中未对图片rgb或者bgr调整，以至于色调出了点问题。
2. 模型在换脸过程中只能较好的识别正脸，以至于侧脸出现扭曲的问题。。
3. 模型在抠图中存在边缘不清晰，导致效果不佳的问题。
 
  敬请期待版本升级...

## Reference
* 不如跳舞：根据照片合成跳舞视频 https://aistudio.baidu.com/aistudio/projectdetail/751132
* PaddleHub创意赛：小丑精彩舞姿与浩瀚宇宙的结合 https://aistudio.baidu.com/aistudio/projectdetail/751148
* PaddleHub人脸检测示例 https://aistudio.baidu.com/aistudio/projectdetail/750356
* AI人像抠图及视频合成：让你体验复仇者联盟的终局之战 https://aistudio.baidu.com/aistudio/projectdetail/751298