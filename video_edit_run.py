# Happy Halloween
# 1. 照片人像识别    # ，替换为skeleton
# 2. 用替换后的照片模仿视频跳舞

import os
import cv2
import numpy as np
import paddlehub as hub
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from moviepy.editor import *
import moviepy.editor as mpe
import shutil

# create new folder
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


# 将视频拆分为图片

def video_split(video_path, pic_path, timeF):
    vc = cv2.VideoCapture(video_path)  # 读入视频文件，命名cv
    n = 1  # 计数
    if vc.isOpened():  # 判断是否正常打开
        rval, frame = vc.read()
    else:
        rval = False
    # timeF = 1  # 视频帧计数间隔频率
    i = 0
    while rval:  # 循环读取视频帧
        rval, frame = vc.read()
        if (n % timeF == 0):  # 每隔timeF帧进行存储操作
            i += 1
            pic_path2 = pic_path + str(i) + '.jpg'
            # print(pic_path2)
            cv2.imwrite(pic_path2, frame)  # 存储为图像
        n = n + 1
        cv2.waitKey(1)
    vc.release()

# 1 拼接成视频的函数


def picvideo(path, fps, size, file_path):
    # path = r'C:\Users\Administrator\Desktop\1\huaixiao\\'#文件路径
    filelist = os.listdir(path)  # 获取该目录下的所有文件名
    #fps = 30
    # size = (591,705) #图片的分辨率片
    file_path = file_path+"finalvideo.mp4"
    # file_path = r"G:/library/baidu_deeplearning/paddle_hub_project/output/finalvideo1.mp4"  # 导出路径
    # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    video = cv2.VideoWriter(file_path, fourcc, fps, size)
    for item in filelist:
        if item.endswith('.jpg'):  # 判断图片后缀是否是.png
            item = path + '/' + item
            # 使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
            img = cv2.imread(item)
            video.write(img)  # 把图片写进视频

    video.release()  # 释放


# 2 执行换脸操作

def get_image_size(image):
    """
    获取图片大小（高度,宽度）
    :param image: image
    :return: （高度,宽度）
    """
    image_size = (image.shape[0], image.shape[1])
    return image_size


def get_face_landmarks(image, module1):
    """
    获取人脸标志，68个特征点
    :param image: image
    :param face_detector: dlib.get_frontal_face_detector
    :param shape_predictor: dlib.shape_predictor
    :return: np.array([[],[]]), 68个特征点
    """
    #dets = face_landmark.keypoint_detection([image])
    # module = hub.Module(name="face_landmark_localization")
    #img = cv2.imread("".join(path))
    #img = cv2.resize(img, (600, img.shape[0] * 600 // img.shape[1]))
    dets = module1.keypoint_detection(images=[image])
    #num_faces = len(dets[0]['data'][0])
    if len(dets) == 0:
        print("Sorry, there were no faces found.")
        return None
    # shape = shape_predictor(image, dets[0])
    face_landmarks = np.array([[p[0], p[1]] for p in dets[0]['data'][0]])
    return face_landmarks


def get_face_mask(image_size, face_landmarks):
    """
    获取人脸掩模
    :param image_size: 图片大小
    :param face_landmarks: 68个特征点
    :return: image_mask, 掩模图片
    """
    mask = np.zeros(image_size, dtype=np.int32)
    points = np.concatenate([face_landmarks[0:16], face_landmarks[26:17:-1]])
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(img=mask, pts=[points], color=255)
    # mask = np.zeros(image_size, dtype=np.uint8)
    # points = cv2.convexHull(face_landmarks)  # 凸包
    # cv2.fillConvexPoly(mask, points, color=255)
    return mask.astype(np.uint8)


def get_affine_image(image1, image2, face_landmarks1, face_landmarks2):
    """
    获取图片1仿射变换后的图片
    :param image1: 图片1, 要进行仿射变换的图片
    :param image2: 图片2, 只要用来获取图片大小，生成与之大小相同的仿射变换图片
    :param face_landmarks1: 图片1的人脸特征点
    :param face_landmarks2: 图片2的人脸特征点
    :return: 仿射变换后的图片
    """
    three_points_index = [18, 8, 25]
    M = cv2.getAffineTransform(face_landmarks1[three_points_index].astype(np.float32),
                               face_landmarks2[three_points_index].astype(np.float32))
    dsize = (image2.shape[1], image2.shape[0])
    affine_image = cv2.warpAffine(image1, M, dsize)
    return affine_image.astype(np.uint8)


def get_mask_center_point(image_mask):
    """
    获取掩模的中心点坐标
    :param image_mask: 掩模图片
    :return: 掩模中心
    """
    image_mask_index = np.argwhere(image_mask > 0)
    miny, minx = np.min(image_mask_index, axis=0)
    maxy, maxx = np.max(image_mask_index, axis=0)
    center_point = ((maxx + minx) // 2, (maxy + miny) // 2)
    return center_point


def get_mask_union(mask1, mask2):
    """
    获取两个掩模掩盖部分的并集
    :param mask1: mask_image, 掩模1
    :param mask2: mask_image, 掩模2
    :return: 两个掩模掩盖部分的并集
    """
    mask = np.min([mask1, mask2], axis=0)  # 掩盖部分并集
    mask = ((cv2.blur(mask, (5, 5)) == 255) * 255).astype(np.uint8)  # 缩小掩模大小
    mask = cv2.blur(mask, (3, 3)).astype(np.uint8)  # 模糊掩模
    return mask


def skin_color_adjustment(im1, im2, mask=None):
    """
    肤色调整
    :param im1: 图片1
    :param im2: 图片2
    :param mask: 人脸 mask. 如果存在，使用人脸部分均值来求肤色变换系数；否则，使用高斯模糊来求肤色变换系数
    :return: 根据图片2的颜色调整的图片1
    """
    if mask is None:
        im1_ksize = 55
        im2_ksize = 55
        im1_factor = cv2.GaussianBlur(
            im1, (im1_ksize, im1_ksize), 0).astype(np.float)
        im2_factor = cv2.GaussianBlur(
            im2, (im2_ksize, im2_ksize), 0).astype(np.float)
    else:
        im1_face_image = cv2.bitwise_and(im1, im1, mask=mask)
        im2_face_image = cv2.bitwise_and(im2, im2, mask=mask)
        im1_factor = np.mean(im1_face_image, axis=(0, 1))
        im2_factor = np.mean(im2_face_image, axis=(0, 1))

    im1 = np.clip((im1.astype(np.float) * im2_factor /
                   np.clip(im1_factor, 1e-6, None)), 0, 255).astype(np.uint8)
    return im1


def change_face(im_name1, im_name2, module1, new_path):
    """
    :param im1: 要替换成的人脸图片或文件名
    :param im_name2: 原始人脸图片文件名
    :param new_path: 替换后图片目录
    """
    if isinstance(im_name1, str):
        im1 = cv2.imread(im_name1)  # face_image
        b,g,r = cv2.split(im1)
        im1 = cv2.merge([r,g,b])
    else:
        im1 = im_name1
    im1 = cv2.resize(im1, (600, im1.shape[0] * 600 // im1.shape[1]))
    landmarks1 = get_face_landmarks(im1, module1=module1)  # 68_face_landmarks

    im1_size = get_image_size(im1)  # 脸图大小
    im1_mask = get_face_mask(im1_size, landmarks1)  # 脸图人脸掩模

    im2 = cv2.imread(im_name2)
    b,g,r = cv2.split(im2)
    im2 = cv2.merge([r,g,b])
    landmarks2 = get_face_landmarks(im2, module1=module1)  # 68_face_landmarks
    if landmarks2 is not None:
        im2_size = get_image_size(im2)  # 摄像头图片大小
        im2_mask = get_face_mask(im2_size, landmarks2)  # 摄像头图片人脸掩模

        affine_im1 = get_affine_image(
            im1, im2, landmarks1, landmarks2)  # im1（脸图）仿射变换后的图片
        affine_im1_mask = get_affine_image(
            im1_mask, im2, landmarks1, landmarks2)  # im1（脸图）仿射变换后的图片的人脸掩模
        union_mask = get_mask_union(im2_mask, affine_im1_mask)  # 掩模合并
        # affine_im1 = skin_color_adjustment(affine_im1, im2, mask=union_mask)  # 肤色调整
        point = get_mask_center_point(
            affine_im1_mask)  # im1（脸图）仿射变换后的图片的人脸掩模的中心点
        seamless_im = cv2.seamlessClone(
            affine_im1, im2, mask=union_mask, p=point, flags=cv2.NORMAL_CLONE)  # 进行泊松融合
        new_im = os.path.join(new_path)
        cv2.imwrite(new_im, seamless_im)
    else:
        shutil.copy(im_name2, new_path)

# 1） 批量抠图


def humanseg(img_path, outputpath, module2):
    input_dict = {"image": img_path}
    #outputpath = 'G:/library/baidu_deeplearning/paddle_hub_project/input/photo_edited1/'
    # execute predict and print the result
    module2.segmentation(data=input_dict, visualization=True,
                        output_dir=outputpath)
    return None
# 2） 并黏贴到背景图


def blend_images(fore_image, base_image, i, path):
    """
    将抠出的人物图像换背景
    fore_image: 前景图片，抠出的人物图片
    base_image: 背景图片
    """
    # 读入图片
    base_image = Image.open(base_image).convert('RGB')
    fore_image = Image.open(fore_image).resize(base_image.size)
    # 图片加权合成
    scope_map = np.array(fore_image)[:, :, -1] / 255
    scope_map = scope_map[:, :, np.newaxis]
    scope_map = np.repeat(scope_map, repeats=3, axis=2)
    res_image = np.multiply(scope_map, np.array(fore_image)[
                            :, :, :3]) + np.multiply((1-scope_map), np.array(base_image))
    # 保存图片
    res_image = Image.fromarray(np.uint8(res_image))
    res_image.save(path+str(i)+".jpg")
    return None


def put_together(img1, img2):
    print(img1.shape, img2.shape)
    if img1.shape == img2.shape:
        rows, cols, c = img1.shape
        newimg = np.zeros([rows, 2*cols, c]).astype(np.uint8)
        newimg[:, 0:cols, :] = img1
        newimg[:, cols:2*cols, :] = img2
        return newimg
    else:
        return img1


# 按帧分解视频图片

# 接入参数
parser = argparse.ArgumentParser(description='Input')
parser.add_argument('--path_to_files', type=str,
                    default=None, help='Directory to load all prepared files.')
parser.add_argument('--video_path_action', type=str,
                    default=None, help='Directory to load dance video.')
parser.add_argument('--timeF_action', type=int, default=None,
                    help='Time between two pictures from dance video.')
parser.add_argument('--video_path_bg', type=str, default=None,
                    help='Directory to load background video, for example black and white movie.')
parser.add_argument('--timeF_bg', type=int, default=None,
                    help='Time between two pictures from background video.')
parser.add_argument('--face_pic_path1', type=str,
                    default=None, help='Dance battle competitor 1.')
parser.add_argument('--face_pic_path2', type=str,
                    default=None, help='Dance battle competitor 2.')
parser.add_argument('--music_path', type=str,
                    default=None, help='Directory to music file.')
parser.add_argument('--fps', type=int, default=None,
                    help='Time between two pictures when connecting video.')
parser.add_argument('--final_video_path', type=str,
                    default=None, help='Directory to save final video.')

args, unknown = parser.parse_known_args()
print(args)

path_to_files = args.path_to_files
video_path_action = args.video_path_action
timeF_action = args.timeF_action
video_path_bg = args.video_path_bg
timeF_bg = args.timeF_bg
face_pic_path1 = args.face_pic_path1
face_pic_path2 = args.face_pic_path2
music_path = args.music_path
fps = args.fps
final_video_path = args.final_video_path

# create necessary folders
video_path_action = path_to_files + video_path_action
video_path_bg = path_to_files + video_path_bg
face_pic_path1 = path_to_files + face_pic_path1
face_pic_path2 = path_to_files + face_pic_path2
music_path = path_to_files + music_path
mkdir(path = video_path_action)
mkdir(path = video_path_bg)
mkdir(path = face_pic_path1)
mkdir(path = face_pic_path2)
# mkdir(path = music_path)
mkdir(path = final_video_path)

module1 = hub.Module(name="face_landmark_localization")
module2 = hub.Module(name="deeplabv3p_xception65_humanseg")

# video_path_action = 'G:/library/baidu_deeplearning/paddle_hub_project/input/videos/video7.mp4'
#pic_path_action = 'G:/library/baidu_deeplearning/paddle_hub_project/input/videos/pic_actionvideo/'
pic_path_action = path_to_files + '/pic_actionvideo/'
mkdir(path = pic_path_action)
# timeF_action = 100
video_split(video_path=video_path_action,
            pic_path=pic_path_action, timeF=timeF_action)

# video_path_bg = 'G:/library/baidu_deeplearning/paddle_hub_project/input/videos/video4.mp4'
#pic_path_bg = 'G:/library/baidu_deeplearning/paddle_hub_project/input/videos/pic_bgvideo/'
pic_path_bg = path_to_files + '/pic_bgvideo/'
mkdir(path =  pic_path_bg)
# timeF_bg = 100
video_split(video_path=video_path_bg,
            pic_path=pic_path_bg, timeF=timeF_bg)

# the smaller number of pictures from two videos will be taken as the length of video
videolength1 = len([lists for lists in os.listdir(
    pic_path_action) if os.path.isfile(os.path.join(pic_path_action, lists))])
videolength2 = len([lists for lists in os.listdir(
    pic_path_bg) if os.path.isfile(os.path.join(pic_path_bg, lists))])
videolength = min(videolength1, videolength2)

# face_pic_path1 = "G:/library/baidu_deeplearning/paddle_hub_project/input/photos/face_img.jpg"
# facechanged_pic_path1 = "G:/library/baidu_deeplearning/paddle_hub_project/input/face_chaged1/"
facechanged_pic_path1 = path_to_files + "/face_chaged1/"
mkdir(path=facechanged_pic_path1)

for i in range(1, videolength):
    path2 = pic_path_action+str(i)+'.jpg'
    new_path = facechanged_pic_path1+str(i)+'.jpg'
    change_face(im_name1=face_pic_path1, im_name2=path2, module1 = module1, new_path=new_path)

# face_pic_path2 = "G:/library/baidu_deeplearning/paddle_hub_project/input/photos/face_img2.jpg"
#facechanged_pic_path2 = "G:/library/baidu_deeplearning/paddle_hub_project/input/face_chaged2/"
facechanged_pic_path2 = path_to_files + "/face_chaged2/"
mkdir(path=facechanged_pic_path2)

for i in range(1, videolength):
    path2 = pic_path_action+str(i)+'.jpg'
    new_path = facechanged_pic_path2+str(i)+'.jpg'
    change_face(im_name1=face_pic_path2, im_name2=path2, module1 = module1, new_path=new_path)
    img = mpimg.imread(new_path)
    img = img[:, ::-1, :]
    cv2.imwrite( new_path, img)

# 4 抠图+视频背景替换
# 设置一个if loop，不存储无人像的图片
def video_bg_change(module2, pic_path, pic_save_path, bg_dir, fore_dir, videolength):
    #module = hub.Module(name="deeplabv3p_xception65_humanseg")
    for i in range(1, videolength):
        img_path = [pic_path+str(i)+'.jpg']
        humanseg(img_path=img_path, outputpath=fore_dir, module2 = module2)

    for i in range(1, videolength):
        fore_image = fore_dir+str(i)+'.png'
        base_image = bg_dir+str(i)+'.jpg'
        blend_images(fore_image, base_image, i, pic_save_path)


# 'G:/library/baidu_deeplearning/paddle_hub_project/input/videos/base_img2/'
# bg_dir = pic_path_bg
# fore_dir1 = 'G:/library/baidu_deeplearning/paddle_hub_project/input/photo_edited_face_chaged1/'
fore_dir1 = path_to_files + '/photo_edited_face_chaged1/'
mkdir(path = fore_dir1)
# "G:/library/baidu_deeplearning/paddle_hub_project/input/face_chaged1/"
# facechanged_pic_path1 = facechanged_pic_path1
#pic_changedbg_path1 = 'G:/library/baidu_deeplearning/paddle_hub_project/input/final1/'
pic_changedbg_path1 = path_to_files + '/final1/'
mkdir(path = pic_changedbg_path1)
video_bg_change(module2 = module2,
                pic_path = facechanged_pic_path1,
                pic_save_path = pic_changedbg_path1,
                bg_dir = pic_path_bg,
                fore_dir = fore_dir1,
                videolength = videolength)


#fore_dir2 = 'G:/library/baidu_deeplearning/paddle_hub_project/input/photo_edited_face_chaged2/'
fore_dir2 = path_to_files + '/photo_edited_face_chaged2/'
mkdir(path = fore_dir2)
# "G:/library/baidu_deeplearning/paddle_hub_project/input/face_chaged2/"
# facechanged_pic_path2 = facechanged_pic_path2
#pic_changedbg_path2 = 'G:/library/baidu_deeplearning/paddle_hub_project/input/final2/'
pic_changedbg_path2 = path_to_files + '/final2/'
mkdir(path = pic_changedbg_path2)

video_bg_change(module2 = module2,
                pic_path = facechanged_pic_path2,
                pic_save_path = pic_changedbg_path2,
                bg_dir = pic_path_bg,
                fore_dir = fore_dir2,
                videolength = videolength)
# 3 dance battle function

# "G:/library/baidu_deeplearning/paddle_hub_project/input/face_chaged/"
# facechanged_pic_path1 = facechanged_pic_path1
# "G:/library/baidu_deeplearning/paddle_hub_project/input/face_chaged2/"
# facechanged_pic_path2 = facechanged_pic_path2
#facechanged_pic_merged = "G:/library/baidu_deeplearning/paddle_hub_project/input/face_chaged_put_together/"
facechanged_pic_merged = path_to_files + '/face_chaged_put_together/'
mkdir(path = facechanged_pic_merged)
for i in range(1, videolength):
    im_name1 = pic_changedbg_path1 + str(i) + '.jpg'
    im_name2 = pic_changedbg_path2 + str(i) + '.jpg'
    new_name2 = facechanged_pic_merged + str(i) + '.jpg'
    img1 = mpimg.imread(im_name1)
    img2 = mpimg.imread(im_name2)
    new_img = put_together(img1, img2)
    import matplotlib.image as mpimg
    mpimg.imsave(new_name2, new_img)

# new_video_path = facechanged_pic_merged # 'G:/library/baidu_deeplearning/paddle_hub_project/input/face_chaged_put_together/'
# final_video_path = 'G:/library/baidu_deeplearning/paddle_hub_project/output/'

img = Image.open(facechanged_pic_merged + '1.jpg')
# img.size
picvideo(path=facechanged_pic_merged,fps=fps,
         size=img.size, file_path=final_video_path)

# adding sound
# music_path = 'G:/library/baidu_deeplearning/paddle_hub_project/input/videos/video7_music.mp3'
if not os.path.exists(music_path):
    video = VideoFileClip(video_path_action)
    audio = video.audio
    audio.write_audiofile(music_path)
else:
    import moviepy.editor as mpe
    audio = mpe.AudioFileClip(music_path)

video2 = VideoFileClip(
    final_video_path+'finalvideo.mp4')
video2 = video2.set_audio(audio)  # 设置
video2.write_videofile(final_video_path+'finalvideo2.mp4')

