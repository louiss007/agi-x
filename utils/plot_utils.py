"""
==========================
# -*- coding: utf8 -*-
# @Author   : louiss007
# @Time     : 2023/12/2 14:57
# @FileName : plot_utils.py
# @Email    : quant_master2000@163.com
==========================
"""
import os
import io
import imghdr
import imageio.v2 as imageio
import numpy as np

from PIL import ImageFont, Image, ImageDraw, ImageSequence


def create_gif(image_list, gif_name, duration=0.35, date='未知时间'):
    frames = []
    result_list = []
    font = ImageFont.truetype('simfang', size=130)
    # 字体颜色
    fillColor = (255, 0, 0)
    # 文字输出位置
    position = (100, 100)
    # 输出内容
    str_ = date
    str_ = str_.encode('utf-8').decode('utf-8')

    for image_name in image_list:
        image_ = Image.fromarray(imageio.imread(image_name))
        temp_draw = ImageDraw.Draw(image_)
        # temp_draw.text(position, str_, font=font, fill=fillColor)  # 写入文字
        image_ = np.asarray(image_)
        result_list.append(image_)
        # frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, result_list, 'GIF', duration=duration)


def main():
    orgin = r'E:\workfiles\temp_img\date_img'  # 首先设置图像文件路径
    files = os.listdir(orgin)  # 获取图像序列
    for file in files:
        file_path = os.path.join(orgin, file).replace('\\', '/')
        if os.path.isdir(file_path):
            image_list = []
            for i in os.listdir(file_path):
                path = os.path.join(file_path, i).replace('\\', '/')
                if imghdr.what(path) == 'png':
                    image_list.append(path)
            image_list = sorted(image_list)
            gif_name = os.path.join(orgin, file, file + '.gif')  # 设置动态图的名字
            duration = 0.5
            print('gif_name:', gif_name)
            print('image_list:', image_list)
            create_gif(image_list, gif_name, duration, file)  # 创建动态图


def watermark_on_gif(in_gif, out_gif, text='scratch8'):
    """本函数给gif动图加水印"""

    frames = []

    # myfont = ImageFont.truetype("msyh.ttf", 12)  # 加载字体对象

    im = Image.open(in_gif)  # 打开gif图形

    # water_im = Image.new("RGBA", im.size)  # 新建RGBA模式的水印图
    #
    # draw = ImageDraw.Draw(water_im)  # 新建绘画层
    #
    # draw.text((10, 10), text, fill='red')

    for frame in ImageSequence.Iterator(im):  # 迭代每一帧
        d = ImageDraw.Draw(frame)
        d.text((50, 100), "Hello World")
        del d

        # frame = frame.convert("RGBA")  # 转换成RGBA模式

        # frame.paste(water_im, None)  # 把水印粘贴到frame
        #
        # frames.append(frame)  # 加到列表中

        b = io.BytesIO()
        frame.save(b, format="GIF")
        frame = Image.open(b)

        # Then append the single frame image to a list of frames
        frames.append(frame)

    newgif = frames[0]  # 第一帧

    # quality参数为质量，duration为每幅图像播放的毫秒时间

    newgif.save(out_gif, save_all=True,
                append_images=frames[1:], quality=85, duration=100)
    # frames[0].save(r'E:\workfiles\temp_img\new_wind_2.gif', save_all=True, append_images=frames[1:])
    im.close()


if __name__ == '__main__':
    main()

