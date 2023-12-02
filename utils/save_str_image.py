"""
==========================
# -*- coding: utf8 -*-
# @Author   : louiss007
# @Time     : 2023/12/2 15:02
# @FileName : save_str_image.py
# @Email    : quant_master2000@163.com
==========================
"""

from PIL import Image, ImageDraw, ImageFont


def save_str_as_image(string, font_size, font_color, background_color, output_path):
    # 创建一个空白图片
    image = Image.new('RGB', (200, 100), background_color)
    draw = ImageDraw.Draw(image)
    # font = ImageFont.truetype('path/to/font.ttf', font_size)

    # 在图片上绘制字符串
    draw.text((50, 50), string, fill=font_color)

    # 保存图片
    image.save(output_path)


# 调用函数保存字符串为图片
# save_string_as_image('Hello, World!', 30, 'green', 'black', 'output.jpg')
