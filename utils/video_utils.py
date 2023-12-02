"""
==========================
# -*- coding: utf8 -*-
# @Author   : louiss007
# @Time     : 2023/12/2 15:40
# @FileName : video_utils.py
# @Email    : quant_master2000@163.com
==========================
"""
from moviepy.editor import *


def convert_video_to_gif(video_file, gif_file):
    video = VideoFileClip(video_file)
    # clip = video.subclip((0, 0), (0, 40)).resize(0.2)
    clip = video.subclip((0, 0), (0, 41))
    clip.write_gif(gif_file, fps=8)


if __name__ == '__main__':
    in_file = '/Users/louiss007/Desktop/q_learning_mp4.mov'
    out_file = '/Users/louiss007/Desktop/q_learning_mp4.gif'
    convert_video_to_gif(in_file, out_file)


