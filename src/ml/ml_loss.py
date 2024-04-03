"""
==========================
# -*- coding: utf8 -*-
# @Author   : louiss007
# @Time     : 2023/12/9 13:24
# @FileName : ml_loss.py
# @Email    : quant_master2000@163.com
==========================
"""
import numpy as np


def mse(yt, yp):
    sq_error = (yp - yt) ** 2
    sum_sq_error = np.sum(sq_error)
    loss = sum_sq_error / yt.size
    return loss


def mae(yt, yp):
    error = yp - yt
    absolute_error = np.absolute(error)
    sum_absolute_error = np.sum(absolute_error)
    loss = sum_absolute_error / yt.size
    return loss


def bce(yt, yp):
    ce_loss = yt * np.log(yp) + (1 - yt) * np.log(1 - yp)
    sum_ce_loss = np.sum(ce_loss)
    loss = - sum_ce_loss / yt.size
    return loss


def cce(yt, yp):
    cce_loss = yt * np.log(yp)
    sum_cce_loss = np.sum(cce_loss)
    loss = - sum_cce_loss / yt.size
    return loss


def hinge(yt, yp):
    hinge_loss = np.sum(max(0, 1 - (yp * yt)))
    return hinge_loss


def huber(yt, yp, delta):
    y_size = yt.size
    tot_error = 0
    for i in range(y_size):
        err = np.absolute(yp[i] - yt[i])
        if err < delta:
            huber_error = (err ** 2) / 2
        else:
            huber_error = (delta * err) / (0.5 * (delta * delta))
        tot_error += huber_error
    loss = tot_error / y_size
    return loss


def zero_one(yt, yp):
    if yt == yp:
        return 0
    else:
        return 1
