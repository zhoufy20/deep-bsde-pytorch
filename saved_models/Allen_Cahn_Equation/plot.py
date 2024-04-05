#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :main.py
# @Time      :2024/3/13 0:17
# @Author    :Feiyu
# @Main      :the plots about the loss of deep_bsde_pytorch and initial value of pdes.

import matplotlib.pyplot as plt
import numpy as np

with open('init.txt', 'r') as file:
    init_lines = file.readlines()
with open('loss.txt', 'r') as file:
    loss_lines = file.readlines()

init_lines = np.array([line.split() for line in init_lines])
loss_lines = np.array([line.split() for line in loss_lines])

init_lines = init_lines.astype(float)
loss_lines = loss_lines.astype(float)

plt.figure(1)
plt.plot( loss_lines, color='blue')
plt.xlabel("Iterations")
plt.ylabel("loss")
plt.title("The loss of deep_bsde_pytorch")
plt.savefig('loss_bsde.png')

plt.figure(2)
plt.plot( init_lines, color='red')
plt.xlabel("Iterations")
plt.ylabel("prediction")
plt.title("The prediction of the initial value")
plt.savefig('prediction.png')

plt.show()
