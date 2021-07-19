"""
Contains functions for displaying and saving plots of both real and generated
data (currently written for tri-axial data but could be easily modified for
applications with a different number of sensor channels)
"""

import matplotlib.patches as mpl_patches
import matplotlib.pyplot as plt
import matplotlib
import tkinter
import numpy as np
from typing import List

matplotlib.use('tkagg')


def plot_results(epochs: List[int],
                 class_acc: List[float],
                 disc_acc: List[float],
                 gen_acc: List[float]):
    plt.figure(figsize=(12, 9))
    plt.style.use('fivethirtyeight')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax = plt.subplot(111)
    ax.set_title('GAN Accuracy Progression')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Model Accuracy')
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_xticks(np.arange(0, len(epochs) + 1))
    ax.plot(epochs, convert_dec_to_percent(class_acc),
            label='Classifier Accuracy')
    ax.plot(epochs, convert_dec_to_percent(disc_acc),
            label='Discriminator Accuracy')
    ax.plot(epochs, convert_dec_to_percent(gen_acc),
            label='Generator-Trick-Discriminator Accuracy')
    vals = ax.get_yticks()
    ax.set_yticklabels(['{}%'.format(x) for x in vals])
    plt.legend()
    plt.show()


def convert_dec_to_percent(decimal_list: List[float]) -> List[float]:
    return list(map(lambda x: x * 100, decimal_list))
