#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:49:50 2018

@author: Chandra Sekhar Ravuri
"""


import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
from prettytable import PrettyTable
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,size=15)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


data = sio.loadmat('/home/hadoop1/Documents/prml/project/UKDALE/we_did_edge.mat')




plt.figure(1)
plt.subplot(121)
plot_confusion_matrix(confusion_matrix(data['rm_state_true'].squeeze(),data['rm_state'].squeeze()),classes=['OFF','ON'], title='running_machine confussion matrix\n edge as feature')
plt.subplot(122)
plot_confusion_matrix(confusion_matrix(data['dw_state_true'].squeeze(),data['dw_state'].squeeze()),classes=['OFF','ON'], title='dish_washer confussion matrix\n edge as feature')
plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

plt.show()


plt.figure(2)
plt.subplot(121)
plot_confusion_matrix(confusion_matrix(data['wm_state_true'].squeeze(),data['wm_state'].squeeze()),classes=['OFF','ON'], title='washing_machine confussion matrix\n edge as feature')
plt.subplot(122)
plot_confusion_matrix(confusion_matrix(data['k_state_true'].squeeze(),data['k_state'].squeeze()),classes=['OFF','ON'], title='kettle confussion matrix\n edge as feature')
plt.subplots_adjust(left=0.1, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)

plt.show()


print(accuracy_score(data['rm_state_true'].squeeze(),data['rm_state'].squeeze()), precision_score(data['rm_state_true'].squeeze(),data['rm_state'].squeeze()), f1_score(data['rm_state_true'].squeeze(),data['rm_state'].squeeze()))

print(accuracy_score(data['k_state_true'].squeeze(),data['k_state'].squeeze()), precision_score(data['k_state_true'].squeeze(),data['k_state'].squeeze()), f1_score(data['k_state_true'].squeeze(),data['k_state'].squeeze()))

print(accuracy_score(data['wm_state_true'].squeeze(),data['wm_state'].squeeze()), precision_score(data['wm_state_true'].squeeze(),data['wm_state'].squeeze()), f1_score(data['wm_state_true'].squeeze(),data['wm_state'].squeeze()))

print(accuracy_score(data['dw_state_true'].squeeze(),data['dw_state'].squeeze()), precision_score(data['dw_state_true'].squeeze(),data['dw_state'].squeeze()), f1_score(data['dw_state_true'].squeeze(),data['dw_state'].squeeze()))





