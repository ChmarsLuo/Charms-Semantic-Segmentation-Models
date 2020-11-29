# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        #print(hist.shape)


        return hist

    def evaluate(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
            # miou
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        miou = np.nanmean(iou)
        # mean acc
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        return acc, acc_cls, iou, miou, fwavacc


if __name__ == '__main__':
    label_path = './build/test/labels/'
    predict_path = './output/build-1-result/'
    pres = os.listdir(label_path)
    labels = []
    predicts = []
    for im in pres:
        if im[-4:] == '.png':
            label_name = im.split('.')[0] + '.png'
            lab_path = os.path.join(label_path, label_name)
            pre_path = os.path.join(predict_path, im)
            label = cv2.imread(lab_path, 0)
            label[label>0] = 1
            pre = cv2.imread(pre_path, 0)
            print(pre)
            pre[pre>0] = 1
            #pre = cv2.resize(pre, (1500, 1500), interpolation=cv2.INTER_NEAREST)
            labels.append(label)
            predicts.append(pre)
    #print(labels)
    el = IOUMetric(2)
    print('ok')
    acc, acc_cls, iou, miou, fwavacc = el.evaluate(predicts, labels)
    print('acc: ', acc)
    print('acc_cls: ', acc_cls)
    print('iou: ', iou)
    print('miou: ', miou)
    print('fwavacc: ', fwavacc)

    pres = os.listdir(predict_path)
    # print(pres)
    init = np.zeros((2, 2))
    for im in pres:
        label_name = im.split('.')[0] + '.png'
        lb_path = os.path.join(label_path, im)
        # print('ss:',lab_path)
        pre_path = os.path.join(predict_path, im)
        lb = cv2.imread(lb_path, 0)
        # lb = lb/255.0
        # lb = lb[0].item()
        # print(lb)
        pre = cv2.imread(pre_path, 0)
        #pre = cv2.resize(pre, (1500, 1500))
        lb[lb > 0] = 1
        pre[pre > 0] = 1
        lb = lb.flatten()
        pre = pre.flatten()
        confuse = confusion_matrix(lb, pre)
        init += confuse

    precision = init[1][1] / (init[0][1] + init[1][1])
    recall = init[1][1] / (init[1][0] + init[1][1])
    accuracy = (init[0][0] + init[1][1]) / init.sum()
    f1_score = 2 * precision * recall / (precision + recall) #模型准确率和召回率的一种加权平均
    print('class_accuracy: ', precision)
    print('class_recall: ', recall)
    print('accuracy: ', accuracy)
    print('f1_score: ', f1_score)
# !/usr/bin/env python
# coding=utf-8
########################################################################################################################
from xlwt import * # 需要xlwt库的支持 # import xlwt
file = Workbook(encoding='utf-8') # 指定file以utf-8的格式打开
table = file.add_sheet('eval')  # 指定打开的文件名
eval = { ## 字典数据
    "1": ['acc', acc],
    "2": ['acc_cls: ', acc_cls],
    "3": ['iou: ', iou],
    "4": ['miou: ', miou],
    "5": ['fwavacc: ', fwavacc],
    "6": ["class_accuracy", precision],
    "7": ["class_recall", recall],
    "8": ["f1_score", accuracy],
    "9": ['f1_score: ', f1_score]
     }
ldata = []
num = [a for a in eval] # for循环指定取出key值存入num中
#num.sort() # 字典数据取出后无需，需要先排序
for x in num:
    # for循环将data字典中的键和值分批的保存在ldata中
    t = [int(x)]
    for a in eval[x]:
        t.append(a)
    ldata.append(t)
for i, p in enumerate(ldata):
    # 将数据写入文件,i是enumerate()函数返回的序号数
    for j, q in enumerate(p):
        # print i,j,q
        table.write(i, j, str(q))
file.save('./output/build-1-log/eval.csv')
