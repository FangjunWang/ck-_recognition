import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
    
def statistics(pred,label,thresh=0.5):
    #confusion matrix
    size = pred.size(0)
    class_no = pred.size(1)
    pred = pred > thresh
    pred = pred.long()
    pred[pred == 0] = -1
    statistics_list = []
    for j in range(class_no):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(size):
            if pred[i][j] == 1:
                if label[i][j] == 1:
                    TP += 1
                elif label[i][j] == -1:
                    FP += 1
            elif pred[i][j] == -1:
                if label[i][j] == 1:
                    FN += 1
                elif label[i][j] == -1:
                    TN += 1
        statistics_list.append({'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN})
    return statistics_list

def update_statistics(old_list,new_list):
    for i in range(len(old_list)):
        old_list[i]['TP'] += new_list[i]['TP']
        old_list[i]['FP'] += new_list[i]['FP']
        old_list[i]['TN'] += new_list[i]['TN']
        old_list[i]['FN'] += new_list[i]['FN']
    return old_list
    
def f1_score(statistics_list):
    #calculate f1 score
    f1_score_list = []
    for i in range(len(statistics_list)):
        TP = statistics_list[i]['TP']
        FP = statistics_list[i]['FP']
        FN = statistics_list[i]['FN']
        precise = TP / (TP + FP + 1e-20)
        recall = TP / (TP + FN + 1e-20)
        f1_score = 2 * precise * recall / (precise + recall + 1e-20)
        f1_score_list.append(f1_score)
    mean_f1_score = sum(f1_score_list) / len(f1_score_list)
    return mean_f1_score, f1_score_list
    
def acc(pred,tar):
    count = 0
    for i in range(len(pred)):
        ind = 0
        m = 0
        for j in range(7):
            if pred[i][j]>m:
                m = pred[i][j]
                ind = j
        if tar[i][ind]==1:
            count+=1
    return count * 1. /len(pred) 