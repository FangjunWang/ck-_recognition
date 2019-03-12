import os

cuda_num = 0

class_number = 7#类别数
#图片大小
height = 224
width = 224
crop_height = 224
crop_width = 224
#学习率
lr = 0.0001
lr_decay_every_epoch = 50
lr_decay_rate = 0.9

model_type = 0

#迭代次数
epoch = 100
#batch size
train_batch_size = 10
test_batch_size = 10

thresh = 0.5

#测试频率
test_every_epoch = 5


data_root = 'data/'
train_info = 'label/train_label.txt'
test_info = 'label/test_label.txt'

image_dir = data_root

