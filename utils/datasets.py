import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import pandas as pd
import datetime
import re
import math

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)



aaa = set()
def cell_time_yuchuli_proc(cell):#修复表格里面的格式问题，生成Xlsx
    pattern = re.compile(r'\W+')#匹配符号1：2这样的
    num_pattern = re.compile(r'\d+')#匹配数字这样的30
    if isinstance(cell, int) or isinstance(cell, float) :
        if cell == 0 or cell == 0.0:
            return "0"
        return  str(int(cell))#预处理时统一处理成STRING
    elif isinstance(cell ,datetime.datetime):
        return "###"#非法字符，返回空，填写原地址
    elif isinstance(cell ,datetime.time):
        return f"{cell.hour}:{cell.minute}"
    elif isinstance(cell, str):
        cell = cell.strip()
        if pattern.search(cell):
            minute, second = pattern.split(cell)#有三个就报错，提醒的作用
            return f"{int(minute)}:{int(second)}" #返回修改成标准格式，带一个空格
        elif num_pattern.search(cell):
            return cell
        elif cell == "0":
            return cell
        if cell.strip():
            print(cell)
        return ""
    return ""

def cell_time_proc(cell):#这里不能打断点
    if isinstance(cell , int) or isinstance(cell , float):
        return int(cell)#不考虑浮点数
    elif isinstance(cell , datetime.time):#datetime.time表示的时间
        return cell.minute + cell.hour*60
        # return cell
    elif isinstance(cell , str):#有字符串表示的时间

        pattern = re.compile(r'\W+')
        if pattern.search(cell):#字符串中有间隔符,则自动计算时间
            minute, second = pattern.split(cell)
            minute = int(minute)
            second = int(second)
            return second + minute*60
        elif cell == "":
            return cell
        else :
            return int(cell) ##字符串中没有间隔符,则直接转换成时间
    else:#1900:00:00  异常数据
        return ""

from enum import Enum
class music_type(Enum):
    kp = 1
    pg = 2
    xs = 3
    gc = 4

kp = "空拍"
pg = "平鼓"
xs = "向上"
gc = "高潮"
def cell_type_proc(cell):
    if cell == "":
        return cell
    elif not cell:
        return 111
    elif isinstance(cell, str) and (cell in [pg, kp , xs ,gc]):
        if cell == kp:
            return 1
        if cell == pg:
            return 2
        if cell == xs:
            return 3
        if cell == gc:
            return 4
        return cell
    # else:
    #     raise

def cell_origin_proc(cell):#这里不能打断点
    if isinstance(cell , str):#有字符串表示的时间
        return cell.replace("\\",'/')


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        sub_file = 'data/custom/' + 'data.xlsx'
        sub_file_yuchuli = 'data/custom/' + 'yuchulihou.xlsx'
        sub_file_all_ndarray = 'data/custom/' + 'all_ndarray.xlsx'

        excel = pd.read_excel(io=sub_file, header=1)  # 0是第一行
        excel = excel.fillna("")
        for col_name in excel.columns.tolist():
            if isinstance(col_name, str):
                if col_name.startswith("time-end") or col_name.startswith("time-start"):
                    excel[col_name] = excel[col_name].apply(cell_time_proc)
                    # excel[col_name] = excel[col_name].apply(cell_time_yuchuli_proc)#预处理和找错误使用
                    # excel[col_name] = excel[col_name].astype(int)#设置单元格格式
                elif col_name.startswith("type-specific"):
                    continue  # 暂时不用预处理这列
                elif col_name.startswith("type"):
                    excel[col_name] = excel[col_name].apply(cell_type_proc)   ##预处理
                elif col_name.startswith("origin") or col_name.startswith("drum"):
                    excel[col_name] = excel[col_name].apply(cell_origin_proc)  ##预处理

        # excel.values[0][4]
        # excel.loc[[0, 1, 2, 3], ['type0', 'type1']].to_numpy()#完美 取4行，
        # excel.to_excel(sub_file_yuchuli , index=False) #预处理，保存
        # excel.to_excel(sub_file_all_ndarray, index=False)  # 全部处理成NDARRAY，保存
        arrary = excel.loc[:, [
                                "origin",
                                'time-start0', 'time-end0', 'type0',   'time-start1', 'time-end1', 'type1',   'time-start2', 'time-end2', 'type2',
                                'time-start3', 'time-end3', 'type3',   'time-start4', 'time-end4', 'type4',   'time-start5', 'time-end5', 'type5',
                                'time-start6', 'time-end6', 'type6',   'time-start7', 'time-end7', 'type7',   'time-start8', 'time-end8', 'type8',
                                'time-start9', 'time-end9', 'type9',   'time-start10', 'time-end10', 'type10',  'time-start11', 'time-end11', 'type11',
                                'time-start12', 'time-end12', 'type12', 'time-start13', 'time-end13', 'type13', 'time-start14', 'time-end14', 'type14',
                                ]].to_json(r'filename1.json', lines=True, orient="records")  # 保存json文件的字典格式，不用

        # 数据清洗

        # 填充0值  #前面已经整体上粗糙的填充了，前面是按值传递，需要赋值
        excel[
            'time-start0'
        ].fillna("", inplace=True)#inplace=True :代表按地址传递

        # 按列限定类型
        # excel = excel.astype({"time-start0": int, "origin": str})
        # print(excel)

        # 截取子的DF，同时限定顺序
        sub_excel = excel.loc[:, [
                                    "origin",
                                    'time-start0', 'time-end0', 'type0',   'time-start1', 'time-end1', 'type1',   'time-start2', 'time-end2', 'type2',
                                    'time-start3', 'time-end3', 'type3',   'time-start4', 'time-end4', 'type4',   'time-start5', 'time-end5', 'type5',
                                    'time-start6', 'time-end6', 'type6',   'time-start7', 'time-end7', 'type7',   'time-start8', 'time-end8', 'type8',
                                    'time-start9', 'time-end9', 'type9',   'time-start10', 'time-end10', 'type10',  'time-start11', 'time-end11', 'type11',
                                    'time-start12', 'time-end12', 'type12', 'time-start13', 'time-end13', 'type13', 'time-start14', 'time-end14', 'type14',
                                    ]]
        print(sub_excel)
        # 保存
        pkl_path = r'sub_data.pkl'
        sub_excel.to_pickle(pkl_path)

        # 加载
        sub = pd.read_pickle(pkl_path)
        for r in sub.values.tolist():
            print(r)


        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
