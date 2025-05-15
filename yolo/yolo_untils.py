import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
    
def find_bounding_boxes(image_path) -> tuple[any, list, list]:
    '''
    Output:
        第三个返回值 normalized_cords
        每个元素是四个0~1的数值, 前两个是矩形的中心点, 后两个分别是宽和高, 
        注意都是经过正则化的, 水平方向是img.shape[1], 竖直方向是img.shape[0]
    '''
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化图像 (白色区域变为255，其他区域变为0)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_img = img.copy()
    bounding_boxes = []
    normalized_cords = []
    
    for contour in contours:
        # 计算最小外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        
        # 过滤掉太小的区域（可以根据需要调整阈值）
        if w > 5 and h > 5:
            # 绘制矩形
            cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            bounding_boxes.append((x, y, w, h))
            normalized_cords.append([(x + w/2) / img.shape[1], (y + h/2) / img.shape[0], w / img.shape[1], h / img.shape[0]])

    return result_img, bounding_boxes, normalized_cords

def create_labels(dataset_path = "/home/stellarlane/main/finetune/clinicDB/archive/PNG/", truth_name = "Ground Truth"):
    '''
    按照yolo看得懂的格式根据clinicDB的ground truth来标数据, 其中每个图都会产生一个txt文件, 每一行第一个为类(这里只有一个, 0, 息肉), 后面为坐标(详见get_bounding_boxes)
    产生的.txt文件存在和Ground Truth和Original平行的label文件夹下
    由于其他数据集
    '''
    os.mkdir(f"{dataset_path}label") if not os.path.exists(f"{dataset_path}label") else None
    total = len(os.listdir(f"{dataset_path}{truth_name}"))
    for i in range(1, total + 1):
        _, _, labels = find_bounding_boxes(f"{dataset_path}{truth_name}/{i}.png")
        label_file_path = f"{dataset_path}label/{i}.txt"
        with open(label_file_path, 'w') as f:
            # 遍历每个检测到的边界框
            for box in labels:
                # YOLO格式：<class_id> <center_x> <center_y> <width> <height>
                # 这里类别ID使用0（息肉）
                line = f"0 {box[0]} {box[1]} {box[2]} {box[3]}\n"
                f.write(line)

def read_txt_lines(file_path) -> list:
    '''
    Output:
        读取txt文件的每一行, 返回一个list
    '''
    lines = []
    with open(file_path, 'r') as f:
        for line in f:
            lines.append(line.strip())  # strip() removes newline characters
    return lines

def build_dataset_multipleDB(ratio: float = 0.8, 
                             target_path: str = "/home/stellarlane/main/finetune/yolo/datasets/clinicDB/", 
                             source_paths: list = "/home/stellarlane/main/finetune/clinicDB/archive/PNG/", 
                             image_folder_name: str = "Original",):
    '''
    ratio: 训练集和验证集的比例
    target_path: 目标路径
    source_paths: 源路径列表, 每个路径下都有一个images和label文件夹
    '''
    if not os.path.exists(target_path):
        os.mkdir(target_path)
        os.mkdir(f"{target_path}images")
        os.mkdir(f"{target_path}images/train")
        os.mkdir(f"{target_path}images/val")
        os.mkdir(f"{target_path}labels")
        os.mkdir(f"{target_path}labels/train")
        os.mkdir(f"{target_path}labels/val")

        prev_train_indices = 0
        prev_val_indices = 0
        for source_path in source_paths:
            all_indices = np.arange(1, len(os.listdir(f"{source_path}{image_folder_name}")) + 1)
            train_indices = np.random.choice(all_indices, int(len(all_indices) * ratio), replace=False)
            val_indices = np.setdiff1d(all_indices, train_indices)
            
            for ind in train_indices:
                shutil.copy(f"{source_path}{image_folder_name}/{ind}.png", f"{target_path}images/train/{prev_train_indices + ind}.png")
                shutil.copy(f"{source_path}label/{ind}.txt", f"{target_path}labels/train/{prev_train_indices + ind}.txt")
            print(f"{source_path}: {len(train_indices)} train images, {len(val_indices)} val images")
            prev_train_indices += len(train_indices)
            
            for ind in val_indices:
                shutil.copy(f"{source_path}{image_folder_name}/{ind}.png", f"{target_path}images/val/{prev_val_indices + ind}.png")
                shutil.copy(f"{source_path}label/{ind}.txt", f"{target_path}labels/val/{prev_val_indices + ind}.txt")
            prev_val_indices += len(val_indices)

def build_standalone_test_dataset(target_path: str, source_path: str):
    '''
    单独的测试数据集
    '''
    if os.path.exists(f"{target_path}"):
        os.mkdir(f"{target_path}images/test/")
        os.mkdir(f"{target_path}labels/test/")
        all_indices = np.arange(1, len(os.listdir(f"{source_path}/images")) + 1)
        for ind in all_indices:
            shutil.copy(f"{source_path}images/{ind}.png", f"{target_path}images/test/{ind}.png")
            shutil.copy(f"{source_path}label/{ind}.txt", f"{target_path}labels/test/{ind}.txt")
        print(f"test: {len(all_indices)} test images")
    else: print("No train&val test")
            
def build_dataset_multipleDB_with_test(ratio: list = [0.8, 0.1, 0.1],
                                       target_path: str = "/home/stellarlane/main/finetune/yolo/datasets/clinicDB/",
                                       source_paths: list = "/home/stellarlane/main/finetune/clinicDB/archive/PNG/",
                                       image_folder_name: str = "Original",):
    '''
    包含了测试集的数据集划分,
    ratio中分别为训练集, 验证集, 测试集的比例
    '''
    if not os.path.exists(target_path):
            os.mkdir(target_path)
            os.mkdir(f"{target_path}images")
            os.mkdir(f"{target_path}images/train")
            os.mkdir(f"{target_path}images/val")
            os.mkdir(f"{target_path}images/test")
            os.mkdir(f"{target_path}labels")
            os.mkdir(f"{target_path}labels/train")
            os.mkdir(f"{target_path}labels/val")
            os.mkdir(f"{target_path}labels/test")

            prev_train_indices = 0
            prev_val_indices = 0
            prev_test_indices = 0
            for source_path in source_paths:
                all_indices = np.arange(1, len(os.listdir(f"{source_path}{image_folder_name}")) + 1)
                train_indices = np.random.choice(all_indices, int(len(all_indices) * ratio[0]), replace=False)
                val_indices = np.setdiff1d(all_indices, train_indices)
                test_indices = np.random.choice(val_indices, int(len(val_indices) * (ratio[2] / (ratio[1] + ratio[2]))), replace=False)
                val_indices = np.setdiff1d(val_indices, test_indices)
                for ind in train_indices:
                    shutil.copy(f"{source_path}{image_folder_name}/{ind}.png", f"{target_path}images/train/{prev_train_indices + ind}.png")
                    shutil.copy(f"{source_path}label/{ind}.txt", f"{target_path}labels/train/{prev_train_indices + ind}.txt")
                print(f"{source_path}: {len(train_indices)} train images, {len(val_indices)} val images", f"{len(test_indices)} test images")
                prev_train_indices += len(train_indices)
                
                for ind in val_indices:
                    shutil.copy(f"{source_path}{image_folder_name}/{ind}.png", f"{target_path}images/val/{prev_val_indices + ind}.png")
                    shutil.copy(f"{source_path}label/{ind}.txt", f"{target_path}labels/val/{prev_val_indices + ind}.txt")
                prev_val_indices += len(val_indices)

                for ind in test_indices:
                    shutil.copy(f"{source_path}{image_folder_name}/{ind}.png", f"{target_path}images/test/{prev_test_indices + ind}.png")
                    shutil.copy(f"{source_path}label/{ind}.txt", f"{target_path}labels/test/{prev_test_indices + ind}.txt")
                prev_test_indices += len(test_indices)
        


def build_dataset_clinicDB(ratio, target_path = "./datasets/clinicDB/", source_path = '../clinicDB/archive/PNG/'):
    '''
    创建一个ratio开的数据集, 测试用
    '''
    if not os.path.exists(target_path):
        os.mkdir(target_path)
        os.mkdir(f"{target_path}images")
        os.mkdir(f"{target_path}images/train")
        os.mkdir(f"{target_path}images/val")
        os.mkdir(f"{target_path}labels")
        os.mkdir(f"{target_path}labels/train")
        os.mkdir(f"{target_path}labels/val")
        all_indices = np.arange(1, len(os.listdir(f"{source_path}Original")) + 1)
        train_indices = np.random.choice(all_indices, int(len(all_indices) * ratio), replace=False)
        val_indices = np.setdiff1d(all_indices, train_indices)
        for ind in train_indices:
            shutil.copy(f"{source_path}Original/{ind}.png", f"{target_path}images/train/{ind}.png")
            shutil.copy(f"{source_path}label/{ind}.txt", f"{target_path}labels/train/{ind}.txt")
        for ind in val_indices:
            shutil.copy(f"{source_path}Original/{ind}.png", f"{target_path}images/val/{ind}.png")
            shutil.copy(f"{source_path}label/{ind}.txt", f"{target_path}labels/val/{ind}.txt")
        print(f"ratio: {ratio}\n"
            f"train: {len(os.listdir(f'{target_path}images/train'))} / {len(os.listdir(f'{source_path}Original'))}\n"
            f"val: {len(os.listdir(f'{target_path}images/val'))} / {len(os.listdir(f'{source_path}Original'))}\n"
            f"total: {len(os.listdir(f'{source_path}Original'))}")