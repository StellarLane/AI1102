import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
    
def seperate_mask(mask_path) -> list:
    '''

    '''
    img = cv2.imread(mask_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化图像 (白色区域变为255，其他区域变为0)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    masks = []
    points = []
    for contour in contours:
        mask = np.zeros(img.shape, dtype=np.uint8)
        pts = np.array([contour], dtype=np.int32)
        mask = cv2.fillPoly(mask, pts, (255, 255, 255))
        _, binary = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
        masks.append(np.array(binary))

        # 计算最小外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        points.append([x + w/2, y + h/2])
        # 将mask转换为2D纯二值掩码，0为黑，1为白
        mask_2d = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_2d = (mask_2d > 0).astype(np.uint8)
        masks[-1] = mask_2d
    return np.array(masks), np.array(points)

def load_one_data(dataset_path = "../clinicDB/", org_folder = "images", mask_folder = "masks") -> tuple[any, list, list]:
    '''
    Output:
        img: 图片
        masks: masks组
        points: 每个mask中的一个点(外接矩形中心)
        labels: 每个mask的label, 这里都是1
    '''
    i = np.random.choice(range(1, len(os.listdir(f'{dataset_path}{org_folder}')) + 1))
    img = cv2.imread(f"{dataset_path}{org_folder}/{i}.png")[...,::-1]  # read image
    masks, points = seperate_mask(f"{dataset_path}{mask_folder}/{i}.png")
    labels = np.ones(len(masks))
    return img, masks, points, labels

def build_dataset(ratio = 0.95, name = "dataset_0424"):
    if not os.path.exists(name):
        os.mkdir(name)
        os.mkdir(f"{name}/train")
        os.mkdir(f"{name}/train/images")
        os.mkdir(f"{name}/train/masks")
        os.mkdir(f"{name}/val")
        os.mkdir(f"{name}/val/images")
        os.mkdir(f"{name}/val/masks")

        j, k = 0, 0
        # clinicDB
        print("Current working directory:", os.getcwd())
        path_org, path_mask = "/home/stellarlane/main/finetune/clinicDB/images", "../clinicDB/Ground Truth"
        train_ind = np.random.choice(range(1, len(os.listdir(path_org)) + 1), int(len(os.listdir(path_org)) * ratio), replace=False)
        for i in range(1, len(os.listdir(path_org)) + 1):
            if i in train_ind:
                j+=1
                shutil.copy(f"{path_org}/{i}.png", f"{name}/train/images/{j}.png")
                shutil.copy(f"{path_mask}/{i}.png", f"{name}/train/masks/{j}.png")
            else:
                k+=1
                shutil.copy(f"{path_org}/{i}.png", f"{name}/val/images/{k}.png")
                shutil.copy(f"{path_mask}/{i}.png", f"{name}/val/masks/{k}.png")
        print(f"clinicDB train: {len(train_ind)}, val: {len(os.listdir(path_org)) - len(train_ind)}")

        # CVC-ColonDB
        path_org, path_mask = "../CVC-ColonDB/images", "../CVC-ColonDB/masks"
        train_ind = np.random.choice(range(1, len(os.listdir(path_org)) + 1), int(len(os.listdir(path_org)) * ratio), replace=False)
        for i in range(1, len(os.listdir(path_org)) + 1):
            if i in train_ind:
                j+=1
                shutil.copy(f"{path_org}/{i}.png", f"{name}/train/images/{j}.png")
                shutil.copy(f"{path_mask}/{i}.png", f"{name}/train/masks/{j}.png")
            else:
                k+=1
                shutil.copy(f"{path_org}/{i}.png", f"{name}/val/images/{k}.png")
                shutil.copy(f"{path_mask}/{i}.png", f"{name}/val/masks/{k}.png")
        print(f"CVC-ColonDB train: {len(train_ind)}, val: {len(os.listdir(path_org)) - len(train_ind)}")

        #ETIS
        path_org, path_mask = "../ETIS/images", "../ETIS/masks"
        train_ind = np.random.choice(range(1, len(os.listdir(path_org)) + 1), int(len(os.listdir(path_org)) * ratio), replace=False)
        for i in range(1, len(os.listdir(path_org)) + 1):
            if i in train_ind:
                j+=1
                shutil.copy(f"{path_org}/{i}.png", f"{name}/train/images/{j}.png")
                shutil.copy(f"{path_mask}/{i}.png", f"{name}/train/masks/{j}.png")
            else:
                k+=1
                shutil.copy(f"{path_org}/{i}.png", f"{name}/val/images/{k}.png")
                shutil.copy(f"{path_mask}/{i}.png", f"{name}/val/masks/{k}.png")
        print(f"ETIS train: {len(train_ind)}, val: {len(os.listdir(path_org)) - len(train_ind)}")


