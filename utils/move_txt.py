import sys
import glob
import os
from tqdm import tqdm
import numpy as np
from PIL import Image

output_paths1 = 'data/train/train'
os.makedirs(output_paths1, exist_ok=True)
output_anno1 = 'data/train/train_annotation.txt'
output_paths2 = 'data/val/val'
os.makedirs(output_paths2, exist_ok=True)
output_anno2 = 'data/val/val_annotation.txt'
target_anno = 'data_v6/all.txt'

anno_file = open(target_anno)
anno_list = anno_file.readlines()

train_num = 0
val_num = 0

for idx, line in enumerate(tqdm(anno_list)):
    # if idx > 30:
    #     break
    oneline = line.strip().split(" ")
    img_path = oneline[0]
    img = Image.open(img_path).convert('RGB')
    if idx % 50 == 0:
        output_path = os.path.join(output_paths2, str(val_num) + ".jpg")
        val_num += 1
        img.save(output_path, format='JPEG', subsampling=0, quality=100)
        oneline[0] = output_path
        oneline = " ".join(oneline)
        # print(idx, oneline)
        oneline += "\n"
        f = open(output_anno2, "a")
        f.write(oneline)
        f.close()
    else:
        output_path = os.path.join(output_paths1, str(train_num) + ".jpg")
        train_num += 1
        img.save(output_path, format='JPEG', subsampling=0, quality=100)
        oneline[0] = output_path
        oneline = " ".join(oneline)
        # print(idx, oneline)
        oneline += "\n"
        f = open(output_anno1, "a")
        f.write(oneline)
        f.close()
print("finish convert %d train to %s" % (train_num, output_paths1))
print("finish convert %d val to %s" % (val_num, output_paths2))
#
# with open(output_anno, "a") as f:
#     for idx, line in enumerate(anno_list):
#         oneline = line.strip().split(" ")
#         img_path = oneline[0]
#         img = Image.open(img_path).convert('RGB')
#         output_path = os.path.join(output_paths, str(idx) + ".jpg")
#         img.save(output_path, format='JPEG', subsampling=0, quality=100)
#         oneline[0] = output_path
#         oneline = " ".join(oneline)
#         print(idx, oneline)
#         oneline += "\n"
#         f.write(oneline)
# f.close()
