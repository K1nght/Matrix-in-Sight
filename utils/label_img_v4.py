import cv2
import numpy as np
import os
import sys
import glob
from matplotlib import pyplot as plt

'''
a: +        m:-     c:*     u:drop      y:matrix
'''
# label_class = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'm', 'c', 'u', 'y']
Customer_DATA = {
    "NUM": 13,  # dataset number
    "CLASSES": {
        '0': '0',
        '1': '1',
        '2': '2',
        '3': '3',
        '4': '4',
        '5': '5',
        '6': '6',
        '7': '7',
        '8': '8',
        '9': '9',
        "left_matrix": 'l',
        "right_matrix": 'o',
        "add": 'a',
        "minus": 'm',
        "multi": 't'
    },  # dataset class
}

# generate the class dictionary
label_class = {}
idx = 0
for ch in Customer_DATA["CLASSES"].values():
    label_class[ch] = str(idx)
    idx += 1

window_name = 'test'
move_x = 50
move_y = 50


def accessPiexl(img):
    height = img.shape[0]
    width = img.shape[1]
    for i in range(height):
        for j in range(width):
            img[i][j] = 255 - img[i][j]
    return img


def accessBinary(img, threshold=128):
    img = accessPiexl(img)
    # 边缘膨胀，不加也可以
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    _, img = cv2.threshold(img, threshold, 0, cv2.THRESH_TOZERO)
    return img


def findBorderContours(path, maxArea=50):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = accessBinary(img)
    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    borders = []
    for contour in contours:
        # 将边缘拟合成一个边框
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > maxArea:
            border = [(x, y), (x + w, y + h)]
            borders.append(border)
    return borders


def label_transfer(label):
    return label_class[label]


def showResults(data_path, anno_path, results=None):
    # img = cv2.imread(data_path)

    data_expr = os.path.join(
        data_path, "*.jpg"
    )
    data_paths = glob.glob(data_expr)
    image_ids = [str(i) for i in range(len(data_paths))]

    maxArea = 30

    for ids, path in enumerate(data_paths):
        imgname = os.path.basename(os.path.splitext(path)[0])

        annotation_dir = os.path.join(anno_path, "annotation_%s" % imgname)
        manual_img_dir = os.path.join(anno_path, "manual")
        # information output dir
        os.makedirs(annotation_dir, exist_ok=True)
        os.makedirs(manual_img_dir, exist_ok=True)
        # txt file output
        annotation_path = os.path.join(annotation_dir, "%s.txt" % imgname)
        img_logpath = os.path.join(manual_img_dir, "%s.jpg" % imgname)

        img = cv2.imread(path)
        img_nxt = img.copy()  # the copy of the original img for processing
        img_now = img.copy()  # the copy of the original img for processing

        print('The image is %s' % imgname)

        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        # cv2.namedWindow(window_name, cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow(window_name, img)
        cv2.moveWindow(window_name, move_x, move_y)
        print('press any key to start, \'p\' to pass :  ')
        pass_cmd = chr(cv2.waitKey(0))
        print(pass_cmd)
        # pass_cmd = input('press any key to start, p to pass :  ')

        if pass_cmd == 'p':
            continue

        img_bor = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img_bor = accessBinary(img_bor)
        _, contours, _ = cv2.findContours(img_bor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        borders = []
        for contour in contours:
            # 将边缘拟合成一个边框
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > maxArea:
                border = [(x, y), (x + w, y + h)]
                borders.append(border)

        new_str = ''
        save_flag = False

        for i, border in enumerate(borders):

            cv2.rectangle(img_nxt, border[0], border[1], (255, 0, 0),thickness=1)  # box in blue to label
            cv2.namedWindow(window_name, 0)
            cv2.imshow(window_name, img_nxt)
            cv2.moveWindow(window_name, move_x, move_y)
            label = chr(cv2.waitKey(0))

            while label != 'u' and label_class.get(label, None) is None:
                print('The input is invalide, re-print the label:')
                label = chr(cv2.waitKey(0))

            print('confirm ***%s***? press any to confirm, r to relabel' %label)
            cv2.rectangle(img_nxt, border[0], border[1], (0, 255, 0), thickness=1)  # box in blue to label
            cv2.namedWindow(window_name, 0)
            cv2.imshow(window_name, img_nxt)
            cv2.moveWindow(window_name, move_x, move_y)

            conf_cmd = chr(cv2.waitKey(0))

            if conf_cmd == 'r':
                print('Re-label****', end=' ')
                label = chr(cv2.waitKey(0))

            # if the label is 'u', then drop the box
            if label == 'u':
                img_nxt = img_now.copy()
                cv2.namedWindow(window_name, 0)
                cv2.imshow(window_name, img_nxt)
                cv2.moveWindow(window_name, move_x, move_y)
                print("drop")
                save_flag = True
                continue

            label_tran = label_transfer(label)
            print('The %d rectangle labeld: %s [%s]' % (i + 1, label, label_tran))

            cv2.rectangle(img_now, border[0], border[1], (0, 0, 255), thickness=1)  # save the labeled box in red
            img_nxt = img_now.copy()
            cv2.namedWindow(window_name, 0)
            cv2.imshow(window_name, img_nxt)
            cv2.moveWindow(window_name, move_x, move_y)

            xmin = str(border[0][0])
            ymin = str(border[0][1])
            xmax = str(border[1][0])
            ymax = str(border[1][1])
            new_str += " " + ",".join(
                [xmin, ymin, xmax, ymax, str(label_tran)]
            )

        if os.path.exists(annotation_path):
            os.remove(annotation_path)
            # print(new_str)

        with open(annotation_path, "a") as f:

            annotation = ''  # path of the img
            annotation += new_str  # label info
            annotation += "\n"  # next line
            print(annotation)
            f.write(annotation)
        if save_flag == True:
            if os.path.exists(img_logpath):
                os.remove(img_logpath)

            cv2.imwrite(img_logpath, img_now, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the training data directory.')
    parser.add_argument('--anno_path', type=str, required=True,
                        help='Path to the resulted annotation')
    opt = parser.parse_args()

    data_path = opt.data_path
    anno_path = opt.anno_path
    print("the data being labeled is in: %s" % data_path)
    print("the annotation txt will be saved in: %s" % anno_path)

    # # without the Terminal to run the code
    # anno_path = './anno_data/two_34_43'
    # data_path = './two_34_43crop'

    print(label_class)
    showResults(data_path, anno_path)
