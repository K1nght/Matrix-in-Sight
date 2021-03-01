import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

STANDARD_COLORS = [
    "Chartreuse",
    "Aqua",
    "Aquamarine",
    "BlueViolet",
    "BurlyWood",
    "CadetBlue",
    "Chocolate",
    "Coral",
    "CornflowerBlue",
    "Cornsilk",
    "Crimson",
    "Cyan",
    "DarkCyan",
    "DarkGoldenRod",
    "DarkGrey",
    "DarkKhaki",
    "DarkOrange",
    "DarkOrchid",
    "DarkSalmon",
    "DarkSeaGreen",
    "DarkTurquoise",
    "DarkViolet",
    "DeepPink",
    "DeepSkyBlue",
    "DodgerBlue",
    "FireBrick",
    "FloralWhite",
    "ForestGreen",
    "Fuchsia",
    "Gainsboro",
    "GhostWhite",
    "Gold",
    ]


txtFile = open("checkpoint/Dec-16[18.19.35]-YOLOv4/log.txt")
txtList = txtFile.readlines()
total_loss = []
loss_ciou = []
loss_conf = []
loss_cls = []
number_map = []
T_map = []
left_map = []
right_map = []
add_map = []
minus_map = []
multi_map = []
total_map = []

# os.makedirs(xmlPath, exist_ok=True)
for idx, line in enumerate(txtList):
    if line.find("total_loss") != -1:
        k = line.find("total_loss")
        k = k + 11
        t = k
        while (line[k] >='0' and line[k] <= '9') or line[k] == '.':

            k = k + 1
        total_loss.append(float(line[t:k]))
    if line.find("loss_ciou") != -1:
        k = line.find("loss_ciou")
        k = k + 10
        t = k
        while (line[k] >='0' and line[k] <= '9') or line[k] == '.':

            k = k + 1
        loss_ciou.append(float(line[t:k]))
    if line.find("loss_conf") != -1:
        k = line.find("loss_conf")
        k = k + 10
        t = k
        while (line[k] >='0' and line[k] <= '9') or line[k] == '.':

            k = k + 1
        loss_conf.append(float(line[t:k]))
    if line.find("loss_cls") != -1:
        k = line.find("loss_cls")
        k = k + 9
        t = k
        while (line[k] >='0' and line[k] <= '9') or line[k] == '.':

            k = k + 1
        loss_cls.append(float(line[t:k]))
    if line.find("number --> mAP : ") != -1:
        k = line.find("number --> mAP : ")
        k = k + len("number --> mAP : ")
        t = k
        while (line[k] >= '0' and line[k] <= '9') or line[k] == '.':
            k = k + 1
        number_map.append(float(line[t:k]))
    if line.find("left_matrix --> mAP : ") != -1:
        k = line.find("left_matrix --> mAP : ")
        k = k + len("left_matrix --> mAP : ")
        t = k
        while (line[k] >= '0' and line[k] <= '9') or line[k] == '.':
            k = k + 1
        left_map.append(float(line[t:k]))
    if line.find("right_matrix --> mAP : ") != -1:
        k = line.find("right_matrix --> mAP : ")
        k = k + len("right_matrix --> mAP : ")
        t = k
        while (line[k] >= '0' and line[k] <= '9') or line[k] == '.':
            k = k + 1
        right_map.append(float(line[t:k]))
    if line.find("add --> mAP : ") != -1:
        k = line.find("add --> mAP : ")
        k = k + len("add --> mAP : ")
        t = k
        while (line[k] >= '0' and line[k] <= '9') or line[k] == '.':
            k = k + 1
        add_map.append(float(line[t:k]))
    if line.find("minus --> mAP : ") != -1:
        k = line.find("minus --> mAP : ")
        k = k + len("minus --> mAP : ")
        t = k
        while (line[k] >= '0' and line[k] <= '9') or line[k] == '.':
            k = k + 1
        minus_map.append(float(line[t:k]))
    if line.find("multi --> mAP : ") != -1:
        k = line.find("multi --> mAP : ")
        k = k + len("multi --> mAP : ")
        t = k
        while (line[k] >= '0' and line[k] <= '9') or line[k] == '.':
            k = k + 1
        multi_map.append(float(line[t:k]))
    if line.find("T --> mAP : ") != -1:
        k = line.find("T --> mAP : ")
        k = k + len("T --> mAP : ")
        t = k
        while (line[k] >= '0' and line[k] <= '9') or line[k] == '.':
            k = k + 1
        T_map.append(float(line[t:k]))
    if line.find(":mAP : ") != -1:
        k = line.find(":mAP : ")
        k = k + len(":mAP : ")
        t = k
        while (line[k] >= '0' and line[k] <= '9') or line[k] == '.':
            k = k + 1
        total_map.append(float(line[t:k]))
print(len(total_loss))
print(len(loss_ciou))
print(len(loss_conf))
print(len(loss_cls))
print(len(number_map))
print(len(T_map))
print(len(total_map))#总mAP

total_loss = np.array(total_loss)
loss_ciou = np.array(loss_ciou)
loss_conf = np.array(loss_conf)
loss_cls = np.array(loss_cls)
number_map = np.array(number_map)
T_map = np.array(T_map)
total_map = np.array(total_map)

# p1, = plt.plot(number_map, c=STANDARD_COLORS[0])
# p2, = plt.plot(T_map, c=STANDARD_COLORS[1])
# p3, = plt.plot(total_map, c=STANDARD_COLORS[2])
# p4, = plt.plot(left_map, c=STANDARD_COLORS[3])
# p5, = plt.plot(right_map, c=STANDARD_COLORS[4])
# p6, = plt.plot(add_map, c=STANDARD_COLORS[5])
# p7, = plt.plot(minus_map, c=STANDARD_COLORS[6])
# p8, = plt.plot(multi_map, c=STANDARD_COLORS[7])
# plt.xlabel("epoch")
# plt.ylabel("mAP")
# plt.legend([p1, p2, p4, p5, p6, p7, p8, p3], ["number", "T", "left_matrix", "right_matrix", "add", "minus", "multi", "Total"])
# plt.grid(True)
# plt.show()

plt.figure(figsize=(18,6))
epochs = np.linspace(0, 50, 1660)
plt.subplot(141)
plt.plot(epochs, total_loss, label="Total loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid(True)
plt.subplot(142)
plt.plot(epochs, loss_ciou, label="Bounding box regression loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid(True)
plt.subplot(143)
plt.plot(epochs, loss_conf, label="Confidence loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid(True)
plt.subplot(144)
plt.plot(epochs, loss_cls, label="Classification loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid(True)

plt.tight_layout()  # 调整整体空白
plt.show()


