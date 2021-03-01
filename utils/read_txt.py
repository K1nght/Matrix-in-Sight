import os
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('seaborn')


txtFile = open("checkpoint/Dec-26[10.52.33]-YOLOv4-good/log.txt")
txtList = txtFile.readlines()
total_loss = []
loss_ciou = []
loss_conf = []
loss_cls = []
number_map = []
T_map = []
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

p1, = plt.plot(number_map)
p2, = plt.plot(T_map)
p3, = plt.plot(total_map)
plt.xlabel("epoch")
plt.ylabel("mAP")
plt.legend([p1, p2, p3], ["number", "T", "Total"])
plt.grid(True)
plt.show()

# plt.figure(figsize=(18,6))
# epochs = np.linspace(0, 50, 1900)
# plt.subplot(141)
# plt.plot(epochs, total_loss, label="Total loss")
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.legend()
# plt.grid(True)
# plt.subplot(142)
# plt.plot(epochs, loss_ciou, label="Bounding box regression loss")
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.legend()
# plt.grid(True)
# plt.subplot(143)
# plt.plot(epochs, loss_conf, label="Confidence loss")
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.legend()
# plt.grid(True)
# plt.subplot(144)
# plt.plot(epochs, loss_cls, label="Classification loss")
# plt.xlabel("epoch")
# plt.ylabel("loss")
# plt.legend()
# plt.grid(True)
#
# plt.tight_layout()  # 调整整体空白
# plt.show()


