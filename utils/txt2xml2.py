from xml.dom.minidom import Document
import os
import cv2

Customer_DATA = {
    "NUM": 7,  # dataset number
    "CLASSES": [
        "number",
        "left_matrix",
        "right_matrix",
        "add",
        "minus",
        "multi",
        "T",
    ],  # dataset class
}

label_class = {}
for idx, s in enumerate(Customer_DATA["CLASSES"]):
    label_class[str(idx)] = s


def add_bndbox(xmlBuilder, annotation, Pwidth, Pheight, Pdepth, name, label, a, b, c, d):

    object = xmlBuilder.createElement("object")
    picname = xmlBuilder.createElement("name")
    nameContent = xmlBuilder.createTextNode(label_class[label])
    picname.appendChild(nameContent)
    object.appendChild(picname)
    pose = xmlBuilder.createElement("pose")
    poseContent = xmlBuilder.createTextNode("Unspecified")
    pose.appendChild(poseContent)
    object.appendChild(pose)
    truncated = xmlBuilder.createElement("truncated")
    truncatedContent = xmlBuilder.createTextNode("0")
    truncated.appendChild(truncatedContent)
    object.appendChild(truncated)
    difficult = xmlBuilder.createElement("difficult")
    difficultContent = xmlBuilder.createTextNode("0")
    difficult.appendChild(difficultContent)
    object.appendChild(difficult)
    bndbox = xmlBuilder.createElement("bndbox")
    xmin = xmlBuilder.createElement("xmin")
    mathData = int(a)
    xminContent = xmlBuilder.createTextNode(str(mathData))
    xmin.appendChild(xminContent)
    bndbox.appendChild(xmin)
    ymin = xmlBuilder.createElement("ymin")
    mathData = int(b)
    yminContent = xmlBuilder.createTextNode(str(mathData))
    ymin.appendChild(yminContent)
    bndbox.appendChild(ymin)
    xmax = xmlBuilder.createElement("xmax")
    mathData = int(c)
    xmaxContent = xmlBuilder.createTextNode(str(mathData))
    xmax.appendChild(xmaxContent)
    bndbox.appendChild(xmax)
    ymax = xmlBuilder.createElement("ymax")
    mathData = int(d)
    ymaxContent = xmlBuilder.createTextNode(str(mathData))
    ymax.appendChild(ymaxContent)
    bndbox.appendChild(ymax)
    object.appendChild(bndbox)

    annotation.appendChild(object)

def makexml(txtPath, xmlPath):  # 读取txt路径，xml保存路径，数据集图片所在路径

    txtFile = open(txtPath)
    txtList = txtFile.readlines()

    os.makedirs(xmlPath, exist_ok=True)
    for idx, line in enumerate(txtList):
        oneline = line.strip().split(" ")
        # print(oneline)
        name = oneline[0].split("\\")
        print(name)
        # top_dir, folder, filename = name
        folder, filename = name
        # print(name)

        xmlBuilder = Document()
        annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
        xmlBuilder.appendChild(annotation)
        img = cv2.imread(oneline[0])
        Pheight, Pwidth, Pdepth = img.shape
        # print(Pheight, Pwidth, Pdepth)
        
        folderC = xmlBuilder.createElement("folder")  # folder标签
        folderContent = xmlBuilder.createTextNode(folder)
        folderC.appendChild(folderContent)
        annotation.appendChild(folderC)

        filenameC = xmlBuilder.createElement("filename")  # filename标签
        filenameContent = xmlBuilder.createTextNode(filename)
        filenameC.appendChild(filenameContent)
        annotation.appendChild(filenameC)

        size = xmlBuilder.createElement("size")  # size标签
        width = xmlBuilder.createElement("width")  # size子标签width
        widthContent = xmlBuilder.createTextNode(str(Pwidth))
        width.appendChild(widthContent)
        size.appendChild(width)
        height = xmlBuilder.createElement("height")  # size子标签height
        heightContent = xmlBuilder.createTextNode(str(Pheight))
        height.appendChild(heightContent)
        size.appendChild(height)
        depth = xmlBuilder.createElement("depth")  # size子标签depth
        depthContent = xmlBuilder.createTextNode(str(Pdepth))
        depth.appendChild(depthContent)
        size.appendChild(depth)
        annotation.appendChild(size)

        segmented = xmlBuilder.createElement("segmented")
        segmentedContent = xmlBuilder.createTextNode(str(0))
        segmented.appendChild(segmentedContent)
        annotation.appendChild(segmented)
        
        for content in oneline[1:]:
            a, b, c, d, label = content.strip().split(",")
            # print(a, b, c, d, label)
            add_bndbox(xmlBuilder, annotation,
                       Pwidth, Pheight, Pdepth, name, label, a, b, c, d)
        xmlfilePath = os.path.join(xmlPath, filename[:-4] + ".xml")
        f = open(xmlfilePath, 'w')
        xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
        f.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--txtpath', type=str, required=True,
                        help='Path to the txt')
    parser.add_argument('--xmlpath', type=str, required=True,
                        help='Path to the xml')

    opt = parser.parse_args()
    makexml(opt.txtpath, opt.xmlpath)
