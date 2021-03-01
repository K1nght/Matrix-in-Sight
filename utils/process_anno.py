import sys
import glob

sys.path.append("..")
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm



Customer_DATA = {
    "NUM": 2,  # dataset number
    "CLASSES": [
        "number",
        "T",
    ],  # dataset class
}


def parse_voc_annotation(
    data_path, file_type, anno_path, use_difficult_bbox=False
):
    classes = Customer_DATA["CLASSES"]

    data_expr = os.path.join(
                data_path, file_type, "*.jpg"
            )
    data_paths = glob.glob(data_expr)
    image_ids = [str(i) for i in range(len(data_paths))]

    with open(anno_path, "a") as f:
        for image_id in tqdm(image_ids):
            new_str = ''
            image_path = os.path.join(
                data_path, file_type, image_id + ".jpg"
            )
            annotation = image_path
            label_path = os.path.join(
                data_path, 'label', image_id + ".xml"
            )
            root = ET.parse(label_path).getroot()
            objects = root.findall("object")
            for obj in objects:
                difficult = obj.find("difficult").text.strip()
                if (not use_difficult_bbox) and (
                    int(difficult) == 1
                ):  # difficult 表示是否容易识别，0表示容易，1表示困难
                    continue
                bbox = obj.find("bndbox")
                class_id = classes.index(obj.find("name").text.lower().strip())
                xmin = bbox.find("xmin").text.strip()
                ymin = bbox.find("ymin").text.strip()
                xmax = bbox.find("xmax").text.strip()
                ymax = bbox.find("ymax").text.strip()
                new_str += " " + ",".join(
                    [xmin, ymin, xmax, ymax, str(class_id)]
                )
            if new_str == '':
                continue
            annotation += new_str
            annotation += "\n"
            # print(annotation)
            f.write(annotation)
    return len(image_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the training data directory.')
    parser.add_argument('--file_type', type=str, required=True, choices=['train', 'val', 'test'],
                        help='[train|val|test]')
    parser.add_argument('--anno_path', type=str, required=True,
                        help='Path to the resulted annotation')
    opt = parser.parse_args()

    data_path = opt.data_path
    file_type = opt.file_type
    annotation_path = os.path.join(opt.anno_path, "%s_annotation.txt" % file_type)
    if os.path.exists(annotation_path):
        os.remove(annotation_path)

    print(
        "The number of images for {0} are: {1}".format(
            file_type,
            parse_voc_annotation(
                data_path,
                file_type,
                annotation_path,
                use_difficult_bbox=False,
            )
        )
    )