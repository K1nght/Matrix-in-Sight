# coding=utf-8
# project
DATA_PATH = "./data/data_v3"
PROJECT_PATH = "data/data_v3/train"
DETECTION_PATH = "./YOLOV4/"

MODEL_TYPE = {
    "TYPE": "YOLOv4"
}  # YOLO type:YOLOv4, Mobilenet-YOLOv4 or Mobilenetv3-YOLOv4

CONV_TYPE = {"TYPE": "DO_CONV"}  # conv type:DO_CONV or GENERAL

ATTENTION = {"TYPE": "NONE"}  # attention type:SEnet、CBAM or NONE

# train
TRAIN = {
    "DATA_TYPE": "Customer",  # DATA_TYPE: Customer
    "TRAIN_IMG_SIZE": 416,
    "AUGMENT": True,
    "BATCH_SIZE": 2,
    "MULTI_SCALE_TRAIN": False,
    "IOU_THRESHOLD_LOSS": 0.5,
    "YOLO_EPOCHS": 50,  # 50
    "Mobilenet_YOLO_EPOCHS": 120,
    "NUMBER_WORKERS": 0,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0005,
    "LR_INIT": 1e-5,
    "LR_END": 1e-7,
    "WARMUP_EPOCHS": 2,  # or None
}


# val
VAL = {
    "TEST_IMG_SIZE": 416,
    "BATCH_SIZE": 2,
    "NUMBER_WORKERS": 0,
    "CONF_THRESH": 0.005,
    "NMS_THRESH": 0.45,
    "MULTI_SCALE_VAL": True,
    "FLIP_VAL": True,
    "Visual": True,
}
#
# Customer_DATA = {
#     "NUM": 7,  # dataset number
#     "CLASSES": [
#         "number",
#         "left_matrix",
#         "right_matrix",
#         "add",
#         "minus",
#         "multi",
#         "T",
#     ],  # dataset class
# }

Customer_DATA = {
    "NUM": 2,  # dataset number
    "CLASSES": [
        "number",
        "T",
    ],  # dataset class
}

# model
MODEL = {
    "ANCHORS": [
        [
            (1.25, 1.625),
            (2.0, 3.75),
            (4.125, 2.875),
        ],  # Anchors for small obj(12,16),(19,36),(40,28)
        [
            (1.875, 3.8125),
            (3.875, 2.8125),
            (3.6875, 7.4375),
        ],  # Anchors for medium obj(36,75),(76,55),(72,146)
        [
            (3.625, 2.8125),
            (4.875, 6.1875),
            (11.65625, 10.1875),
        ],
    ],  # Anchors for big obj(142,110),(192,243),(459,401)
    "STRIDES": [8, 16, 32],
    "ANCHORS_PER_SCLAE": 3,
}
