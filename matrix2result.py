import utils.gpu as gpu
from model.build_model import Build_Model
from utils.tools import *
from eval.evaluator import Evaluator
import argparse
import time
import logging
import config.yolov4_config as cfg
from utils.torch_utils import *
from graph import *
from utils.img_crop_detecv3 import *
from det_numbers.num_model import *

min_score_thresh = 0.6


def get_points(img, bboxes_prd, model, mnist=False):
    if bboxes_prd.shape[0] != 0:
        points = {}
        max_l = 0
        for label in cfg.Customer_DATA["CLASSES"]:
            points[label] = []
        boxes = bboxes_prd[..., :4]
        class_inds = bboxes_prd[..., 5].astype(np.int32)
        scores = bboxes_prd[..., 4]
        sorted_ind = np.argsort(-scores)
        boxes = boxes[sorted_ind]
        scores = scores[sorted_ind]
        classes = class_inds[sorted_ind]

        category_index = {}
        for id_, label_name in enumerate(cfg.Customer_DATA["CLASSES"]):
            category_index[id_] = {"name": label_name}
        for i in range(boxes.shape[0]):
            if scores[i] > min_score_thresh:
                xmin, ymin, xmax, ymax = tuple(boxes[i].tolist())
                xmin, ymin, xmax, ymax = round(xmin), round(ymin), round(xmax), round(ymax)
                max_l = max(xmax - xmin, ymax - ymin, max_l)
                score = scores[i]
                label = category_index[classes[i]]["name"]
                x, y = (xmin + xmax) / 2, (ymin + ymax) / 2
                # 防止数字重叠
                if label == 'number':
                    idx = img_proce(img, xmin, ymin, xmax, ymax, model=model, mnist=mnist)
                    if not mnist and len(idx) == 0:
                        idx = '0'
                    drop = False
                    for i, (idx2, x2, y2, xmin2, ymin2, xmax2, ymax2, score2) in enumerate(points['number']):
                        if (xmin2 < x < xmax2 and ymin2 < y < ymax2) or (xmin < x2 < xmax and ymin < y2 < ymax):
                            if (xmax - xmin) * (ymax - ymin) < (xmax2 - xmin2) * (ymax2 - ymin2):
                                drop = True
                                break
                            else:
                                points['number'][i] = [int(idx), x, y, xmin, ymin, xmax, ymax, score]
                                drop = True
                                break
                    if not drop:
                        points['number'].append([int(idx), x, y, xmin, ymin, xmax, ymax, score])
                # 其他情况取分高的
                else:
                    idx = None
                    drop = False
                    for l in points.keys():
                        for i, (idx2, x2, y2, xmin2, ymin2, xmax2, ymax2, score2) in enumerate(points[l]):
                            if (xmin2 < x < xmax2 and ymin2 < y < ymax2) or (xmin < x2 < xmax and ymin < y2 < ymax):
                                if score < score2:
                                    drop = True
                                    break
                                else:
                                    del points[l][i]
                    if not drop:
                        points[label].append([idx, x, y, xmin, ymin, xmax, ymax, score])

        return points, max_l

    else:
        raise NotImplementedError("Haven't detected any numbers in the picture")


class Evaluation(object):
    def __init__(
            self,
            gpu_id=0,
            model1_path=None,
            model2_path=None,
            data_dir=None,
            # result_dir=None,
            mnist=False,
    ):
        self.__num_class = cfg.Customer_DATA["NUM"]
        self.__conf_threshold = cfg.VAL["CONF_THRESH"]
        self.__nms_threshold = cfg.VAL["NMS_THRESH"]
        self.__device = gpu.select_device(gpu_id)
        self.__multi_scale_val = cfg.VAL["MULTI_SCALE_VAL"]
        self.__flip_val = cfg.VAL["FLIP_VAL"]

        self.__data_dir = data_dir
        print(self.__data_dir)
        self.__classes = cfg.Customer_DATA["CLASSES"]
        self.__mnist = mnist
        self.__model1 = Build_Model().to(self.__device)
        if mnist:
            self.__model2 = torch.load(model2_path).double().cuda()
        else:
            self.__model2 = torch.load(model2_path).cuda()

        self.__load_model_weights(model1_path)

        self.__evalter = Evaluator(self.__model1, showatt=False)
        # self.__result_dir = result_dir

    def __load_model_weights(self, weight_path):
        print("loading weight file from : {}".format(weight_path))

        weight = os.path.join(weight_path)
        chkpt = torch.load(weight, map_location=self.__device)
        self.__model1.load_state_dict(chkpt)
        # print("loading weight file is done")
        del chkpt

    def detection(self):
        if os.path.isdir(self.__data_dir):
            imgs = os.listdir(self.__data_dir)
            print("***********Start Detection****************")
            for v in imgs:
                path = self.__data_dir + "/" + v
                print("val images : {}".format(path))

                img = cv2.imread(path)
                assert img is not None

                bboxes_prd = self.__evalter.get_bbox(img, v)
                points, max_l = get_points(img, bboxes_prd=bboxes_prd, model=self.__model2, mnist=self.__mnist)
                if points is None:
                    return
                # print(points)
                matrix_calculator = calculator(self.__classes)
                matrix_calculator.get_from_points(points, max_l)
                print(matrix_calculator())
        else:
            raise NotImplementedError("The data directory is not exist!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model1_path",
        type=str,
        default="weights/best.pt",
        help="weight file path of model1",
    )
    parser.add_argument(
        "--model2_path",
        type=str,
        default="checkpoint",
        help="weight file path of model2",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=-1,
        help="whither use GPU(eg:0,1,2,3,4,5,6,7,8) or CPU(-1)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="val data path or None",
    )
    # parser.add_argument("--result_dir", type=str, default="./detection_result", help="dir to save detection result")
    parser.add_argument("--mnist", action='store_true', help="whether use mnist to detect")
    opt = parser.parse_args()

    result = Evaluation(
        gpu_id=opt.gpu_id,
        model1_path=opt.model1_path,
        model2_path=opt.model2_path,
        data_dir=opt.data_dir,
        mnist=opt.mnist,
    ).detection()
