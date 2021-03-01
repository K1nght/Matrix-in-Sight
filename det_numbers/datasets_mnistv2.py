import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor, to_pil_image
import random
import string
import os
import glob
from PIL import Image


characters = ' ' + string.digits
n_classes = len(characters)
n_input_length, n_len = 12, 3
img_dir = 'data/num'
txt_path = 'det_numbers/labeled.txt'


class CaptchaDataset(Dataset):
    def __init__(self, characters, input_length, label_length=None, length=None):
        super(CaptchaDataset, self).__init__()
        self.characters = characters
        self.input_length = input_length
        self.n_class = len(characters)
        self.__img = []
        self.__str = []

        # img_expr = img_dir + "/*.jpg"
        # img_paths = glob.glob(img_expr)
        # img_paths = sorted(img_paths)


        txtFile = open(txt_path)
        txtList = txtFile.readlines()

        if length is not None:
            self.length = length
        else:
            self.length = len(txtList)

        for i, oneline in enumerate(txtList):
            if i == length:
                break

            info = oneline.split(' ')
            image_name = info[0]
            rec_number = info[1][:-1]
            # print(rec_number)

            img_path = os.path.join(img_dir, image_name+'.jpg')
            image = Image.open(img_path).convert('RGB')
            self.__img.append(image)
            # print(rec_number)
            self.__str.append(rec_number)

        print(self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image_str = self.__str[index]
        label_length = len(image_str)
        image = to_tensor(self.__img[index])
        target = torch.tensor([self.characters.find(x) for x in image_str], dtype=torch.long)
        input_length = torch.full(size=(1,), fill_value=self.input_length, dtype=torch.long)
        target_length = torch.full(size=(1,), fill_value=label_length, dtype=torch.long)
        return image, target, input_length, target_length


if __name__ == '__main__':
    print(characters, n_classes)
    dataset = CaptchaDataset(characters, n_input_length)
    image, target, input_length, label_length = dataset[-1]
    print(target)
    print(''.join([characters[x] for x in target]), input_length, label_length)
    img = to_pil_image(image)
    img.show()