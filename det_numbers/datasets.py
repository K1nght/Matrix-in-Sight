import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image

from det_numbers.image_12_27 import ImageCaptcha
import random

import string

characters = ' ' + string.digits
width, height, n_len, n_classes = 192, 64, 4, len(characters)
n_input_length = 12


class CaptchaDataset(Dataset):
    def __init__(self, characters, length, width, height, input_length, label_length):
        super(CaptchaDataset, self).__init__()
        self.characters = characters
        self.length = length
        self.width = width
        self.height = height
        self.input_length = input_length
        self.label_length = label_length
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=width, height=height)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        l = random.randint(1, self.label_length)
        random_str = ''.join([random.choice(self.characters[1:]) for _ in range(l)])
        image = to_tensor(self.generator.generate_image(random_str))
        target = torch.tensor([self.characters.find(x) for x in random_str], dtype=torch.long)
        input_length = torch.full(size=(1,), fill_value=self.input_length, dtype=torch.long)
        target_length = torch.full(size=(1,), fill_value=l, dtype=torch.long)
        return image, target, input_length, target_length


if __name__ == '__main__':
    print(characters, width, height, n_len, n_classes)
    dataset = CaptchaDataset(characters, 1, width, height, n_input_length, n_len)
    image, target, input_length, l = dataset[0]
    print(''.join([characters[x] for x in target]), input_length, l)
    img = to_pil_image(image)
    img.show()