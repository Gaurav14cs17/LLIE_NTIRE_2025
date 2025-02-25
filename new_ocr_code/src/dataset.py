from PIL import Image
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from src.tokenizer import Tokenizer

img_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=(0.229)),
])


class Lplate_Dataset(Dataset):
    def __init__(self, images_path=None , width = 400, height = 200,  max_len = 11 ):
        self.images_path = images_path
        self.images_names = list(os.listdir(self.images_path))
        self.size = len(self.images_names)
        self.n_chars = max_len
        self.chars = list('1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.tokenizer = Tokenizer(self.chars)
        self.first_run = True
        self.image_width_height = (width, height)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.images_path, self.images_names[idx]))
        image = image.convert("L")
        image = image.resize(self.image_width_height)
        image = img_trans(image)
        text = self.images_names[idx].split('.')[0]
        label_encoder = torch.full((self.n_chars + 2,), self.tokenizer.EOS_token, dtype=torch.long)
        ts = self.tokenizer.tokenize(text)
        label_encoder[:ts.shape[0]] = torch.tensor(ts)
        return image, label_encoder
