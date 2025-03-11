from PIL import Image
import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class Container_Dataset(Dataset):
    def __init__(self, images_path, width=256, height=256, max_len=17):
        self.images_path = images_path
        self.images_names = [f for f in os.listdir(images_path) if "+" not in f and "#" not in f]
        self.size = len(self.images_names)
        self.n_chars = max_len
        self.chars = list('1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.tokenizer = Tokenizer(self.chars)
        self.image_width_height = (width, height)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.images_path, self.images_names[idx]))
        image = image.resize(self.image_width_height)
        image = self.transform(image)
        text = self.images_names[idx].split('.')[0]
        label_encoder = torch.full((self.n_chars + 2,), self.tokenizer.EOS_token, dtype=np.compat.long)
        ts = self.tokenizer.tokenize(text)
        label_encoder[:ts.shape[0]] = torch.tensor(ts)
        return image, label_encoder
