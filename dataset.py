from torch.utils.data import Dataset
from PIL import Image
import os
import random


def pil_loader(img_str, str='RGB'):
    with Image.open(img_str) as img:
        img = img.convert(str)
    return img


class DatasetWithMeta(Dataset):
    def __init__(self, root_dir, meta_file, transform=None):
        super(DatasetWithMeta, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        with open(meta_file) as f:
            lines = f.readlines()
        self.images = []
        self.cls_idx = []
        self.classes = set()

        for line in lines:
            segs = line.strip().split(' ')
            self.images.append(' '.join(segs[:-1]))
            self.cls_idx.append(int(segs[-1]))
            self.classes.add(int(segs[-1]))
        self.num = len(self.images)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        filename = os.path.join(self.root_dir, self.images[idx])

        try:
            img = pil_loader(filename)
        except:
            print(filename)
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        # transform
        if self.transform is not None:
            img = self.transform(img)
        return img, self.cls_idx[idx]
