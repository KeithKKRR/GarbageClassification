from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class GarbageDataset(Dataset):
    class_dict = {"battery": 0,
                  "biological": 1,
                  "brown-glass": 2,
                  "cardboard": 3,
                  "clothes": 4,
                  "green-glass": 5,
                  "metal": 6,
                  "paper": 7,
                  "plastic": 8,
                  "shoes": 9,
                  "trash": 10,
                  "white-glass": 11}

    def __init__(self, dataset_path):
        with open(dataset_path, "r") as f:
            self.data_text = f.read().splitlines()

        self.transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

    def __getitem__(self, item):
        img_path = self.data_text[item]
        img_label = GarbageDataset.class_dict[img_path.split("/")[1]]
        img_tensor = self.transform(Image.open(img_path).convert("RGB"))
        return img_tensor, img_label

    def __len__(self):
        return len(self.data_text)