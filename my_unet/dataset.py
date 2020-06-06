from torch.utils.data import Dataset
import PIL.Image as Image
import os

def make_dataset(root):
    imgs = []
    # return [(os.path.join(root,"cappi_ref_201705040736_2500_0.png"),os.path.join(root,"cappi_ref_201705040736_2500_0_mask.png"))]
    for file in sorted(os.listdir(root)):
        if not file[:2]=="pd":
            # name = file.split('.')[0]
            img = os.path.join(root,file)
            # mask = os.path.join(root,name+"_mask.png")
            imgs.append((img))
    return imgs

class LiverDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path = self.imgs[index]
        img_x = Image.open(x_path).convert('RGB')
        img_x = img_x.resize((64,64))
        name = x_path.split('/')[-1].split('.')[0]
        if self.transform is not None:
            img_x = self.transform(img_x)
        return img_x, name

    def __len__(self):
        return len(self.imgs)