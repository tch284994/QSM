r""" FSS-1000 few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np


class DatasetFSS(Dataset):
    def __init__(self, datapath, transform, split, shot):
        self.split = split
        self.benchmark = 'fss'
        self.shot = shot

        self.base_path = os.path.join(datapath, 'FSS-1000')

        # Given predefined test split, load randomly generated training/val splits:
        # (reference regarding trn/val/test splits: https://github.com/HKUSTCV/FSS-1000/issues/7))
        with open('./data_util/splits/fss/%s.txt' % split, 'r') as f:
            self.categories = f.read().split('\n')[:-1]
        self.categories = sorted(self.categories)

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()

        self.transform = transform

    # Returns the length of the dataset
    def __len__(self):
        return len(self.img_metadata)

    # Carry out pretreatment and conversion
    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)  # Sample a set of episode
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)  # Ensure the same size
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        # Constructing a batch dictionary for meta learning
        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,

                 'class_id': torch.tensor(class_sample)}

        return batch

    def load_frame(self, query_name, support_names):
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        query_id = os.path.splitext(os.path.basename(query_name))[0]
        query_name = os.path.join(os.path.dirname(query_name), query_id) + '.png'
        support_ids = [os.path.splitext(os.path.basename(name))[0] for name in support_names]
        support_names = [os.path.join(os.path.dirname(name), sid) + '.png' for name, sid in zip(support_names, support_ids)]

        query_mask = self.read_mask(query_name)  # Read the mask of the query image and the supporting image
        support_masks = [self.read_mask(name) for name in support_names]

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):  # Read the mask of the image and convert it into a gray image
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    # Generate sample episode
    def sample_episode(self, idx):
        query_name = self.img_metadata[idx]
        class_sample = self.categories.index(query_name.split(os.path.sep)[-2])
        if self.split == 'val':
            class_sample += 520
        elif self.split == 'test':
            class_sample += 760

        support_names = []
        while True:  # Cycle sampling the support set until the number of samples in the support set reaches n-shot
            support_name = np.random.choice(range(1, 11), 1, replace=False)[0]
            support_name = os.path.join(os.path.dirname(query_name), str(support_name)) + '.jpg'
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_sample

    # Index list used to build category samples
    def build_class_ids(self):
        if self.split == 'trn':
            class_ids = range(0, 520)
        elif self.split == 'val':
            class_ids = range(520, 760)
        elif self.split == 'test':
            class_ids = range(760, 1000)
        return class_ids

    # Used to build an image metadata list
    def build_img_metadata(self):
        img_metadata = []
        for cat in self.categories:  # Traverse each category and get the image path under each category
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'jpg':
                    img_metadata.append(img_path)

        print('Total (%s) images are from FSS: %d ' % (self.split, len(img_metadata)))

        return img_metadata
