r""" Chest X-ray few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np


class DatasetLung(Dataset):
    def __init__(self, datapath, transform, split, shot, num=600):
        self.split = split
        self.benchmark = 'lung'
        self.shot = shot
        self.num = num

        self.base_path = os.path.join(datapath, 'LungSegmentation')
        self.img_path = os.path.join(self.base_path, 'CXR_png')
        self.ann_path = os.path.join(self.base_path, 'masks')

        self.categories = ['1']  # Number of categories of data set

        self.class_ids = range(0, 1)
        self.img_metadata_classwise = self.build_img_metadata_classwise()

        self.transform = transform

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        # Build a batch dictionary
        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,

                 'class_id': torch.tensor(class_sample)}

        return batch

    # Used to load query images and supporting images
    def load_frame(self, query_name, support_names):
        query_mask = self.read_mask(query_name)
        support_masks = [self.read_mask(name) for name in support_names]

        if query_name[-22] == 'C':
            query_id = query_name[:-9] + '.png'
        else:
            query_id = query_name
        query_img = Image.open(os.path.join(self.img_path, os.path.basename(query_id))).convert('RGB')
        
        support_ids = []
        for name in support_names:
            if name[-22] == 'C':
                support_id = os.path.basename(name)[:-9] + '.png'
            else:
                support_id = os.path.basename(name)
            support_ids.append(support_id)

        support_names = [os.path.join(self.img_path, sid) for sid in support_ids]
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        return query_img, query_mask, support_imgs, support_masks

    # Read the mask of the image and convert it into a grayscale image
    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    # Generate an episode
    def sample_episode(self, idx):
        class_id = idx % len(self.class_ids)
        class_sample = self.categories[class_id]

        query_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        support_names = []
        while True:
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_id

    # Constructing image metadata
    def build_img_metadata(self):
        img_metadata = []
        for cat in self.categories:
            os.path.join(self.base_path, cat)

            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.img_path, cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'png':
                    img_metadata.append(img_path)

        return img_metadata

    # Building image metadata by category
    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        total_images = 0  # Total number of pictures to be processed

        for cat in self.categories:
            img_metadata_classwise[cat] = []

        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % self.ann_path)])
            total_images += len(img_paths)

            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'png':
                    img_metadata_classwise[cat] += [img_path]

        print('Total (%s) images are from Lung: %d ' % (self.split, total_images))

        return img_metadata_classwise
