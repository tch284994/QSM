r""" Dataloader builder for few-shot semantic segmentation dataset  """
from torchvision import transforms
from torch.utils.data import DataLoader

from .pascal import DatasetPASCAL
from .fss import DatasetFSS
from .deepglobe import DatasetDeepglobe
from .isic import DatasetISIC
from .lung import DatasetLung


class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath):

        cls.datasets = {
            'pascal': DatasetPASCAL,
            'fss': DatasetFSS,
            'deepglobe': DatasetDeepglobe,
            'isic': DatasetISIC,
            'lung': DatasetLung,
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath

        cls.transform = transforms.Compose([transforms.Resize(size=(img_size, img_size)),       ##img_size = (400,400)
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, split, shot=1):

        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility

        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0

        dataset = cls.datasets[benchmark](cls.datapath, transform=cls.transform, split=split, shot=shot)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader
