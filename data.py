import h5py
import numpy as np

from torch.utils.data import Dataset
import torchvision.transforms as T

class GalaxiesDataset(Dataset):
    def __init__(self, path):
        # Télécharge toutes les données (le dataset est assez petit donc on peut se le permettre)
        with h5py.File(path, 'r') as F:
            self.images = np.array(F['images'])
            self.labels = np.array(F['ans'])
        # Définit les opérations de preprocessing
        # self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform = T.ToTensor()

    def __getitem__(self, index):
        image = self.transform(self.images[index])
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.labels)