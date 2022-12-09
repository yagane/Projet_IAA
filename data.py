import h5py
import numpy as np

from torch.utils.data import Dataset

class GalaxiesDataset(Dataset):
    def __init__(self, path):
        # Télécharge toutes les données (le dataset est assez petit donc on peut se le permettre)
        with h5py.File(path, 'r') as F:
            self.images = np.array(F['images'])
            self.labels = np.array(F['ans'])

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.labels)