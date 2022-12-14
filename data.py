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

class Galaxies3Dataset(Dataset):
    def __init__(self, path):
        # Télécharge toutes les données (le dataset est assez petit donc on peut se le permettre)
        with h5py.File(path, 'r') as F:
            all_images = np.array(F['images'])
            all_labels = np.array(F['ans'])

        # *** Réduit le problème de 10 classes à un problème de 3 classes
        # Enlève les classes merging et disturbed
        new_images = all_images[np.logical_and(all_labels>=2, all_labels<=9)]
        new_labels = all_labels[np.logical_and(all_labels>=2, all_labels<=9)]
        # Round
        new_labels[np.logical_and(new_labels>=2, new_labels<=4)] = 0 
        # Elliptical
        new_labels[np.logical_and(new_labels>=5, new_labels<=7)] = 1
        # Edge-on
        new_labels[np.logical_and(new_labels>=8, new_labels<=9)] = 2
        
        self.images = new_images
        self.labels = new_labels

        # Définit les opérations de preprocessing
        #self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transform = T.ToTensor()

    def __getitem__(self, index):
        image = self.transform(self.images[index])
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.labels)