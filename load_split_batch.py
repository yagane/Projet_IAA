from time import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from data import GalaxiesDataset

label_dict = {
    0: "Disturbed Galaxies",
    1: "Merging Galaxies",
    2: "Round Smooth Galaxies",
    3: "In-between Round Smooth Galaxies",
    4: "Cigar Shaped Smooth Galaxies",
    5: "Barred Spiral Galaxies",
    6: "Unbarred Tight Spiral Galaxies",
    7: "Unbarred Loose Spiral Galaxies",
    8: "Edge-on Galaxies without Bulge",
    9: "Edge-on Galaxies with Bulge"
}

# *** Charger les données ***
# Comme la classe GalaxiesDataset met toutes les données sur la mémoire vive, cette étape prend un peu de temps (1 minute sur HDD)
# Mais ça permet ensuite de chercher des batchs très rapidement.
start_time = time()
dataset = GalaxiesDataset('Galaxy10_DECals.h5')
total_time = time() - start_time
print("Loading time : %.3f" %total_time)

# *** Diviser en données d'entraînement et de test ***
start_time = time()
train_test_ratios = [0.7, 0.3]
generator = torch.Generator().manual_seed(42)
train_set, test_set = random_split(dataset=dataset, lengths=train_test_ratios, generator=generator)
total_time = time() - start_time
print("Split time : %.3f" %total_time)

# *** Echantilloner un batch de 64 données d'entraînement aléatoirement ***
params = {'batch_size': 64, 
          'shuffle': True
          }

start_time = time()
train_loader = DataLoader(train_set, **params)
batch = next(iter(train_loader))
total_time = time() - start_time
print("Sampling batch time : %.3f" %total_time)
