import h5py
import numpy as np
import time

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
time1 = time.time()

X_train, y_train = [], []

X_test, y_test = [], []

with h5py.File('Galaxy10_DECals.h5', 'r') as F:
    X = F['images'][:,:,:,:]
    y = F['ans'][:]

time2 = time.time()

for i in range(700):
        k = np.random.choice(17736)
        X_train.append(X[k].tolist())
        y_train.append(y[k].tolist())

for i in range(300):
        k = np.random.choice(17736)
        X_test.append(X[k].tolist())
        y_test.append(y[k].tolist())

time3 = time.time()

print(time2 - time1)

print(time3 - time1)