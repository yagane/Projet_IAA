import h5py
import numpy as np
import matplotlib.pyplot as plt

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

i = np.random.choice(17736)

# To get the image i and its label
with h5py.File('Galaxy10_DECals.h5', 'r') as F:
    image = F['images'][i,:,:,:]
    label = F['ans'][i]
    
imgplot = plt.imshow(image)
plt.show()
print("This is a galaxy of type %s" %label_dict[label])
