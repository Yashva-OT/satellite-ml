import os
import tempfile

import matplotlib.pyplot as plt

from torchgeo.datasets import EuroSAT100
from torchgeo.transforms import AppendNDBI, AppendNDVI, AppendNDWI


root = os.path.join(tempfile.gettempdir(), 'eurosat100')
ds = EuroSAT100(root, download=True)
sample = ds[21]

# plot true color (rgb) of the sample
ds.plot(sample)
plt.show()
plt.close()


# plot NDVI
# NDVI is appended to channel dimension (dim=0)
index = AppendNDVI(index_nir=7, index_red=3)
image = sample['image']
image = index(image)[0]

print(image)
print(image.shape)

# Normalize from [-1, 1] -> [0, 1] for visualization
image[-1] = (image[-1] + 1) / 2

plt.imshow(image[-1], cmap='RdYlGn')
plt.axis('off')
plt.show()
plt.close()

# plot NDBI
# NDBI is appended to channel dimension (dim=0)
index = AppendNDBI(index_swir=11, index_nir=7)
image = index(image)[0]

# Normalize from [-1, 1] -> [0, 1] for visualization
image[-1] = (image[-1] + 1) / 2

plt.imshow(image[-1], cmap='terrain')
plt.axis('off')
plt.show()
plt.close()
