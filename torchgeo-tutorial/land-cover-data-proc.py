import os
import tempfile
from datetime import datetime

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from torchgeo.datasets import CDL, BoundingBox, Landsat7, Landsat8, stack_samples
from torchgeo.datasets.utils import download_and_extract_archive
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler

# semantic segmentation

# More concretely, imagine you would like to combine a set of Landsat 7 and 8 scenes with the Cropland Data Layer (CDL). This presents a number of challenges for a typical machine learning pipeline:

## We may have hundreds of partially overlapping Landsat images that need to be mosaiced together

## We have a single CDL mask covering the entire continental US

## Neither the Landsat input or CDL output will have the same geospatial bounds

## Landsat is multispectral, and may have a different resolution for each spectral band

## Landsat 7 and 8 have a different number of spectral bands

## Landsat and CDL may have a differerent CRS

## Every single Landsat file may be in a different CRS (e.g., multiple UTM zones)

## We may have multiple years of input and output data, and need to ensure matching time spans

'''
dataset download and load.
'''

# landsat

landsat_root = os.path.join(tempfile.gettempdir(), 'landsat')

url = 'https://hf.co/datasets/torchgeo/tutorials/resolve/ff30b729e3cbf906148d69a4441cc68023898924/'
landsat7_url = url + 'LE07_L2SP_022032_20230725_20230820_02_T1.tar.gz'
landsat8_url = url + 'LC08_L2SP_023032_20230831_20230911_02_T1.tar.gz'

download_and_extract_archive(landsat7_url, landsat_root)
download_and_extract_archive(landsat8_url, landsat_root)

landsat7_bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']
landsat8_bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']

landsat7 = Landsat7(paths=landsat_root, bands=landsat7_bands)
landsat8 = Landsat8(paths=landsat_root, bands=landsat8_bands)

print(landsat7)
print(landsat8)

print(landsat7.crs)
print(landsat8.crs)

# CDL

cdl_root = os.path.join(tempfile.gettempdir(), 'cdl')

cdl_url = url + '2023_30m_cdls.zip'

download_and_extract_archive(cdl_url, cdl_root)

cdl = CDL(paths=cdl_root)

print(cdl)
print(cdl.crs)

# Composing datasets


landsat = landsat7 | landsat8
print(landsat)
print(landsat.crs)

dataset = landsat & cdl
print(dataset)
print(dataset.crs)


# Fetching and plotting the data

import ipdb;
ipdb.set_trace()

size = 256

xmin = 925000
xmax = xmin + size * 30
ymin = 4470000
ymax = ymin + size * 30
tmin = datetime(2023, 1, 1).timestamp()
tmax = datetime(2023, 12, 31).timestamp()

bbox = BoundingBox(xmin, xmax, ymin, ymax, tmin, tmax)
sample = dataset[bbox]

landsat8.plot(sample)
cdl.plot(sample)
plt.show()


# Samplers and sampling - as manually writing bbox is not feasible each time.

train_sampler = RandomGeoSampler(dataset, size=size, length=1000)
next(iter(train_sampler))


# as testing requires complete and non-repeated pixel set.
test_sampler = GridGeoSampler(dataset, size=size, stride=size)
next(iter(test_sampler))



# Data Loaders

# Note - All of these abstractions (GeoDataset and GeoSampler) are fully compatible with all of the rest of PyTorch. 

train_dataloader = DataLoader(
    dataset, batch_size=128, sampler=train_sampler, collate_fn=stack_samples
)
test_dataloader = DataLoader(
    dataset, batch_size=128, sampler=test_sampler, collate_fn=stack_samples
)





