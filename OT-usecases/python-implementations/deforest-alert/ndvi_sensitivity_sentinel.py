from torchgeo.datasets import Sentinel2
from torchgeo.datasets.utils import BoundingBox
import pyproj
import matplotlib.pyplot as plt
import json
import torch

def toCRS(bbox, crs):
    # implementation tested for "EPSG:32642" and "EPSG:32643"
    lon_min = bbox[0]
    lat_min = bbox[1]
    lon_max = bbox[2]
    lat_max = bbox[3]

    # Define the CRS for WGS84 (EPSG:4326) and UTM zone
    wgs84 = pyproj.CRS("EPSG:4326")
    utm_zone = pyproj.CRS(f"EPSG:{crs}")

    # Create a transformer to convert lat/lon to UTM coordinates
    transformer = pyproj.Transformer.from_crs(wgs84, utm_zone, always_xy=True)

    # Convert lat/lon to projected coordinates (UTM Zone 42N)
    x_min, y_min = transformer.transform(lon_min, lat_min)
    x_max, y_max = transformer.transform(lon_max, lat_max)

    return x_min, y_min, x_max, y_max


def get_NDVI(data_dir, bands, bbox, crs):

    # Initialize your Sentinel dataset
    dataset = Sentinel2(paths=data_dir, bands = bands)

    # get time bounds
    maxt, mint = dataset.bounds[-1], dataset.bounds[-2]

    x_min, y_min, x_max, y_max = toCRS(bbox, crs)

    # Create a BoundingBox using projected coordinates
    roi = BoundingBox(minx=x_min, maxx=x_max, miny=y_min, maxy=y_max, mint=mint, maxt=maxt)

    # For sanity
    print(roi.minx >= dataset.bounds.minx)
    print(roi.maxx <= dataset.bounds.maxx)
    print(roi.miny >= dataset.bounds.miny)
    print(roi.maxy <= dataset.bounds.maxy)


    # Filter the dataset using the bounding box
    sample = dataset[roi]

    ndvi = sample['image'][3,:,:]  # NDVI band

    # Normalize NDVI values for visualization
    ndvi_normalized = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min())

    return ndvi_normalized




if __name__ == "__main__":

    with open("./configs/ndvi_sensitivity.json", 'r') as f:
        infer_conf = json.load(f)

    data_dir1 = infer_conf["data_dir1"]
    data_dir2 = infer_conf["data_dir2"]
    bands = infer_conf["bands"]
    bbox = infer_conf["bbox"]
    crs = infer_conf["crs"]
    threshold = infer_conf["threshold"]


    ndvi_norm1 = get_NDVI(data_dir1, bands, bbox, crs)
    ndvi_norm2 = get_NDVI(data_dir2, bands, bbox, crs)

    initial_green = ndvi_norm1[ndvi_norm1>threshold]
    final_green = ndvi_norm2[ndvi_norm1>threshold]

    diff = final_green - initial_green

    average_change = sum(diff) / len(diff)

    print("MAX INCERESE OBSERVED / TILE",max(diff))
    print("MAX DECREASE OBSERVED / TILE", min(diff))
    
    print("STANDARD DEVIATION OF CHANGE",torch.std(diff))
    print("MEAN CHANGE ACROSS TILE",average_change)