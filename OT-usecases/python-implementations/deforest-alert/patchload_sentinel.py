from torchgeo.datasets import Sentinel2
from torchgeo.datasets.utils import BoundingBox
import pyproj
import matplotlib.pyplot as plt
import json

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




if __name__ == "__main__":

    with open("./configs/patchload.json", 'r') as f:
        infer_conf = json.load(f)

    data_dir = infer_conf["data_dir"]
    bands = infer_conf["bands"]
    bbox = infer_conf["bbox"]
    crs = infer_conf["crs"]
    ndvi = infer_conf["ndvi"]
    figname = infer_conf["figname"]

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

    if ndvi:
        ndvi = sample['image'][3,:,:]  # NDVI band

        # Normalize NDVI values for visualization
        ndvi_normalized = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min())

        # Plotting the NDVI map
        plt.figure(figsize=(10, 6))
        plt.imshow(ndvi_normalized.cpu().numpy(), cmap='RdYlGn')  # Use RdYlGn colormap for greenery
        plt.colorbar(label='NDVI')
        plt.title('Greenery Map (NDVI)')
        plt.axis('off')  # Turn off axis labels
        plt.savefig(figname)