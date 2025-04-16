from torchgeo.datasets import Sentinel2
from torchgeo.datasets.utils import BoundingBox
import pyproj
import matplotlib.pyplot as plt
import json
import os

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

def plot_indice(indice_normalized, indice_name, plot_dir):

    plt.figure(figsize=(10, 6))
    plt.imshow(indice_normalized.cpu().numpy(), cmap='RdYlGn')  # Use RdYlGn colormap for greenery
    plt.colorbar(label=f'{indice_name.upper()}')
    plt.title(f'Greenery Map ({indice_name.upper()})')
    plt.axis('off')  # Turn off axis labels
    plt.savefig(f"{plot_dir}/{indice_name}")


if __name__ == "__main__":

    with open("./configs/patchload.json", 'r') as f:
        infer_conf = json.load(f)

    data_dir = infer_conf["data_dir"]
    bands = infer_conf["bands"]
    bbox = infer_conf["bbox"]
    crs = infer_conf["crs"]
    indices = infer_conf["indices"]
    rgb = infer_conf["rgb"]

    #if data_dir contains band with multiple res, transform them to a common res (currently-10m)
    multi_res = infer_conf["multi_res"] 

    if multi_res:
        from change_resolution import change_res
        # currently hardcoded to ->10m
        change_res(data_dir, 10)


    # Initialize your Sentinel dataset
    dataset = Sentinel2(paths=data_dir, bands = bands)

    # get time bounds
    maxt, mint = dataset.bounds[-1], dataset.bounds[-2]

    x_min, y_min, x_max, y_max = toCRS(bbox, crs)

    # Create a BoundingBox using projected coordinates
    roi = BoundingBox(minx=x_min, maxx=x_max, miny=y_min, maxy=y_max, mint=mint, maxt=maxt)

    # Filter the dataset using the bounding box
    sample = dataset[roi]
    image = sample['image']

    if rgb or len(indices):
        # create plot dir
        data_dir_name = data_dir.split('/')[-1]
        plot_dir = f"plots/{data_dir_name}"
        os.makedirs(plot_dir, exist_ok=True)
        
    if rgb:

        dataset.plot(sample)
        plt.savefig(f"{plot_dir}/rgb")

    if len(indices):
        from indices import INDICE_MAP
        for indice, indice_args in indices.items():
            transform = INDICE_MAP[indice](*indice_args)
            image_modified = transform(image)
            indice_tensor = image_modified[0,-1,:,:]  # indice
            indice_normalized = (indice_tensor - indice_tensor.min()) / (indice_tensor.max() - indice_tensor.min())

            plot_indice(indice_normalized, indice, plot_dir)