import os
import urllib.request
from urllib.parse import urlparse, unquote
import planetary_computer
import pystac_client
import json


if __name__=="__main__":
    with open("./configs/download.json", 'r') as f:
        download_conf = json.load(f)

    bbox = download_conf["bbox"]
    datetime = download_conf["datetime"]
    query = download_conf["query"]
    bands = download_conf["bands"]
    download_dir = download_conf["download_dir"]

    # Connect to Planetary Computer STAC
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # Search parameters (adjust these)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,  # Gujarat bounding box
        datetime=datetime,
        query=query,
    )

    items = search.item_collection()
    print(f"Found {len(items)} items")

    # Create output directory if it doesn't exist, TODO: Get these name from json.
    os.makedirs(download_dir, exist_ok=True)

    for item in items:
        for band in bands:
            try:
                asset = item.assets[band]
                signed_href = planetary_computer.sign(asset.href)
                
                # Parse the URL to extract the filename
                parsed_url = urlparse(signed_href)
                filename = unquote(os.path.basename(parsed_url.path))  # Decode URL-encoded characters
                
                # Construct full output path
                output_path = os.path.join(download_dir, filename)
                
                # Download the file and save it with its original name
                urllib.request.urlretrieve(signed_href, output_path)
                print(f"Downloaded {filename} to {download_dir}")
                
            except Exception as e:
                print(f"Failed to download {band}: {str(e)}")
