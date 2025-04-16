import os
import json
import re
import rasterio
from rasterio.enums import Resampling


def change_res(input_dir, out_res):
    for fname in os.listdir(input_dir):
        match = re.search(r"_(\d+)m\.tif$", fname)
        if not match or match.group(1) == f"{out_res}":
            continue

        in_res = int(match.group(1))
        scale = in_res / out_res
        input_path = os.path.join(input_dir, fname)
        output_fname = re.sub(r"_(\d+)m\.tif$", f"_{out_res}m.tif", fname)
        output_path = os.path.join(input_dir, output_fname)

        with rasterio.open(input_path) as src:
            data = src.read(
                out_shape=(src.count, int(src.height * scale), int(src.width * scale)),
                resampling=Resampling.bilinear
            )
            transform = src.transform * src.transform.scale(1/scale, 1/scale)
            profile = src.profile
            profile.update({
                "height": data.shape[1],
                "width": data.shape[2],
                "transform": transform
            })
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(data)

        print(f"✅ Converted {fname} -> {output_fname}")


if __name__ == "__main__":

    with open("./configs/change_resolution.json", 'r') as f:
        resolution_conf = json.load(f)

    input_dir = resolution_conf['input_dir']
    out_res = resolution_conf['out_res']

    change_res(input_dir, out_res)
