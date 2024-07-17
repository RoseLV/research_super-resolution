import glob
from pathlib import Path

import tiffile as tif
import torch
import torchvision.transforms.functional as tf
from tqdm import tqdm


def crop_one(ifile: str, ofile: str, top: int, left: int, size: int):
    img = tif.imread(ifile)
    oimg = tf.crop(torch.tensor(img), top, left, size, size)
    tif.imwrite(ofile, oimg.numpy())


def crop(input_dir: str, output_dir: str, top: int, left: int, size: int):
    input_pattern = Path(input_dir) / "*.tiff"
    input_files = glob.glob(str(input_pattern))

    for file in tqdm(input_files):
        output_file = Path(output_dir) / file.split("/")[-1]
        crop_one(file, str(output_file), top, left, size)


if __name__ == "__main__":
    crop("/data/share/PPT_raw_yearly/", "/data/share/PPT_4km_128/", 240, 789, 128)
    crop_one(
        "/data/share/topo_4000.tiff", "/data/share/topo_4km_128.tiff", 240, 789, 128
    )
