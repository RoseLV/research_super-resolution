import glob
from pathlib import Path

import tiffile as tif
import torch
import torchvision.transforms.functional as tf
from tqdm import tqdm


def crop(input_dir: str, output_dir: str, top: int, left: int, size: int):
    input_pattern = Path(input_dir) / "*.tiff"
    input_files = glob.glob(str(input_pattern))

    for file in tqdm(input_files):
        output_file = Path(output_dir) / file.split("/")[-1]
        img = tif.imread(file)
        oimg = tf.crop(torch.tensor(img), top, left, size, size)
        tif.imwrite(output_file, oimg.numpy())


if __name__ == "__main__":
    crop("/data/share/PPT_raw_yearly/", "/data/share/PPT_4km_128/", 240, 789, 128)
