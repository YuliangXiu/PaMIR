from glob import glob
from tqdm import tqdm
import os

pbar = tqdm(glob("/home/yxiu/BigDisk/DCPIFu_data/cape/*/*/*/png_square"))

for indir in pbar:
    outdir = indir.replace("png_square", "pamir")
    pbar.set_description(indir)
    os.makedirs(outdir, exist_ok=True)
    os.system(f"CUDA_VISIBLE_DEVICES=1 python main_test.py -indir {indir} -outdir {outdir} > cache")