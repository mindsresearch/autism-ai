''' Converts nii.gz files of 3D MRI scans into folders of graph images.

RUN ONLY FROM COMMAND LINE!

Run with -h flag for commandline args help

LICENSE:
    CC BY-NC-SA 4.0
    https://creativecommons.org/licenses/by-nc-sa/4.0/
'''
import os
import argparse

from tqdm import tqdm
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser(prog='visualize.py',
                                 description='Converts nii.gz files into folders of images',
                                 epilog='''MINDS Research; Noah Duggan Erickson;
                                 Q1 2024; CC BY-NC-SA 4.0''')
parser.add_argument('-i',  '--nii_in', required=True, dest='i_root', metavar='NII_PATH', type=str,
                    help='Path to directory containing the *.nii.gz files')
parser.add_argument('-o', '--img_out', required=True, dest='o_root', metavar='IMG_PATH', type=str,
                    help='Directory where image files will be written')
parser.add_argument('-d',  '--sd_csv', required=True, dest='d_path', metavar='CSV_PATH', type=str,
                    help="Path to 'subject_data.csv' file")
parser.add_argument('-m', '--img_max', required=False, dest='i_max', metavar='X',        type=int,
                    help="Only convert X nii.gz files into images (default: 25)", default=25)
args = parser.parse_args()

sd = pd.read_csv(args.d_path, na_values=[-9999])

# Locate and enumerate *.nii.gz files at i_root directory
#
niipaths = []
for root, dirs, files in os.walk(args.i_root):
    for file in files:
        if file.endswith('.gz'):
            niipaths.append((root, file))

niipaths = niipaths[:(args.i_max)]
# process those files
#
for x in tqdm(niipaths, desc='file'):
    ip = os.path.join(x[0], x[1])
    sub_fn = x[1][:-12]
    op = os.path.join(args.o_root, x[1][:-7])
    tg = list(sd[sd['FILE_ID'] == sub_fn]['DX_GROUP'])[0]
    if not os.path.exists(op):
        os.makedirs(op)
        img = nib.load(ip)
        data = img.get_fdata()
        for z in tqdm(list(range(data.shape[2])), desc='z', leave=False):
            plt.imshow(data[:, :, z], cmap='gray')
            plt.title(f"ID:{sub_fn} group:{tg} z={z}")
            plt.savefig(os.path.join(op, f"{z}.png"))
            plt.close()
