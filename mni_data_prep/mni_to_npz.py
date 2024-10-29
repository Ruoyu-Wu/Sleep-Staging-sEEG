import argparse
import glob
import math
import ntpath
import os
import shutil
import pyedflib


from datetime import datetime

import numpy as np
import pandas as pd

from mne.io import concatenate_raws, read_raw_edf

stage_dict = {
    "Wakefulness": 0,
    "N2": 1,
    "N3": 2,
    "REM": 3,
}

EPOCH_SEC_SIZE = 30

# Define the main folder and the output folder
main_folder = '/Volumes/T7 Shield/mni_sEEG'
output_folder = os.path.join(main_folder, 'npz_folder')

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

subfolders = ['N2_AllRegions', 'N3_AllRegions', 'REM_AllRegions', 'Wakefulness_AllRegions']

 # Iterate through each subfolder
for subfolder in subfolders:
    subfolder_path = os.path.join(main_folder, subfolder)
    # Check if the subfolder exists
    if os.path.exists(subfolder_path):
        # Iterate through all files in the subfolder
        edf_files = [f for f in os.listdir(subfolder_path) if f.endswith('.edf') and not f.startswith('._')]
        for filename in edf_files:
            file_path = os.path.join(subfolder_path, filename)
            # Check if it's a file (not a directory)
            if os.path.isfile(file_path):
                # Load the data from the file
                # raw = pyedflib.EdfReader(file_path)
                # n = raw.signals_in_file
                # signal_labels = raw.getSignalLabels()
                # data = np.zeros((n, raw.getNSamples()[0]))
                # for i in np.arange(n):
                #     data[i, :] = raw.readSignal(i)
                raw = read_raw_edf(file_path) # or use np.load, pd.read_csv, etc.
                data = raw.get_data()
                data = data[:,:6800]
                # Create a .npz filename
                for channel_no in range(data.shape[0]):
                    npz_filename = f"{subfolder.split('_')[0]}_{filename.split('.')[0]}_chan{channel_no}.npz"
                    npz_path = os.path.join(output_folder, npz_filename)
                    state = subfolder[:-11]
                    np.savez(npz_path, X=data, y = stage_dict[state],region = filename[:-6])
                    print(f"Created {npz_path}")
    else:
        print(f"Subfolder {subfolder} not found.")

print("Finished creating .npz files.")

file_name = pyedflib.data.get_generator_filename()
f = pyedflib.EdfReader(file_name)
n = f.signals_in_file
signal_labels = f.getSignalLabels()
sigbufs = np.zeros((n, f.getNSamples()[0]))
for i in np.arange(n):
        sigbufs[i, :] = f.readSignal(i)

