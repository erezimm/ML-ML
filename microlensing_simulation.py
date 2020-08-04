import os
import random

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.table
import astropy.io.ascii

from tqdm import tqdm

"""
This module makes use of Erez' class lc_cat (to be updated) in order to read the hdf5 files and somhow tell which one is
a variable star and which isn't. Loading the deseired number of curves from all hdf5 files (euler1 should be mounted),
it saves the final training/validation/whatever set in the deseired location, some of the lcs having gone through
synthetic microlensing.
"""


class lc_cat:
    def __init__(self, h5_file):
        self.cat_file = h5py.File(h5_file, 'r')
        self.ind_cols = ['RA', 'Dec', 'I1', 'I2', 'Nep', 'ID', 'FilterID', 'Field', 'RcID', 'MeanMag', 'StdMag',
                         'RStdMag', 'MaxMag', 'MinMag', 'Chi2', 'MaxPower', 'FreqMaxPower']
        self.lc_data_name = 'AllLC'
        self.lc_cols = ['HMJD', 'Mag', 'MagErr', 'ColorCoef', 'Flags']
        self.ind_cat = self.load_ind()

    def __getitem__(self, idx):
        lc_idx = self.ind_cat.iloc(0)[idx]
        lc_start = int(lc_idx['I1'])
        lc_end = int(lc_idx['I2'])
        lc_raw = np.array(self.cat_file[self.lc_data_name][0:5, lc_start - 1:lc_end])
        lc = pd.DataFrame(lc_raw.transpose(), columns=self.lc_cols)
        return lc

    def __len__(self):
        return len(self.ind_cat)

    def load_ind(self, ind_name='IndAllLC'):
        inds = np.array(self.cat_file[ind_name])
        ind_cat = pd.DataFrame(inds.transpose(), columns=self.ind_cols)
        return ind_cat


def limit_data_points(file_name, min_epochs, num_lcs_toget):
    """picks a random set of size num_lcs_toget out of the lcs in the file, who satisfy #epochs > num_epochs"""
    table = lc_cat(file_name)
    nep = table.ind_cat['Nep']
    idx = nep[nep > min_epochs].index
    idx = random.sample(list(idx), k=min(num_lcs_toget, len(idx)))

    new_tab = []
    for i in idx:
        new = table[i]
        new_tab.append((new['HMJD'], new['Mag'], new['MagErr']))

    return new_tab


def get_lightcurves(num, filesdir, min_epochs):
    """
    Reads the hdf5 files to return light curves (lcs). Can be parallelized but it doesn't matter much because it'd only
    run once.

    Args:
        num: how many lcs to return
        filesdir: directory of hdf5 files
        min_epochs: minimal number or data points the light curves will have
    Returns:
        lcs: a list of (timestamps, magintudes, errors)
    """
    filenames = os.listdir(filesdir)
    random.shuffle(filenames)
    max_from_file = int(np.ceil(num / len(filenames)))

    lcs = []
    # loop over all of the files in the directory
    for filename in tqdm(filenames):
        if len(lcs) >= num:
            break
        num_lcs_toget = min(max_from_file, num - len(lcs))
        new_lcs = limit_data_points(os.path.join(filesdir, filename), min_epochs, num_lcs_toget=num_lcs_toget)
        lcs.extend(new_lcs)

    return lcs


def microlensingsimulation(timestamps, magnitudes, errors, showplot=False):
    """
    Simulates flux magnification resulting from microlensing.
    [1] https://arxiv.org/pdf/2004.14347.pdf

    Args:
        timestamps: epochs, times of observation, in days
        magnitudes: non-amplified magnitudes
        errors: their errors
        showplot: whether to plot the result

    Returns:
        amplified_magnitudes
    """

    conditions_met = False
    while not conditions_met:
        # generate parameters: p4.3 in [1]
        u0 = random.uniform(0, 1)  # unitless
        tE = random.gauss(mu=30, sigma=10)  # days
        g = random.uniform(1, 10)  # unitless
        t0 = random.uniform(np.percentile(timestamps, 10), np.percentile(timestamps, 90))  # days

        between = [i for i, timestamp in enumerate(timestamps) if t0 - tE < timestamp < t0 + tE]
        c1 = len(between) >= 7
        if not c1:
            continue

        # compute model: p2 in [1]
        u = lambda t: np.sqrt(u0 ** 2 + ((t - t0) / tE) ** 2)
        A = lambda t: (u(t) ** 2 + 2) / (u(t) * np.sqrt(u(t) ** 2 + 4))
        Aobs = lambda t: (A(t) + g) / (1 + g)
        aobs = [Aobs(timestamp) for timestamp in timestamps]
        aobs = np.array(aobs)

        # conditions: p4.3 in [1]
        photometric_error = np.max(errors)
        c2 = np.prod([aobs[i] for i in between]) ** (1 / len(between)) > 10 ** (0.05 / 2.5)
        if photometric_error == 0:
            c3 = True
        else:
            c3 = len([i for i in between if 2.5 * np.log10(aobs[i]) > 3 * photometric_error]) > len(between) / 3
        c4 = np.max(aobs) > 2 * g / (1 + g)

        conditions_met = c1 and c2 and c3 and c4

    amplified_magnitudes = magnitudes - 2.5 * np.log10(aobs)

    if showplot:
        plt.errorbar(timestamps, amplified_magnitudes, yerr=errors, fmt='.')
        plt.gca().invert_yaxis()
        plt.xlabel('Obsevation Day')
        plt.ylabel('Amplified Apparent Magnitude')
        plt.title('Simulated Microlensing of a lightcurve')
        plt.grid()
        plt.show()

    return amplified_magnitudes


def getdataset(num_clean, num_microlensed, min_epochs, filesdir, saveto):
    """
     Produces a list of lightcurves from the variable stars data set, some as they are and some after microlensing.
     Saved as plain column files (HMJD, Mag, MagErr) in the provided directory, with names "clean_*" or "microlensed_*".

    Args:
        num_clean: how many lcs to return directly from the ZTF dataset
        num_microlensed: how many to return after synthetic microlensing
        min_epochs: minimal number or data points the light curves will have
        filesdir: directory of hdf5 files
        saveto: where to save the results
    """
    print('Getting lcs from hdf5 files...')
    lcs = get_lightcurves(num_clean + num_microlensed, filesdir, min_epochs)

    if not os.path.isdir(saveto):
        os.makedirs(saveto)

    print('Saving clean lcs...')
    for i, lc in tqdm(enumerate(lcs[0:num_clean])):
        tbl = astropy.table.Table()
        tbl.add_columns(cols=lc)
        astropy.io.ascii.write(tbl, os.path.join(saveto, 'clean_' + str(i)), format='no_header')

    print('Saving microlensed lcs...')
    for i, lc in tqdm(enumerate(lcs[num_clean:num_clean + num_microlensed])):
        amplified_magnitudes = microlensingsimulation(*lc, showplot=False)
        tbl = astropy.table.Table()
        tbl.add_columns(cols=[lc[0], amplified_magnitudes, lc[2]])
        astropy.io.ascii.write(tbl, os.path.join(saveto, 'microlensed_' + str(i)), format='no_header')


if __name__ == "__main__":
    getdataset(num_clean=500, num_microlensed=500, min_epochs=20,
               filesdir='/home/ofekb/euler1mnt/var/www/html/data/catsHTM/ZTF/LCDR1', saveto='asciis')
