import os
import random

import astropy.io.ascii
import astropy.table
import matplotlib.pyplot as plt
import numpy as np
from joblib import delayed, Parallel
from tqdm import tqdm

from lc_cat import LcCat

id2filter = {2:'r', 1:'g'}
def make_dataset(num_const, num_var, filt, min_epochs, filesdir, saveto, files_to_use=-1, n_jobs=16):
    """
    Produces a list of lightcurves from the variable stars data set, variable-candidate stars as they are and constant
    stars after microlensing. Saved as plain column files (HMJD, Mag, MagErr) in the provided directory, with names
    'microlensedconst_*' or 'cleanvar_*'.

    Args:
        num_const: number of constant star lcs to put through synthetic microlensing and write to a file
        num_var: number of variable (candidate) stars lcs to write to a file as they are
        min_epochs: minimal number or data points the light curves will have
        filesdir: directory of hdf5 files
        saveto: where to save the results
        files_to_use: from how many (randomly chosen) files to get the deseired number of lcs (equally divided), -1 (default) means all
        n_jobs: for parallelization
    """

    if not os.path.isdir(saveto):
        os.makedirs(saveto)

    filenames = os.listdir(filesdir)
    if files_to_use == -1:
        files_to_use = len(filenames)
    filenames = random.sample(filenames, k=files_to_use)

    asciiname = {'const': 'microlensedconst_', 'var': 'cleanvar_'}
    num = {'const': num_const, 'var': num_var}
    num_from_file = {'const': {}, 'var': {}}
    for k in ('const', 'var'):
        floor = num[k] // len(filenames)
        toadd = num[k] - floor * len(filenames)
        for i, filename in enumerate(filenames):
            num_from_file[k][filename] = floor + (i < toadd)

    def handlefile(filename):
        try:
            if num_from_file['const'][filename] == num_from_file['var'][filename] == 0:
                return
            filecat = LcCat(os.path.join(filesdir, filename), min_Nep=min_epochs)
            for k in ('const', 'var'):
                this_indcat = filecat.constants[filecat.constants['FilterID'] == filt] if k == 'const' else \
                filecat.variable_candidates[filecat.variable_candidates['FilterID'] == filt]
                # this_indcat = filecat.constants if k == 'const' else filecat.variable_candidates
                nep = this_indcat['Nep']
                idx = nep[nep > min_epochs].index
                idx = random.sample(list(idx), k=int(min(num_from_file[k][filename], len(idx))))
                for i in idx:
                    lc = filecat[i]
                    times, magnitudes, magerrs = lc['HMJD'], lc['Mag'], lc['MagErr']
                    # plt.scatter(times, magnitudes)
                    # plt.show()
                    if k == 'const':
                        magnitudes = microlensingsimulation(times, magnitudes, magerrs, showplot=False)
                    tbl = astropy.table.Table()
                    tbl.add_columns(cols=[times, magnitudes, magerrs])
                    outname = asciiname[k] + filename + '_' + str(i)
                    astropy.io.ascii.write(tbl, os.path.join(saveto, outname), format='no_header', overwrite=True)
        except:
            print('Error during processing file:', filename)

    if n_jobs > 1:
        Parallel(n_jobs=16, verbose=11)(delayed(handlefile)(filename) for filename in filenames)
    else:
        for filename in tqdm(filenames):
            handlefile(filename)


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

        between = [i for i in range(len(timestamps)) if t0 - tE < timestamps[i] < t0 + tE]
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


if __name__ == "__main__":
    saveto = '/home/ofekb/data/data_r'

    make_dataset(num_const=20000, num_var=20000, min_epochs=20, files_to_use=-1, filt=2,
                 filesdir='/home/ofekb/euler1mnt/var/www/html/data/catsHTM/ZTF/LCDR1', saveto=saveto,n_jobs=16)

    var = const = 0
    for nm in os.listdir(saveto):
        if nm.startswith('microlensedconst_'):
            const += 1
        elif nm.startswith('cleanvar_'):
            var += 1
    print(const, 'microlensed constant stars and', var, 'clean variable candidates')
