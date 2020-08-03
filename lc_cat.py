import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class lc_cat:
    def __init__(self,h5_file):
        self.cat_file = h5py.File(h5_file,'r')
        self.deg_to_rad = 0.017453292519943295
        self.arcsec_to_rad = 4.84813681109536e-06
        self.ind_cols = ['RA','Dec','I1','I2','Nep','ID','FilterID','Field','RcID','MeanMag','StdMag','RStdMag','MaxMag','MinMag','Chi2','MaxPower','FreqMaxPower']
        self.lc_data_name = 'AllLC'
        self.lc_cols = ['HMJD','Mag','MagErr','ColorCoef','Flags']
        self.ind_cat = self.load_ind()

    def __getitem__(self,idx):
        lc_idx = self.ind_cat.iloc(0)[idx]
        lc_start = int(lc_idx['I1'])
        lc_end = int(lc_idx['I2'])
        lc_raw = np.array(self.cat_file[self.lc_data_name][0:5,lc_start-1:lc_end])
        lc = pd.DataFrame(lc_raw.transpose(),columns=self.lc_cols)
        return lc

    def __len__(self):
        return len(self.ind_cat)

    def load_ind(self,ind_name='IndAllLC'):
        inds = np.array(self.cat_file[ind_name])
        ind_cat = pd.DataFrame(inds.transpose(),columns=self.ind_cols)
        return ind_cat

    def plot_lc(self,idx):
        # get the filter
        color = 'r'
        filt = self.ind_cat[table.ind_cat.index==idx]['FilterID'].to_numpy()
        if filt == 2:
            color='g'
        #get the light-curve
        lc = self.__getitem__(idx)
        #plot
        plt.errorbar(lc['HMJD'],lc['Mag'],yerr=lc['MagErr'],color=color,marker='o',ls='none')
        plt.gca().invert_yaxis()
        plt.xlabel('HJD',FontSize=16)
        plt.ylabel('Magnitude',FontSize=16)
        title_str = 'Lightcurve #'+str(idx)
        plt.title(title_str,FontSize=16)
        plt.show()

    def find_lc(self,ra,dec,radius=1.5,coo_unit='deg'):
        if coo_unit == 'deg':
            ra = ra*self.deg_to_rad
            dec = dec*self.deg_to_rad
        radius = radius*self.arcsec_to_rad
        table = self.ind_cat[(self.ind_cat['RA']<ra+radius) & (self.ind_cat['RA']>ra-radius) 
                                & (self.ind_cat['Dec']>dec-radius) & (self.ind_cat['Dec']<dec+radius)]
        return table