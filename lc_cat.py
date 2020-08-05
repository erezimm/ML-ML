import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class lc_cat:
    """
    This Class is made to acess the lightcurve catalogue from arXiv:2007.01537. 
    More info on the structure of the files can be found in the paper
    Use this class to access individual files from the catalogue.
    Class attributes are as follows:
        cat_file -  the hdf5 file instance from which it accesses the data
        deg_to_rad - the number conversion from degress to radians
        arcsec_to_rad - the number conversion from arcseconds to radians
        ind_cols - the column names for the index table
        lc_data_name - the dataset name of the lightcurve data within the hdf5 catalogue file
        lc_cols - the light curve data columns
        ind_cat - the catalogue indexes. contains all the metadata of each light curve and it's position in the
                  h5df catalogue file
        variable_threshold - a dictionary containing the threshold criteria to classify a star as variable as a function fo filter and magnitude. 
                             The keys are magnitudes and the values are arrasy with the criteria for the robust std to classify a lightcurve as variable.
                             first value in the array is g band and second is r band.
        variable_candidates - the ind_cat catalogue of all variable candidate stars based on robust Std
    """

    def __init__(self,h5_file,min_Nep=24):
        self.cat_file = h5py.File(h5_file,'r')
        self.deg_to_rad = 0.017453292519943295
        self.arcsec_to_rad = 4.84813681109536e-06
        self.ind_cols = ['RA','Dec','I1','I2','Nep','ID','FilterID','Field','RcID','MeanMag','StdMag','RStdMag','MaxMag','MinMag','Chi2','MaxPower','FreqMaxPower']
        self.lc_data_name = 'AllLC'
        self.lc_cols = ['HMJD','Mag','MagErr','ColorCoef','Flags']
        self.ind_cat = self.load_ind()
        self.variable_threshold = {13:[0.0658,0.0411],
                                13.5:[0.0605,0.0409],
                                14:[0.0598,0.0421],
                                14.5:[0.0615,0.0437],
                                15:[0.0639,0.0452],
                                15.5:[0.0662,0.0464],
                                16:[0.0683,0.0481],
                                16.5:[0.0708,0.0514],
                                17:[0.0751,0.0581],
                                17.5:[0.0831,0.0705],
                                18:[0.0976,0.0915],
                                18.5:[0.1222,0.1245],
                                19:[0.1609,0.1736],
                                19.5:[0.2189,0.2435],
                                20:[0.3016,0.3393]}
        self.variable_candidates,self.constants = self.find_variables(min_Nep)

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
        """
        This function loads the index table from the hdf5 file
        inputs:
            ind_name - the name of the data set which contains the indexes of all the light curves in the hdf5 file
        """
        inds = np.array(self.cat_file[ind_name])
        ind_cat = pd.DataFrame(inds.transpose(),columns=self.ind_cols)
        return ind_cat

    def plot_lc(self,idx):
        """
        A function that plots the light curve at index idx
        inputs:
            idx - the index of the light curve
        """
        # get the filter
        color = 'r'
        filt = self.ind_cat[self.ind_cat.index==idx]['FilterID'].to_numpy()
        if filt == 1:
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
        """
        A function that finds all the lightcurves within certain coordinates
        inputs:
            ra - right acesnsion of the lightcurve
            dec - declination of the lightcurve
            radius - the radius of search from these coordinates in arcseconds
            coo_unit - the units of the ccordinates, default is deg.
                       if the input is in radians one should change from deg to something else.
        """
        if coo_unit == 'deg':
            ra = ra*self.deg_to_rad
            dec = dec*self.deg_to_rad
        radius = radius*self.arcsec_to_rad
        table = self.ind_cat[(self.ind_cat['RA']<ra+radius) & (self.ind_cat['RA']>ra-radius) 
                                & (self.ind_cat['Dec']>dec-radius) & (self.ind_cat['Dec']<dec+radius)]
        return table
    
    def round_05(self,number):
        """
        A function the rounds a number to the nearest half integer.
        """       
        return round(number*2)/2
    
    def find_variables(self,min_Nep=24):
        """
        A function that finds variable sources candidates from the ind_cat catalogue.
        This is done by searching for light curves that pass the thershold of robust std in magnitude according to the criteria in the paper.
        Also returns an indicator table of what is percieved to be constant lightcurve sources.
        inputs:
            min_Nepp - minimum number of epochs per lightcurve to be considered a variable source
        """       
        meanmags = self.ind_cat['MeanMag'].apply(self.round_05)
        var_indexes = []
        for group in self.ind_cat.groupby([meanmags,'FilterID']):
            meanmag= group[0][0]
            filt = int(group[0][1])
            table = group[1]
            if meanmag in self.variable_threshold.keys():
                thershold = self.variable_threshold[meanmag][filt-1]
                g_variables = table[(table['RStdMag']>thershold) & (table['Nep'] >= min_Nep)]
                var_indexes.extend(g_variables.index.to_list())
        variables = self.ind_cat[self.ind_cat.index.isin(var_indexes)]
        constants = self.ind_cat[np.logical_not(self.ind_cat.index.isin(var_indexes))]
        return variables,constants