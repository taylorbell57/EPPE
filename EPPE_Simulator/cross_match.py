import numpy as np
import pandas as pd
import astropy.constants as const


# This is the function you can use to cross-match the two exoplanet archive tables
def cross_match_tables():
    data = pd.read_csv('planets.csv', comment='#')

    data2 = pd.read_csv('compositepars.csv', comment='#')

    # This will list all of the columns that I downloaded from the website 
        # if it's missing something, you can download your own table from
        # https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=compositepars

    # Uncomment the following if you want to see all of the column names
    # list(data) # not the composite table, but has inclinations
    # list(data2) # the composite (good) table, currently has no inclinations, but we'll insert that now


    # make a new column to hold the inclinations
    data2['fpl_orbincl'] = np.nan
    data2['fpl_orblper'] = np.nan
    data2['fpl_tranmid'] = np.nan

    # cross-match the tables to insert the known inclinations and lon. of peri. (takes several minutes......)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
        for i in range(data.shape[0]):
            try:
                loc = np.where(np.logical_and(data['pl_hostname'][i]==data2['fpl_hostname'],
                                              data['pl_letter'][i]==data2['fpl_letter']))[0][0]
#                 data2['fpl_orbincl'][loc] = data['pl_orbincl'][i]
#                 data2['fpl_orblper'][loc] = data['pl_orblper'][i]
#                 data2['fpl_tranmid'][loc] = data['pl_tranmid'][i]
                data2.loc[loc, 'fpl_orbincl'] = data.loc[i, 'pl_orbincl']
                data2.loc[loc, 'fpl_orblper'] = data.loc[i, 'pl_orblper']
                data2.loc[loc, 'fpl_tranmid'] = data.loc[i, 'pl_tranmid']
            except:
                continue
    
    data2.to_csv('compositepars_crossMatched.csv')


# data = pd.read_csv('compositepars_with_inclinations.csv', comment='#')

# # Grab the specific columns I need and convert them into SI units
# radii = np.array(data['fpl_radj'])*const.R_jup.value # m
# masses = np.array(data['fpl_bmassj'])*const.M_jup.value # kg
# a = np.array(data['fpl_smax'])*const.au.value # m
# per = np.array(data['fpl_orbper']) # days
# inc = np.array(data['fpl_orbincl']) # degrees
# e = np.array(data['fpl_eccen']) # None
# dist = np.array(data['fst_dist'])*const.pc.value # m
# teff = np.array(data['fst_teff']) # K
# rstar = np.array(data['fst_rad'])*const.R_sun.value # m