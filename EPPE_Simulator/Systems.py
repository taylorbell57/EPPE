# Author: Taylor James Bell
# Last Update: 2019-01-28

import numpy as np
import astropy.constants as const
import pandas as pd

class Systems(object):
    def __init__(self, load=True, fname='compositepars_with_inclinations.csv', complete=True, comment='#', nPlanets=300):
        if load:
            self.catalogue = self.load_planet_catalogue(fname=fname, complete=complete, comment=comment)
            # self.catalogue = self.load_crossmatch_planet_catalogue(complete=complete, comment=comment)
        else:
            self.catalogue = self.generate_planet_catalogue(nPlanets)
        return
    
    def load_planet_catalogue(self, fname='compositepars_with_inclinations.csv', complete=True, comment='#'):
        data = pd.read_csv(fname, comment=comment)
        good = np.where(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(
                            np.logical_or(np.isfinite(data['fpl_radj']),np.isfinite(data['fpl_bmassj'])),
                            np.isfinite(data['fst_dist'])),
                            np.isfinite(data['fpl_orbper'])),
                            np.isfinite(data['fst_teff'])),
                            (data['fpl_orbper'] < 100.)),
                            np.isfinite(data['fst_rad'])),
                            np.isfinite(data['fpl_smax'])))[0]
        
        data = data.iloc[good]
        
        name = np.array(data['fpl_hostname'])+' '+np.array(data['fpl_letter'])
        radii = np.array(data['fpl_radj'])*const.R_jup.value
        masses = np.array(data['fpl_bmassj'])*const.M_jup.value/const.M_earth.value
        a = np.array(data['fpl_smax'])*const.au.value
        per = np.array(data['fpl_orbper'])
        inc = np.array(data['fpl_orbincl'])
        orbAxisAng = np.random.uniform(0,360,len(name))
        e = np.array(data['fpl_eccen'])
        argp = np.random.uniform(0,360,len(name))
        dist = np.array(data['fst_dist'])*const.pc.value
        teff = np.array(data['fst_teff'])
        rstar = np.array(data['fst_rad'])*const.R_sun.value
        
        e[np.isnan(e)] = 0
        inc[np.isnan(inc)] = np.arccos(np.random.uniform(0,1,len(inc[np.isnan(inc)])))*180/np.pi
        
        Omega = 270.*np.ones(len(a))
        
        if False:
            i = 1
            Omega = 1
        
        if complete:
            slope1 = 0.2790
            slope2 = 0.589
            slope3 = -0.044
            slope4 = 0.881

            trans1 = 2.04
            trans2 = 0.414*const.M_jup.value/const.M_earth.value
            trans3 = 0.08*const.M_sun.value/const.M_earth.value

            const1 = 1.008*const.R_earth.value
            const2 = const1*trans1**(slope1)/trans1**(slope2)
            const3 = const2*trans2**(slope2)/trans2**(slope3)
            const4 = const3*trans3**(slope3)/trans3**(slope4)

            nonTransiting = np.where(np.logical_not(np.isfinite(radii)))[0]
            transiting = np.where(np.isfinite(radii))[0]
            for i in nonTransiting:
                if masses[i] < trans1:
                    radii[i] = const1*masses[i]**(slope1)
                elif masses[i] < trans2:
                    radii[i] = const2*masses[i]**(slope2)
                elif masses[i] < trans3:
                    radii[i] = const3*masses[i]**(slope3)
                else:
                    radii[i] = const4*masses[i]**(slope4)
        
        albedo = 1.*np.ones_like(radii)
        polEff = 1.*np.ones_like(radii)
        
        catalogue = {'name': name, 'rp': radii, 'a': a, 'per': per,
                     'inc': inc, 'orbAxisAng': orbAxisAng, 'e': e,
                     'dist': dist, 'teff': teff, 'rstar': rstar,
                     'albedo': albedo, 'polEff': polEff}
        
        return catalogue
    
    def generate_planet_catalogue(self, nPlanets=300):
        
        radii = 1.*const.R_jup.value*np.ones(nPlanets)
        a = 0.05*const.au.value*np.ones(nPlanets)
        per = 3.*np.ones(nPlanets)
        inc = 90.*np.ones(nPlanets)
        e = 0.*np.ones(nPlanets)
        dist = 10.*np.ones(nPlanets)*const.pc.value
        teff = 5000.*np.ones(nPlanets)
        rstar = 1.*const.R_sun.value*np.ones(nPlanets)
        
        orbAxisAng = np.random.uniform(0,360,nPlanets)
        
        albedo = 1.*np.ones_like(radii)
        polEff = 1.*np.ones_like(radii)
        
        name = np.arange(nPlanets).astype(str)
        
        catalogue = {'name': name, 'rp': radii, 'a': a, 'per': per,
                     'inc': inc, 'orbAxisAng': orbAxisAng, 'e': e,
                     'dist': dist, 'teff': teff, 'rstar': rstar,
                     'albedo': albedo, 'polEff': polEff}
        
        return catalogue
    
    def Fstar(self, filterObj, tBrights=None):
        """Calculate the stellar photon flux from each system.
        
        Args:
            filterObj (dictionary): The filter object containing the filter wavelengths and throughputs.
            tBrights (ndarray): The brightness temperatures to use if not stellar effective temperature.
        
        Returns:
            ndarray: The photon flux from each system.
        
        """
        
        return self.integrate_spec( self.Fstar_spec(filterObj, tBrights=None), filterObj )
    
    def Fp(self, filterObj, tBrights=None):
        """Calculate the planetary photon flux from each system.
        
        Args:
            filterObj (dictionary): The filter object containing the filter wavelengths and throughputs.
            tBrights (ndarray): The brightness temperatures to use if not stellar effective temperature.
        
        Returns:
            ndarray: The photon flux from each system.
        
        """
        
        return self.integrate_spec( self.Fp_spec(filterObj, tBrights=None), filterObj )
    
    def Fstar_spec(self, filterObj, tBrights=None):
        """Calculate the stellar spectral flux from each system.
        
        Args:
            filterObj (dictionary): The filter object containing the filter wavelengths and throughputs.
            tBrights (ndarray): The stellar brightness temperatures to use if not stellar effective temperature.
        
        Returns:
            ndarray: The spectral flux from each system.
        
        """
        
        if tBrights is None:
            tBrights = self.catalogue['teff']
        tBrights = tBrights.reshape(-1,1)
        wavs = filterObj['wavs'].reshape(1,-1)
        tputs = filterObj['tput'].reshape(1,-1)
        
        a = (2.*const.h.value*const.c.value**2/wavs**5)
        b = (const.h.value*const.c.value)/(wavs*const.k_B.value)
        
        B_wav = a/np.expm1(b/tBrights)
        
        return tputs * B_wav * (np.pi*self.catalogue['rstar']**2).reshape(-1,1)

    def Fp_spec(self, filterObj, tBrights=None):
        """Calculate the reflected photon flux from each system.
        
        Args:
            filterObj (dictionary): The filter object containing the filter wavelengths and throughputs.
            tBrights (ndarray): The stellar brightness temperatures to use if not stellar effective temperature.
        
        Returns:
            ndarray: The reflected photon flux from each system
        
        """
        
        fstar = self.Fstar_spec(filterObj, tBrights)
        albedo = self.catalogue['albedo'].reshape(-1,1)
        rp = self.catalogue['rp'].reshape(-1,1)
        a = self.catalogue['a'].reshape(-1,1)
        
        # factor of 4 missing from fstar and semi-major axis squared, so they cancel out
        return albedo * fstar * (rp/a)**2

    def Fobs(self, fluxes):
        
        return fluxes/(4*np.pi*self.catalogue['dist']**2)
    
    def integrate_spec(self, flux, filterObj):
        
        dwav = filterObj['dwav'].reshape(1,-1)
        energies = const.h.value*const.c.value/filterObj['wavs'].reshape(1,-1)
        
        return np.sum(flux/energies*dwav, axis=1) 
    
    