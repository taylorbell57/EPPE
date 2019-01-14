# Author: Taylor James Bell
# Last Update: 2019-01-14

import numpy as np
import astropy.constants as const
import pandas as pd

class Systems(object):
    def __init__(self, load=True, fname='planets.csv', complete=True, comment='#', nPlanets=3000):
        if load:
            self.catalogue = self.load_planet_catalogue(fname=fname, complete=complete, comment=comment)
        else:
            self.catalogue = self.generate_planet_catalogue(nPlanets)
        return
    
    def load_planet_catalogue(self, fname='planets.csv', complete=True, comment='#'):
        data = pd.read_csv(fname, comment=comment)
        good = np.where(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(
                            np.logical_or(np.isfinite(data['pl_radj']),np.isfinite(data['pl_bmassj'])),
                            np.logical_or(np.isfinite(data['st_dist']), np.isfinite(data['gaia_dist']))),
                            np.isfinite(data['pl_orbper'])),
                            np.isfinite(data['st_teff'])),
                            np.isfinite(data['st_rad'])),
                            np.isfinite(data['pl_orbsmax'])))[0]
        
        data = data.iloc[good]
        
        radii = np.array(data['pl_radj'])*const.R_jup.value
        masses = np.array(data['pl_bmassj'])*const.M_jup.value/const.M_earth.value
        a = np.array(data['pl_orbsmax'])*const.au.value
        per = np.array(data['pl_orbper'])*3600*24
        gaia_dist = np.array(data['gaia_dist'])*const.pc.value
        dist = np.array(data['st_dist'])*const.pc.value
        teff = np.array(data['st_teff'])
        rstar = np.array(data['st_rad'])*const.R_sun.value

        dist[np.isfinite(gaia_dist)] = gaia_dist[np.isfinite(gaia_dist)]
        
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
        
        albedo = np.ones_like(radii)
        polEff = np.ones_like(radii)
        
        catalogue = {'rp': radii, 'a': a, 'per': per, 'dist': dist, 'teff': teff, 'rstar': rstar, 'albedo': albedo, 'polEff': polEff}
        
        return catalogue
    
    def generate_planet_catalogue(self, nPlanets=3000):
        
        radii = const.R_jup.value*np.ones(nPlanets)
        a = 0.02*const.au.value*np.ones(nPlanets)
        per = 3*np.ones(nPlanets)
        dist = 10*np.ones(nPlanets)
        teff = 5000*np.ones(nPlanets)
        
        albedo = np.ones_like(radii)
        polEff = np.ones_like(radii)
        
        catalogue = {'rp': radii, 'a': a, 'per': per, 'dist': dist, 'teff': teff, 'albedo': albedo, 'polEff': polEff}
        
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
        dwav = filterObj['dwav']
        wavs = filterObj['wavs'].reshape(1,-1)
        tputs = filterObj['tput'].reshape(1,-1)
        
        a = (2.*const.h.value*const.c.value**2/wavs**5)
        b = (const.h.value*const.c.value)/(wavs*const.k_B.value)
        
        B_wav = a/np.expm1(b/tBrights)
        
        return tputs * B_wav * np.pi*self.catalogue['rstar'].reshape(-1,1)**2

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
        
        # factor of 4 missing from fstar and semi-major axis, so they cancel out
        return albedo * fstar * (rp/a)**2

    def Fobs(self, fluxes):
        
        return fluxes/(4*np.pi*self.catalogue['dist']**2)
    
    def integrate_spec(self, flux, filterObj):
        
        dwav = filterObj['dwav']
        energies = const.h.value*const.c.value/filterObj['wavs'].reshape(1,-1)
        
        return np.sum(flux/energies*dwav, axis=1) 
    
    