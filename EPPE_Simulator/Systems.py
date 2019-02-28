# Author: Taylor James Bell
# PHOENIX stellar spectra taken from https://doi.org/10.1051/0004-6361/201219058
    # and downloaded from http://phoenix.astro.physik.uni-goettingen.de
# Last Update: 2019-02-26

import numpy as np
import astropy.constants as const
import pandas as pd
from astropy.io import fits

class Systems(object):
    def __init__(self, load=True, fname='compositepars_crossMatched.csv', complete=True, comment='#', nPlanets=300,
                 albedo=0.3, polEff=1.0, randomOrientation=False):
        
        if albedo=='theo':
            albedo = 0.3
            self.updateFlag = True
        else:
            self.updateFlag = False
        
        if load:
            self.catalogue = self.load_planet_catalogue(fname=fname, complete=complete, comment=comment,
                                                        polEff=polEff, randomOrientation=randomOrientation,
                                                        albedo=albedo)
        else:
            self.catalogue = self.generate_planet_catalogue(nPlanets, polEff=polEff, randomOrientation=randomOrientation,
                                                            albedo=albedo)
            
        return
    
    def load_planet_catalogue(self, fname='compositepars_crossMatched.csv', complete=True, comment='#',
                              albedo=0.3, polEff=0.5, randomOrientation=False):
        data = pd.read_csv(fname, comment=comment)
        good = np.where(np.logical_and(np.logical_and(np.logical_and(
                        np.logical_and(np.logical_and(np.logical_and(np.logical_and(
                            np.logical_or(np.isfinite(data['fpl_radj']),np.isfinite(data['fpl_bmassj'])),
                            np.isfinite(data['fst_dist'])),
                            np.isfinite(data['fpl_orbper'])),
                            np.isfinite(data['fst_teff'])),
                            (data['fpl_orbper'] < 100.)),
                            np.isfinite(data['fst_rad'])),
                            np.isfinite(data['fst_optmag'])),
                            np.isfinite(data['fpl_smax'])))[0]
        
        data = data.iloc[good]
        
        name = np.array(data['fpl_hostname'])+' '+np.array(data['fpl_letter'])
        radii = np.array(data['fpl_radj'])*const.R_jup.value
        masses = np.array(data['fpl_bmassj'])*const.M_jup.value/const.M_earth.value
        a = np.array(data['fpl_smax'])*const.au.value
        per = np.array(data['fpl_orbper'])
        t0 = np.array(data['fpl_tranmid'])
        inc = np.array(data['fpl_orbincl'])
        e = np.array(data['fpl_eccen'])
        argp = np.array(data['fpl_orblper'])
        dist = np.array(data['fst_dist'])*const.pc.value
        teff = np.array(data['fst_teff'])
        rstar = np.array(data['fst_rad'])*const.R_sun.value
        logg = np.array(data['fst_logg'])
        logg[np.isnan(logg)] = 4.5
        logg = logg - (logg%0.5) + np.rint((logg%0.5)*2)/2.
        feh = np.array(data['fst_met'])
        feh[np.isnan(feh)] = 0.
        feh = (feh - (feh%0.5) + np.rint((feh%0.5)*2)/2.)
        feh[feh<-2.] = (feh[feh<-2.] - (feh[feh<-2.]%1) + np.rint((feh[feh<-2.]%1)))
        optMag = np.array(data['fst_optmag'])
        optMagBand = np.array(data['fst_optmagband'])
        nirMag = np.array(data['fst_nirmag'])
        nirMagBand = np.array(data['fst_nirmagband'])
        ra = np.array(data['ra'])
        dec = np.array(data['dec'])
        
        t0[np.isnan(t0)] = 0.
        e[np.isnan(e)] = 0.
        argp[e==0] = 90.
        argp[np.isnan(argp)] = np.random.uniform(0.,360.,len(argp[np.isnan(argp)]))
        inc[np.isnan(inc)] = np.arccos(np.random.uniform(0.,1.,len(inc[np.isnan(inc)])))*180./np.pi
        nirMagBand[np.isnan(nirMag)] = optMagBand[np.isnan(nirMag)]
        nirMag[np.isnan(nirMag)] = optMag[np.isnan(nirMag)]
        
        Omega = 270.*np.ones(len(a))
        if randomOrientation:
            orbAxisAng = np.random.uniform(0,360,len(name))
        else:
            orbAxisAng = np.zeros(len(name))
        
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
        
        albedo = albedo*np.ones_like(radii)
        polEff = polEff*np.ones_like(radii)
        teq = 0.25**0.25*teff*np.sqrt(rstar/a)
        
        catalogue = {'name': name, 'rp': radii, 'a': a, 'per': per,
                     'inc': inc, 'orbAxisAng': orbAxisAng,
                     't0': t0, 'e': e, 'argp': argp, 'teq': teq,
                     'dist': dist, 'teff': teff, 'rstar': rstar,
                     'logg': logg, 'feh': feh,
                     'optMag': optMag, 'optMagBand': optMagBand,
                     'nirMag': nirMag, 'nirMagBand': nirMagBand,
                     'ra': ra, 'dec': dec,
                     'albedo': albedo, 'polEff': polEff}
        
        return catalogue
    
    def generate_planet_catalogue(self, nPlanets=300, albedo=0.3, polEff=0.5, randomOrientation=False):
        
        radii = 1.*const.R_jup.value*np.ones(nPlanets)
        a = 0.05*const.au.value*np.ones(nPlanets)
        per = 3.*np.ones(nPlanets)
        t0 = 0.*np.ones(nPlanets)
        inc = 90.*np.ones(nPlanets)
        e = 0.*np.ones(nPlanets)
        argp = 90.*np.ones(nPlanets)
        dist = 10.*np.ones(nPlanets)*const.pc.value
        teff = 5000.*np.ones(nPlanets)
        rstar = 1.*const.R_sun.value*np.ones(nPlanets)
        feh = 0.*np.ones(nPlanets)
        logg = 4.5*np.ones(nPlanets)
        optMag = 6.*np.ones(nPlanets)
        optMagBand = np.array(['V' for i in range(nPlanets)])
        nirMag = 7.*np.ones(nPlanets)
        nirMagBand = np.array(['J' for i in range(nPlanets)])
        ra = np.random.uniform(0.,360.,nPlanets)
        dec = np.random.uniform(-90.,90.,nPlanets)
        
        if randomOrientation:
            orbAxisAng = np.random.uniform(0.,360.,nPlanets)
        else:
            orbAxisAng = np.zeros(nPlanets)
        
        albedo = albedo*np.ones_like(radii)
        polEff = polEff*np.ones_like(radii)
        teq = 0.25**0.25*teff*np.sqrt(rstar/a)
        
        name = np.arange(nPlanets).astype(str)
        
        catalogue = {'name': name, 'rp': radii, 'a': a, 'per': per,
                     'inc': inc, 'orbAxisAng': orbAxisAng,
                     't0': t0, 'e': e, 'argp': argp, 'teq': teq,
                     'dist': dist, 'teff': teff, 'rstar': rstar,
                     'logg': logg, 'feh': feh,
                     'optMag': optMag, 'optMagBand': optMagBand,
                     'nirMag': nirMag, 'nirMagBand': nirMagBand,
                     'ra': ra, 'dec': dec,
                     'albedo': albedo, 'polEff': polEff}
        
        return catalogue
    
    def updateAlbedos(self, filt):
        """Update the planetary albedos using the predictions from Sudarsky.
        
        Args:
            filt: A string containing the filter's name
        
        Returns:
            None
        
        """
        
        teq = self.catalogue['teq']
        
        jup = self.catalogue['rp']/const.R_earth.value > 4
        cat1 = np.logical_and(teq<=150, jup)
        cat2 = np.logical_and(np.logical_and(teq>150, teq<=350), jup)
        cat3 = np.logical_and(np.logical_and(teq>350, teq<=900), jup)
        cat4 = np.logical_and(np.logical_and(teq>900, teq<=1500), jup)
        cat5 = np.logical_and(np.logical_and(teq>1500, teq<=2000), jup)
        cat6 = np.logical_and(teq>2000, jup)
        
        u1 = 0.87
        u2 = 1.00
        u3 = 0.50
        u4 = 0.20
        u5 = 0.45
        u6 = 0.05

        b1 = 0.65
        b2 = 0.98
        b3 = 0.20
        b4 = 0.07
        b5 = 0.65
        b6 = 0.05

        v1 = 0.70
        v2 = 0.95 
        v3 = 0.10
        v4 = 0.05 # Tres-2b (Ag = 0.0253 +/- 0.0072)
        v5 = 0.40 # Kepler-7b (Ag = 0.32 +/- 0.03); tau Boo (Ag < 0.12)
        v6 = 0.05 # WASP-12b (Ag < 0.064)

        r1 = 0.65
        r2 = 0.95
        r3 = 0.05
        r4 = 0.05
        r5 = 0.40
        r6 = 0.05

        i1 = 0.60
        i2 = 0.85
        i3 = 0.02
        i4 = 0.10
        i5 = 0.55
        i6 = 0.05
        
        if filt=='U':
            self.catalogue['albedo'][cat1] = u1
            self.catalogue['albedo'][cat2] = u2
            self.catalogue['albedo'][cat3] = u3
            self.catalogue['albedo'][cat4] = u4
            self.catalogue['albedo'][cat5] = u5
            self.catalogue['albedo'][cat6] = u6
        elif filt=='B':
            self.catalogue['albedo'][cat1] = b1
            self.catalogue['albedo'][cat2] = b2
            self.catalogue['albedo'][cat3] = b3
            self.catalogue['albedo'][cat4] = b4
            self.catalogue['albedo'][cat5] = b5
            self.catalogue['albedo'][cat6] = b6
        elif filt=='R':
            self.catalogue['albedo'][cat1] = r1
            self.catalogue['albedo'][cat2] = r2
            self.catalogue['albedo'][cat3] = r3
            self.catalogue['albedo'][cat4] = r4
            self.catalogue['albedo'][cat5] = r5
            self.catalogue['albedo'][cat6] = r6
        elif filt=='I':
            self.catalogue['albedo'][cat1] = i1
            self.catalogue['albedo'][cat2] = i2
            self.catalogue['albedo'][cat3] = i3
            self.catalogue['albedo'][cat4] = i4
            self.catalogue['albedo'][cat5] = i5
            self.catalogue['albedo'][cat6] = i6
        else:
            self.catalogue['albedo'][cat1] = v1
            self.catalogue['albedo'][cat2] = v2
            self.catalogue['albedo'][cat3] = v3
            self.catalogue['albedo'][cat4] = v4
            self.catalogue['albedo'][cat5] = v5
            self.catalogue['albedo'][cat6] = v6
            
        self.catalogue['albedo'][np.logical_not(jup)] = 0.3
        
        return
    
    def Fstar(self, filterObj, usePhoenix=True, tBrights=None):
        """Calculate the stellar photon flux from each system.
        
        Args:
            filterObj (dictionary): The filter object containing the filter wavelengths and throughputs.
            usePhoenix (bool, optional): Whether or not to use PHOENIX stellar models for fluxes.
            tBrights (ndarray, optional): The brightness temperatures to use if not stellar effective temperature.
        
        Returns:
            ndarray: The photon flux from each system.
        
        """
        
        return self.integrate_spec( self.Fstar_spec(filterObj, usePhoenix=usePhoenix, tBrights=tBrights), filterObj )
    
    def Fp(self, filterObj, tBrights=None):
        """Calculate the planetary photon flux from each system.
        
        Args:
            filterObj (dictionary): The filter object containing the filter wavelengths and throughputs.
            tBrights (ndarray): The brightness temperatures to use if not stellar effective temperature.
        
        Returns:
            ndarray: The photon flux from each system.
        
        """
        
        return self.integrate_spec( self.Fp_spec(filterObj, tBrights=None), filterObj )
    
    def Fstar_spec(self, filterObj, usePhoenix=True, tBrights=None):
        """Calculate the stellar spectral flux from each system.
        
        Args:
            filterObj (dictionary): The filter object containing the filter wavelengths and throughputs.
            usePhoenix (bool, optional): Whether or not to use PHOENIX stellar models for fluxes.
            tBrights (ndarray): The stellar brightness temperatures to use if not stellar effective temperature.
        
        Returns:
            ndarray: The spectral flux from each system.
        
        """
        
        if usePhoenix:
            teff = self.catalogue['teff']
            logg = -self.catalogue['logg']
            feh = self.catalogue['feh']
            
            wavs = np.logspace(np.log10(3000), np.log10(25000), 212027, endpoint=True)/1e10
#             inds = np.where(np.logical_and(wavCent-wavWidth/2. < wavs, wavs < wavCent+wavWidth/2.))
            inds = np.where(np.logical_and(np.min(filterObj['wavs']) <= wavs, wavs <= np.max(filterObj['wavs'])))[0]
            
            #lte11800-6.00-4.0.PHOENIX-ACES-AGSS-COND-2011-HiRes
            
            teffStr = np.copy(teff)
            teffStr[teff<=7000] = teffStr[teff<=7000] - (teffStr[teff<=7000]%100) + np.rint((teffStr[teff<=7000]%100)/100)*100
            teffStr[teff>7000] = teffStr[teff>7000] - (teffStr[teff>7000]%200) + np.rint((teffStr[teff>7000]%200)/200)*200
            teffStr[teff>12000] = 12000
            
            folder = '/home/taylor/Documents/Research/PHOENIX/MedResFITS/R10000FITS/'
            files = [folder+'lte'+str(int(teffStr[i])).zfill(5)
                     +("{0:+.02f}".format(logg[i]) if logg[i]!=0 else '-0.00')
                     +("{0:+.01f}".format(feh[i]) if feh[i]!=0 else '-0.0')
                     +'.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits' for i in range(len(teff))]

            files_unique, inverse_inds = np.unique(files, return_inverse=True)
            
            B_wav_unique = []
            for fname in files_unique:
                with fits.open(fname) as file:
                    B_wav_unique.append(file[0].data[inds])
            
            B_wav = np.array([B_wav_unique[i] for i in inverse_inds])
            
        else:
            if tBrights is None:
                tBrights = self.catalogue['teff']
            tBrights = tBrights.reshape(-1,1)
            wavs = filterObj['wavs'].reshape(1,-1)

            a = (2.*const.h.value*const.c.value**2/wavs**5)
            b = (const.h.value*const.c.value)/(wavs*const.k_B.value)

            B_wav = a/np.expm1(b/tBrights) * np.pi #pi for stellar circle of unit radius
        
        tputs = filterObj['tput'].reshape(1,-1)
        
        return tputs * B_wav * (self.catalogue['rstar']**2).reshape(-1,1)

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
        """Account for the d^-2 drop-off in flux.
        
        Args:
            fluxes (ndarray): The fluxes from each system.
        
        Returns:
            ndarray: The observed fluxes from each system.
        
        """
        
        return fluxes/(4*np.pi*self.catalogue['dist']**2)
    
    def integrate_spec(self, flux, filterObj):
        """Convert the spectral flux into photons/s.
        
        Args:
            flux (ndarray): The spectral flux from each system.
            filterObj (dictionary): The filter object containing the filter wavelengths and throughputs.
        
        Returns:
            ndarray: The photon flux from each system.
        
        """
        
        dwav = filterObj['dwav'].reshape(1,-1)
        energies = const.h.value*const.c.value/filterObj['wavs'].reshape(1,-1)
        
        return np.sum(flux/energies*dwav, axis=1) 
    
    def name_to_index(self, name):
        """Get the array index of a particular planet.
        
        Args:
            name (str): Name of planet in system array.
        
        Returns:
            int: The array index of the particular planet.
        
        """
        
        if np.any(self.catalogue['name']==name):
            i = np.where(self.catalogue['name']==name)[0][0]
        else:
            print('No such planet!')
            i = None
        
        return i
    
    def index_details(self, i):
        """Print some details about a particular planet.
        
        Args:
            i (int): Index of planet in system array.
        
        Returns:
            None
        
        """
        
        if i >= len(self.catalogue['name']) or i < 0:
            print('Index '+str(i)+' out of range!')
        else:
            print('Name:', self.catalogue['name'][i])
            print('Radius: '+str(np.round(self.catalogue['rp'][i]/const.R_jup.value, 2))+' Rjup')
            print('Period:', np.round(self.catalogue['per'][i], 2), 'days')
            print('Equilibrium Temperature: '+str(int(np.rint(self.catalogue['teq'][i])))+' K')
            print('Transit Depth: '+str(np.round((self.catalogue['rp'][i]/self.catalogue['rstar'][i])**2*100, 3))+'%')
            print('Distance: '+str(int(np.rint(self.catalogue['dist'][i]/const.pc.value)))+' pc')
        
        return
    
    def name_details(self, name):
        """Print some details about a particular planet.
        
        Args:
            name (str): Name of planet in system array.
        
        Returns:
            None
        
        """
        
        i = self.name_to_index(name)
        
        if i is not None:
            self.index_details(i)
        
        return
    