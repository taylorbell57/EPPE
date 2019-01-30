# Author: Taylor James Bell
# Last Update: 2019-01-28

import numpy as np
import astropy.constants as const
from .Polarimetry import *
from .KeplerOrbit import KeplerOrbit


class EPPE(object):
    def __init__(self, rad=0.15, trans=0.7, filterObj=None, sensFloor=None):
        
        self.rad = rad #m
        self.trans = trans #dimensionless (0-1)
        self.filterObj = filterObj
        self.sensFloor = sensFloor
        
        if self.filterObj is None:
            wavCent = 551e-9 #m
            wavWidth = 88e-9 #m
            nwav = 100
            dwav = wavWidth/nwav*np.ones(nwav)
            wavs = np.linspace(wavCent-wavWidth/2., wavCent+wavWidth/2., nwav)
            tput = np.ones_like(wavs)
            
            self.filterObj = {'wavs': wavs, 'tput': tput, 'dwav': dwav}
        
        return
    
    def expose_photometric(self, systems, expTime=1.):
        
        teleFactor = self.trans*(expTime*3600)*(np.pi*self.rad**2)
        
        fstars = teleFactor*systems.Fobs(systems.Fstar(self.filterObj))
        fplanets = teleFactor*systems.Fobs(systems.Fp(self.filterObj))
        fPhotNoise = np.sqrt(fstars+fplanets)
        
        return np.rint(fplanets), np.rint(fstars), np.rint(fPhotNoise)
    
    def observe_photometric(self, systems, expTime=1., intTime=24., photonNoise=True):
        
        fplanets, fstars, _ = self.expose_photometric(systems, expTime)
        
        fplanetsObs = []
        fstarsObs = []
        timesList = []
        phasesList = []
        
        for i in range(len(fplanets)):
            # t0 = systems.catalogue['t0'][i]
            t0 = 0 # FIX
            dist = systems.catalogue['dist'][i]
            Porb = systems.catalogue['per'][i]
            a = systems.catalogue['a'][i]
            inc = systems.catalogue['inc'][i]
            e = systems.catalogue['e'][i]
            # argp = systems.catalogue['argp'][i]
            argp = 90 # FIX
            orbAxisAng = systems.catalogue['orbAxisAng'][i]
            
            nPoints = int(np.rint(intTime/expTime))
            
            times = np.random.uniform(0,1,1)*Porb+np.linspace(0, intTime/24., nPoints)
            phases = t_to_phase(times, Porb)
            timesList.append(times)
            phasesList.append(phases)
            
            orb = KeplerOrbit(Porb=Porb, a=a, inc=inc, e=e, argp=argp, Omega=270., t0=t0)
            r = np.array(orb.xyz(times))
            angs = xyz_to_scatAngle(r, dist)
            lambertCurve = lambert_scatter(angs+180, fplanets[i])
            
            if photonNoise:
                pNoise = np.sqrt(lambertCurve)*np.random.normal(loc=0, scale=1, size=nPoints) 
                sNoise = np.random.normal(loc=0, scale=np.sqrt(fstars[i]), size=nPoints)
            else:
                pNoise = sNoise = np.ones(nPoints)
            
            fplanetsObs.append(lambertCurve+pNoise)
            fstarsObs.append(fstars[i]+sNoise)
            
        return fplanetsObs, fstarsObs, timesList, phasesList

    def observe_polarization(self, systems, expTime=1., intTime=24., photonNoise=True):
        
        fplanetObs, fstarObs, _ = self.expose_photometric(systems, expTime)
        
        nPoints = int(np.rint(intTime/expTime))
        
        stokesCurves = []
        
        for i in range(len(fplanetObs)):
            
            # t0 = systems.catalogue['t0'][i]
            t0 = 0 # FIX
            dist = systems.catalogue['dist'][i]
            Porb = systems.catalogue['per'][i]
            a = systems.catalogue['a'][i]
            inc = systems.catalogue['inc'][i]
            e = systems.catalogue['e'][i]
            # argp = systems.catalogue['argp'][i]
            argp = 90 # FIX
            orbAxisAng = systems.catalogue['orbAxisAng'][i]
            polEff = systems.catalogue['polEff'][i]
            
            times = np.random.uniform(0,1,1)*Porb+np.linspace(0, intTime/24., nPoints)
            phases = t_to_phase(times, Porb)
            
            if photonNoise:
                nPhotons = fplanetObs[i] + np.random.normal(loc=0, scale=np.sqrt(fplanetObs[i]), size=nPoints)
                nPhotons_stokes = np.random.normal(loc=0, scale=np.sqrt(fplanetObs[i]/3.)/2., 
                                                   size=3*nPoints).reshape(3,nPoints)
                stokes = np.append(nPhotons[np.newaxis,:], nPhotons_stokes, axis=0)

                nPhotonsStar = fstarObs[i] + np.random.normal(loc=0, scale=np.sqrt(fstarObs[i]), size=nPoints)
                nPhotonsStar_stokes = np.random.normal(loc=0, scale=np.sqrt(fstarObs[i]/3.)/2.,
                                                       size=3*nPoints).reshape(3,nPoints)
                stokesStar = np.append(nPhotonsStar[np.newaxis, :], nPhotonsStar_stokes, axis=0)
            else:
                stokes = np.array([fplanetObs[i], 0, 0, 0]).reshape(4,1)
                stokesStar = np.array([fstarObs[i], 0, 0, 0]).reshape(4,1)
            
            stokesCurve = polarization_apparentAngles(times, stokes, polEff, dist, Porb, a,
                                                      inc, e, argp, orbAxisAng, t0)
            stokesCurve += stokesStar
            stokesCurve = np.rint(stokesCurve)
            
            stokesCurve = np.append(stokesCurve, times[np.newaxis,:], axis=0)
            stokesCurve = np.append(stokesCurve, phases[np.newaxis,:], axis=0)
            
            stokesCurves.append(stokesCurve)
            
        return stokesCurves
    