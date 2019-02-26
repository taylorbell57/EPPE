# Author: Taylor James Bell
# Last Update: 2019-02-26

import numpy as np
import astropy.constants as const
from .Polarimetry import *
from .KeplerOrbit import KeplerOrbit


class EPPE(object):
    def __init__(self, systems, rad=0.15, trans=0.7, filt='V', filterObj=None, sensFloor=None):
        
        self.systems = systems
        
        self.rad = rad #m
        self.trans = trans #dimensionless (0-1)
        self.filterObj = filterObj
        self.sensFloor = sensFloor
        
        if filt=='V' or (filt is None and self.filterObj is None):
            wavCent = 551e-9 #m
            wavWidth = 88e-9 #m
        elif filt=='U':
            wavCent = 365e-9 # m
            wavWidth = 66e-9 # m
        elif filt=='B':
            wavCent = 445e-9 # m
            wavWidth = 94e-9 # m
        elif filt=='R':
            wavCent = 658e-9 # m
            wavWidth = 138e-9 # m
        elif filt=='I':
            wavCent = 806e-9 # m
            wavWidth = 149e-9 # m
        if self.filterObj is None:
            nwav = 100
            dwav = wavWidth/nwav*np.ones(nwav)
            wavs = np.linspace(wavCent-wavWidth/2., wavCent+wavWidth/2., nwav)
            tput = np.ones_like(wavs)
            
            self.filterObj = {'wavs': wavs, 'tput': tput, 'dwav': dwav}
            
        if self.systems.updateFlag:
            self.systems.updateAlbedos(filt)
            self.systems.updateFlag = False
        
        return
    
    def expose_photometric(self, expTime=1.):
        
        teleFactor = self.trans*(expTime*3600)*(np.pi*self.rad**2)
        
        fstars = teleFactor*self.systems.Fobs(self.systems.Fstar(self.filterObj))
        fplanets = teleFactor*self.systems.Fobs(self.systems.Fp(self.filterObj))
        fPhotNoise = np.sqrt(fstars+fplanets)
        
        return np.rint(fplanets), np.rint(fstars), np.rint(fPhotNoise)
    
    def observe_photometric(self, expTime=1., intTime=None, photonNoise=True, pStart=None):
        
        fplanets, fstars, _ = self.expose_photometric(expTime)
        
        fplanetsObs = []
        fstarsObs = []
        timesList = []
        phasesList = []
        
        for i in range(len(fplanets)):
            t0 = self.systems.catalogue['t0'][i]
            dist = self.systems.catalogue['dist'][i]
            Porb = self.systems.catalogue['per'][i]
            a = self.systems.catalogue['a'][i]
            inc = self.systems.catalogue['inc'][i]
            e = self.systems.catalogue['e'][i]
            argp = self.systems.catalogue['argp'][i]
            orbAxisAng = self.systems.catalogue['orbAxisAng'][i]
            
            if intTime is None:
                intTimeTemp = Porb*24.
            else:
                intTimeTemp = intTime
            
            nPoints = int(np.rint(intTimeTemp/expTime))
            
            if pStart is None:
                pStartTemp = np.random.uniform(0,1,1)
            else:
                pStartTemp = pStart
            
            times = t0+pStartTemp*Porb+np.linspace(0, intTimeTemp/24., nPoints)
            phases = t_to_phase(times, Porb, t0)
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

    def observe_polarization(self, expTime=1., intTime=None, photonNoise=True, pStart=None):
        
        fplanetObs, fstarObs, _ = self.expose_photometric(expTime)
        
        stokesCurves = []
        
        for i in range(len(fplanetObs)):
            
            t0 = self.systems.catalogue['t0'][i]
            dist = self.systems.catalogue['dist'][i]
            Porb = self.systems.catalogue['per'][i]
            a = self.systems.catalogue['a'][i]
            inc = self.systems.catalogue['inc'][i]
            e = self.systems.catalogue['e'][i]
            argp = self.systems.catalogue['argp'][i]
            orbAxisAng = self.systems.catalogue['orbAxisAng'][i]
            polEff = self.systems.catalogue['polEff'][i]
            
            if intTime is None:
                intTimeTemp = np.min([Porb*24., 30*24.])
            else:
                intTimeTemp = intTime
            
            nPoints = int(np.rint(intTimeTemp/expTime))
            
            if pStart is None:
                pStartTemp = np.random.uniform(0,1,1)
            else:
                pStartTemp = pStart
            
            times = t0+pStartTemp*Porb+np.linspace(0, intTimeTemp/24., nPoints)
            phases = t_to_phase(times, Porb, t0)
            
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
    
    def compute_SNR(self, expTime=1., photonNoise=True, amps=None, noise=None):
        if amps is None:
            amps = self.compute_amps(expTime)
        if noise is None:
            noise = self.compute_noise(expTime, photonNoise)
        
        return amps/(np.sqrt(2)*noise)
    
    def compute_amps(self, expTime=1.):
        stokesCurves_ideal = self.observe_polarization(expTime, None, photonNoise=False, pStart=0)
        
        amps = []
        
        for i in range(len(stokesCurves_ideal)):
            truth = np.sqrt(stokesCurves_ideal[i][1]**2+stokesCurves_ideal[i][2]**2)/stokesCurves_ideal[i][0]
            amps.append(np.max(truth)-np.min(truth))
            
        return np.array(amps)
    
    def compute_noise(self, expTime=1., photonNoise=True):
        
        fplanetObs, fstarObs, _ = self.expose_photometric(expTime)
        
        stokesNoises = []
        
        for i in range(len(fplanetObs)):
            
            nPoints = int(1e3)
            
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
            
            stokes = np.rint(stokes + stokesStar)
            pol = np.sqrt(stokes[1]**2+stokes[2]**2)/stokes[0]
            stokesNoises.append(np.std(pol))
            
        return np.array(stokesNoises)
        