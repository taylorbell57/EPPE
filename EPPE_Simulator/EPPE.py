# Author: Taylor James Bell
# Last Update: 2019-02-26

import numpy as np
import astropy.constants as const
from scipy.interpolate import interp1d
from .Polarimetry import *
from .KeplerOrbit import KeplerOrbit


class mission(object):
    def __init__(self, systems, rad=0.15, trans=0.7, filt='V', filterObj=None, pemCent=None, usePhoenix=True,
                 fNoiseFloor=0, pNoiseFloor=0, fNoiseMultiplier=1, pNoiseMultiplier=1):
        
        self.systems = systems
        self.usePhoenix = usePhoenix
        
        self.rad = rad #m
        self.trans = trans #dimensionless (0-1)
        self.filterObj = filterObj
        self.fNoiseFloor = fNoiseFloor
        self.pNoiseFloor = pNoiseFloor
        self.fNoiseMultiplier = fNoiseMultiplier
        self.pNoiseMultiplier = pNoiseMultiplier
        self.pemCent = pemCent
        qe = 1.
        
        if (filt=='V' or filt is None) and self.filterObj is None:
            wavCent = 551e-9 #m
            wavWidth = 88e-9 #m
        elif filt=='U' and self.filterObj is None:
            wavCent = 365e-9 # m
            wavWidth = 66e-9 # m
        elif filt=='B' and self.filterObj is None:
            wavCent = 445e-9 # m
            wavWidth = 94e-9 # m
        elif filt=='R' and self.filterObj is None:
            wavCent = 658e-9 # m
            wavWidth = 138e-9 # m
        elif filt=='I' and self.filterObj is None:
            wavCent = 806e-9 # m
            wavWidth = 149e-9 # m
        elif filt=='VR' and self.filterObj is None:
            wavCent = 617e-9 # m
            wavWidth = 220e-9 # m
        elif filt=='LP590' and self.filterObj is None:
            wavCent = 765e-9 # m
            wavWidth = 350e-9 # m
        elif filt=='EPPE' and self.filterObj is None:
            wavCent = 650e-9 # m
            wavWidth = 500e-9 # m
            with open('EPPE_Simulator/APD_QE.csv', 'r') as f:
                qe_wav = []
                qe = []
                for line in f.readlines():
                    line = line.split(',')
                    qe_wav.append(float(line[0]))
                    qe.append(float(line[1]))
                qe_wav = np.array(qe_wav)
                qe = np.array(qe)[np.argsort(qe_wav)]
                qe_wav = np.sort(qe_wav)
        else:
            print('ERROR: Filter with name \''+str(filt)+'\' is not recognized.')
            return None
        
        if self.filterObj is None:
            wavs = np.logspace(np.log10(3000), np.log10(25000), 212027, endpoint=True)/1e10
            wavs = wavs[np.logical_and(wavCent-wavWidth/2. < wavs, wavs < wavCent+wavWidth/2.)]
            nwav = len(wavs)
            dwav = (np.roll(wavs, -1)-wavs)/2.
            dwav[0] = dwav[1]
            dwav[-1] = dwav[-2]
#             if self.usePhoenix:
#                 wavs = np.logspace(np.log10(3000), np.log10(25000), 212027, endpoint=True)/1e10
#                 wavs = wavs[np.logical_and(wavCent-wavWidth/2. < wavs, wavs < wavCent+wavWidth/2.)]
#                 nwav = len(wavs)
#                 dwav = (np.roll(wavs, -1)-wavs)/2.
#                 dwav[0] = dwav[1]
#                 dwav[-1] = dwav[-2]
#             else:
#                 nwav = 100
#                 dwav = wavWidth/nwav*np.ones(nwav)/2.
#                 wavs = np.linspace(wavCent-wavWidth/2., wavCent+wavWidth/2., nwav)
            
            if self.pemCent is None:
                totals = np.array([(np.sum(retardance_efficiency(wavs[::100], pemCent)*dwav[::100])
                                    /np.sum(dwav[::100])) for pemCent in wavs[::100]])
                self.pemCent = wavs[::100][np.argmax(totals)]
    
            tput = retardance_efficiency(wavs, self.pemCent)#np.ones_like(wavs)
            
            self.filterObj = {'wavs': wavs, 'tput': tput, 'dwav': dwav}
        else:
            wavs = self.filterObj['wavs']
            
            if self.pemCent is None:
                totals = np.array([(np.sum(retardance_efficiency(wavs[::100], pemCent)*dwav[::100])
                                    /np.sum(dwav[::100])) for pemCent in wavs[::100]])
                self.pemCent = wavs[::100][np.argmax(totals)]
    
            self.filterObj['tput'] *= retardance_efficiency(wavs, self.pemCent)
        
        if type(qe) != float:
            qe_interp = interp1d(qe_wav, qe, fill_value='extrapolate')
            self.filterObj['tput'] *= qe_interp(wavs*1e6)
        
        if self.systems.updateFlag:
            self.systems.updateAlbedos(filt)
            self.systems.updateFlag = False
        
        return
    
    def expose_photometric(self, expTime=1., rnd=False):
        
        teleFactor = self.trans*(expTime*3600)*(np.pi*self.rad**2)
        
        fstars = teleFactor*self.systems.Fobs(self.systems.Fstar(self.filterObj, usePhoenix=self.usePhoenix))
        fplanets = teleFactor*self.systems.Fobs(self.systems.Fp(self.filterObj, usePhoenix=self.usePhoenix))
        fPhotNoise = np.sqrt(fstars+fplanets)*self.fNoiseMultiplier + (fstars+fplanets)*self.fNoiseFloor
        
        if rnd:
            return np.rint(fplanets), np.rint(fstars), np.rint(fPhotNoise)
        else:
            return fplanets, fstars, fPhotNoise
    
    def observe_photometric(self, expTime=1., intTime=None, photonNoise=True, pStart=None, rnd=False):
        
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
            
            fp = lambertCurve
            fs = fstars[i]
            
            if photonNoise:
                planetNoise = np.random.normal(loc=0,
                                               scale=(np.sqrt(lambertCurve)*self.fNoiseMultiplier
                                                      +lambertCurve*self.fNoiseFloor),
                                               size=nPoints)
                starNoise = np.random.normal(loc=0, scale=(np.sqrt(fstars[i])*self.fNoiseMultiplier
                                                           +fstars[i]*self.fNoiseFloor), 
                                             size=nPoints)
                
                fp += planetNoise
                fs += starNoise
            
            if rnd:
                fp = np.rint(fp)
                fs = np.rint(fs)
            
            fplanetsObs.append(fp)
            fstarsObs.append(fs)
            
        return fplanetsObs, fstarsObs, timesList, phasesList

    def observe_polarization(self, expTime=1., intTime=None, pStart=None,
                             photonNoise=True, stellarVariability=False, stellarAmp=0.001, stellarPeriod=4*np.pi, rnd=False):
        
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
                pStartTemp = np.random.uniform(0.,1.,1)
            else:
                pStartTemp = pStart
            
            times = t0+pStartTemp*Porb+np.linspace(0., intTimeTemp/24., nPoints)
            phases = t_to_phase(times, Porb, t0)
            
            if photonNoise:
                fNoise = np.sqrt(fplanetObs[i])*self.fNoiseMultiplier + fplanetObs[i]*self.fNoiseFloor
                nPhotons = fplanetObs[i] + np.random.normal(loc=0., scale=fNoise, size=nPoints)
                pNoise = np.sqrt(fplanetObs[i]/3.)/2.*self.fNoiseMultiplier+fplanetObs[i]/3./4.*self.fNoiseFloor
                pNoise = pNoise*self.pNoiseMultiplier+fplanetObs[i]/3./4.*self.pNoiseFloor
                nPhotons_stokes = np.random.normal(loc=0, scale=pNoise, size=3*nPoints).reshape(3,nPoints)
                stokes = np.append(nPhotons[np.newaxis,:], nPhotons_stokes, axis=0)

                if stellarVariability:
                    stFluct = 1+stellarAmp*np.cos(times/stellarPeriod*2*np.pi+np.random.uniform()*2*np.pi)
                    nPhotonsStar = fstarObs[i]*stFluct
                else:
                    nPhotonsStar = fstarObs[i]
                
                fNoise = (np.sqrt(fstarObs[i])*self.fNoiseMultiplier + fstarObs[i]*self.fNoiseFloor)
                nPhotonsStar += np.random.normal(loc=0, scale=fNoise, size=nPoints)
                pNoise = np.sqrt(fstarObs[i]/3.)/2.*self.fNoiseMultiplier+fstarObs[i]/3./4.*self.fNoiseFloor
                pNoise = pNoise*self.pNoiseMultiplier+fstarObs[i]/3./4.*self.pNoiseFloor
                nPhotonsStar_stokes = np.random.normal(loc=0, scale=pNoise, size=3*nPoints).reshape(3,nPoints)
                stokesStar = np.append(nPhotonsStar[np.newaxis, :], nPhotonsStar_stokes, axis=0)
            else:
                stokes = np.array([fplanetObs[i], 0, 0, 0]).reshape(4,1)
                stokesStar = np.array([fstarObs[i], 0, 0, 0]).reshape(4,1)
                
                if stellarVariability:
                    stFluct = 1+stellarAmp*np.cos(times/stellarPeriod*2*np.pi+np.random.uniform()*2*np.pi)
                    stokesStar *= stFluct
            
            stokesCurve = polarization_apparentAngles(times, stokes, polEff, dist, Porb, a,
                                                      inc, e, argp, orbAxisAng, t0)
            stokesCurve += stokesStar
            if rnd:
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
    
    def compute_amps(self):
        expTime = 0.1
        stokesCurves_ideal = self.observe_polarization(expTime=expTime, intTime=None, pStart=0, photonNoise=False)
        
        amps = []
        
        for i in range(len(stokesCurves_ideal)):
            truth = np.sqrt(stokesCurves_ideal[i][1]**2+stokesCurves_ideal[i][2]**2)/stokesCurves_ideal[i][0]
            amps.append(np.max(truth)-np.min(truth))
            
        return np.array(amps)
    
    def compute_noise(self, expTime=1.):
        
        fplanetObs, fstarObs, _ = self.expose_photometric(expTime)
        
        stokesNoises = []
        
        for i in range(len(fplanetObs)):
            
            nPoints = int(1e3)
            
            fNoise = np.sqrt(fplanetObs[i])*self.fNoiseMultiplier + fplanetObs[i]*self.fNoiseFloor
            nPhotons = fplanetObs[i] + np.random.normal(loc=0., scale=fNoise, size=nPoints)
            pNoise = np.sqrt(fplanetObs[i]/3.)/2.*self.fNoiseMultiplier+fplanetObs[i]/3./4.*self.fNoiseFloor
            pNoise = pNoise*self.pNoiseMultiplier+fplanetObs[i]/3./4.*self.pNoiseFloor
            nPhotons_stokes = np.random.normal(loc=0, scale=pNoise, size=3*nPoints).reshape(3,nPoints)
            stokes = np.append(nPhotons[np.newaxis,:], nPhotons_stokes, axis=0)

            fNoise = (np.sqrt(fstarObs[i])*self.fNoiseMultiplier + fstarObs[i]*self.fNoiseFloor)
            nPhotonsStar = fstarObs[i] + np.random.normal(loc=0, scale=fNoise, size=nPoints)
            pNoise = np.sqrt(fstarObs[i]/3.)/2.*self.fNoiseMultiplier+fstarObs[i]/3./4.*self.fNoiseFloor
            pNoise = pNoise*self.pNoiseMultiplier+fstarObs[i]/3./4.*self.pNoiseFloor
            nPhotonsStar_stokes = np.random.normal(loc=0, scale=pNoise, size=3*nPoints).reshape(3,nPoints)
            stokesStar = np.append(nPhotonsStar[np.newaxis, :], nPhotonsStar_stokes, axis=0)
            
            stokes = np.rint(stokes + stokesStar)
            pol = np.sqrt(stokes[1]**2+stokes[2]**2)/stokes[0]
            stokesNoises.append(np.std(pol))
            
        return np.array(stokesNoises)
        