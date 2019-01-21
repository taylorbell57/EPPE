# Author: Taylor James Bell
# Last Update: 2019-01-21

import numpy as np
import astropy.constants as const
from .Polarimetry import *
from .KeplerOrbit import KeplerOrbit


class EPPE(object):
    def __init__(self, rad=0.3, trans=0.7, filterObj=None, sensFloor=None):
        
        self.rad = rad #m
        self.trans = trans #dimensionless
        self.filterObj = filterObj
        self.sensFloor = sensFloor
        
        if self.filterObj is None:
            wavCent = 551e-6 #m
            wavWidth = 88e-6 #m
            nwav = 100
            dwav = wavWidth/nwav
            wavs = np.linspace(wavCent-wavWidth/2., wavCent+wavWidth/2., nwav)
            tput = np.ones_like(wavs)
            
            self.filterObj = {'wavs': wavs, 'tput': tput, 'dwav': dwav}
        
        return
    
    def observe_photometric(self, systems, expTime=1):
        
        teleFactor = self.trans*(expTime/24)*(np.pi*self.rad**2)
        
        fstarObs = teleFactor*systems.Fobs(systems.Fstar(self.filterObj))
        fplanetObs = teleFactor*systems.Fobs(systems.Fp(self.filterObj))
        fPhotNoise = np.sqrt(fstarObs+fplanetObs)
        
        return fplanetObs, fstarObs, fPhotNoise

    def observe_polarization(self, systems, expTime=1):
        
        teleFactor = self.trans*(expTime/24)*(np.pi*self.rad**2)
        
        fstarObs = teleFactor*systems.Fobs(systems.Fstar(self.filterObj))
        fplanetObs = teleFactor*systems.Fobs(systems.Fp(self.filterObj))
        fPhotNoise = np.sqrt(fstarObs+fplanetObs)
        
        stokesCurves = []
        
        for i in range(len(fplanetObs)):
            
            stokes = np.array([fplanetObs[i], 0, 0, 0]).reshape(4,1)
            
            # t0 = systems.catalogue['t0'][i]
            t0 = 0 # FIX
            dist = systems.catalogue['dist'][i]
            Porb = systems.catalogue['per'][i]
            a = systems.catalogue['a'][i]
            inc = systems.catalogue['inc'][i]
            e = systems.catalogue['e'][i]
            # argp = systems.catalogue['argp'][i]
            argp = 90 # FIX
            # Omega = systems.catalogue['Omega'][i]
            Omega = 270 # FIX
            
            # FIX: randomly generate start time, and use determined integration duration
            times = np.random.uniform(0,1,1)*Porb+np.linspace(0, Porb, np.rint(Porb*24./expTime))
            
            phases = t_to_phase(times, Porb)
            stokesCurve = polarization(times, stokes, dist, Porb, a, inc=inc, e=e, argp=argp, Omega=Omega, t0=0)
            
            stokesCurve[0,:] += fstarObs[i]
            
            stokesCurve = np.append(stokesCurve, times[np.newaxis,:], axis=0)
            stokesCurve = np.append(stokesCurve, phases[np.newaxis,:], axis=0)
            
            stokesCurves.append(stokesCurve)
            
            # plt.plot(phases, np.sqrt(stokesCurve[1,:]**2+stokesCurve[2,:]**2)/stokesCurve[0,:]*100)
            # plt.ylabel('P (%)')
            # plt.xlabel('Orbital Phase')
            # plt.ylim(0,100)
            # plt.xlim(0,1)
            # plt.gca().set_xticks(np.linspace(0,1,11,endpoint=True), minor=True)
            # plt.show()

            # plt.plot(phases, stokesCurve[1,:]/stokesCurve[0,:], label='q')
            # plt.plot(phases, stokesCurve[2,:]/stokesCurve[0,:], label='u')
            # plt.plot(phases, stokesCurve[3,:]/stokesCurve[0,:], label='v')
            # plt.ylabel('Flux')
            # plt.xlabel('Orbital Phase')
            # plt.xlim(0,1)
            # plt.ylim(-1,1)
            # plt.gca().set_xticks(np.linspace(0,1,11,endpoint=True), minor=True)
            # plt.legend(loc=6, bbox_to_anchor=(1,0.5))
            # plt.show()

            # orb = eppe.KeplerOrbit(Porb=Porb, a=a, inc=inc, e=e, argp=argp, Omega=Omega, t0=t0)
            # angles = eppe.compute_scatPlane_angle(times, orb, dist)

            # plt.plot(phases, angles)
            # plt.ylabel('Scattering Plane Angle')
            # plt.xlabel('Orbital Phase')
            # plt.xlim(0,1)
            # plt.ylim(-180,180)
            # plt.gca().set_xticks(np.linspace(0,1,11,endpoint=True), minor=True)
            # plt.show()
            
        return stokesCurves#np.array(stokesCurves)
