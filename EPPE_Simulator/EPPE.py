# Author: Taylor James Bell
# Last Update: 2019-01-16

import numpy as np
import astropy.constants as const


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
    
    def observe(self, systems, expTime=1):
        
        teleFactor = self.trans*(expTime/24)*(np.pi*self.rad**2)
        
        fstarObs = teleFactor*systems.Fobs(systems.Fstar(self.filterObj))
        fplanetObs = teleFactor*systems.Fobs(systems.Fp(self.filterObj))
        fPhotNoise = np.sqrt(fstarObs+fplanetObs)
        
        return fplanetObs, fstarObs, fPhotNoise


