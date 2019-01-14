# Author: Taylor James Bell
# Last Update: 2019-01-11

import numpy as np

def lambertFlux(scatAngle):
    """Compute the Lambertian scattering phase function.
    
    Args:
        scatAngle (ndarray): The scattering angle.
    
    Returns:
        ndarray: The Lambertian phase function evaluated at scatAngle.
    
    """
    
    return np.abs((np.sin(scatAngle)+(np.pi-scatAngle)*np.cos(scatAngle))/np.pi)


def rayleighFlux(scatAngle):
    """Compute the Rayleigh scattering phase function.
    
    Args:
        scatAngle (ndarray): The scattering angle.
    
    Returns:
        ndarray: The Rayleigh phase function evaluated at scatAngle.
    
    """
    
    return np.sin(alpha)**2/(1+np.cos(alpha)**2)


def rayleighPolarization(scatAngle):
    return