# Author: Taylor James Bell
# Last Update: 2019-01-11

import numpy as np

def lambert_scatter(theta, stokes):
    """Compute the Lambertian scattering flux phase function.
    
    Args:
        scatAngle (ndarray): The scattering angle.
    
    Returns:
        ndarray: The Lambertian scattering flux phase function evaluated at scatAngle.
    
    """
    
    return np.abs((np.sin(theta*np.pi/180)+(np.pi-theta*np.pi/180)*np.cos(theta*np.pi/180))/np.pi)*stokes

def rayleigh_scatter(theta, stokes):
    """Compute the Rayleigh scattering polarization phase function.
    
    Args:
        scatAngle (ndarray): The scattering angle.
    
    Returns:
        ndarray: The Rayleigh scattering polarization phase function evaluated at theta.
    
    """
    matrix = 3./4.*np.array([[np.cos(theta*np.pi/180)**2+1, np.cos(theta*np.pi/180)**2-1, 0, 0],
                            [np.cos(theta*np.pi/180)**2-1, np.cos(theta*np.pi/180)**2+1, 0, 0],
                            [0, 0, 2*np.cos(theta*np.pi/180), 0],
                            [0, 0, 0, 2*np.cos(theta*np.pi/180)]])
    return np.matmul(matrix, stokes).reshape(4,-1)

def rotate(phi, stokes):
    Qprime = stokes[1]*np.cos(2*phi*np.pi/180) + stokes[2]*np.sin(2*phi*np.pi/180)
    Uprime = -stokes[1]*np.sin(2*phi*np.pi/180) + stokes[2]*np.cos(2*phi*np.pi/180)
    return np.array([stokes[0], Qprime, Uprime, stokes[3]]).reshape(4,-1)

def get_rotate_matrix(phi):
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(2*phi*np.pi/180), np.sin(2*phi*np.pi/180), 0],
                     [0, -np.sin(2*phi*np.pi/180), np.cos(2*phi*np.pi/180), 0],
                     [0, 0, 0, 0]])

def rotate2(phi, stokes, rotate_matrix=None):
    if rotate_matrix is None:
        rotate_matrix = get_rotate_matrix(phi)
    return np.matmul(rotate_matrix, stokes).reshape(4,-1)

def xyz_to_scatAngle(r, dist):
    d = np.array([dist, 0, 0])[:,np.newaxis]
    scatAngle = 180/np.pi*np.arccos(np.sum(r*(d-r), axis=0)
                                        /(np.sqrt(np.sum(r**2, axis=0))*np.sqrt(np.sum((d-r)**2, axis=0))))
    return scatAngle
