# Author: Taylor James Bell
# Last Update: 2019-01-21

import numpy as np
from .KeplerOrbit import KeplerOrbit

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

def compute_scatPlane_angle(times, orb, dist):
    times = times.copy()
    
    # fix scattering plane during transit and eclipse
    times[np.abs(times%orb.Porb-orb.t_trans)<(1./24./3600.)] += (1./24./3600.)
    times[np.abs(times%orb.Porb-orb.t_ecl)<(1./24./3600.)] += (1./24./3600.)

    r = np.array(orb.xyz(times))
    d = np.array([dist, 0, 0])[:,np.newaxis]
    Z = np.array([0, 0, 1])[:,np.newaxis]
    Y = np.array([0, 1, 0])[:,np.newaxis]
    norm = np.cross(r, d, axis=0)
    
    return np.sign(np.sum(Y*norm, axis=0))*180/np.pi*np.arccos(np.sum(Z*norm, axis=0)
                               /(np.sqrt(np.sum(Z**2, axis=0))*np.sqrt(np.sum(norm**2, axis=0))))

def t_to_phase(times, Porb, t0=0):
    return (times-t0)/Porb % 1.

def polarization(times, stokes, dist, Porb, a, inc=90, e=0, argp=90, Omega=270, t0=0):
    orb = KeplerOrbit(Porb=Porb, a=a, inc=inc, e=e, argp=argp, Omega=Omega, t0=t0)
    
    Ii = np.array([1, 0, 0, 0]).reshape(-1,1)
    
    r = np.array(orb.xyz(times))
    angs = xyz_to_scatAngle(r, dist)
    
    lambertCurve = lambert_scatter(angs+180, stokes)[0]
    rayStokesCurve = np.array([rayleigh_scatter(ang, Ii) for ang in angs])[:,:,0].T*lambertCurve[np.newaxis,:]
    
    scatPlane_angle = compute_scatPlane_angle(times, orb, dist)
    rayStokesCurve = rotate(scatPlane_angle, rayStokesCurve)
    
    return rayStokesCurve
