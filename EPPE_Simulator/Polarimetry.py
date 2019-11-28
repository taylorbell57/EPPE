# Author: Taylor James Bell
# Last Update: 2019-02-26

from .KeplerOrbit import KeplerOrbit

import numpy as np
from astropy.convolution import convolve, Box1DKernel
import matplotlib.pyplot as plt
from scipy.special import jv as Jv

def lambert_scatter(theta, stokes):
    """Compute the Lambertian scattering flux phase function.
    
    Args:
        theta (ndarray): The scattering angle.
    
    Returns:
        ndarray: The Lambertian scattering flux phase function evaluated at scatAngle.
    
    """
    
    return np.abs((np.sin(theta*np.pi/180)+(np.pi-theta*np.pi/180)*np.cos(theta*np.pi/180))/np.pi)*stokes

def rayleigh_scatter(theta):
    """Compute the Rayleigh scattering polarization phase function.
    
    Args:
        theta (ndarray): The scattering angle.
    
    Returns:
        ndarray: The Rayleigh scattering polarization phase function evaluated at theta.
    
    """
    
    return np.sin(theta*np.pi/180)**2/(1+np.cos(theta*np.pi/180)**2)

# def rayleigh_scatter_website(theta, stokes):
#     """Compute the Rayleigh scattering polarization phase function.
    
#     Args:
#         theta (float): The scattering angle.
    
#     Returns:
#         ndarray: The Rayleigh scattering polarization phase function evaluated at theta.
    
#     """
#     matrix = 3./4.*np.array([[np.cos(theta*np.pi/180)**2+1, np.cos(theta*np.pi/180)**2-1, 0, 0],
#                             [np.cos(theta*np.pi/180)**2-1, np.cos(theta*np.pi/180)**2+1, 0, 0],
#                             [0, 0, 2*np.cos(theta*np.pi/180), 0],
#                             [0, 0, 0, 2*np.cos(theta*np.pi/180)]])
#     return np.matmul(matrix, stokes).reshape(4,-1)

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

def polarization_keplerAngles(times, stokes, polEff, dist, Porb, a, inc=90, e=0, argp=90, Omega=270, t0=0):
    orb = KeplerOrbit(Porb=Porb, a=a, inc=inc, e=e, argp=argp, Omega=Omega, t0=t0)
    
    Ii = np.array([1, 0, 0, 0]).reshape(-1,1)
    
    r = np.array(orb.xyz(times))
    angs = xyz_to_scatAngle(r, dist)
    
    lambertCurve = lambert_scatter(angs+180, stokes)[0]
    rayStokesCurve = np.array([rayleigh_scatter(ang, Ii) for ang in angs])[:,:,0].T*lambertCurve[np.newaxis,:]
    rayStokesCurve[1:] *= polEff
    
    scatPlane_angle = compute_scatPlane_angle(times, orb, dist)
    rayStokesCurve = rotate(scatPlane_angle, rayStokesCurve)
    
    return rayStokesCurve

def polarization_apparentAngles(times, stokes, polEff, dist, Porb, a, inc=90, e=0, argp=90, orbAxisAng=0, t0=0):
    orb = KeplerOrbit(Porb=Porb, a=a, inc=inc, e=e, argp=argp, Omega=270, t0=t0)
    
    Ii = np.array([1, 0, 0, 0]).reshape(-1,1)
    
    r = np.array(orb.xyz(times))
    angs = xyz_to_scatAngle(r, dist)
    
    rayStokesCurve = lambert_scatter(angs+180, stokes)
    rayStokesCurve[1] += rayStokesCurve[0]*rayleigh_scatter(angs)*polEff
    
    scatPlane_angle = compute_scatPlane_angle(times, orb, dist)
    rayStokesCurve = rotate(scatPlane_angle, rayStokesCurve)
    rayStokesCurve = rotate(orbAxisAng, rayStokesCurve)
    
    return rayStokesCurve

def retardance_efficiency(wavs, wavCent):
    return Jv(2,np.pi*wavCent/wavs)*np.sqrt(2)

def plot_lightcurve(stokesCurve, filt, fstar=None, stokesCurve_ideal=None, highPassSize=None, lines=False):
    
    stokesCurve = np.copy(stokesCurve)
    if stokesCurve_ideal is not None:
        stokesCurve_ideal = np.copy(stokesCurve_ideal)
    
    if fstar is None:
        fstar = stokesCurve[0]
    
    fig, ax = plt.subplots(1, 1, figsize=(12,4))

    x = stokesCurve[-1]
    F = (stokesCurve[0]/np.median(fstar)-1)*1e6
    order = np.argsort(x)
    F = F[order]
    x = x[order]
    
    ax.plot(x, F, '.', c='k')
    
    if lines and highPassSize is not None:
        F_sig = np.append(np.append(F,F),F)
        smoothed_F = convolve(F_sig, Box1DKernel(highPassSize))[len(F):2*len(F)]
        
        ax.plot(x, smoothed_F, '-', lw=3, c='k')
    
    if lines and (stokesCurve_ideal is not None):
        x_ideal = stokesCurve_ideal[-1]
        F_ideal = (stokesCurve_ideal[0]/np.median(fstar)-1)*1e6
        order_ideal = np.argsort(x_ideal)
        F_ideal = F_ideal[order_ideal]
        x_ideal = x_ideal[order_ideal]

        ax.plot(x_ideal, F_ideal, '--', lw=3, c='k')


#     ax.plot([0,1], [0,0], lw=1, c='k')
    ax.set_ylabel(r'$\rm F_p/F_*~(ppm);~'+filt+'\mbox{-}Band$')
    ax.set_xlabel(r'$\rm Orbital~Phase$')
    ax.set_xlim(0,1)
    
    return fig
#     plt.show()
#     plt.close(fig)

def plot_QU(stokesCurve, filt, stokesCurve_ideal=None, highPassSize=None, lines=False, normed=False):
    
    stokesCurve = np.copy(stokesCurve)
    if stokesCurve_ideal is not None:
        stokesCurve_ideal = np.copy(stokesCurve_ideal)
    
    fig, ax = plt.subplots(1, 1, figsize=(12,4))

    x = stokesCurve[-1]
    Q = stokesCurve[1]
    U = stokesCurve[2]
    if normed:
        Q /= stokesCurve[0]
        U /= stokesCurve[0]
    order = np.argsort(x)
    Q = Q[order]
    U = U[order]
    x = x[order]
    
    if normed:
        Qlabel = 'q'
        Ulabel = 'u'
        norm = 1e6
    else:
        Qlabel = 'Q'
        Ulabel = 'U'
        norm = 1.
    
    ax.plot(x, Q*norm, '.', c='teal', label=r'$\rm '+Qlabel+'_'+filt+'$')
    ax.plot(x, U*norm, '.', c='darkorange', label=r'$\rm '+Ulabel+'_'+filt+'$')
    
    if lines and highPassSize is not None:
        Q_sig = np.append(np.append(Q,Q),Q)
        U_sig = np.append(np.append(U,U),U)
        smoothed_Q = convolve(Q_sig, Box1DKernel(highPassSize))[len(Q):2*len(Q)]
        smoothed_U = convolve(U_sig, Box1DKernel(highPassSize))[len(U):2*len(U)]
        
        ax.plot(x, smoothed_Q*norm, '-', lw=3, c='teal', label=r'$\rm '+Qlabel+'_'+filt+'~Smoothed$')
        ax.plot(x, smoothed_U*norm, '-', lw=3, c='darkorange', label=r'$\rm '+Ulabel+'_'+filt+'~Smoothed$')

    if lines and (stokesCurve_ideal is not None):
        x_ideal = stokesCurve_ideal[-1]
        Q_ideal = stokesCurve_ideal[1]
        U_ideal = stokesCurve_ideal[2]
        if normed:
            Q_ideal /= stokesCurve_ideal[0]
            U_ideal /= stokesCurve_ideal[0]
        order_ideal = np.argsort(x_ideal)
        Q_ideal = Q_ideal[order_ideal]
        U_ideal = U_ideal[order_ideal]
        x_ideal = x_ideal[order_ideal]

        ax.plot(x_ideal, Q_ideal*norm, '--', lw=3, c='teal', label=r'$\rm '+Qlabel+'_'+filt+'~ideal$')
        ax.plot(x_ideal, U_ideal*norm, '--', lw=3, c='darkorange', label=r'$\rm '+Ulabel+'_'+filt+'~ideal$')
    
    ax.plot([0,1], [0,0], lw=1, c='k')
    
    if normed:
        ylabel = r'$\rm Normalized~Polarized~Flux~(ppm)$'
    else:
        ylabel = r'$\rm Polarized~Flux~(photons)$'
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r'$\rm Orbital~Phase$')
    ax.set_xlim(0,1)
    ax.legend(loc=6, bbox_to_anchor=(1,0.5))
    
    return fig
#     plt.show()
#     plt.close(fig)


def plot_Q(stokesCurve, filt, stokesCurve_ideal=None, highPassSize=None, lines=False, normed=False):
    
    stokesCurve = np.copy(stokesCurve)
    if stokesCurve_ideal is not None:
        stokesCurve_ideal = np.copy(stokesCurve_ideal)
    
    fig, ax = plt.subplots(1, 1, figsize=(12,4))

    x = stokesCurve[-1]
    Q = stokesCurve[1]
    if normed:
        Q /= stokesCurve[0]
    order = np.argsort(x)
    Q = Q[order]
    x = x[order]
    
    if normed:
        Qlabel = 'q'
        norm = 1e6
    else:
        Qlabel = 'Q'
        norm = 1.
    
    ax.plot(x, Q*norm, '.', c='teal', label=r'$\rm '+Qlabel+'_'+filt+'$')
    
    if lines and highPassSize is not None:
        Q_sig = np.append(np.append(Q,Q),Q)
        smoothed_Q = convolve(Q_sig, Box1DKernel(highPassSize))[len(Q):2*len(Q)]
        
        ax.plot(x, smoothed_Q*norm, '-', lw=3, c='teal', label=r'$\rm '+Qlabel+'_'+filt+'~Smoothed$')

    if lines and (stokesCurve_ideal is not None):
        x_ideal = stokesCurve_ideal[-1]
        Q_ideal = stokesCurve_ideal[1]
        if normed:
            Q_ideal /= stokesCurve_ideal[0]
        order_ideal = np.argsort(x_ideal)
        Q_ideal = Q_ideal[order_ideal]
        x_ideal = x_ideal[order_ideal]

        ax.plot(x_ideal, Q_ideal*norm, '--', lw=3, c='teal', label=r'$\rm '+Qlabel+'_'+filt+'~ideal$')
    
    ax.plot([0,1], [0,0], lw=1, c='k')
    
    if normed:
        ylabel = r'$\rm Normalized~Polarized~Flux~(ppm)$'
    else:
        ylabel = r'$\rm Polarized~Flux~(photons)$'
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel(r'$\rm Orbital~Phase$')
    ax.set_xlim(0,1)
    ax.legend(loc=6, bbox_to_anchor=(1,0.5))
    
    return fig
#     plt.show()
#     plt.close(fig)


def plot_P(stokesCurve, filt, stokesCurve_ideal=None, highPassSize=None, lines=False):
    
    stokesCurve = np.copy(stokesCurve)
    if stokesCurve_ideal is not None:
        stokesCurve_ideal = np.copy(stokesCurve_ideal)
    
    fig, ax = plt.subplots(1, 1, figsize=(12,4))

    x = stokesCurve[-1]
    F = stokesCurve[0]
    Q = stokesCurve[1]
    U = stokesCurve[2]
    P = np.sqrt(Q**2+U**2)/F
    order = np.argsort(x)
    F = F[order]
    Q = Q[order]
    U = U[order]
    P = P[order]
    x = x[order]
    
    ax.plot(x, P*1e6, '.', c='k', label=r'$\rm P_'+filt+'$')
    
    if lines and highPassSize is not None:
        F_sig = np.append(np.append(F,F),F)
        Q_sig = np.append(np.append(Q,Q),Q)
        U_sig = np.append(np.append(U,U),U)
        P_sig = np.append(np.append(P,P),P)
        
        smoothed_F = convolve(F_sig, Box1DKernel(highPassSize))[len(F):2*len(F)]
        smoothed_Q = convolve(Q_sig, Box1DKernel(highPassSize))[len(Q):2*len(Q)]
        smoothed_U = convolve(U_sig, Box1DKernel(highPassSize))[len(U):2*len(U)]
        smoothed_P = convolve(P_sig, Box1DKernel(highPassSize))[len(P):2*len(P)]
        presmoothed_P = np.sqrt(smoothed_Q**2+smoothed_U**2)/smoothed_F
        
        ax.plot(x, smoothed_P*1e6, '-', lw=3, c='k', label=r'$\rm P_'+filt+'~Smoothed$')
        ax.plot(x, presmoothed_P*1e6, '-', lw=3, c='b', label=r'$\rm P_'+filt+'~Pre\mbox{-}Smoothed$')
    
    if lines and (stokesCurve_ideal is not None):
        x_ideal = stokesCurve_ideal[-1]
        P_ideal = np.sqrt(stokesCurve_ideal[1]**2+stokesCurve_ideal[2]**2)/stokesCurve_ideal[0]
        order_ideal = np.argsort(x_ideal)
        P_ideal = P_ideal[order_ideal]
        x_ideal = x_ideal[order_ideal]

        ax.plot(x_ideal, P_ideal*1e6, '--', lw=3, c='k', label=r'$\rm P_'+filt+'~ideal$')
    
    ax.set_ylabel(r'$\rm Polarization~Fraction~(ppm)$')
    ax.set_xlabel(r'$\rm Orbital~Phase$')
    ax.set_xlim(0,1)
    ax.set_ylim(0)
    ax.legend(loc=6, bbox_to_anchor=(1,0.5))
        
    return fig
#     plt.show()
#     plt.close(fig)
