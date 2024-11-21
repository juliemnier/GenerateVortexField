import numpy as np
import sys
import matplotlib.pyplot as plt
import random as rd
from numpy.fft import *

class Vortex_Field:

    def __init__(self, resol, *args, **kwargs):
        self.grid = [] 
        self.vort = [] 
        self.resol = resol

    def distance(self, pos1, pos2):
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def gaussian_vortex(self, x, y, x0, y0, amplitude, sigma):
        """Generate a 2D Gaussian vortex of given amplitude and radius"""
        return amplitude * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    def rankine_vortex(x, y, x0, y0, amplitude, sigma):
        """Generate a 2D Rankine vortex of given amplitude and radius"""
        xx,yy = np.meshgrid(x,y)
        return amplitude * (np.sqrt((xx - x0)**2 + (yy - y0)**2) <= sigma )

    def random_vortex_placing(self, xx, yy, N, amplitude, sigma, margin, type_vortex = 'gaussian'):
        """ Randomly positions N dipoles of vortices of opposite signs in the (x,y) field
            N: number of vortices
            amplitude, sigma: gaussian parameters 

            ajouter marge pres des bords pour decompter un vortex de plus si reintroduit par periodicité
            speedup + margin pour ne pas mettre de vortex aux bords 
        """
        if N % 2 == 1:
            sys.exit("Not a dipole. Choose even number")
        ### initialization
        # output field
        Nx, Ny = np.shape(xx)
        vorticity_field = np.zeros((Nx,Ny))
        # sampling pool
        lx = xx[0,:]
        mask = np.zeros((Nx, Ny), dtype=bool)

        # margin around border, then with periodicity it will properly fill the borders 
        grid_margin = (xx**2 + yy**2 <= (margin*sigma/2)**2)
        halo_cells = int(np.sqrt(len(grid_margin.nonzero()[0])))
        mask[halo_cells:Nx-halo_cells,halo_cells:Ny-halo_cells] = True
        pooly, poolx = mask.nonzero() 
        # for loop
        sgn = 1
        for vortex in range(N):
            # check if current vortex density allows no overlap
            Npool = len(poolx)
            if Npool<=4*sigma*margin*Nx/(2*np.pi):
                sys.exit("Not enough space for number of dipoles")
                return
            rd_pair_index = rd.choice(range(Npool))
            xn, yn = lx[poolx[rd_pair_index]], lx[pooly[rd_pair_index]]
            if type_vortex == 'gaussian':
                vorticity_field+= gaussian_vortex(xx, yy, xn, yn, sgn*amplitude, sigma)
            elif type_vortex == 'rankine':
                vorticity_field+= rankine_vortex(xx, yy, xn, yn, sgn*amplitude, sigma)
            
            # vectorized for performance
            distance_sq = (xx - xn)**2 + (yy - yn)**2
            mask &= distance_sq > (sigma * margin)**2

            pooly, poolx = mask.nonzero() 
            sgn *= -1
        
        return vorticity_field
    

    def handle_periodicity(self):

        
        resol = np.shape(self.vort)[0]
        periodic_vort = np.zeros((3 * resol , 3 * resol))

        # Array at center
        periodic_vort[resol:2 * resol, resol:2 * resol] = self.vort

        # Use np.roll to fill the periodic boundaries
        periodic_vort[:resol, resol:2 * resol] = np.roll(self.vort, shift=-resol, axis=0)  # Top
        periodic_vort[2 * resol:, resol:2 * resol] = np.roll(self.vort, shift=resol, axis=0)  # Bottom
        periodic_vort[resol:2 * resol, :resol] = np.roll(self.vort, shift=-resol, axis=1)  # Left
        periodic_vort[resol:2 * resol, 2 * resol:] = np.roll(self.vort, shift=resol, axis=1)  # Right

        # Fill the corners
        periodic_vort[:resol, :resol] = np.roll(self.vort, shift=(-resol, -resol), axis=(0, 1))  # Top-left
        periodic_vort[:resol, 2 * resol:] = np.roll(self.vort, shift=(-resol, resol), axis=(0, 1))  # Top-right
        periodic_vort[2 * resol:, :resol] = np.roll(self.vort, shift=(resol, -resol), axis=(0, 1))  # Bottom-left
        periodic_vort[2 * resol:, 2 * resol:] = np.roll(self.vort, shift=(resol, resol), axis=(0, 1))  # Bottom-right

        # The center of the array is the periodic part of the field
        periodic_field = periodic_vort[resol//2:3 * resol//2,resol//2:3 * resol//2]

        return periodic_field
    

