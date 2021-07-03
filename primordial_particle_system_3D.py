# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 10:08:43 2021

@author: alexa
"""

# -*- coding: utf-8 -*-

import pygame
from pygame.locals import *
import numpy as np
import itertools
import matplotlib.pyplot as plt
from datetime import datetime

print('Press p to pause/unpause. Add new cells in pause mode by right clicking.')

def R_x(a):
    return np.moveaxis(
        np.array([[ np.cos(a/2), -1j*np.sin(a/2) ],[ -1j*np.sin(a/2), np.cos(a/2) ]]),
        -1,0)

def R_y(a):
    return np.moveaxis(
        np.array([[np.cos(a/2), -np.sin(a/2)],[np.sin(a/2), np.cos(a/2)]]),
        -1,0)

def R_z(a):
    return np.moveaxis(
        np.array([[np.exp(-1j*a/2), np.zeros(a.shape)],[np.zeros(a.shape),np.exp(1j*a/2)]]),
        -1,0)

def R3D(a_,phi,theta):
    res = np.identity(2)
    for R in [
            R_z(phi),
            R_y(theta),
            R_z(a_),
            np.transpose(R_y(theta),(0,2,1)),
            np.transpose(R_z(phi),(0,2,1))]:
        res = np.matmul(res,R)    
    return res

def get_cartesian(phi_,theta_):
    # y, x, z
    return np.stack([np.sin(phi_)+np.sin(theta_),np.cos(phi_)+np.sin(theta_),np.cos(theta_)],axis=-1)

def phi_theta_from_qubit(q):
    mod_z0 = np.sqrt(q[:,:,0].real**2 + q[:,:,0].imag**2).flatten() # theta
    mod_z1 = np.sqrt(q[:,:,1].real**2 + q[:,:,1].imag**2).flatten() # phi
    phi_ = 2*np.arctan(mod_z1/mod_z0)
    arg_z0 = np.arctan( q[:,:,0].imag / q[:,:,0].real )
    arg_z1 = np.arctan( q[:,:,1].imag / q[:,:,1].real )
    theta_ = arg_z1 - arg_z0
    return phi, theta

w,h  = 800,800
n_particles = int(1400*0.75*0.75)
z = 75
zoom = 0.8
spread = 1
#np.random.seed(42)
pos = np.stack([
                    np.random.uniform(0+z*(1-spread)/2,z*(1-(1-spread)/2),n_particles),
                    np.random.uniform(0+z*(1-spread)/2,z*(1-(1-spread)/2),n_particles),
                    np.random.uniform(0+z*(1-spread)/2,z*(1-(1-spread)/2),n_particles)
                    ]).T
#np.ones((n_particles,2))*np.array([[h/2,w/2]])
phi = np.random.uniform(0,2*np.pi,n_particles)
theta = np.random.uniform(0,np.pi,n_particles)
orient = get_cartesian(phi,theta)
alpha = 180*np.pi/180
beta = 17*np.pi/180
neighbor_radius = 5
cm = plt.get_cmap('viridis')
col = 255*np.ones((n_particles,3),dtype=np.int32)
v = 0.67
print('density: %.3f%%'%(100*n_particles/((z*spread)**2)))

def move(pos,orient):
    pos = pos + v*orient
    pos[pos < 0] += 2*v
    pos[pos > z] -= 2*v
    return pos


pygame.init()
screen = pygame.display.set_mode((w,h))
#cell = pygame.Surface((w,cellsize))
#cell.fill((255,255,255))
running = True
pause = True
stoptime = pygame.time.get_ticks()
t = 0
t0,t1,t2,t3, t4 = [datetime.now()]*5
it = 0
while running:
    #screen.fill((50,50,50))
    if not pause:
        t3 = datetime.now()
        screen.fill((0,0,0))
        #pygame.time.wait(5)
        znew = np.max(np.max(pos,0)-np.min(pos,0))
        grid_contents = np.zeros((int(znew/neighbor_radius)+1,int(znew/neighbor_radius)+1,int(znew/neighbor_radius)+1,n_particles),dtype=bool)
        grid_idx = ((pos - np.min(pos,0))/neighbor_radius).astype(int)
        isparticle = np.concatenate([grid_idx, np.arange(n_particles).reshape(-1,1)],axis=1)
        grid_contents[isparticle[:,0],isparticle[:,1],isparticle[:,2],isparticle[:,3]] = True
        neighbors = np.zeros((n_particles,n_particles),dtype=bool)
        rotationpart = np.zeros(n_particles,dtype=np.float32)
        for i in range(grid_contents.shape[0]):
            for j in range(grid_contents.shape[1]):
                for k in range(grid_contents.shape[2]):
                    particles = np.argwhere(grid_contents[i,j,k]).flatten()
                    if len(particles) == 0:
                        continue
                    adj_particles = np.argwhere(grid_contents[
                       max(0,i-1):min(grid_contents.shape[0],i+1)+1,
                       max(0,j-1):min(grid_contents.shape[1],j+1)+1,
                       max(0,k-1):min(grid_contents.shape[2],k+1)+1
                    ])[:,-1]
                    #print(i,j,particles,adj_particles)
                    for p in particles:
                        adj_particles_p = np.setdiff1d(adj_particles,[p])
                        if len(adj_particles_p) > 0:
                            idx = np.sum((pos[p] - pos[adj_particles_p])**2,axis=1) < neighbor_radius**2
                            neighbors[p,adj_particles_p[idx]] = True
                            xprod = np.cross(orient[p,None,None],pos[adj_particles_p,None,None]-pos[p,None,None])
                            R = np.sum(xprod >= 0)
                            L = np.sum(xprod < 0)
                            rotationpart[p] += R - L
        rotationpart[rotationpart > 0] = 1
        rotationpart[rotationpart < 0] = -1
        t4 = datetime.now()
        for i in range(n_particles):
            x = pos[i,1]
            y = pos[i,0]
            col = 255*np.array(cm((pos[i,2] - 0+z*(1-spread)/2) / (z*(1-(1-spread)/2)))[:3])
            pygame.draw.circle(screen,col,(int(((1-zoom)/2)*w + x*zoom*w/z),int(((1-zoom)/2)*h + zoom*y*h/z)),int(w/400))
        param_font = pygame.font.Font('freesansbold.ttf', w//50)
        param_text = param_font.render('max_iter = %d,  t1-t0: %.2f,  t2-t1: %.2f,  t3-t2: %.2f,  t4-t3: %.2f  t0-t4: %.2f'%(
                t,(t1-t0).microseconds/1000,(t2-t1).microseconds/1000,(t3-t2).microseconds/1000,(t4-t3).microseconds/1000,(t4-t0).microseconds/1000
            ), True, (255,255,255))
        screen.blit(param_text,param_text.get_rect(bottomleft=(10,h-10)))
        pygame.image.save(screen,'../../PPS/%05d.png'%it)
        it += 1
        t0 = datetime.now()
        d_angle = alpha + beta*rotationpart*neighbors.sum(-1)
        Rmat = R3D(d_angle,phi,theta)
        t1 = datetime.now()
        qubit = np.array([np.cos(theta/2),np.exp(1j*phi)*np.sin(theta/2)]).T
        qubit = np.einsum('ijk,ikl->ijl',qubit.reshape(n_particles,1,2),Rmat)
        phi, theta = phi_theta_from_qubit(qubit)
        orient = get_cartesian(phi,theta)
        pos = move(pos,orient)
        t += 1
        t2 = datetime.now()
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
        pressed_keys = pygame.key.get_pressed()
        if pressed_keys[K_p]: 
                pause = not pause    
        elif event.type == QUIT:
            running = False
    pygame.display.flip()
    
pygame.quit()