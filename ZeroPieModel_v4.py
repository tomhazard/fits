# Zero-pie model based on PHYSICAL REVIEW B 90, 094518 (2014)

import numpy as np
import scipy as sp
import time
import matplotlib.pyplot as plt

#Zero-pi potential
def V_potential(EL, EJ, flux_ext, phi, theta):
    return - 2 * EJ * np.cos(theta) * np.cos(phi - flux_ext / 2) + EL * phi ** 2 + 2 * EJ 
    
    
q_e = 1.602e-19 # electron charge in C
h = 6.626e-34 # Planck constant in Js
hbar = h /  (2 * np.pi) # Planck constant / 2p in Js
phi_0 = hbar / (2 * q_e) # flux quantum / 2p in Wb
   
L = 1.25e-6 # large inductance  in H
C = 1e-13 # shunting capacitance in F
LJ = 4e-8 # single Josephson junction inductance in H

CJ = 7.5e-23 / LJ # Josephson junction capacitance in F - based on previous measurements LJ * CJ = 7.5e-23 or with other words w_p = 2pi * 18 GHz
CS = CJ + C # sum capacitance   

EL = phi_0 ** 2 / L / h * 1e-9 # in GHz units (EL[Hz] = EL[J] / h)
EJ = phi_0 ** 2 / LJ / h * 1e-9  # in GHz units (EJ[Hz] = EJ[J] / h
ECJ = q_e ** 2 / (2 * CJ) / h * 1e-9 # in GHz units (ECJ[Hz] = ECJ[J] / h)
ECS = q_e ** 2 / (2 * CS) / h * 1e-9 # in GHz units (ECS[Hz] = ECS[J] / h)

# PRB paper parameters

#w_p = 2 * np.pi * 40 # 2pi * 40 GHz
#EL = hbar * w_p * 1e-3 / h # 0.04 GHz
#ECS = hbar * w_p * 1e-3 / h # 0.04 GHz
#EJ = hbar * w_p / 3.95 / h # 10.1 GHz
#ECJ = (hbar * w_p) ** 2 / (8 * EJ) / h ** 2 # 19.8 GHz

N = 60 # number of points in both theta and phi directions
N_flux = 21 # number of flux points

flux_ext_vector = np.linspace(0*np.pi, 2 * np.pi, N_flux)  #linspace makes evenly spaced list from nstart, nstop,npoints
phi = np.linspace(-8*np.pi, 8*np.pi, N)
theta = np.linspace(-0.8*np.pi, 1.8*np.pi, N)
PHI, THETA = np.meshgrid(phi, theta)
    
dp = (phi[-1] - phi[0]) / (N-1) # generates step size ([-1] returns last value in array)
dt = (theta[-1] - theta[0]) / (N-1)

Bdiag = (4 * ECJ / dp ** 2 + 4 * ECS / dt ** 2) * np.eye(N) #eye(N) returns N-d identity matrix
Bupper = np.diag([-2 * ECJ / dp ** 2] * (N - 1), 1) #diag returns a 2D array of first number on the 2nd number diagonal
Blower = np.diag([-2 * ECJ / dp ** 2] * (N - 1), -1)
B = Bdiag + Bupper + Blower
blst = [B] * N

Dupper = np.diag([-2 * ECS / dt ** 2] * N * (N - 1), N)
Dlower = np.diag([-2 * ECS / dt ** 2] * N * (N - 1), -N)

E0 = []
E1 = []
E2 = []
E3 = []
E4 = []
E5 = []

Degeneracy = []

for flux_ind in range(N_flux):
    start = time.time() # measure the elapsed time
    
    H = sp.linalg.block_diag(*blst) + Dupper + Dlower

    V_elements = []
    for ind in range(N):
        V_elements.extend(V_potential(EL, EJ, flux_ext_vector[flux_ind], phi, theta[ind]))
    V = np.diag(V_elements)

    H += V 

    E, PSI = np.linalg.eig(H) #calc eigenvalues
    E_sorted = np.sort(E)
    PSI_sorted = PSI[:, E.argsort()]

    E0.append(E_sorted[0])
    E1.append(E_sorted[1])
    E2.append(E_sorted[2])    
    E3.append(E_sorted[3])
    E4.append(E_sorted[4])
    E5.append(E_sorted[5])
     
    deg = np.log10((E_sorted[2] - E_sorted[0])/(E_sorted[1] - E_sorted[0]))
    Degeneracy.append(deg)
    
    print(str(flux_ind+1), '/', str(N_flux), ': Elapsed time ' + str(round(time.time()-start)) + ' sec')


#%% Plot the dispersion relation and degeneracy

E_zero = np.mean(E0)

fig = plt.figure(1)
ax  = fig.add_subplot(111)
plt.xlabel('External flux', fontsize=16)
plt.ylabel('Energy (GHz)', fontsize=16)
plt.plot(flux_ext_vector, E0 - E_zero, 'b')
plt.plot(flux_ext_vector, E1 - E_zero, 'b')
plt.plot(flux_ext_vector, E2 - E_zero, 'b')
plt.plot(flux_ext_vector, E3 - E_zero, 'b')
plt.plot(flux_ext_vector, E4 - E_zero, 'b')
plt.plot(flux_ext_vector, E5 - E_zero, 'b')

ax.set_xticks([0, np.pi, 2*np.pi])
ax.set_xticklabels([0, r"${\pi}$", r"$2{\pi}$"] , fontsize=14)

fig = plt.figure(2)
ax  = fig.add_subplot(111)
plt.xlabel('External flux', fontsize=16)
plt.ylabel('Degeneracy', fontsize=16)
plt.plot(flux_ext_vector, Degeneracy, 'r')
ax.set_xticks([0, np.pi, 2*np.pi])
ax.set_xticklabels([0, r"${\pi}$", r"$2{\pi}$"] , fontsize=14)
            
   
#%% Plot the wavefunctions for n = 0, 1, 2, 3 energy levels
theta_tick = [0, np.pi]
theta_label = [0, r"${\pi}$"]
phi_tick = [-8*np.pi, -6*np.pi, -4*np.pi, -2*np.pi, 0, 2*np.pi, 4*np.pi, 6*np.pi, 8*np.pi]
phi_label = [r"$-8{\pi}$", r"$-6{\pi}$", r"$-4{\pi}$", r"$-2{\pi}$", 0, r"$2{\pi}$", r"$4{\pi}$", r"$6{\pi}$", r"$8{\pi}$"]              


V_matrix = np.asarray(V_elements).reshape(N,N)    
    
fig = plt.figure(3)
ax  = fig.add_subplot(111)
ax.set_title('V($\Theta$, $\Phi$)')
plt.pcolor(PHI, THETA, V_matrix, cmap='afmhot')
#plt.colorbar()
plt.axis('scaled')
plt.xlabel('$\Phi$', fontsize=16)
plt.ylabel('$\Theta$', fontsize=16)
ax.set_xticks(phi_tick)
ax.set_xticklabels(phi_label, fontsize=14)               
ax.set_yticks(theta_tick)
ax.set_yticklabels(theta_label, fontsize=14)

E_zero = np.real(E_sorted[0])

for ind in range(6):
    fig = plt.figure(4+ind)
    ax  = fig.add_subplot(111)
    ax.set_title('$E_%s$ = %s GHz' %(ind, np.round(np.real(E_sorted[ind]) - E_zero,3)), fontsize=16)
    plt.pcolor(PHI, THETA, np.real(-PSI_sorted[:, ind].reshape(N,N)), cmap='bwr')
    plt.clim(-0.3,0.3)#plt.colorbar()
    plt.axis('scaled')
    plt.xlabel('$\Phi$', fontsize=16)
    plt.ylabel('$\Theta$', fontsize=16)
    ax.set_xticks(phi_tick)
    ax.set_xticklabels(phi_label, fontsize=14)               
    ax.set_yticks(theta_tick)
    ax.set_yticklabels(theta_label, fontsize=14)

