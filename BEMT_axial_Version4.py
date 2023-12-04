#
# code version 3 BEMT
'''
# @author : Dahia Chibouti
# november 2023 @Paris
'''

import numpy as np
import matplotlib.pyplot as plt



is_custom_input = False

if is_custom_input:
    print('CUSTOM input files used')
    # Custom input function goes here
else:
    print('DEFAULT input files used')

    # Rotor parameters
    R = 60/3.28084/2    # 9.14  #1.15  # in m
    Nb = 4              # 6 # blade number
    Omega_rad = 21.62   # rotation speed in rad/s
    Omega_rpm = 30*Omega_rad/np.pi #1921.795  # in rpm
    vel_i = 21.62    # in m/s

    dCldA = 2 * np.pi   # d_Cl/d_alpha in radians
    alpha0_deg = 0.0    # angle of attack (deg)
    twist = -10         # angle of twist (deg)


    # -- AR or chord --
    # AR = ?
    c = 2/3.28084
    # -- theta distribution (theta_deg = 8)
    theta_deg = 17.5

    # Environment parameters
    rho = 1.225        # in kg/m3

    # solver parameters
    nx = 50             # No. of stations along blade

    # For accounting tip loss
    # (change accordingly when prandtlTipLoss_switch is 0)
    root_cut = 0.15  # r/R  # cutout
    tip_cut = 0.999  # r/R

    # Feature switches
    spacing_switch = 1 # [1]Equispaced [2]Cosine [3]aTan
    #prandtlTipLoss_switch = 1

# Spacing blade stations
if spacing_switch == 1:
    # Equi-spaced
    r_prime= np.linspace(root_cut, tip_cut, nx)
    dr_prime= (r_prime[1] - r_prime[0]) * np.ones_like(r_prime)

# elif spacing_switch == 2:
#     # Cosine spaced
#     theta_cosine = np.linspace(0, np.pi, nx)
#     r_prime= R * 0.5 - R * 0.5 * np.cos(theta_cosine)
#     dr_prime= np.diff(r_prime)
#     dr_prime= np.append(dr_prime, 0)

# elif spacing_switch == 3:
#     # atan spaced
#     x_spacing = np.linspace(np.tan(-1), np.tan(1), nx)
#     r_prime= R * 0.5 + R * 0.5 * np.arctan(x_spacing)
#     dr_prime= np.diff(r_prime)
#     dr_prime= np.append(dr_prime, 0)

#-----------------------------------------
# Calculated Rotor Parameters
if 'AR' in locals():
    c = np.ones(nx) * R / AR
elif 'c' in locals():
    c = np.ones(nx) * c

theta_deg = np.ones(nx) * theta_deg
#-----------------------------------------

# params calc
alphaCamber = -1.0 * alpha0_deg * twist*np.pi / 180.0
theta = theta_deg * np.pi / 180 + alphaCamber  # Corrected for cambered airfoils

solidity = Nb * c / (np.pi * R)
Omega_rad = Omega_rpm * np.pi / 30  # in rad per sec
lam_c = vel_i / (R * Omega_rad)

# Feature switches
prandtlTipLoss_switch = 1

# Inflow computation
if prandtlTipLoss_switch == 0:
    # Inflow ratio from BEMT
    const1 = solidity * dCldA / 16
    lam_c_05 = lam_c * 0.5
    lam = -(const1 - lam_c_05) + np.sqrt((const1 - lam_c_05)**2 + 2*const1 * theta * r_prime)
    phi = lam / r_prime
    alf = theta - phi
    prandtl_F = np.ones_like(r_prime)  #r_prime[cutout:end_blade]

elif prandtlTipLoss_switch == 1:
    prandtl_F = np.ones_like(r_prime)  # Initial value
    prandtl_residual = 1.0
    const1 = solidity * dCldA / 16
    lam_c_05 = lam_c * 0.5
    counter = 1

    while prandtl_residual > 0.001 and counter < 21:
        lam = -(const1 / prandtl_F - lam_c_05) + np.sqrt(
            (const1 / prandtl_F - lam_c_05) ** 2 + 2 * const1 / prandtl_F * theta * r_prime
        )
        prandtl_Fsmall = 0.5 * Nb * (1.0 - r_prime) / lam

        prandtl_F_prev = prandtl_F
        prandtl_F = (2 / np.pi) * np.arccos(np.exp(-prandtl_Fsmall))
        prandtl_residual = np.abs(np.linalg.norm(prandtl_F - prandtl_F_prev))
        counter += 1

    if counter == 21:
        print('Warning: Prandtl tip-loss factor failed to converge')

    phi = lam / r_prime
    alf = theta - phi


# Using Momentum theory
ct_vec = prandtl_F * 4 * lam * (lam - lam_c) * r_prime* dr_prime
lam_mean = 2.0 * np.trapz(r_prime, lam * r_prime) / (1 - root_cut * root_cut)
CT_MT = np.sum(ct_vec)

# Using BEMT
ct_vec = 0.5 * solidity * dCldA * dr_prime* (r_prime** 2) * alf
CT_BEMT = np.sum(ct_vec)

# Sectional lift distribution
cl_vec = 2.0 * ct_vec / (solidity * r_prime** 2 * dr_prime)
cl_max, imx = np.max(cl_vec), np.argmax(cl_vec)
gamma_max = 0.5 *cl_max*c[imx] * (R * dr_prime[imx])*np.linalg.norm([r_prime[imx], lam[imx]])*R*Omega_rad

# Thrust (livre force)
# T = sys.mvf*(np.pi*p.R**2)*(p.Omega*p.R)**2*IntCT 0.5*x**2*p.sigma*cl

Thrust = CT_MT * (rho * np.pi * R ** 2 * (R * Omega_rad) ** 2)

# Check between MT and BEMT
if CT_MT - CT_BEMT > np.finfo(float).eps:
    print('Warning: Discrepancy between CT calculated using BEMT and MT')


# Results
# print('\nColl. pitch (deg) =', theta_deg)
# print('solidity     =', solidity)
print('CT  MT       =', CT_MT)
print('CT  BEMT     =', CT_BEMT)
print('inflow ratio =', lam_mean)
print('Thrust (N)   =', Thrust)
print('inflow (m/s) =', lam_mean * R * Omega_rad)
print('CL max       =', cl_max)
print('Gamma max    =', gamma_max)


# Plots with a custom layout (rearranged subfigures)
plt.subplot(2, 2, 1)
plt.plot(r_prime, lam, 'k')
plt.grid(True)
plt.xlabel('r/R')
plt.ylabel('Inflow Ratio')

plt.subplot(2, 2, 2)
plt.plot(r_prime, (alf - alphaCamber) * 180 / np.pi, 'k')
plt.grid(True)
plt.xlabel('r/R')
plt.ylabel('Alpha (deg)')

plt.subplot(2, 2, 3)
plt.plot(r_prime, cl_vec, 'k')
plt.grid(True)
plt.xlabel('r/R')
plt.ylabel('Sectional CL')

plt.subplot(2, 2, 4)
plt.plot(r_prime, ct_vec / Nb, 'k')
plt.grid(True)
plt.xlabel('r/R')
plt.ylabel('Sectional CT')

plt.tight_layout()  # Adjust subplots for better layout

# Save or display the plots
plt.savefig('inflow_plots.png')  # Change the filename and format as needed
plt.show()





