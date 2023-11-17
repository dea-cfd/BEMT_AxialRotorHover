#
# code version 3 BEMT_Axial
# @author : Dahia Chibouti
#
import numpy as np
import matplotlib.pyplot as plt



is_custom_input = False

if is_custom_input:
    print('CUSTOM input files used')
    # Custom input function goes here
else:
    print('DEFAULT input files used')

    # Rotor parameters
    R = 1.15 # 0.518  # in m
    Nb = 6  # blade number
    Omega_rpm = 1921.795  # in rpm
    vel_climb = 15.0  # in m/s

    dCldA = 2 * np.pi  # d_Cl/d_alpha in radians
    alpha0_deg = 0.0

    # Selective parameters
    # -- AR or chord --
    # AR = 6  # Aspect ratio
    # c = 0
    cByR_dist = np.array([
        [0.2, 0.116],
        [0.3, 0.109],
        [0.4, 0.101],
        [0.5, 0.094],
        [0.6, 0.087],
        [0.7, 0.080],
        [0.8, 0.072],
        [0.9, 0.065],
        [1.0, 0.058],
    ])
    # -- theta distribution or constant angle --
    # theta_deg = 8
    theta_deg_dist = np.array([
        [0.2, 14.97],
        [0.3, 13.97],
        [0.4, 12.97],
        [0.5, 11.97],
        [0.6, 10.97],
        [0.7, 9.97],
        [0.8, 8.97],
        [0.9, 7.97],
        [1.0, 6.97],
    ])
    # ------

    # Environment parameters
    rho = 1.2249  #0.022  # in kg/m3

    # solver parameters
    nx = 50  # No. of stations along blade

    # Feature switches
    spacing_switch = 1  # [1]Equispaced [2]Cosine [3]aTan
    prandtlTipLoss_switch = 1

    # For accounting tip loss
    # (change accordingly when prandtlTipLoss_switch is 0)
    root_cut = 0.15  # r/R
    tip_cut = 0.999  # r/R

# Spacing blade stations
if spacing_switch == 1:
    # Equi-spaced
    r_bar = np.linspace(root_cut, tip_cut, nx)
    dr_bar = (r_bar[1] - r_bar[0]) * np.ones_like(r_bar)

elif spacing_switch == 2:
    # Cosine spaced
    theta_cosine = np.linspace(0, np.pi, nx)
    r_bar = R * 0.5 - R * 0.5 * np.cos(theta_cosine)
    dr_bar = np.diff(r_bar)
    dr_bar = np.append(dr_bar, 0)

elif spacing_switch == 3:
    # atan spaced
    x_spacing = np.linspace(np.tan(-1), np.tan(1), nx)
    r_bar = R * 0.5 + R * 0.5 * np.arctan(x_spacing)
    dr_bar = np.diff(r_bar)
    dr_bar = np.append(dr_bar, 0)

# Calculated Rotor Parameters
if 'AR' in locals():
    c = np.ones(nx) * R / AR
elif 'c' in locals():
    c = np.ones(nx) * c
elif 'cByR_dist' in locals():
    cByR_poly = np.polyfit(cByR_dist[:, 0], cByR_dist[:, 1], 4)
    cByR = np.polyval(cByR_poly, r_bar)
    c = cByR * R

if 'theta_deg' in locals():
    theta_deg = np.ones(nx) * theta_deg
elif 'theta_deg_dist' in locals():
    if theta_deg_dist.shape[0] == 2:
        theta_poly = np.polyfit(theta_deg_dist[:, 0], theta_deg_dist[:, 1], 1)
    else:
        theta_poly = np.polyfit(theta_deg_dist[:, 0], theta_deg_dist[:, 1], 4)
    theta_deg = np.polyval(theta_poly, r_bar)
    plt.plot(theta_deg_dist[:, 0], theta_deg_dist[:, 1], 'ro')
    plt.plot(r_bar, theta_deg, 'k')
    plt.grid(True)
    plt.xlabel('Theta (deg)')
    plt.ylabel('r bar')
    plt.tight_layout()  # Adjust subplots for better layout
    # Save plots
    plt.savefig('theta_plots.png') 
    plt.show()
    plt.close()

alphaCamber = -1.0 * alpha0_deg * np.pi / 180.0
theta = theta_deg * np.pi / 180 + alphaCamber  # Corrected for cambered airfoils
solidity = Nb * c / (np.pi * R)
Omega_rad = Omega_rpm * np.pi / 30  # in rad per sec
lam_climb = vel_climb / (R * Omega_rad)

# Inflow computation
if prandtlTipLoss_switch == 0:
    # Inflow ratio from BEMT
    const1 = solidity * dCldA / 16
    const_climb = lam_climb * 0.5
    lam = -(const1 - const_climb) + np.sqrt((const1 - const_climb) ** 2 + 2 * const1 * theta * r_bar)
    phi = lam / r_bar
    alf = theta - phi
    prandtl_F = np.ones_like(r_bar)

elif prandtlTipLoss_switch == 1:
    prandtl_F = np.ones_like(r_bar)  # Initial value
    prandtl_residual = 1.0
    const1 = solidity * dCldA / 16
    const_climb = lam_climb * 0.5
    counter = 1

    while prandtl_residual > 0.001 and counter < 21:
        lam = -(const1 / prandtl_F - const_climb) + np.sqrt(
            (const1 / prandtl_F - const_climb) ** 2 + 2 * const1 / prandtl_F * theta * r_bar
        )
        prandtl_Fsmall = 0.5 * Nb * (1.0 - r_bar) / lam

        prandtl_F_prev = prandtl_F
        prandtl_F = (2 / np.pi) * np.arccos(np.exp(-prandtl_Fsmall))
        prandtl_residual = np.abs(np.linalg.norm(prandtl_F - prandtl_F_prev))
        counter += 1

    if counter == 21:
        print('Warning: Prandtl tip-loss factor failed to converge')

    phi = lam / r_bar
    alf = theta - phi

# Using Momentum theory
ct_vec = prandtl_F * 4 * lam * (lam - lam_climb) * r_bar * dr_bar
lam_mean = 2.0 * np.trapz(r_bar, lam * r_bar) / (1 - root_cut * root_cut)
CT_MT = np.sum(ct_vec)

# Using BEMT
ct_vec = 0.5 * solidity * dCldA * dr_bar * (r_bar ** 2) * alf
CT_BEMT = np.sum(ct_vec)

Thrust = CT_MT * (rho * np.pi * R ** 2 * (R * Omega_rad) ** 2)

# Check between MT and BEMT
if CT_MT - CT_BEMT > np.finfo(float).eps:
    print('Warning: Discrepancy between CT calculated using BEMT and MT')

# Sectional lift distribution
cl_vec = 2.0 * ct_vec / (solidity * r_bar ** 2 * dr_bar)
cl_max, imx = np.max(cl_vec), np.argmax(cl_vec)
gamma_max = 0.5 * cl_max * c[imx] * (R * dr_bar[imx]) * np.linalg.norm([r_bar[imx], lam[imx]]) * R * Omega_rad

# Results
# print('\nColl. pitch (deg) =', theta_deg)
# print('solidityidity =', solidity)
print('CT           =', CT_MT)
print('CT           =', CT_BEMT)
print('inflow ratio =', lam_mean)
print('Thrust (N)   =', Thrust)
print('inflow (m/s) =', lam_mean * R * Omega_rad)
print('CL max       =', cl_max)
print('Gamma max    =', gamma_max)

# # Generate plots
# plt.subplot(2, 2, 1)
# plt.plot(r_bar, lam, 'k')
# plt.grid(True)
# plt.xlabel('r/R')
# plt.ylabel('Inflow Ratio')

# plt.subplot(2, 2, 2)
# plt.plot(r_bar, ct_vec / Nb, 'k')
# plt.grid(True)
# plt.xlabel('r/R')
# plt.ylabel('Sectional CT')

# plt.subplot(2, 2, 3)
# plt.plot(r_bar, (alf - alphaCamber) * 180 / np.pi, 'k')
# plt.grid(True)
# plt.xlabel('r/R')
# plt.ylabel('Alpha (deg)')

# plt.subplot(2, 2, 4)
# plt.plot(r_bar, cl_vec, 'k')
# plt.grid(True)
# plt.xlabel('r/R')
# plt.ylabel('Sectional CL')

# plt.show()


# Begin plots with a custom layout (rearranged subfigures)
plt.subplot(2, 2, 1)
plt.plot(r_bar, lam, 'k')
plt.grid(True)
plt.xlabel('r/R')
plt.ylabel('Inflow Ratio')

plt.subplot(2, 2, 2)
plt.plot(r_bar, (alf - alphaCamber) * 180 / np.pi, 'k')
plt.grid(True)
plt.xlabel('r/R')
plt.ylabel('Alpha (deg)')

plt.subplot(2, 2, 3)
plt.plot(r_bar, cl_vec, 'k')
plt.grid(True)
plt.xlabel('r/R')
plt.ylabel('Sectional CL')

plt.subplot(2, 2, 4)
plt.plot(r_bar, ct_vec / Nb, 'k')
plt.grid(True)
plt.xlabel('r/R')
plt.ylabel('Sectional CT')

plt.tight_layout()  # Adjust subplots for better layout

# Save or display the plots
plt.savefig('inflow_plots.png')  # Change the filename and format as needed
plt.show()
