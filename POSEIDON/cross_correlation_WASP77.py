from __future__ import absolute_import, unicode_literals, print_function
from POSEIDON.high_res import cross_correlate_sysrem, fast_filter
import math, os
import numpy as np
import pickle
import pickle
from scipy import constants
from numba import jit
from astropy.io import fits
from scipy import interpolate
from POSEIDON.core import create_star, create_planet, define_model, make_atmosphere, read_opacities, wl_grid_constant_R, wl_grid_line_by_line, compute_spectrum
from POSEIDON.constants import R_Sun
from POSEIDON.visuals import plot_stellar_flux
from POSEIDON.constants import R_J, M_J
import numpy as np
from spectres import spectres
from tqdm import tqdm
from multiprocessing import Pool

K_p = 192.06
N_K_p = 200
d_K_p = 1
K_p_arr = (np.arange(N_K_p) - (N_K_p-1)//2) * d_K_p + K_p # making K_p_arr (centered on published or predicted K_p)
# K_p_arr = [92.06 , ..., 191.06, 192.06, 193.06, ..., 291.06]

V_sys = 0
N_V_sys = 200
d_V_sys = 1
V_sys_arr = (np.arange(N_V_sys) - (N_V_sys-1)//2) * d_V_sys + V_sys # making V_sys_arr (centered on published or predicted V_sys (here 0 because we already added V_sys in V_bary))

N_cores = 1 # Change how many cores to use here.
core_indices = np.arange(N_cores)
V_sys_arr_split = np.array_split(V_sys_arr, N_cores)

def batch_cross_correlate(core_index, args):
    F_s_obs, F_p_obs, wl = args
    for i in range(core_index, len(V_sys_arr_split), N_cores):
        print("Core "+str(i)+" is running.")
        output_path = './CC_output/WASP-77Ab' # Could modify output path here.
        data_path = './reference_data/observations/WASP-77Ab'
        os.makedirs(output_path, exist_ok=True)

        wl_grid, data_raw = pickle.load(open(data_path+'/data_RAW.pic', 'rb'))
        Phi = pickle.load(open(data_path+'/ph.pic','rb'))                    # Time-resolved phases
        V_bary = pickle.load(open(data_path+'/rvel.pic','rb'))               # Time-resolved Earth-star velocity (V_bary+V_sys) constructed in make_data_cube.py; then V_sys = V_sys_literature + d_V_sys
        residuals, Us, Ws = fast_filter(data_raw, iter=15)
        # We could also just use V_bary instead, then V_sys is just V_sys (not around zero anymore)
        log_L_arr, CCF_arr = cross_correlate_sysrem(F_s_obs, F_p_obs, wl, K_p_arr, V_sys_arr_split[i], wl_grid, residuals, Us, V_bary, Phi)
        pickle.dump([V_sys_arr_split[i], K_p_arr, CCF_arr, log_L_arr], open(output_path+'/test_sysrem_'+str(i)+'.pic','wb')) # N_cores of these produced. Will be read in by plot_CCF.py

# The code below will only be run on one core to get the model spectrum.
if __name__ == '__main__':
    R_s = 1.21*R_Sun      # Stellar radius (m)
    T_s = 5605.0          # Stellar effective temperature (K)
    Met_s = -0.04         # Stellar metallicity [log10(Fe/H_star / Fe/H_solar)]
    log_g_s = 4.56        # Stellar log surface gravity (log10(cm/s^2) by convention)

    # Create the stellar object
    star = create_star(R_s, T_s, log_g_s, Met_s, stellar_spectrum = True, stellar_grid = 'phoenix')

    F_s = star['F_star']
    wl_s = star['wl_star']
    R_s = star['stellar_radius']


    #***** Define planet properties *****#

    planet_name = 'WASP-77Ab'  # Planet name used for plots, output files etc.

    R_p = 1.21*R_J      # Planetary radius (m)
    M_p = 0.07*M_J      # Mass of planet (kg)
    g_p = 4.3712        # Gravitational field of planet (m/s^2)
    T_eq = 1043.8       # Equilibrium temperature (K)

    # Create the planet object
    planet = create_planet(planet_name, R_p, mass = M_p, gravity = g_p, T_eq = T_eq)

    # If distance not specified, use fiducial value
    if (planet['system_distance'] is None):
        planet['system_distance'] = 1    # This value only used for flux ratios, so it cancels
    d = planet['system_distance']
    
    #***** Define model *****#

    model_name = 'Simple_dayside'  # Model name used for plots, output files etc.

    bulk_species = ['H2', 'He']    # H2 + He comprises the bulk atmosphere
    # param_species = ['H2O', 'CO', 'CH4', 'H2S', 'NH3', 'HCN']
    param_species = ['H2O', 'CO']  # H2O, CO as in Brogi & Line

    # Create the model object
    model = define_model(model_name, bulk_species, param_species,
                        PT_profile = 'Madhu')

    # Check the free parameters defining this model
    print("Free parameters: " + str(model['param_names']))
                                

    #***** Wavelength grid *****#

    wl_min = 1.3      # Minimum wavelength (um)
    wl_max = 2.6      # Maximum wavelength (um)
    R = 250000          # Spectral resolution of grid

    # wl = wl_grid_line_by_line(wl_min, wl_max)
    wl = wl_grid_constant_R(wl_min, wl_max, R)

    #***** Read opacity data *****#

    opacity_treatment = 'opacity_sampling'

    # Define fine temperature grid (K)
    T_fine_min = 500     # 400 K lower limit suffices for a typical hot Jupiter
    T_fine_max = 3000    # 2000 K upper limit suffices for a typical hot Jupiter
    T_fine_step = 20     # 20 K steps are a good tradeoff between accuracy and RAM

    T_fine = np.arange(T_fine_min, (T_fine_max + T_fine_step), T_fine_step)

    # Define fine pressure grid (log10(P/bar))
    log_P_fine_min = -6.0   # 1 ubar is the lowest pressure in the opacity database
    log_P_fine_max = 2.5    # 100 bar is the highest pressure in the opacity database
    log_P_fine_step = 0.2   # 0.2 dex steps are a good tradeoff between accuracy and RAM

    log_P_fine = np.arange(log_P_fine_min, (log_P_fine_max + log_P_fine_step),
                        log_P_fine_step)

    # Now we can pre-interpolate the sampled opacities (may take up to a minute)
    opac = read_opacities(model, wl, opacity_treatment, T_fine, log_P_fine)

    # Specify the pressure grid of the atmosphere
    P_min = 1.0e-7    # 0.1 ubar
    P_max = 100       # 100 bar
    N_layers = 100    # 100 layers

    # We'll space the layers uniformly in log-pressure
    P = np.logspace(np.log10(P_max), np.log10(P_min), N_layers)

    # Specify the reference pressure and radius
    P_ref = 10.0   # Reference pressure (bar)
    R_p_ref = R_p  # Radius at reference pressure

    
    # log_H2O, log_CO, log_CH4, log_H2S, log_NH3, log_HCN, a1, a2, log_P1, log_P2, log_P3, T_deep = params
    # log_H2O, log_CO, a1, a2, log_P1, log_P2, log_P3, T_deep
    params = (-3.93, -3.77, 0.38, 0.56, 0.17, -1.39, 0.36, 931) # Using maxmimum likelihood values from Brogi & Line
    log_H2O, log_CO, a1, a2, log_P1, log_P2, log_P3, T_deep = params

    # Provide a specific set of model parameters for the atmosphere
    PT_params = np.array([a1, a2, log_P1, log_P2, log_P3, T_deep])     # a1, a2, log_P1, log_P2, log_P3, T_deep
    # log_X_params = np.array([[log_H2O, log_CO, log_CH4, log_H2S, log_NH3, log_HCN]])
    log_X_params = np.array([[log_H2O, log_CO]])
    
    atmosphere = make_atmosphere(planet, model, P, P_ref, R_p_ref, PT_params, log_X_params)

    # Generate planet surface flux
    F_p_obs = compute_spectrum(planet, star, model, atmosphere, opac, wl, spectrum_type='direct_emission')
    
    F_s_interp = spectres(wl, wl_s, F_s)
    F_s_obs = (R_s / d)**2 * F_s_interp # observed flux of star on earth

    # Passing stellar spectrum, planet spectrum, wavelenght grid to each core, thus saving time for reading the opacity again
    from itertools import repeat
    pool = Pool(processes=N_cores)
    pool.starmap(batch_cross_correlate, zip(core_indices, repeat((F_s_obs, F_p_obs, wl))))