"""

POSEIDON functions that have been modified to make TRIDENT compatible 
with GCM outputs

"""
# transmission.py imports 
import numpy as np

from numba.core.decorators import jit
from .utility import prior_index

# additional POSEIDON imports 
from .core import check_atmosphere_physical
from .utility import closest_index



# special imports 
import copy




def compute_spectrum_gcm(planet, star, model, atmosphere, opac, wl,
                         spectrum_type='transmission', save_spectrum=False,
                         disable_continuum=True, suppress_print=False,
                         Gauss_quad=2, use_photosphere_radius=True,
                         device='cpu', y_p=np.array([0.0])):
    '''
    Calculate extinction coefficients, then solve the radiative transfer
    equation to compute the spectrum of the model atmosphere.

    disable_continuum set to true so continuum opacities dont have to be calculated yet.

    My notes
    ---------
    planet:
    if not defined initially, planet impact parameter is set to 0

    atmosphere (object) properties:
    Need: P (1D, array); T(3D, array); r, r_up, r_low (3D, array (m))
    look at profiles function

    model (object):
    should be unnecessary as it is used to make the atmosphere object.





    Args:
        planet (dict):
            Collection of planetary properties used by POSEIDON.
        star (dict):
            Collection of stellar properties used by POSEIDON.
        model (dict):
            A specific description of a given POSEIDON model.
        atmosphere (dict):
            Collection of atmospheric properties.
        opac (dict):
            Collection of cross sections and other opacity sources.
        wl (np.array of float):
            Model wavelength grid (μm).
        spectrum_type (str):
            The type of spectrum for POSEIDON to compute
            (Options: transmission / emission / direct_emission /
                      transmission_time_average).
        save_spectrum (bool):
            If True, writes the spectrum to './POSEIDON_output/PLANET/spectra/'.
        disable_continuum (bool):
            If True, turns off CIA and Rayleigh scattering opacities.
        suppress_print (bool):
            if True, turn off opacity print statements (in line-by-line mode).
        Gauss_quad (int):
            Gaussian quadrature order for integration over emitting surface
            * Only for emission spectra *
            (Options: 2 / 3).
        use_photosphere_radius (bool):
            If True, use R_p at tau = 2/3 for emission spectra prefactor.
        device (str):
            Experimental: use CPU or GPU (only for emission spectra)
            (Options: cpu / gpu)
        y_p (np.array of float):
            Coordinate of planet centre along orbit at the time the spectrum
            is computed (y_p = 0, the default, corresponds to mid-transit).
            For non-grazing transits of uniform stellar disks, the spectrum
            is identical at all times due to translational symmetry, so y_p = 0
            is good for all times post second contact and pre third contact.
            Units are in m, not in stellar radii.

    Returns:
        spectrum (np.array of float):
            The spectrum of the atmosphere (transmission or emission).

    '''

    # Check if the atmosphere is unphysical (e.g. temperature out of bounds)
    if (check_atmosphere_physical(atmosphere, opac) == False):
        print('Spectrum is unphysical :(')
        spectrum = np.empty(len(wl))
        spectrum[:] = np.NaN
        return spectrum  # Unphysical => reject model

    # Check that the requested spectrum model is supported
    if (spectrum_type not in ['transmission', 'emission', 'direct_emission',
                              'dayside_emission', 'nightside_emission',
                              'transmission_time_average']):
        raise Exception("Only transmission spectra and emission " +
                        "spectra are currently supported.")

    # Unpack planet and star properties
    R_p = planet['planet_radius']
    b_p = planet['planet_impact_parameter']
    d = planet['system_distance']

    if (star is not None):
        R_s = star['R_s']

    # Check that a distance is provided if user wants a direct spectrum
    if (d is None) and ('direct' in spectrum_type):
        raise Exception("Must provide a system distance when computing a " +
                        "direct emission spectrum.")

    # Unpack atmospheric properties needed for radiative transfer
    r = atmosphere['r']
    r_low = atmosphere['r_low']
    r_up = atmosphere['r_up']
    dr = atmosphere['dr']
    n = atmosphere['n']
    T = atmosphere['T']
    P = atmosphere['P']
    P_surf = atmosphere['P_surf']
    X = atmosphere['X']
    X_active = atmosphere['X_active']
    X_CIA = atmosphere['X_CIA']
    X_ff = atmosphere['X_ff']
    X_bf = atmosphere['X_bf']
    N_sectors = atmosphere['N_sectors']
    N_zones = atmosphere['N_zones']
    phi_edge = atmosphere['phi_edge']
    theta_edge = atmosphere['theta_edge']
    H = atmosphere['H']

    # added params from atmosphere dictionary
    phi = atmosphere['phi']
    theta = atmosphere['theta']
    dphi = atmosphere['dphi']

    # Check if a surface is enabled
    if (P_surf != None):
        enable_surface = 1
    else:
        enable_surface = 0
        P_surf = 100.0  # Set surface pressure to 100 bar if not defined

    # ***** Calculate extinction coefficients *****#

    # Unpack lists of chemical species in this model
    chemical_species = model['chemical_species']
    active_species = model['active_species']
    CIA_pairs = model['CIA_pairs']
    ff_pairs = model['ff_pairs']
    bf_species = model['bf_species']

    # If computing line-by-line radiative transfer, use lbl optimised functions
    if (opac['opacity_treatment'] == 'line_by_line'):
        print('You have chosen... poorly')

    # If using opacity sampling, we can use pre-interpolated cross sections
    elif (opac['opacity_treatment'] == 'opacity_sampling'):

        # Unpack pre-interpolated cross sections
        sigma_stored = opac['sigma_stored']
        CIA_stored = opac['CIA_stored']
        Rayleigh_stored = opac['Rayleigh_stored']
        ff_stored = opac['ff_stored']
        bf_stored = opac['bf_stored']

        # Also unpack fine temperature and pressure grids from pre-interpolation
        T_fine = opac['T_fine']
        log_P_fine = opac['log_P_fine']

        # Running POSEIDON on the CPU
        if (device == 'cpu'):

            # Generate empty arrays so the dark god numba is satisfied
            n_aerosol = []
            sigma_ext_cloud = []

            n_aerosol.append(np.zeros_like(r))
            sigma_ext_cloud.append(np.zeros_like(wl))

            n_aerosol = np.array(n_aerosol)
            sigma_ext_cloud = np.array(sigma_ext_cloud)

            w_cloud = np.zeros_like(wl)
            g_cloud = np.zeros_like(wl)

            # Calculate extinction coefficients in standard mode

            kappa_gas, kappa_Ray, kappa_cloud = extinction_gcm(chemical_species, active_species,
                                                               CIA_pairs, ff_pairs, bf_species,
                                                               n, T, P, wl, X, X_active, X_CIA,
                                                               X_ff, X_bf,
                                                               sigma_stored,
                                                               CIA_stored, Rayleigh_stored,
                                                               ff_stored, bf_stored,
                                                               N_sectors, N_zones, T_fine,
                                                               log_P_fine, P_surf)
            # NOTE: kappa cloud is just an array of zeros
            # change this later to more motivated kappa arrays
            kappa_gas[np.isnan(kappa_gas)] = 0
            kappa_Ray[np.isnan(kappa_Ray)] = 0

    # Generate transmission spectrum
    if (spectrum_type == 'transmission'):

        if (device == 'gpu'):
            raise Exception("GPU transmission spectra not yet supported.")

        # Call the core TRIDENT routine to compute the transmission spectrum
        spectrum = TRIDENT_gcm_split(P, r, r_up, r_low, dr, wl, (kappa_gas + kappa_Ray), kappa_cloud,
                               phi, theta, dphi,
                               N_sectors, N_zones,
                               # kappa_cloud, enable_deck, enable_haze,
                               b_p, y_p[0],
                               R_s,
                               # f_cloud, phi_cloud_0, theta_cloud_0,
                               phi_edge, theta_edge)

    return spectrum



@jit(nopython=True)
def extinction_gcm(chemical_species, active_species, cia_pairs, ff_pairs, bf_species,
                   n, T, P, wl, X, X_active, X_cia, X_ff, X_bf,
                   # a, gamma, P_cloud, kappa_cloud_0,
                   sigma_stored, cia_stored, Rayleigh_stored, ff_stored,
                   bf_stored,
                   # enable_haze, enable_deck, enable_surface,
                   N_sectors, N_zones, T_fine, log_P_fine, P_surf,
                   # enable_Mie, n_aerosol_array, sigma_Mie_array,
                   P_deep=1000.0):
    '''
    Modified extinction function for GCM input.
    Stripped down and simplified so cloud opacities are removed


        Main function to evaluate extinction coefficients for molecules / atoms,
        Rayleigh scattering, hazes, and clouds for parameter combination
        chosen in retrieval step.

        Takes in cross sections pre-interpolated to 'fine' P and T grids
        before retrieval run (so no interpolation is required at each step).
        Instead, for each atmospheric layer the extinction coefficient
        is simply kappa = n * sigma[log_P_nearest, T_nearest, wl], where the
        'nearest' values are the closest P_fine, T_fine points to the
        actual P, T values in each layer. This results in a large speed gain.

        The output extinction coefficient arrays are given as a function
        of layer number (indexed from low to high altitude), terminator
        sector, and wavelength.

    '''

    # Store length variables for mixing ratio arrays
    N_species = len(chemical_species)  # Number of chemical species
    N_species_active = len(active_species)  # Number of spectrally active species
    N_cia_pairs = len(cia_pairs)  # Number of cia pairs included
    N_ff_pairs = len(ff_pairs)  # Number of free-free pairs included
    N_bf_species = len(bf_species)  # Number of bound-free species included

    N_wl = len(wl)  # Number of wavelengths on model grid
    N_layers = len(P)  # Number of layers

    # Define extinction coefficient arrays
    kappa_gas = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))
    kappa_Ray = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))
    kappa_cloud = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))

    # Fine temperature grid (for pre-interpolating opacities)
    N_T_fine = len(T_fine)
    N_P_fine = len(log_P_fine)

    # Find index of deep pressure below which atmosphere is opaque
    i_bot = np.argmin(np.abs(P - P_deep))

    # For each terminator sector (terminator plane)
    for j in range(N_sectors):

        # For each terminator zone (along day-night transition)
        for k in range(N_zones):

            # For each layer, find closest pre-computed cross section to P_fine, T_fine
            for i in range(i_bot, N_layers):

                n_level = n[i, j, k]

                # Find closest index in fine temperature array to given layer temperature
                idx_T_fine = closest_index(T[i, j, k], T_fine[0], T_fine[-1], N_T_fine)
                idx_P_fine = closest_index(np.log10(P[i]), log_P_fine[0], log_P_fine[-1], N_P_fine)

                # For each collisionally-induced absorption (CIA) pair
                for q in range(N_cia_pairs):

                    n_cia_1 = n_level * X_cia[0, q, i, j, k]  # Number density of first cia species in pair
                    n_cia_2 = n_level * X_cia[1, q, i, j, k]  # Number density of second cia species in pair
                    n_n_cia = n_cia_1 * n_cia_2  # Product of number densities of cia pair

                    # For each wavelength
                    for l in range(N_wl):
                        # Add CIA to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_gas[i, j, k, l] += n_n_cia * cia_stored[q, idx_T_fine, l]

                # For each free-free absorption pair
                for q in range(N_ff_pairs):

                    n_ff_1 = n_level * X_ff[0, q, i, j, k]  # Number density of first species in ff pair
                    n_ff_2 = n_level * X_ff[1, q, i, j, k]  # Number density of second species in ff pair
                    n_n_ff = n_ff_1 * n_ff_2  # Product of number densities of ff pair

                    # For each wavelength
                    for l in range(N_wl):
                        # Add free-free to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_gas[i, j, k, l] += n_n_ff * ff_stored[q, idx_T_fine, l]

                # For each source of bound-free absorption (photodissociation)
                for q in range(N_bf_species):

                    n_q = n_level * X_bf[q, i, j, k]  # Number density of dissociating species

                    # For each wavelength
                    for l in range(N_wl):
                        # Add bound-free to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_gas[i, j, k, l] += n_q * bf_stored[q, l]

                # For each molecular / atomic species with active absorption features
                for q in range(N_species_active):

                    n_q = n_level * X_active[q, i, j, k]  # Number density of this active species

                    # For each wavelength
                    for l in range(N_wl):
                        # Add chemical opacity to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_gas[i, j, k, l] += n_q * sigma_stored[q, idx_P_fine, idx_T_fine, l]

                # For each molecular / atomic species
                for q in range(N_species):

                    n_q = n_level * X[q, i, j, k]  # Number density of given species

                    # For each wavelength
                    for l in range(N_wl):
                        # Add Rayleigh scattering to total extinction in layer i, sector j, zone k, for each wavelength
                        kappa_Ray[i, j, k, l] += n_q * Rayleigh_stored[q, l]

    return kappa_gas, kappa_Ray, kappa_cloud




def TRIDENT_gcm_split(P, r, r_up, r_low, dr, wl, kappa_clear, kappa_cloud,
                phi_grid, theta_grid, dphi_grid,
                N_sectors, N_zones,
                b_p, y_p, R_s,
                phi_edge, theta_edge):
    '''
    @Char: attempt 1 at spatially resolved spectra
    Plan: Change phi to 2 sectors (1 sector = atmosphere, 2nd sector = no atmosphere)
    Hopefully instead of computing the radiative transfer through the 2nd atmosphere, I can arbitrarily set the
    transmittance and path tensors to reduce the computation. It would be unnecessary to compute over all theta for
    an empty atmosphere.)

    Main function used by the TRIDENT forward model to solve the equation
    of radiative transfer for exoplanet transmission spectra.

    This function implements the tensor dot product method derived in
    MacDonald & Lewis (2022).

    Note: This function is purely the atmospheric contribution to the spectrum.
          The 'contamination factor' contributions (e.g. stellar heterogeneity)
          are handled afterwards in 'core.py'.

    Args:
        P (np.array of float):
            Atmosphere pressure array (bar).
        r (3D np.array of float):
            Radial distance profile in each atmospheric column (m).
        r_up (3D np.array of float):
            Upper layer boundaries in each atmospheric column (m).
        r_low (3D np.array of float):
            Lower layer boundaries in each atmospheric column (m).
        dr (3D np.array of float):
            Layer thicknesses in each atmospheric column (m).
        wl (np.array of float):
            Model wavelength grid (μm).
        kappa_clear (4D np.array of float):
            Extinction coefficient from the clear atmosphere (combination of
            line absorption, CIA, bound-free and free-free absorption, and
            Rayleigh scattering) (m^-1).
        kappa_cloud (4D np.array of float):
            Extinction coefficient from the cloudy / haze contribution (m^-1).
        enable_deck (int):
            1 if the model contains a cloud deck, else 0.
        enable_haze (int):
            1 if the model contains a haze, else 0.
        b_p (float):
            Impact parameter of planetary orbit (m) -- NOT in stellar radii!
        y_p (float):
            Perpendicular distance from planet centre to the point where d = b_p
            (y coord. of planet centre as seen by observer in stellar z-y plane).
        R_s (float):
            Stellar radius (m).
        f_cloud (float):
            Terminator azimuthal cloud fraction for 2D/3D models.
        phi_0 (float):
            Azimuthal angle in terminator plane, measured clockwise from the
            North pole, where the patchy cloud begins for 2D/3D models (degrees).
        theta_0 (float):
            Zenith angle from the terminator plane, measured towards the
            nightside, where the patchy cloud begins for 2D/3D models (degrees).
        phi_edge (np.array of float):
            Boundary angles for each sector (radians).
        theta_edge (np.array of float):
            Boundary angles for each zone (radians).

    Returns:
        transit_depth (np.array of float):
            Atmospheric transit depth as a function of wavelength.

    '''
    # ***** Step 1: Initialise key quantities *****#

    # Compute projected distance from stellar centre to planet centre (z-y plane)
    d_sq = (b_p ** 2 + y_p ** 2)
    d = np.sqrt(d_sq)
    # print('d = ', d)

    # Load number of wavelengths where transit depth desired
    N_wl = len(wl)

    # Initialise transit depth array
    transit_depth = np.zeros(shape=(N_wl))

    # Compute squared stellar radius
    R_s_sq = R_s * R_s

    # Store number of layers
    N_layers = len(P)

    # Find index of deep pressure below which atmosphere is homogenous (usually 10 bar)
    i_bot = 0  # np.argmin(np.abs(P - P_deep))

    # @char: a quick aside: r[-1, :, 0] doesn't work for finding j_top because the highest pressure
    # level values is often nan in the gcm output. Therefore, need to find new usage for calculation
    # of b and db.
    # removed need for it to be dayside sector or top array as the max radial extent will be these things anyway
    # j top is now an array, not an index.
    j_top = np.unravel_index(np.nanargmax(r), r.shape)

    # Compute maximum radial extent of atmosphere
    # R_max = r_up[-1, j_top, 0]  # Maximal radial extent across all sectors
    # @char: Replaced R_max calculation
    R_max = r_up.flatten()[np.nanargmax(r_up)]
    R_max_sq = R_max * R_max  # Squared maximal radial extent

    # Initialise impact parameter array to radial array in sector with maximal extent
    b_init = r_up[:, j_top[1], j_top[2]]  # Impact parameters given by upper layer boundaries in dayside
    # remove nan values from b array
    b = b_init[~np.isnan(b_init)]
    db_init = dr[:, j_top[1], j_top[2]]  # Differential impact parameter array
    db = db_init[~np.isnan(db_init)]
    N_b = b.shape[0]  # Length of impact parameter array

    # integration will be at the resolution of the gcm grid

    # @char: point at which I am changing the code

    # to store number of sectors where RT needs to be calculated
    N_phi = N_sectors
    # Store number of distinct background sectors for azithumal integrals
    N_sectors_back = r.shape[1]

    # Store number of distinct background zones
    N_zones_back = r.shape[2]

    theta_edge_all = theta_edge_back = theta_edge
    # No difference between background atmosphere and clouds
    phi_edge_all = phi_edge_back = phi_edge

    # set cloudy_sectors and cloudy_zones array to zero (no clouds)

    # Define array specifying which zones contain clouds (0 -> clear, 1-> cloudy)
    cloudy_zones = np.zeros(N_zones).astype(np.int64)
    # Define array specifying which sectors contain clouds (0 -> clear, 1-> cloudy)
    cloudy_sectors = np.zeros(N_sectors).astype(np.int64)

    # @char: this is taken from extend_rad_transfer_grids
    # Initialise array containing zone indices each angular slice falls in
    k_zone_back = np.zeros(shape=(N_zones)).astype(np.int64)

    # For each zone angle
    for k in range(N_zones):
        # Find the zone in which this angle lies, store for later referencing in radiative transfer
        k_zone_back[k] = prior_index(theta_grid[k], theta_edge_back, 0)  # Background atmosphere only

    # Now create arrays storing which original sector and background sector
    # a given angle lies in (to avoid computing transmissivities multiple times)

    # Initialise array containing sector indices each angular slice falls in
    j_sector = np.zeros(shape=(N_phi)).astype(np.int64)
    j_sector_back = np.zeros(shape=(N_phi)).astype(np.int64)

    # For each polar angle
    for j in range(N_phi):

        # Find the terminator sector in which this angle lies
        j_sector_in = prior_index(phi_grid[j], phi_edge_all, 0)  # All sectors (including clouds)
        j_sector_back_in = prior_index(phi_grid[j], phi_edge_back, 0)  # Background atmosphere only

        # Find equivalent background sector in northern hemisphere
        if (j_sector_back_in >= N_sectors_back):
            j_sector_back_in = 2 * (N_sectors_back - 1) - j_sector_back_in

        # Store sector indices for later referencing in radiative transfer
        j_sector[j] = j_sector_in
        j_sector_back[j] = j_sector_back_in

    # ***** Step 2: Compute planetary area overlapping the star *****#)

    # If planet does not overlap star, do not need to do any computations
    if (d >= (R_s + R_max)):

        return np.zeros(shape=(N_wl))  # Transit depth zero at all wavelengths

    # If planet fully overlaps star
    elif (d <= (R_s - R_max)):

        # Area of overlap just pi*R_max^2 in this case
        # @char: changing this to the area of the sector of the circle.
        A_overlap = np.pi * R_max_sq /N_sectors

    # ***** Step 3: Calculate the delta_ray matrix *****#


    transmission_matrix = np.zeros([len(wl), N_phi])

    for i in range(N_sectors): #N_sectors 
        sector_idx = i
        #@char take these out of for loop up to path when fixed.
        delta_ray = delta_ray_geom_gcm_split(N_b, b, b_p, y_p, [phi_grid[i]], R_s_sq)

        # ***** Step 4: Calculate atmosphere area matrices *****#

        # Polar coordinate system area for multi-dimensional atmosphere
        dA_atm = np.outer((b * db), dphi_grid[i]).ravel()

        # Find overlapping area matrix of atmosphere (zero if rays don't intersect the star)
        dA_atm_overlap = delta_ray * dA_atm

        # ***** Step 5: Calculate path distribution tensor *****#

        Path = path_distribution_geometric_gcm_split(b, r_up[:,i,:], r_low[:,i,:], dr[:,i,:], i_bot, j_sector_back[i],
                                           N_layers, 1, N_zones, k_zone_back, theta_edge_all)

        # ***** Step 6: Calculate vertical optical depth tensor *****#

        dr_no_nan = copy.deepcopy(dr)
        dr_no_nan[np.isnan(dr_no_nan)] = 0.0

        tau_vert = compute_tau_vert_gcm_split(1, N_layers, N_zones, N_wl, [j_sector[i]],
                                    [j_sector_back[i]], k_zone_back, cloudy_zones,
                                    [cloudy_sectors[i]], kappa_clear, kappa_cloud, dr_no_nan)

        # ***** Step 7: Calculate transmittance tensor *****#

        # Trans = np.zeros(shape=(N_b, 1, N_wl))
        Trans = np.zeros(shape=(N_b, N_wl))

        # @char: this has been changed at my peril
        Trans[:, :] = np.exp(-1.0 * np.tensordot(Path[:, :, :], tau_vert[:, :, :], axes=([2, 1], [0, 1])))

        # Delete vertical optical depth and path distribution tensors to free memory
        del tau_vert, Path

        # ***** Step 8: Finally, compute the transmission spectrum *****#

        # Calculate effective overlapping area of atmosphere at each wavelength
        # A_atm_overlap_eff = np.tensordot(Trans, dA_atm_overlap, axes=([0, 1], [0, 1]))
        # @char: reduced to a standard dot product of A^T.B
        A_atm_overlap_eff = np.tensordot(Trans, dA_atm_overlap, axes=([0],[0]))

        # Compute the transmission spectrum
        transit_depth = (A_overlap - A_atm_overlap_eff) / (np.pi * R_s_sq/N_phi)

        transmission_matrix[:,i] = transit_depth


    return transmission_matrix




@jit(nopython=True)
def delta_ray_geom_gcm_split(N_b, b, b_p, y_p, phi_grid, R_s_sq):
    '''
    Only changed delta_ray array (collapsed the N_phi)

    Compute the ray tracing Kronecker delta factor in the geometric limit.

    Args:
        N_phi (int):
            Number of azimuthal integration elements (not generally the same as
            N_sectors, especially when the planet partially overlaps the star).
        N_b (int):
            Number of impact parameters (length of b).
        b (np.array of float):
            Stellar ray impact parameters.
        b_p (float):
            Impact parameter of planetary orbit (m) -- NOT in stellar radii!
        y_p (float):
            Perpendicular distance from planet centre to the point where d = b_p
            (y coord. of planet centre as seen by observer in stellar z-y plane).
        phi_grid (np.array of float):
            Angles in the centre of each azimuthal integration element (radians).
        R_s_sq (float):
            Square of the stellar radius (m^2).

    Returns:
        delta_ray (2D np.array of float):
            1 if a given ray traces back to the star, 0 otherwise.

    delta_ray = delta_ray_geom_gcm_split(1, N_b, b, b_p, y_p, phi_grid[i], R_s_sq)
    '''

    delta_ray = np.zeros(shape=(N_b))

    # For each polar angle
    for j in range(1):

        # For each atmospheric layer
        for i in range(N_b):

            # Compute distance from stellar centre to centre of area element
            d_ij_sq = (b[i] ** 2 + b_p ** 2 + y_p ** 2 +
                       2.0 * b[i] * (b_p * np.cos(phi_grid[j] - np.pi / 2.0) +
                                     y_p * np.sin(phi_grid[j] - np.pi / 2.0)))

            # If planet area element has star in the background
            if (d_ij_sq <= R_s_sq):

                # Ray traces back to stellar surface => 1
                delta_ray[i] = 1.0

            # If area element falls off stellar surface
            else:

                # No illumination => 0
                delta_ray[i] = 0.0

    return delta_ray





@jit(nopython=True)
def path_distribution_geometric_gcm_split(b, r_up, r_low, dr, i_bot, j_sector_back,
                                N_layers, N_sectors_back, N_zones, k_zone_back, theta_edge_all):
    '''
    @char: I have modified this function so if s1,2,3,4 are nan values (due to out of bounds pressures
    from a GCM input), the s value is changed to zero so the path element doesn't return nans

    @char: r_up, low and dr already have phi dependence specified when called so
    dont need sector idx to be specified.

    Compute the path distribution tensor, in the geometric limit where rays
    travel in straight lines, using the equations in MacDonald & Lewis (2022).

    Args:
        b (np.array of float):
            Stellar ray impact parameters.
        r_up (3D np.array of float):
            Upper layer boundaries in each atmospheric column (m).
        r_low (3D np.array of float):
            Lower layer boundaries in each atmospheric column (m).
        dr (3D np.array of float):
            Layer thicknesses in each atmospheric column (m).
        i_bot (int):
            Layer index of bottom of atmosphere (rays with b[i < i_bot] are
            ignored). By default, i_bot = 0 so all rays are included in the
            radiative transfer calculation.
        j_sector_back (np.array of int):
            Indices encoding which background atmosphere sector each azimuthal
            integration element falls in (accounts for north-south symmetry of
            background atmosphere, since the path distribution need only be
            calculated once for the northern hemisphere).
        N_layers (int):
            Number of layers.
        N_sectors_back (int):
            Number of azimuthal sectors in the background atmosphere.
        N_zones_back (int):
            Number of zenith zones in the background atmosphere.
        N_phi (int):
            Number of azimuthal integration elements (not generally the same as
            N_sectors, especially when the planet partially overlaps the star).
        k_zone_back (np.array of int):
            Indices encoding which background atmosphere zone each zenith
            integration element corresponds to.
        theta_edge_all (np.array of float):
            Zenith angles at the edge of each zone (radians).

    Returns:
        Path (4D np.array of float):
            Path distribution tensor for 3D atmospheres.


    '''
    # Store length of impact parameter vector
    N_b = b.shape[0]

    # Initialise path distribution tensor
    Path = np.zeros(shape=(N_b, N_zones, N_layers))

    # Compute squared radial layer boundary vectors
    r_up_sq = r_up * r_up
    r_low_sq = r_low * r_low

    # Initialise squared impact parameter array
    b_sq = b * b

    # If the rays traverse only a single zone, symmetry gives a factor of 2 in path length
    if (N_zones == 1):
        symmetry_factor = 2.0  # Factor of 2 for ray paths into and out of atmosphere
    elif (N_zones >= 2):
        symmetry_factor = 1.0  # No factor of 2 when inwards and outwards directions separately treated

    # ***** Define minimum and maximum angles of zone boundaries *****#
    # Max angles given by removing the terminator plane from array
    theta_edge_max = np.delete(theta_edge_all, np.where(theta_edge_all == 0.0)[0])
    # print('theta_edge_max = ', theta_edge_max)

    # Min angles given by clipping equators and adding an extra 0.0 (two zones adjacent to the terminator plane share theta = 0)
    theta_edge_min = np.sort(np.append(theta_edge_all[1:-1], 0.0))
    # print('theta_edge_min =', theta_edge_min)

    # Compute maximum and minimum radial extent each ray can possesses in each zone
    r_min, r_max = zone_boundaries_gcm_split(N_b, N_zones, b, r_up,
                                   k_zone_back, theta_edge_min, theta_edge_max)

    # Compute squared radial zone boundary vectors
    r_min_sq = r_min * r_min
    r_max_sq = r_max * r_max

    # Refresh sector count
    j_sector_last = -1  # This counts the angular index where the transmissivity was last computed

    # Find which asymmetric terminator sector this angle lies in
    j_sector_back_in = j_sector_back
    # print('j_sector_back_in = ', j_sector_back_in)

    # For each zone along tangent ray
    for k in range(N_zones):

        # Extract index of background atmosphere arrays this sub-zone is in (e.g. radial arrays)
        k_in = k_zone_back[k]  # Only differs from k when a cloud splits a zone
        # print('k_in = ', k_in)

        # For each ray impact parameter
        for i in range(N_b):

            # For each atmosphere layer
            for l in range(i_bot, N_layers):

                # Check for layers falling outside of region sampled by ray
                if ((r_low[l, k_in] >= r_max[i, k]) or
                        (r_up[l, k_in] <= r_min[i, k]) or
                        (b[i] >= r_max[i, k])):
                    # print('b')

                    Path[i, k, l] = 0.0  # No path if layer outside region


                # For other cases, we always subtract two terms to compute traversed distance
                else:
                    # print('c')

                    if (r_up[l, k_in] >= r_max[i, k]):
                        # print('d')

                        s1 = np.sqrt(r_max_sq[i, k] - b_sq[i])
                        if np.isnan(s1):
                            s1 = 0.0
                        s2 = 0.0

                    else:  # elif (r_up[l,j,k_in] < r_max[i,j,k]):

                        s2 = np.sqrt(r_up_sq[l, k_in] - b_sq[i])
                        if np.isnan(s2):
                            s2 = 0.0
                        s1 = 0.0

                    if (r_low[l, k_in] > r_min[i, k]):
                        # print('f')

                        s3 = np.sqrt(r_low_sq[l,k_in] - b_sq[i])
                        if np.isnan(s3):
                            s3 = 0.0
                        s4 = 0.0

                    else: 

                        s4 = np.sqrt(r_min_sq[i, k] - b_sq[i])
                        if np.isnan(s4):
                            s4 = 0.0
                        s3 = 0.0

                    # Conditions have been placed so nan values of s1,2,3,4 evaluate to 0
                    # Final condition, if dr = nan, path evaluates to zero
                    if np.isnan(dr[l, k_in]):
                        Path[i, k, l] = 0.0
                    else:
                        # print('s1,s2,s3,s4=', s1,s2,s3,s4)
                        Path[i, k, l] = symmetry_factor * (s1 + s2 - s3 - s4) / dr[
                            l, k_in]

    return Path




@jit(nopython=True)
def zone_boundaries_gcm_split(N_b, N_zones, b, r_up, k_zone_back,
                    theta_edge_min, theta_edge_max):
    '''
    Compute the maximum and minimal radial distance from the centre of the
    planet that a ray at impact parameter b experiences in each azimuthal
    sector and zenith zone.

    @char sector idx currently isn't used because the r arrays are called in
    already specifying the phi dependence. May change this later.

    These quantities are 'r_min' and 'r_max' in MacDonald & Lewis (2022).

    Args:
        N_b (int):
            Number of impact parameters (length of b).
        N_sectors (int):
            Number of azimuthal sectors.
        N_zones (int):
            Number of zenith zones.
        b (np.array of float):
            Stellar ray impact parameters.
        r_up (3D np.array of float):
            Upper layer boundaries in each atmospheric column (m).
        k_zone_back (np.array of int):
            Indices encoding which background atmosphere zone each zenith
            integration element corresponds to.
        theta_edge_min (np.array of float):
            Minimum zenith angle of each zone (radians).
        theta_edge_max (np.array of float):
            Maximum zenith angle of each zone (radians).

    Returns:
        r_min (3D np.array of float):
            Minimum radial extent encountered by a ray when traversing a given
            layer in a column defined by its sector and zone (m).
        r_max (3D np.array of float):
            Maximum radial extent encountered by a ray when traversing a given
            layer in a column defined by its sector and zone (m).

    '''

    r_min = np.zeros(shape=(N_b, N_zones))
    r_max = np.zeros(shape=(N_b, N_zones))

    for k in range(N_zones):

        # Trigonometry to compute maximum r, given b and angle to terminator
        denom_min = np.cos(theta_edge_min[k])
        denom_max = np.cos(theta_edge_max[k])

        # Extract index of background atmosphere arrays this sub-zone is in (e.g. radial arrays)
        k_in = k_zone_back[k]  # Only differs from k when a cloud splits a zone

        for i in range(N_b):

            # Trigonometry to compute maximum r, given b and angle to terminator
            r_min_geom = b[i] / (denom_min + 1.0e-250)  # Denominator one loop up for efficiency
            r_max_geom = b[i] / (
                    denom_max + 1.0e-250)  # Additive factor prevents division by zeros for dayside and nightside


            # If geometric expressions go above the maximum altitude, set to top of atmosphere
            # r_min[i, j, k] = np.minimum(r_up[-1, j, k_in], r_min_geom)
            r_min[i, k] = np.nanmin([r_up[-1, k_in], r_min_geom])

            # If in the dayside or nightside, max radial extent given by top of atmosphere
            if ((k == 0) or (k == N_zones - 1)):
                if np.isnan(r_up[-1, k_in]):
                    #@char: this may need changing
                    r_up_max = np.nanmax(r_up)
                    r_max[i, k] = np.nanmin([r_up_max, r_max_geom])
                else:
                    r_max[i, k] = r_up[-1, k_in]  # Top of atmosphere in sector j, zone k

            else:  # For all other zones

                # If geometric expressions go above the maximum altitude, set to top of atmosphere
                r_max[i, k] = np.nanmin([r_up[-1, k_in], r_max_geom])

    return r_min, r_max




@jit(nopython=True)
def compute_tau_vert_gcm_split(N_phi, N_layers, N_zones, N_wl, j_sector, j_sector_back,
                     k_zone_back, cloudy_zones, cloudy_sectors, kappa_clear,
                     kappa_cloud, dr):
    """
    Computes the vertical optical depth tensor across each layer within
    each column as a function of wavelength.

    Args:
        N_phi (int):
            Number of azimuthal integration elements.
        N_layers (int):
            Number of atmospheric layers.
        N_zones (int):
            Number of zenith zones.
        N_wl (int):
            Number of wavelengths.
        j_sector (np.array of int):
            Indices specifying which sector each azimuthal integration element
            falls in (for the full list of sectors, including cloud sectors).
        j_sector_back (np.array of int):
            Indices specifying which sector of the background atmosphere each
            azimuthal integration element falls in (clear atmosphere only).
        k_zone_back (np.array of int):
            Indices encoding which background atmosphere zone each zenith
            integration element corresponds to.
        cloudy_zones (np.array of int):
            0 if a given zone is clear, 1 if it contains a cloud.
        cloudy_sectors (np.array of int):
            0 if a given sector is clear, 1 if it contains a cloud.
        kappa_clear (4D np.array of float):
            Extinction coefficient from the clear atmosphere (combination of
            line absorption, CIA, bound-free and free-free absorption, and
            Rayleigh scattering) (m^-1).
        kappa_cloud (4D np.array of float):
            Extinction coefficient from the cloudy / haze contribution (m^-1).
        dr (3D np.array of float):
            Layer thicknesses in each atmospheric column (m).

    Returns:
        tau_vert (4D np.array of float):
            Vertical optical depth tensor.

    """

    tau_vert = np.zeros(shape=(N_layers, N_zones, N_wl))

    # For each sector around terminator
    for j in range(N_phi):

        # Refresh sector count
        j_sector_last = -1  # This counts the angular index where the transmissivity was last computed

        # Find which asymmetric terminator sector this angle lies in
        j_sector_in = j_sector[j]
        j_sector_back_in = j_sector_back[j]

        # For each zone along line of sight
        for k in range(N_zones):

            # Extract index of background atmosphere this sub-zone is in
            k_zone_back_in = k_zone_back[k]  # Only differs from k when a cloud splits a zone

            # If zone and sector angles lie within cloudy region
            if ((cloudy_zones[k] == 1) and  # If zone is cloudy and
                    (cloudy_sectors[j_sector_in] == 1)):  # If sector is cloudy

                # For each wavelength
                for q in range(N_wl):
                    # Populate vertical optical depth tensor
                    tau_vert[:, k, q] = ((kappa_clear[:, j_sector_back_in, k_zone_back_in, q] +
                                             kappa_cloud[:, j_sector_back_in, k_zone_back_in, q]) *
                                            dr[:, j_sector_back_in, k_zone_back_in])

            # For clear regions, do not need to add cloud opacity
            elif ((cloudy_zones[k] == 0) or  # If zone is clear or
                  (cloudy_sectors[j_sector_in] == 0)):  # If sector is clear

                # For each wavelength
                for q in range(N_wl):
                    # Populate vertical optical depth tensor
                    tau_vert[:, k, q] = (kappa_clear[:, j_sector_back_in, k_zone_back_in, q] *
                                            dr[:, j_sector_back_in, k_zone_back_in])

    return tau_vert
