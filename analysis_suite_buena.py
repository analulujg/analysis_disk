import numpy as np
import sarracen as sar
import matplotlib.pyplot as plt
import cmasher as cmr
import os
from scipy import stats
from scipy.stats import t

#setup values

hacc = 1   #accretion radius of the accretor in rsun
x_acc = 221.526229 #x position of the accretor in rsun
x_don =  -44.8137709 #x position of the donor in rsun
mass_proton_cgs = 1.67262158e-24 
mass_electron_cgs = 9.10938291e-28
kboltz = 1.38066e-16  #erg/K
gamma = 1.1
mu = 2.381
nrad = 100 # Number of radius I want to divided the simulation
yr= 3.1558149984e7
pi = 3.1415926535897932385
sun_mass = 1.989e33 
sun_rad = 6.96e10
a = 266.34
m_d = 6.9733669*sun_mass        
m_a = 1.4133930*sun_mass 
vel_scaling = 4.3669e7 #in cm/s
u_scaling =  1.907e15 # in erg/g
gg  = 6.6742867e-8
omega = np.sqrt(gg*(m_d+m_a)/(a*sun_rad)**3)
r_min = 0.0 
r_max = 90
n_r = 100         
theta_min = 0.0
theta_max =  2* np.pi
n_theta = 20  

#FUNCTIONS!!

#---------------Calculte scale height----------------------------------
def compute_scale_height(snapshot, x_acc):
    df= sar.read_phantom(snapshot)
    df['xnew'] = df['x'] - x_acc
    df['rmag'] = np.sqrt(df['xnew']**2 + df['y']**2)
    df['theta'] = np.mod(np.arctan2(df['y'], df['xnew']), 2 * np.pi)
    df.calc_density()
    df['rho'] = df['rho'] * 5.901 # Convert to g/cm^3
   
    r_values = np.linspace(r_min, r_max, n_r)
    theta_values = np.linspace(theta_min, theta_max, n_theta)

    dr = (r_max- r_min)/(2*n_r)
    dtheta = (theta_max - theta_min)/(2*n_theta) 

    scale_heights_per_r_theta = []
    max_rho_per_r_theta = []
    sigma_per_r_theta =[]

    for k in r_values:
        for j in theta_values:
            #finding particles in dr and dtheta
            mask = ((df['rmag'] >= k - dr) & (df['rmag'] <= k + dr) & (df['theta'] >= j - dtheta) & (df['theta'] <= j + dtheta))
            df_filtered = df[mask]

            if len(df_filtered) == 0:
                scale_heights_per_r_theta.append(0.0)
                max_rho_per_r_theta.append(0.0)
                sigma_per_r_theta.append(0.0)
                continue

            max_rho = df_filtered['rho'].max()
            #total_rho= df_filtered['rho'].sum()
            sigma_per_theta = np.sum(df_filtered['rho'] * 2 * (df_filtered['h']*sun_rad))
            h_rho = max_rho/np.sqrt(np.e)

            # Find particle with rho closest to h_rho
            mask_pos = (df_filtered['z'] >= 0.0) 
            mask_neg = (df_filtered['z'] < 0.0)
            df_pos = df_filtered[mask_pos]
            df_neg = df_filtered[mask_neg]

            if len(df_pos) == 0 or len(df_neg) == 0:
                scale_heights_per_r_theta.append(0.0)
                max_rho_per_r_theta.append(0.0)
                sigma_per_r_theta.append(0.0)
                continue

            H_pindex = (df_pos['rho'] - h_rho).abs().idxmin()
            H_nindex = (df_neg['rho'] - h_rho).abs().idxmin()
            H_p = df_pos.loc[H_pindex]
            H_n = df_neg.loc[H_nindex] 
            Hmean  = (np.abs(H_p['z'])+np.abs(H_n['z']))/2
            #dz = (df_filtered['z'].max() - df_filtered['z'].min())*sun_rad
            sigma = sigma_per_theta
            scale_heights_per_r_theta.append(Hmean)
            max_rho_per_r_theta.append(max_rho)
            sigma_per_r_theta.append(sigma)
    
    max_rho_array = np.array(max_rho_per_r_theta).reshape(len(r_values), len(theta_values))
    scale_heights_array = np.array(scale_heights_per_r_theta).reshape(len(r_values), len(theta_values))
    sigma_array = np.array(sigma_per_r_theta).reshape(len(r_values), len(theta_values))
    H_r = np.median(scale_heights_array, axis=1)
    max_rho_r = np.median(max_rho_array, axis=1)
    sigma_r = np.median(sigma_array, axis=1)

    return r_values, H_r, max_rho_r, sigma_r


#------------------Calculate cumulative mass per radii-----------------
def compute_mass(snapshot,radius, H):
    df = sar.read_phantom(snapshot)
    df['xnew'] = df['x'] - x_acc
    df['rmag'] = np.sqrt(df['xnew']**2 + df['y']**2)
    df['theta'] = np.mod(np.arctan2(df['y'], df['xnew']), 2 * np.pi)
    df.create_mass_column()
    
    mass_per_radius = []
    mini_mask = ((df['rmag'] > 0.0) & (df['rmag'] <= radius[0]) & (df['z'] >= -H[0]) & (df['z'] <= H[0]))
    df_mini = df[mini_mask]
    mass_per_radius.append(df_mini['m'].sum())
    
    for k in range(1,len(radius)):
        mask = ((df['rmag'] >radius[k-1]) & (df['rmag'] <= radius[k]) & (df['z'] >= -H[k]) & (df['z'] <= H[k]))
        df_filtered = df[mask]
        count = len(df_filtered)
        if count == 0:
            mass_per_radius.append(0.0)
        else: 
            mass_now = df_filtered['m'].sum()
            mass_per_radius.append(mass_now+ mass_per_radius[k-1])
 
    mass_r = np.array(mass_per_radius)
 
   # m1 = (mass_per_radius[-1]-mass_per_radius[0])/(radius[-1]-radius[0])
   # grad1 = np.gradient(mass_per_radius)/np.gradient(radius)
   # disk_idx=idx = (np.abs(grad1[:93] - m1)).argmin() 
    for i in range(20,len(mass_r)):
        grad = (mass_r[i]-mass_r[i-1])/mass_r[i]
        if grad>0.01:
            disk_idx = int(len(mass_r)/2)
        else:
            disk_idx = i
            break

    return mass_r, disk_idx

#----------Calculate inflection point for the cumulative mass plot-------------
def find_nearest(array, value): 
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_inflection_point(mass, rad):
    m1 = (mass[-1]-mass[0])/(rad[-1]-rad[0])
    grad = np.gradient(mass)/np.gradient(rad)
    pt = np.where(grad == find_nearest(grad[20:90],m1))[0][0]
    return rad[pt], mass[pt]


#-----------Calculate radial and tangential velocity per radii--------------------------
def coordTransf(x,y,alpha):
    x_transf = x*np.cos(alpha)+y*np.sin(alpha)
    y_transf = -x*np.sin(alpha) +y*np.cos(alpha)

    return x_transf, y_transf

def getradtanvelocity(x,y,v_x,v_y):
    phi = np.arctan2(y,x)
    v_rad = (coordTransf(v_x,v_y,phi)[0])
    v_phi = (coordTransf(v_x,v_y,phi)[1])

    return v_rad,v_phi

def kepler_vt(r):
    kep_vt = np.sqrt(gg*m_a/(r))/1e5  # Convert to km/s
    return kep_vt

def compute_vrad_vphi(snapshot):
    sdf= sar.read_phantom(snapshot)
    sdf['rmag']= np.sqrt((sdf['x']-x_acc)**2 + sdf['y']**2 + sdf['z']**2)*sun_rad  # in cm
    
    rx = (sdf['x'] - x_acc)*sun_rad # in cm
    ry = sdf['y'] * sun_rad # in cm
    rz = sdf['z']* sun_rad # in cm

    vx = sdf['vx']*4.367e7 # in cm/s
    vy = sdf['vy']*4.367e7 # in cm/s
    vz = sdf['vz']*4.367e7 # in cm/s
    vmag = np.sqrt(vx**2 + vy**2 + vz**2)

    # Get radial and tangential velocities
    v_rad, v_phi = getradtanvelocity(rx, ry, vx, vy)
    sdf['v_rad'] = v_rad/1e5 # Convert to km/s
    sdf['v_phi'] = v_phi/1e5 # Convert to km/s
    sdf['f_vphi'] = v_phi/vmag
    sdf['f_vrad'] = v_rad/vmag

    rmax = rmax * sun_rad  # in cm
    rstep = (rmax/n_r) * sun_rad  # in cm
    rmin = rmin * sun_rad # in cm
    dr = (r_max- r_min)/(2*n_r)

    rr = np.linspace(rmin,rmax,n_r)
    r_vp = np.array([])
    r_vr = np.array([])
    r_fvp = np.array([])
    r_fvr = np.array([])
   
    for r in rr:
        # Calculate the mean tangential velocity of all data in r-step
        filter = (sdf['rmag']<(r+dr)) & (sdf['rmag']>(r-dr))
        r_vp = np.append(r_vp, np.mean(sdf['v_phi'][filter]))
        r_vr = np.append(r_vr, np.mean(sdf['v_rad'][filter]))
        r_fvp = np.append(r_fvp, np.mean(sdf['f_vphi'][filter]))
        r_fvr = np.append(r_fvr, np.mean(sdf['f_vrad'][filter]))

    return rr/sun_rad, r_vp, r_vr, r_fvp, r_fvr


#---------Calculate mean temperature per radii-----------
def compute_temperature(snapshot, H):
    sdf= sar.read_phantom(snapshot)
    sdf['rmag']= np.sqrt((sdf['x']-x_acc)**2 + sdf['y']**2 + sdf['z']**2)  # in cm
    sdf['u'] = sdf['u'] * 1.907e15 # Convert to erg/g
    sdf['temp'] = (sdf['u']*mu*mass_proton_cgs*(gamma-1.0))/kboltz 

    dr = (r_max- r_min)/(2*n_r)

    # make rr array
    rr = np.linspace(r_min,r_max, n_r)
    r_temp = np.array([])

    # for each r in the array, calculate temperature
    for k in range(0,len(rr)):
        filter = ((sdf['rmag']<(rr[k]+dr)) & (sdf['rmag']>(rr[k]-dr))& (sdf['z'] >= -H[k]) & (sdf['z'] <= H[k]))
        r_temp = np.append(r_temp, np.mean(sdf['temp'][filter]))

    return rr, r_temp

#----------------Calculate temperature equation----------------

def linear_fit_temperature(r, temp, confidence=0.95):
    x = np.log10(r)
    y = np.log10(temp)
    res = stats.linregress(x, y)
    dof = len(x) - 2  # degrees of freedom
    alpha = 1 - confidence
    tval = abs(t.ppf(alpha/2, dof))  # two-tailed

    slope = res.slope
    intercept = res.intercept
    slope_err = tval * res.stderr
    intercept_err = tval * res.intercept_stderr

    return slope, slope_err, intercept, intercept_err


#----------------Calculate mass transfer rate per radii----------

def compute_mass_transfer_rate(snapshot, r, H):
    r= r*sun_rad #in cm 
    H= H*sun_rad #in cm 
    sdf= sar.read_phantom(snapshot)
    sdf.calc_density()
    sdf.create_mass_column()
    sdf['m'] = sdf['m'] * sun_mass  # in g
    sdf['rho'] = sdf['rho'] * 5.901 # Convert to g/cm^3
    rx = (sdf['x'] - x_acc)*sun_rad # in cm
    ry = sdf['y'] * sun_rad # in cm
    sdf['rmag'] = np.sqrt(rx**2 + ry**2)  # in cm
    sdf['u'] = sdf['u'] * 1.907e15 # Convert to erg/g
    sdf['temp'] = (sdf['u']*mu*mass_proton_cgs*(gamma-1.0))/kboltz
    sdf['cs'] = np.sqrt(kboltz * sdf['temp'] / (mu * mass_proton_cgs))  # in cm/s

    vx = sdf['vx']*4.367e7 # in cm/s
    vy = sdf['vy']*4.367e7 # in cm/s

    # Get radial and tangential velocities
    v_rad, v_phi = getradtanvelocity(rx, ry, vx, vy)
    sdf['v_rad'] = v_rad 
    sdf['v_phi'] = v_phi
    sdf['flux'] = sdf['rho']*sdf['v_rad'] 

    r_min = r[0] 
    r_max = r[-1]
    n_r = len(r)         
    dr = (r_max- r_min)/(n_r)
    mdot_per_r = []
    cs_per_r = []
    sigma_per_r =[]
    flux_per_r = []

    
    for i in range(len(H)):
        mask = (
        (sdf['rmag'] < (r[i] + dr/2)) &
        (sdf['rmag'] > (r[i] - dr/2)) &
        (sdf['z'] >= -H[i]) &
        (sdf['z'] <=  H[i])
    )
        filtered = sdf[mask]
        flux = filtered['flux'].mean()
        mdot =  -2*pi*r[i]*(2*H[i])*flux
        cs = filtered['cs'].mean()
        sigma = filtered['m'].sum()/(2*pi*r[i]*(dr))
        flux_per_r.append(flux)
        mdot_per_r.append(mdot)
        cs_per_r.append(cs)
        sigma_per_r.append(sigma)

    mdot_r = np.array(mdot_per_r)
    cs_r = np.array(cs_per_r)
    sigma_r = np.array(sigma_per_r)
    flux_r = np.array(flux_per_r)

    alpha_r = (mdot_r /(3.0* pi * sigma_r * cs_r * H)) * (1- np.sqrt(1.0*sun_rad/(r)))

    nu_r = alpha_r*cs_r*H

    mdot_r = mdot_r *(yr/sun_mass)  # in Msun/yr

    return mdot_r, alpha_r, flux_r, nu_r

#---------------Calculte opacity at certain positions along the z-axis----------------------------------
def compute_opacity(snapshot, x_acc, r_line, theta_line):
    kappa = 0.34 #in cm^2/g electron scattering opacity for sun meatalicity
    c = 2.997924e10  
    df= sar.read_phantom(snapshot)
    df['xnew'] = df['x'] - x_acc
    df['rmag'] = np.sqrt(df['xnew']**2 + df['y']**2)
    df['theta'] = np.mod(np.arctan2(df['y'], df['xnew']), 2 * np.pi)
    df.calc_density()
    df['rho'] = df['rho'] * 5.901 # Convert to g/cm^3

    z_min = -30.0  
    z_max = 30.0
    n_z = 100
    z_values = np.linspace(z_min, z_max, n_z)
    dz = (z_max - z_min)/(2*n_z)
    opacity_per_z =[]
    time_per_z =[]

    dr = (r_max- r_min)/(2*n_r)
    dtheta = (theta_max - theta_min)/(2*n_theta) 

    mask = ((df['rmag'] >= r_line - dr) & (df['rmag'] <= r_line + dr) & (df['theta'] >= theta_line - dtheta) & (df['theta'] <= theta_line + dtheta))
    column_df = df[mask].copy()
    if len(column_df) == 0:
        return 0.0
    
    for k in z_values:
        #finding particles in dz
        mask_z = (column_df['z'] >= k - dz) & (column_df['z'] <= k + dz)
        df_filtered = column_df[mask_z]

        if len(df_filtered) == 0:
            opacity_per_z.append(0.0)
            continue
        rho_mean = df_filtered['rho'].mean()
        dz_cm = dz*sun_rad
        opacity = kappa * rho_mean*dz_cm
        t_phot = opacity*((df_filtered['z']- z_min)*sun_rad)/c

        opacity_per_z.append(opacity)
        time_per_z.append(t_phot)
    
    opacity_total = np.sum(opacity_per_z)
    time_photon = np.sum(time_per_z)

    return opacity_total, time_photon, opacity_per_z, z_values


#------------------Calculate energies of the  dumpfile-----------------

def kernel_softening(q):
    if q< 1.0:
        phi = 2.*(q**2)/3. - 3.*(q**4)/10. + (q**5)/10. - 7./5.
    elif q < 2.0:
        phi = 4.*(q**2)/3. - (q**3) +3.*(q**4)/10. - (q**5)/10 - 8./5. + 1./(15.*q)
    else:
        phi = -1./q
    
    return phi

def fix_potential(x,y,z):
    x_acc = 221.526229*sun_rad #x position of the accretor in cm
    x_don =  -44.8137709*sun_rad #x position of the donor in cm
    h_acc = 1.0*sun_rad
    h_don = 100.0*sun_rad
    r_acc = np.sqrt((x - x_acc)**2 + y**2 + z**2)
    r_don = np.sqrt((x - x_don)**2 + y**2 + z**2)
    q_acc = r_acc/h_acc
    q_don = r_don/h_don

    phi_acc = kernel_softening(q_acc)
    phi_don = kernel_softening(q_don)

    phi_acc = gg*m_a*phi_acc/h_acc
    phi_don = gg*m_d*phi_don/h_don

    poten = phi_acc +phi_don
    
    return poten

def compute_energies(snapshot): 
    sdf= sar.read_phantom(snapshot)
    sdf.calc_density()
    sdf.create_mass_column()
    sdf['m'] *=sun_mass #Convert to g
    sdf['rho'] *= 5.901 # Convert to g/cm^3
    sdf['u'] *= 1.907e15 # Convert to erg/g
    sdf['x'] *= sun_rad
    sdf['y'] *= sun_rad
    sdf['z'] *= sun_rad
    sdf['vx'] *= 4.367e7 #Convert to cm/s
    sdf['vy'] *= 4.367e7 #Convert to cm/s
    sdf['vz'] *= 4.367e7 #Convert to cm/s
    total_mass = sdf["m"].sum() 
    pmass= total_mass / len(sdf)
    #inertial frame velocities v + (Omega x r)   [everything is a vector] (Omega x r) = (-Omega y, Omega x, 0 )
    sdf['vx_nr'] = sdf['vx'] - omega*(sdf['y']*sun_rad)  # in cm/s
    sdf['vy_nr'] = sdf['vy'] + omega*(sdf['x']*sun_rad) # in cm/s
    sdf['vz_nr'] = sdf['vz']
    sdf['vmag'] = np.sqrt(sdf['vx_nr']**2 + sdf['vy_nr']**2 + sdf['vz_nr']**2) # speed in cm/s in the inertial frame

    #kinetic energy
    sdf['ekin'] = sdf['m']*0.5*sdf['vmag']**2

    #thermal energy (no radiation)
    sdf['eth'] = sdf['m']*sdf['u']

    #potential energy (no self-gravity)
    epot_values = []
    for i in range(len(sdf)):
        x, y, z, m = sdf.iloc[i][['x', 'y', 'z', 'm']]
        phii =  fix_potential(x, y, z)
        epot_values.append(m*phii)

    sdf['epot'] = epot_values
    
    #Total energies 
    E_tot_kin = sdf['e_kin'].sum()
    E_tot_pot_external = sdf['e_pot'].sum()
    E_tot_thermal = sdf['e_th'].sum()
    E_tot = E_tot_kin + E_tot_pot_external + E_tot_thermal

    return E_tot_kin, E_tot_pot_external, E_tot_thermal, E_tot, total_mass

#------------------Calcultate bound and unbound mass per snapshot-----------------
###----------------WRONG!! NEEDS TO BE FIXED  ----------------------------------
def compute_bound_unbound(snapshot):
    sdf= sar.read_phantom(snapshot, ignore_inactive=False)
    sdf.create_mass_column()
    total_mass = sdf["m"].sum() 
    pmass= total_mass / len(sdf)
    sdf['r_acc'] = np.sqrt((sdf['x'] - x_acc)**2 + sdf['y']**2 + sdf['z']**2)* sun_rad
    sdf['r_don'] = np.sqrt((sdf['x'] - x_don)**2 + sdf['y']**2 + sdf['z']**2)* sun_rad
    sdf['E_kin'] = 0.5 * (sdf['vx']**2 + sdf['vy']**2 + sdf['vz']**2) * vel_scaling**2
    sdf['E_pot'] = -gg*m_a/sdf['r_acc'] - gg*m_d/sdf['r_don']
    sdf['E_the'] = sdf['u'] * u_scaling
    sdf['E_tot'] = sdf['E_kin'] + sdf['E_pot']

    mask_alive_b = (sdf['h'] > 0.0) & (sdf['E_tot'] < 0.0) 
    mask_death_b = (sdf['h'] <= 0.0) & (sdf['E_tot'] < 0.0)
    mask_alive_u = (sdf['h'] > 0.0) & (sdf['E_tot'] >= 0.0) 
    mask_death_u = (sdf['h'] <= 0.0) & (sdf['E_tot'] >= 0.0)

    df_alive_b = sdf[mask_alive_b]
    df_death_b = sdf[mask_death_b]
    df_alive_u = sdf[mask_alive_u]
    df_death_u = sdf[mask_death_u]

    m_alive_b = df_alive_b['m'].sum()
    m_death_b = df_death_b['m'].sum()
    m_alive_u = df_alive_u['m'].sum()
    m_death_u = df_death_u['m'].sum()

    return m_alive_b, m_death_b, m_alive_u, m_death_u, pmass


#------------------Calculate alpha parameter per radii and angle--------------------------
def alpha_calculation(snapshot, n_angles): #My way of calculating alpha = <vr*vphi>/cs^2 per angle sector
    dir_vel = '/Users/lourdesjuarez/disk_sph/big_nozz/cs_simulations/cs_pm10/alpha/over_time/'
    vr_mean = np.loadtxt(f"{dir_vel}{n_angles}_vr_mean.txt", skiprows=1, delimiter='\t')
    vphi_mean = np.loadtxt(f"{dir_vel}{n_angles}_vphi_mean.txt", skiprows=1, delimiter='\t')
    df = sar.read_phantom(snapshot)
    df.calc_density()
    df.create_mass_column()
    df['m'] *= sun_mass # in g
    df['rho']*= 5.901 #in g/cm^3
    df['u'] *= 1.907e15 # Convert to erg/g
    df['pr'] = (gamma-1.0)*df['rho']*df['u']
    df['x_pacc'] = (df['x'] - x_acc)*sun_rad #in cm 
    df['y'] *= sun_rad #in cm 
    df['z'] *= sun_rad #in cm 
    df['h'] *= sun_rad #in cm
    df['vx'] *= 4.367e7 #in cm/s
    df['vy'] *= 4.367e7 #in cm/s
    df['vz'] *= 4.367e7 #in cm/s
    df['temp'] = (df['u']*mu*mass_proton_cgs*(gamma-1.0))/kboltz
    df['phi'] = np.mod(np.arctan2(df['y'], df['x_pacc']), 2*np.pi) #in rad

    r_bin, H_bin= compute_scale_height(snapshot, x_acc)[:2]

    df['rad'] = np.sqrt(df['x_pacc']**2 + df['y']**2)
    df['vr'] = (df['x_pacc']*df['vx'] + df['y']*df['vy'])/df['rad']
    df['vphi'] = (df['x_pacc']*df['vy'] - df['y']*df['vx'])/df['rad']
    
    df['cs'] = np.sqrt(kboltz * df['temp'] / (mu * mass_proton_cgs)) 

    dr = r_bin[1]- r_bin[0]
    sigma_bin = np.zeros(len(r_bin))
    phi_angles =  np.linspace(0, 2*np.pi, n_angles + 1)

    T_rphi = np.zeros((len(r_bin), n_angles))
    alpha = np.zeros((len(r_bin), n_angles))
    vr_mean_array = np.zeros((len(r_bin), n_angles))
    vphi_mean_array = np.zeros((len(r_bin), n_angles))
    vrvphi_mean = np.zeros((len(r_bin), n_angles))
    vr_fluc_array = np.zeros((len(r_bin), n_angles))
    vphi_fluc_array = np.zeros((len(r_bin), n_angles))


    df['vr_fluc'] = 0.0
    df['vphi_fluc'] = 0.0
    df['flux_rphi'] = 0.0

    for j in range(n_angles):
        for i in range(len(r_bin)):
            mask = ((df['rad'] <= (r_bin[i] + dr/2)*sun_rad) &
                    (df['rad'] > (r_bin[i] - dr/2)*sun_rad) &
                    (df['z'] >= (-H_bin[i]*sun_rad)) &
                    (df['z'] <= (H_bin[i]*sun_rad)) &
                    (df['phi'] >= phi_angles[j]) &
                    (df['phi'] < phi_angles[j+1])
                    )
            df_bin = df[mask].copy()
            #vr_mean = df_bin['vr'].mean()
            #vphi_mean = df_bin['vphi'].mean()

            df_bin['vr_fluc'] = df_bin['vr'] - vr_mean[i,j+1]
            df_bin['vphi_fluc'] = df_bin['vphi'] - vphi_mean[i,j+1]

            df_bin['flux_rphi'] = df_bin['rho']*df['rad']*df_bin['vr_fluc']*df_bin['vphi_fluc']

            df.loc[mask, 'vr_fluc'] = df_bin['vr_fluc']
            df.loc[mask, 'vphi_fluc'] = df_bin['vphi_fluc']
            df.loc[mask, 'flux_rphi'] = df_bin['flux_rphi']

            
            #print(df_bin['vr_fluc'][10:20])
            #df_bin['rey'] = df_bin['rho']*df_bin['vr_fluc']*df_bin['vphi_fluc']

            #df['vr_fluc'] = np.where(mask, df_bin['vr_fluc'].values, np.nan)
            #df['vphi_fluc'] = np.where(mask, df_bin['vphi_fluc'].values, np.nan)
            #test = df.merge(df_bin[['iorig', 'vr_fluc', 'vphi_fluc']], on='iorig', how='left')

            #print((np.sum(df_bin['vphi'])/len(df_bin))/1e5,(np.sum(abs(df_bin['vphi_fluc']))/len(df_bin))/1e5)

            #df.iloc[mask, 'vr_fluc'] = df_bin['vr_fluc'].values
            #df.iloc[mask, 'vphi_fluc'] = df_bin['vphi_fluc'].values

            #T_rphi[i] = (df_bin['rey'].mean())
            sigma_bin[i] = (df_bin['rho'].mean())*2.0*H_bin[i]*sun_rad #in g/cm^2
            T_rphi[i,j] = (sigma_bin[i]*((df_bin['vr_fluc']*df_bin['vphi_fluc']).mean()))
            alpha[i,j] = (df_bin['vr_fluc']*df_bin['vphi_fluc']).mean()/(df_bin['cs'].mean())**2 #Orsolas fix to alpha = |<vr_fluc*vphi_fluc>|/ cs^2

            #vr_mean_array[i,j] = vr_mean
            #vphi_mean_array[i,j] = vphi_mean
            #vrvphi_mean[i,j] = (abs(df_bin['vr_fluc']*df_bin['vphi_fluc'])).mean()
            #vr_fluc_array[i,j] = (abs(df_bin['vr_fluc'])).mean()
            #vphi_fluc_array[i,j] = (abs(df_bin['vphi_fluc'])).mean()

    alpha_avg = np.zeros(len(r_bin))
    for i in range(len(r_bin)):
        alpha_avg[i] = alpha[i,:].sum()/n_angles

    return r_bin, T_rphi, alpha, alpha_avg #r_bin, vr_mean_array, vphi_mean_array, vrvphi_mean, vr_fluc_array, vphi_fluc_array, alpha



