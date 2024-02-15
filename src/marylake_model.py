import numpy as np
import pandas as pd
import os
from math import pi, exp, sqrt
from scipy.interpolate import interp1d
from copy import deepcopy
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit

#os.chdir("/home/robert/Projects/1D-AEMpy/src")
#os.chdir("C:/Users/ladwi/Documents/Projects/R/1D-AEMpy/src")
#os.chdir("D:/bensd/Documents/Python_Workspace/1D-AEMpy/src")
os.chdir("C:/Users/au740615/Documents/Projects/n2_mary/src")
from processBased_lakeModel_functions import get_hypsography, provide_meteorology, initial_profile, run_wq_model, wq_initial_profile, provide_phosphorus, do_sat_calc, calc_dens #, heating_module, diffusion_module, mixing_module, convection_module, ice_module


## lake configurations
zmax = 21 # maximum lake depth
nx = zmax * 2 # number of layers we will have
dt = 3600 # 24 hours times 60 min/hour times 60 seconds/min
dx = zmax/nx # spatial step

## area and depth values of our lake 
area, depth, volume = get_hypsography(hypsofile = '../bcs/bathymetry.csv',
                            dx = dx, nx = nx)
                           
## atmospheric boundary conditions
meteo_all = provide_meteorology(meteofile = '../bcs/meteodriverdata.csv',
                    secchifile = None, 
                    windfactor = 1.0)
                     
## time step discretization                      
hydrodynamic_timestep = 24 * dt
total_runtime =  (364) * hydrodynamic_timestep/dt  #365 *1 # 14 * 365
startTime =  1 #150 * 24 * 3600
endTime =  (startTime + total_runtime) # * hydrodynamic_timestep/dt) - 1

startingDate = meteo_all[0]['date'][startTime] #* hydrodynamic_timestep/dt]
endingDate = meteo_all[0]['date'][(endTime-1)]#meteo_all[0]['date'][(startTime + total_runtime)]# * hydrodynamic_timestep/dt -1]

times = pd.date_range(startingDate, endingDate, freq='H')

nTotalSteps = int(total_runtime)

## here we define our initial profile
u_ini = np.ones(nx)*4.0

n2_ini = 1e-5 * volume

Start = datetime.datetime.now()

    
res = run_wq_model(  
    u = deepcopy(u_ini),
    o2 = n2_ini,
    docr = n2_ini,
    docl = 1.0 * volume,
    pocr = 0.5 * volume,
    pocl = 0.5 * volume,
    alg = 10/1000 * volume,
    nutr = 0.5 * volume,
    startTime = startTime, 
    endTime = endTime, 
    area = area,
    volume = volume,
    depth = depth,
    zmax = zmax,
    nx = nx,
    dt = dt,
    dx = dx,
    daily_meteo = meteo_all[0],
    secview = meteo_all[1],
    ice = False,
    Hi = 0,
    Hs = 0,
    Hsi = 0,
    iceT = 6,
    supercooled = 0,
    coupled = 'off',
    diffusion_method = 'pacanowskiPhilander',#'pacanowskiPhilander',# 'hendersonSellers', 'munkAnderson' 'hondzoStefan'
    scheme ='implicit',
    km = 1.4 * 10**(-7), # 4 * 10**(-6), 
    k0 = 1 * 10**(-2), #1e-2
    weight_kz = 0.5,
    kd_light = 2, 
    denThresh = 1e-2,
    albedo = 0.1,
    eps = 0.97,
    emissivity = 0.97,
    sigma = 5.67e-8,
    sw_factor = 1.0,
    wind_factor = 0.7,#1.0,
    at_factor = 1.0,
    turb_factor = 1.0,
    p2 = 1,
    B = 0.61,
    g = 9.81,
    Cd = 0.0013, # momentum coeff (wind)
    meltP = 1,
    dt_iceon_avg = 0.8,
    Hgeo = 0.1, # geothermal heat 
    KEice = 0,
    Ice_min = 0.1,
    pgdl_mode = 'off',
    rho_snow = 250,
    p_max = 1/86400,
    IP = 3e-2/86400 ,#0.1, 3e-6
    theta_npp = 1.0, #1.08,
    theta_r = 1.08, #1.08,
    conversion_constant = 1e-4,#0.1
    sed_sink =0.005 / 86400, #0.01
    k_half = 3.1, #0.5,
    resp_docr = 0.008/86400, # 0.08 0.001 0.0001
    resp_docl = 0.008/86400, # 0.01 0.05
    resp_pocr = 0.004/86400, # 0.04 0.1 0.001 0.0001
    resp_pocl = 0.004/86400,
    grazing_rate = 0.9/86400, #1e-1/86400, # 3e-3/86400
    pocr_settling_rate = 1e-3/86400,
    pocl_settling_rate = 1e-3/86400,
    algae_settling_rate = 1e-5/86400,
    sediment_rate = 10/86400,
    piston_velocity = 1.0/86400,
    light_water = 0.125,
    light_doc = 0.02,
    light_poc = 0.7,
    mean_depth = sum(volume)/max(area),
    W_str = None,
    tp_inflow = 0,#np.mean(tp_boundary['tp'])/1000 * volume[0] * 1/1e6,
    alg_inflow = 0.1 * volume[0] * 5/1e6,
    pocr_inflow = 0.1 * volume[0] * 1/1e7,
    pocl_inflow = 0.1 * volume[0] * 1/1e7,
    f_sod = 0.01 / 86400,
    d_thick = 0.001,
    growth_rate = 0.5/86400, # 1.0e-3
    grazing_ratio = 0.1,
    alpha_gpp = 0.03/86400,
    beta_gpp = 0.00017/86400,
    o2_to_chla = 2.15/3600)

temp=  res['temp']
o2=  res['o2']
docr=  res['docr']
docl =  res['docl']
pocr=  res['pocr']
pocl=  res['pocl']
alg=  res['alg']
nutr=  res['nutr']
diff =  res['diff']
avgtemp = res['average'].values
temp_initial =  res['temp_initial']
temp_heat=  res['temp_heat']
temp_diff=  res['temp_diff']
temp_mix =  res['temp_mix']
temp_conv =  res['temp_conv']
temp_ice=  res['temp_ice']
meteo=  res['meteo_input']
buoyancy = res['buoyancy']
icethickness= res['icethickness']
snowthickness= res['snowthickness']
snowicethickness= res['snowicethickness']
npp = res['npp']
algae_growth = res['algae_growth']
algae_grazing = res['algae_grazing']
docr_respiration = res['docr_respiration']
docl_respiration = res['docl_respiration']
pocr_respiration = res['pocr_respiration']
pocl_respiration = res['pocl_respiration']
kd = res['kd_light']
thermo_dep = res['thermo_dep']
energy_ratio = res['energy_ratio']


End = datetime.datetime.now()
print(End - Start)

    

# heatmap of temps  
N_pts = 6



fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(temp, cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("Water Temperature  ($^\circ$C)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
time_label = times[::nelement]
#time_label = time_label[::nelement]
#ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
time_label = times[np.array(ax.get_xticks()).astype(int)]
ax.set_xticklabels(time_label, rotation=90)
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()


fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(diff, cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = diff.min(), vmax = diff.max())
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("Diffusivity  (m2/s)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
ax.set_xticklabels(time_label, rotation=0)
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()


# plt.plot(o2[0,:]/volume[0])
# ax = plt.gca()
# ax.set_ylim(0,20)
# plt.show()

fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(np.transpose(np.transpose(o2)/volume), cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("Nitrogen  (g/m3)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
ax.set_xticklabels(time_label, rotation=0)
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()
