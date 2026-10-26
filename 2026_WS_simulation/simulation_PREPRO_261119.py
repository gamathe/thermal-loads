import pandas as pd                 # Data tables
import os                           # Returns current directory, create files
import numpy as np                  # Arrays
import matplotlib.pyplot as plt	    # Graphs

from math import sqrt, atan, log, exp, sin, cos, tan, trunc

from simulation_MATERIALS_261119 import *
from simulation_PREPRO_VF_261119 import *


pi = np.pi

# Air properties
v_a   = 0.8401 # [m^3/kg] "specific volume of humid air per kg of dry air"
c_p_a = 1020 # [J/kg-K] "specific heat capacity of humid air per kg of dry air"
rho_a = 1.197 #[kg/m^3]
sigma_boltzman = 5.67E-8


# Boundary layers
h_c   = 3 #[W/m2/K]
h_r   = 5 # [W/m^2-K]
h_in  = 8 # [W/m^2-K]
h_out = 23 # [W/m^2-K]

# glazing
rho_glazing    = 2500
lambda_glazing = 1.0
c_glazing      = 840


def weather_data(weather):
    
    # Weather

    if weather == 'CW' : # Winter hour UTC + 1
        w = pd.read_excel('cold_wave_UTC.xlsx', index_col=0)
    else : # Winter hour UTC + 1
        w = pd.read_excel('mean_january_UTC.xlsx', index_col=0) 

    # tz_localize() does not shift the time — it just says “these naive timestamps are actually UTC”.
    # tz_convert() shifts the localized times to 'Europe/Brussels' .
    
    w.index = (
        w.index
          .tz_localize('UTC')  
          .tz_convert('Europe/Brussels') # or 'Europe/Brussels' / your actual UTC+1 zone
    )
    
    w['ts'] = 3600 * np.arange(len(w)) 
    
    return w


def weather_name(weather) :
    
    if weather == 'CW' :
        weather_name = 'Cold wave' 
    else :
        weather_name = 'Average January'
        
    return weather_name

def prepro_tinit(weather, t_set_heating, t_set_min) :
    
    if weather == 'CW' :
        weather_name = 'Cold wave' 
        t_in_init = t_set_min
    else  :
        weather_name = 'Average January'
        t_in_init = (t_set_min + t_set_heating) / 2
        
    return t_in_init


def prepro_VF(L_rm, B_rm, H_rm, dfwl, dfwi, dfwd):
    
    # Wall areas

    # length on 6h-0h direction
    length_6h_0h = L_rm
    # length on 9h-3h direction
    length_9h_3h = B_rm
    # Room height
    Height_room  = H_rm
    
    area_wall_0h = length_9h_3h * Height_room
    area_wall_3h = length_6h_0h * Height_room
    area_wall_6h = length_9h_3h * Height_room
    area_wall_9h = length_6h_0h * Height_room
    area_floor   = length_6h_0h * length_9h_3h
    area_ceiling = length_6h_0h * length_9h_3h
    
    areas_uncarved_ext_walls = np.array([area_wall_0h, area_wall_3h, area_wall_6h, area_wall_9h, area_floor, area_ceiling])

    n_ext_walls = 6
    n_int_walls = len(dfwi)
    n_op_walls  = n_ext_walls + n_int_walls
    n_wis = len(dfwi)
    n_wds = len(dfwd)
    
    areas_wds = dfwd['area'].values
    host_wds  = dfwd['host'].values

    areas_wis = dfwi['area'].values
    host_wis  = dfwi['host'].values
    
    # Final areas of external walls when windows and internal walls are carved into the external walls
    areas_op_walls  = np.append(np.zeros(n_ext_walls), areas_wis)
    
    for i in range(n_ext_walls) :
        areas_op_walls[i] = areas_uncarved_ext_walls[i] - np.array([areas_wds[l] for l in range(n_wds) if host_wds[l] == i]).sum()  \
                                                        - np.array([areas_wis[l] for l in range(n_wis) if host_wis[l] == i]).sum() 

    # --- Walls parameters ---

    # 'facades': ['0h', '3h', '6h', '9h', 'floor', 'ceiling/roof']

    lst_cols = ['T']
    dfwlwi   = pd.concat([dfwl[lst_cols], dfwi[lst_cols]]).reset_index().copy()

    # Wall types
    nowall   = (dfwlwi['T'] == 'N').astype(int).values

    # View Factors

    lst_cols = ['height [m]', 'breadth [m]', 'sill height [m]', 'dist corner [m]', 'host', 'area' ]
    dfwdwi = pd.concat([dfwd[lst_cols], dfwi[lst_cols]]).reset_index().copy()
                      
    VF_wdwi                       = View_factors_windows(L_rm, B_rm, H_rm, dfwdwi)
    VF_walls_wds, areas_walls_wds = View_factors_multiple_wds(L_rm, B_rm, H_rm, dfwdwi, VF_wdwi)
    n_walls_wds                   = len(areas_walls_wds)
    
    # VF_walls_1wd : all windows gathered in one window at the end of the wall list
    n_walls_1wd = n_op_walls + 1
    VF_walls_1wd                           = np.zeros((n_walls_1wd, n_walls_1wd))
    VF_walls_1wd[:n_op_walls, :n_op_walls] = VF_walls_wds[:n_op_walls, :n_op_walls]
    VF_walls_1wd[:n_op_walls,  n_op_walls] = 1 - np.sum(VF_walls_wds[:n_op_walls, :n_op_walls], axis=1)
    VF_walls_1wd[n_op_walls,  :n_op_walls] = areas_walls_wds[:n_op_walls] * VF_walls_1wd[:n_op_walls, n_op_walls] / areas_wds.sum() \
                                            if areas_wds.sum() > 0 else 0
    VF_walls_1wd[n_op_walls,   n_op_walls] = 1 - np.sum(VF_walls_1wd[n_op_walls, :n_op_walls])

    areas_walls_1wd              = np.zeros(n_walls_1wd)
    areas_walls_1wd[:n_op_walls] = areas_walls_wds[:n_op_walls]
    areas_walls_1wd[n_op_walls]  = np.sum(areas_walls_wds[n_op_walls:])
    

    # Radiative and convective heat tansfer coefficients (walls inside boundary layers)

    nowall_1wd     = np.append(nowall, [0])
    Ah_r_walls_1wd = h_r * (1 - nowall_1wd)[:, None] * (np.diag(areas_walls_1wd) @ VF_walls_1wd) * (1 - nowall_1wd)[None, :]
    
    return VF_walls_1wd, Ah_r_walls_1wd


def prepro_layers(wt_layers, materials):

    # Time period for separating adiabatic walls
    T_h = 24
    time_period_s = T_h*3600
    omega_rad  = 2 * pi / time_period_s
    
    # dfwt        = pd.merge(wt_layers, materials, on = 'material', how='left')

    dfwt = wt_layers.join(materials.set_index('material'), on='material', how='left')
    dfwt = dfwt.reset_index()
    
    dfwt['R']   = dfwt['thickness'] / dfwt['lambda']
    dfwt['C']   = dfwt['thickness'] * dfwt['rho'] * dfwt['c']
    dfwt['diffusivity'] = dfwt['lambda'] /dfwt['rho'] /dfwt['c']
    
    # Compute the damping factor at each layer interface in order to find the nul heat flow plane for adiabatic walls
    # Compute damping at the layer interface, with number of interfaces equals (number of layers+1)
    dfwt['diffusivity'] = dfwt['lambda'] /dfwt['rho'] /dfwt['c']
    dfwt['damp_factor'] = np.exp(- dfwt['thickness'] * (omega_rad / (2 * dfwt['diffusivity']))**0.5 )
    
    # Cumulative damping factors from inside to outside
    dfwt['damp_factor_cumprod'] = dfwt.groupby('wall_type')['damp_factor'].cumprod()
    
    # Reverse cumulative product of damping factors
    dfwt['reverse_damp_factor_cumprod'] = (   
        dfwt[::-1]                                         # reverse the DataFrame
        .groupby(dfwt['wall_type'])[ 'damp_factor']        # group using the original 'group'
        .cumprod()                                         # cumprod in reverse order 
    )

    
    # Iterate over each subset defined by 'wall_type'
    for wall_type, group in dfwt.groupby('wall_type'):
        damp_in_to_out = np.concatenate([[1], group['damp_factor_cumprod'].values]) 
        damp_out_to_in = np.concatenate([group['reverse_damp_factor_cumprod'].values, [1]]) 
    
        delta_damp = damp_in_to_out - damp_out_to_in
        f_weight_layer = np.ones(len(group))
    
        for k in range(len(group)):
            if (delta_damp[k] > 0) & (delta_damp[k + 1] < 0) & (delta_damp[k] != delta_damp[k+1]):
                f_weight_layer[k] = delta_damp[k] / (delta_damp[k] - delta_damp[k+1])
            elif (delta_damp[k] > 0) & (delta_damp[k + 1] > 0):
                f_weight_layer[k] = 1
            elif (delta_damp[k] <= 0) & (delta_damp[k + 1] <= 0):
                f_weight_layer[k] = 0
    
        # Assign back to dfwt using .loc with the group's index
        dfwt.loc[group.index, 'fi'] = f_weight_layer
    
    dfwt['Ci'] = dfwt['fi'] * dfwt['C']
    dfwt['Ri'] = dfwt['fi'] * dfwt['R']
    dfwt = dfwt[['wall_type', 'thickness', 'R', 'C', 'Ri', 'Ci']].groupby(['wall_type']).sum()
    dfwt['Ce'] = dfwt['C'] - dfwt['Ci'] 
    dfwt['Re'] = dfwt['R'] - dfwt['Ri'] 

    return dfwt
    
    
def prepro_walls(A_floor, L_rm, B_rm, H_rm, dfwl, dfwi, dfwd):
    
    # Wall areas

    # length on 6h-0h direction
    length_6h_0h = L_rm
    # length on 9h-3h direction
    length_9h_3h = B_rm
    # Room height
    Height_room  = H_rm
    
    area_wall_0h = length_9h_3h * Height_room
    area_wall_3h = length_6h_0h * Height_room
    area_wall_6h = length_9h_3h * Height_room
    area_wall_9h = length_6h_0h * Height_room
    area_floor   = length_6h_0h * length_9h_3h
    area_ceiling = length_6h_0h * length_9h_3h
    
    areas_uncarved_ext_walls = np.array([area_wall_0h, area_wall_3h, area_wall_6h, area_wall_9h, area_floor, area_ceiling])

    n_ext_walls = 6
    n_int_walls = len(dfwi)
    n_op_walls  = n_ext_walls + n_int_walls
    n_wis = len(dfwi)
    n_wds = len(dfwd)
    
    areas_wds = dfwd['area'].values
    host_wds  = dfwd['host'].values

    areas_wis = dfwi['area'].values
    host_wis  = dfwi['host'].values
    
    # Final areas of external walls when windows and internal walls are carved into the external walls
    areas_op_walls  = np.append(np.zeros(n_ext_walls), areas_wis)
    
    for i in range(n_ext_walls) :
        areas_op_walls[i] = areas_uncarved_ext_walls[i] - np.array([areas_wds[l] for l in range(n_wds) if host_wds[l] == i]).sum()  \
                                                        - np.array([areas_wis[l] for l in range(n_wis) if host_wis[l] == i]).sum() 

    # Indoor air capacity
    C_in = 5 * Height_room * length_6h_0h * length_9h_3h * c_p_a * rho_a
    
    # Glazing capacity
    thickness_gl  = 0.004 
    f_frame = 0.1
    C_gl   = ((1 - f_frame) * areas_wds).sum()  * thickness_gl * rho_glazing * c_glazing


    # --- Walls parameters ---

    # 'facades': ['0h', '3h', '6h', '9h', 'floor', 'ceiling/roof']

    lst_cols = ['T', 'R', 'Ri', 'Re', 'C', 'Ci', 'Ce', 'rad heating [W/m2]']
    dfwlwi   = pd.concat([dfwl[lst_cols], dfwi[lst_cols]]).reset_index().copy()

    # Wall types
    external = (dfwlwi['T'] == 'E').astype(int).values
    cond     = (dfwlwi['T'] == 'C').astype(int).values
    uncond   = (dfwlwi['T'] == 'U').astype(int).values
    belowgr  = (dfwlwi['T'] == 'B').astype(int).values
    nowall   = (dfwlwi['T'] == 'N').astype(int).values

    # Wall thermal resistances and capacities
    R_walls_no_bl = dfwlwi['R'].values
    R_walls_ew_mw = dfwlwi['Re'].values
    R_walls_mw_iw = dfwlwi['Ri'].values
    C_walls       = dfwlwi['C'].values  * (1 - nowall) * areas_op_walls 
    C_walls_ew    = dfwlwi['Ce'].values * (1 - nowall) * areas_op_walls 
    C_walls_mw    = dfwlwi['Ci'].values * (1 - nowall) * areas_op_walls  / 2
    C_walls_iw    = dfwlwi['Ci'].values * (1 - nowall) * areas_op_walls  / 2

    U_walls_ew_mw  = np.divide(np.ones_like(R_walls_ew_mw), R_walls_ew_mw, out=np.zeros_like(R_walls_ew_mw), where = R_walls_ew_mw != 0)
    AU_walls_ew_mw = (1 - nowall) * areas_op_walls * U_walls_ew_mw

    U_walls_mw_iw  = np.divide(np.ones_like(R_walls_mw_iw), R_walls_mw_iw, out=np.zeros_like(R_walls_mw_iw), where = R_walls_mw_iw != 0)
    AU_walls_mw_iw = (1 - nowall) * areas_op_walls * U_walls_mw_iw

    # External walls AU-values
    h_out_bl = np.where(dfwlwi['T'] == 'E', h_out, h_in)
    U_walls  = 1 / (R_walls_no_bl + 1/h_in + 1/h_out_bl) 
    AU_walls = (1 - nowall) * areas_op_walls * U_walls

    # Walls radiative heating and cooling powers
    f_area_int_ext = A_floor / L_rm / B_rm
    Q_dot_rad_heating_walls   = dfwlwi['rad heating [W/m2]'].values * (1 - nowall) * areas_op_walls * f_area_int_ext

    # Radiative and convective heat tansfer coefficients (walls inside boundary layers)

    Ah_c_op_walls  = h_c * (1 - nowall) * areas_op_walls
    Ah_c_wd        = h_c * areas_wds.sum()
    

    # Windows parameters
    area_wd      = areas_wds.sum()

    walls_param  = [n_op_walls, area_wd, C_in, C_gl, Ah_c_wd]

    walls_arrays = [areas_op_walls, external, cond, uncond, belowgr, nowall,   \
                    AU_walls_ew_mw, AU_walls_mw_iw, AU_walls, C_walls_ew, C_walls_mw, C_walls_iw, \
                    Q_dot_rad_heating_walls,  Ah_c_op_walls]
    
    return walls_param, walls_arrays















    




