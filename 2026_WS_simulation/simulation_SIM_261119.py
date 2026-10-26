import pandas as pd                 # Data tables
import os                           # Returns current directory, create files
import numpy as np                  # Arrays
import matplotlib.pyplot as plt	    # Graphs

from math import sqrt, atan, log, exp, sin, cos, tan, trunc

from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import *
from scipy import interpolate

import matplotlib.dates as mdates

from simulation_INTERFACE_261119 import *
from simulation_MATERIALS_261119 import *
from simulation_PREPRO_261119 import *
from simulation_WEATHER_261119 import *


pi = np.pi

# Air properties
v_a   = 0.8401 # [m^3/kg] "specific volume of humid air per kg of dry air"
c_p_a = 1020 # [J/kg-K] "specific heat capacity of humid air per kg of dry air"
rho_a = 1.197 #[kg/m^3]
sigma_boltzman = 5.67E-8

H_fg_w = 2500000


# Boundary layers
h_c   = 3 #[W/m2/K]
h_r   = 5 # [W/m^2-K]
h_in  = 8 # [W/m^2-K]
h_out = 23 # [W/m^2-K]


# Convective emission
def X_h(t_in, t_set, db):
    exponent = ( t_in - t_set - db / 2 ) / db / 0.125 
    safe_exponent = np.clip(exponent, -700, 700)  # Avoid overflow
    return 1 / (1 + np.exp(safe_exponent))


# Restart heating
def X_rh(nh_remaining_for_ti, nh_ti_rh, t_in, t_set):
    db = 1
    nh_reheat_ti = nh_ti_rh * (t_set - t_in)
    exponent = ( nh_remaining_for_ti - nh_reheat_ti - db / 2 ) / db / 0.125 
    safe_exponent = np.clip(exponent, -700, 700)  # Avoid overflow
    return 1 / (1 + np.exp(safe_exponent))


control_laws = [ X_h, X_rh]



def from_hbh_to_sbs(weather, w, dfwl, dfwi, dfh, gen_data, vent_data, ctrl_data, walls_param, control_laws) :
    
    [CP, t_set_heating, t_set_min, n_50, U_wd, qc_heating] = gen_data
    [q_occ_m3h, A_fl_occ, epsilon_hx] = vent_data
    [CO2_vent_control, WEEK_END_set_back, nh_ti_rh, nh_te_rh] = ctrl_data


    # Walls parameters
    [n_op_walls, area_wd , C_in, C_gl, Ah_c_wd] = walls_param


    # CONTROL LAWS
    [ X_h, X_rh] = control_laws 
    
    
    # Expand data from hour by hour to step by step

    
    # --- Hour by hour weather data --- 
    
    t_out       = w['t_out'].values
    c_sky       = w['c_sky'].values
    h_out_var   = w['h_out'].values


    # ---  Hour by hour window U-values --- 
    
    # Window when inside and outside boundary layers are suppressed
    R_wd_no_blayers  = max(0, 1/U_wd - 1/h_in - 1/h_out)
    # Window when inside boundary layer is suppressed
    R_wd_no_ilayer  = R_wd_no_blayers + 1/h_out_var 
    U_wd_no_ilayer  = 1/ R_wd_no_ilayer   
    AU_wd_no_ilayer = area_wd * U_wd_no_ilayer 

    
    # ---  Hour by hour occupancy profiles --- 
    
    # Occupancy and ventilation data (24h profiles)
    f_occ_24h  = dfh['occupancy'].values
    f_vent_24h = dfh['ventilation'].values
    
    # Hourly occupancy profiles from 24h schedules
    dfhbh = w[['ts']].copy()
    dfhbh['hour']    = dfhbh.index.hour
    dfhbh['weekday'] = dfhbh.index.weekday  # Monday=0, Sunday=6
    dfhbh['f_occ']   = [f_occ_24h[h]  for h in dfhbh['hour']]
    dfhbh['f_vent']  = [f_vent_24h[h] for h in dfhbh['hour']]
    
    # If WEEK_END_set_back is True → set weekend occupancy factors to 0
    if WEEK_END_set_back: dfhbh.loc[dfhbh['weekday'] >= 5, 'f_occ'] = 0

    # Store hourly occupancy profiles in arrays
    f_occ  = dfhbh['f_occ'].values
    f_vent = dfhbh['f_vent'].values

    flag_occ = np.where(f_occ > 0, 1, 0)
    
    # ---  Hour by hour fraction of ventilation maximum air flow ---  
    f_fresh  = f_occ if CO2_vent_control else f_vent 
    
    # --- Hour by hour temperature set points --- 
    t_set_h_step = t_set_heating * flag_occ + t_set_min * (1-flag_occ)
    t_set_h  = t_set_h_step
    

    # --- Hour by hour dataframe --- 
    
    # Store hour by hour arrays in a dataframe
    dfhbh = dfhbh[['ts']].copy()

    # Hour by hour arrays
    arrays = [t_out, c_sky,
              h_out_var, AU_wd_no_ilayer, 
              f_occ, f_vent,  f_fresh, 
              t_set_h_step, t_set_h]

    # Hour by hour dataframe
    names =  ['t_out', 'c_sky',
            'h_out_var', 'AU_wd_no_ilayer', 
            'f_occ', 'f_vent', 'f_fresh',
            't_set_h_step', 't_set_h']
    
    dfhbh[names] = np.column_stack(arrays)

    
    # --- Step by step dataframe --- 
    
    # Step by step data (step = 15 min)
    df = dfhbh.resample('15min').asfreq()
    df = df.interpolate(method='time')


    return df 


def sim(weather, gen_data, vent_data, ctrl_data, dfh, wl, wd, wt_layers, materials) :

    [CP, t_set_heating, t_set_min, n_50, U_wd, qc_heating] = gen_data
    [q_occ_m3h, A_fl_occ, epsilon_hx] = vent_data
    [CO2_vent_control, WEEK_END_set_back, nh_ti_rh, nh_te_rh] = ctrl_data

    # Weather data
    w = weather_data(weather)

    # Temperature factors

    t_ext = we.loc[we['CP'] == CP, 'theta_e_base'].iloc[0]
    t_min = we.loc[we['CP'] == CP, 'theta_e_min'].iloc[0] 
    t_avg = we.loc[we['CP'] == CP, 'theta_e_avg'].iloc[0]

    f_temp_min = (20 - t_min) / (20 - t_ext)
    f_temp_avg = (20 - t_avg) / (20 - t_ext)
 

    # Envelope geometric data
    # Dimensions introduced by the user
    # Horizontal dimensions are external dimensions
    # Vertical dimensions are internal dimensions
    # The area of the vertical walls is modified in the calculation to add the effect of the floor and roof thicknesses

    L_rm    = wl.loc[wl['type'] == 'wall',  'width_m'].max()
    B_rm    = wl.loc[wl['type'] == 'wall',  'width_m'].min()
    H_in    = wl.loc[wl['type'] == 'wall',  'height_m'].mean()

    azimuth_0h_deg = wl.loc[wl['type'] == 'wall'].loc[wl['width_m'].idxmin(), 'azimuth_deg']

    wl.loc[wl['width_m'].idxmin(),     'azimuth'] = 'az0'
    wl.loc[wl['width_m'].idxmin() + 1, 'azimuth'] = 'az0+90'
    wl.loc[wl['width_m'].idxmin() + 2, 'azimuth'] = 'az0+180'
    wl.loc[wl['width_m'].idxmin() + 3, 'azimuth'] = 'az0-90'
    wl.loc[wl['type'] == 'floor', 'azimuth'] = 'floor'
    wl.loc[wl['type'] ==  'roof', 'azimuth'] = 'roof'
    
    mask = wl['type'] == 'panel'
    wl.loc[mask, 'azimuth'] = wl.loc[mask, 'parent'].map(wl.set_index('id')['azimuth'])
    wd['azimuth']  = wd['parent'].map(wl.set_index('id')['azimuth'])

    # Wall thicknesses
    dftk   = wt_layers.groupby(level='wall_type')['thickness'].sum().reset_index()
    thk    = dict(zip(dftk['wall_type'], dftk['thickness']))
    df_thk = pd.DataFrame(list(thk.items()), columns=['wall_type', 'thickness']).set_index('wall_type')
    df_thk.loc['no', 'thickness'] = 0.0

    wl['thickness'] = wl['wall_type'].map(df_thk['thickness'])

    # full thickness if external environmental temperature, half thickness otherwise
    wl['contrib'] = np.where(wl['t_env'] == 't_ext',
                                wl['thickness'],
                                0.5 * wl['thickness'])

    mask = wl['type'] == 'wall'

    L_in = (
        L_rm
        - wl.loc[mask & (wl['azimuth'] == 'az0'),     'contrib'].iloc[0]
        - wl.loc[mask & (wl['azimuth'] == 'az0+180'), 'contrib'].iloc[0]
    )
    
    B_in = (
        B_rm
        - wl.loc[mask & (wl['azimuth'] == 'az0+90'), 'contrib'].iloc[0]
        - wl.loc[mask & (wl['azimuth'] == 'az0-90'), 'contrib'].iloc[0]
    )
    
    H_rm = (
        H_in
        + wl.loc[wl['type'] == 'floor', 'contrib'].iloc[0]
        + wl.loc[wl['type'] == 'roof',  'contrib'].iloc[0]
    )


    # Ventilation data
    A_floor    = L_in * B_in
    n_occ      = A_floor / A_fl_occ  # Number of occupants
    qv_su_m3h  = n_occ * q_occ_m3h # ventilation fan supply air flow  (m³/h)
    qv_ex_m3h  = n_occ * q_occ_m3h # ventilation fan extracted air flow  (m³/h)


    ### PREPROCESSING

    wi = wl.loc[wl['type'] == 'panel'].copy()
    wl = wl.loc[wl['type'] != 'panel'].copy()

    wl = dfwl_final(wl, azimuth_0h_deg)
    wi = dfwi_final(wi, azimuth_0h_deg, wl, L_rm, B_rm, H_rm)
    wd = dfwd_final(wd, azimuth_0h_deg, wl, L_rm, B_rm, H_rm)

    # View Factors and radiative heat exchange coefficients  (2D arrays)
    VF_walls_1wd, Ah_r_walls_1wd  = prepro_VF(L_rm, B_rm, H_in, wl, wi, wd)

    dfwt = prepro_layers(wt_layers, materials)

    dfwl = pd.merge(wl, dfwt, left_on='wall_type',  right_index=True,  how='left').fillna(0)
    dfwi = pd.merge(wi, dfwt, left_on='wall_type',  right_index=True,  how='left').fillna(0)
    dfwd = wd.fillna(0)

    # Lists of wall parameters and of wall 1D arrays 
    walls_param, walls_arrays = prepro_walls(A_floor, L_rm, B_rm, H_rm, dfwl, dfwi, dfwd)

    # Wall parameters
    n_op_walls, area_wd , C_in, C_gl, Ah_c_wd = walls_param

    # Wall 1D arrays
    areas_op_walls, external, cond, uncond, belowgr, nowall,   \
                        AU_walls_ew_mw, AU_walls_mw_iw, AU_walls, C_walls_ew, C_walls_mw, C_walls_iw, \
                        Q_dot_rad_heating_walls,  Ah_c_op_walls = walls_arrays

    
    df = from_hbh_to_sbs(weather, w, \
        dfwl, dfwi, dfh, gen_data, vent_data, ctrl_data, walls_param, control_laws)

    time_step = 900 
    max_step  = 900



    ###  SIMULATION

    t_out    = df["t_out"].values
    c_sky    = df['c_sky'].values

    h_out_var = df["h_out_var"].values
    AU_wd_no_ilayer = df["AU_wd_no_ilayer"].values

    f_occ   = df["f_occ"].values
    f_vent  = df["f_vent"].values
    f_fresh = df["f_fresh"].values
    t_set_h = df["t_set_h"].values

    # Additional IR heat losses due to the temperature difference between the sky and the air
        
    epsilon_ir = 0.9
    dt_sky_cs = -21
    h_r = 5 # W/m²K

    # Walls
    index_walls = ['0h', '3h', '6h', '9h', 'floor', 'roof']
    Q_dot_wl_ir_box = np.zeros((len(df), 6))
    
    for k in range(6):
        if (dfwl.loc[index_walls[k], 'T'] == 'E') :
            net_area  = float(dfwl.loc[index_walls[k], 'net_area_m2'].item())
            slope_deg = float(dfwl.loc[index_walls[k], 'slope'].item())
            p_wl_deg = np.abs(slope_deg - 180)
            p_wl_rad = p_wl_deg*pi/180
            cos_p    = np.cos(p_wl_rad)
            Q_dot_wl_ir_box[:, k] = net_area * (1 - c_sky) * epsilon_ir * h_r * dt_sky_cs * (1 + cos_p) / 2

    # Panels
    Q_dot_wi_ir = np.zeros((len(df), len(dfwi)))

    for index, row in dfwi.iterrows():
        if row['T'] == 'E' :
            area = row['area']
            Q_dot_wi_ir[k, :] = area * (1 - c_sky) * epsilon_ir * h_r * dt_sky_cs / 2

    Q_dot_wl_ir  = np.hstack((Q_dot_wl_ir_box, Q_dot_wi_ir)) # Result = original array Q_dot_wl_ir_box with added columns of Q_dot_wi_ir

    # Windows
    Q_dot_wd_ir = dfwd['area'].sum() * (1 - c_sky) * epsilon_ir * h_r * dt_sky_cs / 2
    
    # --- CONSTANT PARAMETERS ---
    
    # Maximum infiltration 
    e_leak      = 0.05 
    f_fac       = 12 if (dfwl['T'][0:4] == 'E').astype(int).sum() == 1 else 8 # Facade factor for infiltration
    q_inf50_m3h = n_50 * L_in * B_in * H_in
    q_leak_m3h  = e_leak * q_inf50_m3h # -->  If no ventilation or if balanced ventilation

    # Minimum infiltration when only ventilation system is working
    qv_max_m3h      = max(qv_su_m3h, qv_ex_m3h)
    qv_min_m3h      = min(qv_su_m3h, qv_ex_m3h)
    qv_unbal_m3h    = qv_max_m3h - qv_min_m3h
    qv_leak_min_m3h = q_leak_m3h / (1 + f_fac / e_leak * (qv_unbal_m3h / q_inf50_m3h)**2)  

    # Maximum heating convective power 
    Q_dot_conv_heating = qc_heating * A_floor
    

    # --- STEP BY STEP VARIABLE QUANTITIES--- 
    
    # --- Restart of the heating system ---
    
    df['start_high_set_point'] = ((df['t_set_h_step'] > df['t_set_h_step'].shift()).astype(int)).where(df['t_set_h_step'] == t_set_heating, 0)
    
    # Get the next heating start as a Series aligned with df
    next_high_set_point_index = df.index.to_series().where(df['start_high_set_point'] == 1)
    next_high_set_point_index = next_high_set_point_index.bfill()  # forward-fill to get the next start for each row

    # Compute hours difference directly (tz-aware safe)
    df['nhours_before_high_set_point'] = ((next_high_set_point_index - df.index) / pd.Timedelta(hours=1)).clip(lower=0)
    df['nhours_before_high_set_point'] = df['nhours_before_high_set_point'].fillna(0)  # After the last start event, keep countdown at 0 instead of NaN
    nhours_before_high_set_point = df['nhours_before_high_set_point'].values

    # Jump of set point temperature at the reheating start
    if (nh_te_rh > 0) & (nh_ti_rh  > 0) :
        Dt_set_reheat = t_set_heating - t_set_min
    else :
        Dt_set_reheat = 0
    
    t_e_base = 10
    dte_base = np.clip(t_e_base - t_out, 0, None)
    nhours_reheat_te = nh_te_rh * dte_base # 5 min per °C difference
    nhours_remaining_for_dti = np.clip(nhours_before_high_set_point - nhours_reheat_te, 0, None)

    dayofweek = df.index.dayofweek.values

    # --- PRECOMPUTE Hv ---

    # fraction of ventilation system air flow required for air renweal 
    Xvent = f_fresh
    flag_vent_only = f_vent
    V_dot_inf_m3h  = q_leak_m3h - flag_vent_only * Xvent * (q_leak_m3h - qv_leak_min_m3h)

    # Effective fresh air flow
    # Ventilation sypply plus additional infiltration
    V_dot_vent_m3h = Xvent * qv_max_m3h

    # Ventilation air flow through recovery HX (bypassed during free-cooling)
    V_dot_hx_m3h   = Xvent * qv_min_m3h

    # Ventilation thermal load 
    Hv = rho_a * c_p_a / 3600 * (V_dot_inf_m3h + V_dot_vent_m3h - V_dot_hx_m3h * epsilon_hx)

    # --- PRECOMPUTE h_env ---

    h_env = np.zeros((n_op_walls, len(df))) \
            + np.outer(external, h_out_var) \
            + np.outer(uncond, h_in) \
            + np.outer(belowgr, h_in) \
            + np.outer(cond, h_in) 

    
    # --- SIMULATION --- 

    ts = df['ts'].values

    t_in_init = prepro_tinit(weather, t_set_heating, t_set_min)
    w_in_init = 0.007  # 2°C, 50%, p_atm
    
    # n_flows = 7
    n_flows = 4

    M_a_in = C_in / c_p_a
    
    Ti0  = [t_in_init, t_in_init] # internal node t and  window 
    Ti0.extend(t_in_init * np.ones(n_op_walls)) # internal opaque walls 
    Ti0.extend(t_in_init * np.ones(n_op_walls)) # middle opaque walls 
    Ti0.extend(t_in_init * np.ones(n_op_walls)) # external opaque walls
    Ti0.extend([0] * n_flows) # flows

    t_s_walls_1wd = np.empty(n_op_walls + 1)

    def get_index(t):
        return min(np.searchsorted(ts, t), len(ts) - 1)

    def model_ode(t, Ti, ind_start, ind_stop):

        ind = get_index(t)
        ind = min(ind, ind_stop)
    
        t_a_in    = Ti[0]
        t_s_wd    = Ti[1]
        t_iw      = Ti[2 : 2 + n_op_walls]
        t_mw      = Ti[2 + n_op_walls : 2 + 2 * n_op_walls]
        t_ew      = Ti[2 + 2 * n_op_walls : 2 + 3 * n_op_walls]
    
        t_si_wall = nowall * t_a_in + (1 - nowall) * t_iw
        t_sm_wall = nowall * t_a_in + (1 - nowall) * t_mw
        t_se_wall = nowall * t_a_in + (1 - nowall) * t_ew

        t_s_walls_1wd[:-1] = t_si_wall
        t_s_walls_1wd[-1]  = t_s_wd
        
        ind = min(int((t - ts[0]) / time_step), len(ts)-1)

        T0 = t_out[ind]

        # Reheat control
        Xrh = X_rh(nhours_remaining_for_dti[ind], nh_ti_rh, t_a_in, t_set_heating)
        Dtset = Dt_set_reheat * Xrh
        Xhe = X_h(t_a_in, t_set_h[ind] + Dtset, 0.5)
    
        Q_dot_h_conv = Xhe * Q_dot_conv_heating # scalar 
        Q_dot_h_rad  = Xhe * Q_dot_rad_heating_walls # array

        Q_dot_vent   = Hv[ind] * (T0 - t_a_in)
    
        # Walls and window convective heat exchanges with indoor air 
        Q_dot_c_in_to_walls = Ah_c_op_walls * (t_a_in - t_si_wall) 
        Q_dot_c_in_to_wd    = Ah_c_wd * (t_a_in - t_s_wd)
        
        # Q_dot_r_i_to_j_walls_1wd = np.diag(t_s_walls_1wd) @ Ah_r_walls_1wd - Ah_r_walls_1wd @ np.diag(t_s_walls_1wd)
        Q_dot_r_i_to_j_walls_1wd = (t_s_walls_1wd[:, None] - t_s_walls_1wd[None, :]) * Ah_r_walls_1wd
    
        Q_dot_r_i_to_j      = Q_dot_r_i_to_j_walls_1wd[:-1, :-1] 
        Q_dot_r_wd_to_walls = Q_dot_r_i_to_j_walls_1wd[-1, :-1] 
        Q_dot_r_walls_to_wd = Q_dot_r_i_to_j_walls_1wd[:-1, -1]
              
        # Wall surface node heat balance 
        Q_dot_in_to_walls_tot  = (Q_dot_h_rad \
                        + Q_dot_c_in_to_walls \
                        + Q_dot_r_wd_to_walls - Q_dot_r_walls_to_wd   \
                        + np.sum(Q_dot_r_i_to_j, axis=0) - np.sum(Q_dot_r_i_to_j, axis=1) ) # array 

     
        # Window heat exchange to external air
        Q_dot_tr_wd    = AU_wd_no_ilayer[ind] * (T0 - t_s_wd)
    
        # Conductive heat exchanges between internal and external wall surfaces
        Q_dot_ew_to_mw = AU_walls_ew_mw * (t_se_wall - t_sm_wall)
        Q_dot_mw_to_iw = AU_walls_mw_iw * (t_sm_wall - t_si_wall)

        t_env = np.ones(n_op_walls) * T0  \
                + uncond  * (1 - f_temp_min) * (t_a_in - T0) \
                + belowgr * (1 - f_temp_avg) * (t_a_in - T0) \
                + cond    * (t_a_in - T0)

        Q_dot_tr_walls = areas_op_walls * h_env[:, ind] * (t_env - t_se_wall)
        
    
        # Internal air 
        Q_dot_in = Q_dot_vent + Q_dot_h_conv - np.sum(Q_dot_c_in_to_walls) - Q_dot_c_in_to_wd
        dTin_t   = Q_dot_in / C_in if C_in > 0 else 0

        # Window 
        Q_dot_wd = Q_dot_tr_wd + Q_dot_wd_ir[ind] + Q_dot_c_in_to_wd \
                   + np.sum(Q_dot_r_walls_to_wd) - np.sum(Q_dot_r_wd_to_walls)
        C_wd     = C_gl 
        dTwd_t   = Q_dot_wd / C_wd if C_wd > 0 else 0
        
        # Internal capacities
        Q_dot_iw = Q_dot_mw_to_iw + Q_dot_in_to_walls_tot 
        C_iw     = C_walls_iw
        dTiw_t   = np.divide(Q_dot_iw, C_iw, out=np.zeros_like(C_iw), where=C_iw!=0) 

        # Middel capacities
        Q_dot_mw = Q_dot_ew_to_mw - Q_dot_mw_to_iw
        C_mw     = C_walls_mw
        dTmw_t   = np.divide(Q_dot_mw, C_mw, out=np.zeros_like(C_mw), where=C_mw!=0) 
    
        # External capacities
        Q_dot_ew = Q_dot_tr_walls + Q_dot_wl_ir[ind, :] - Q_dot_ew_to_mw 
        C_ew     = C_walls_ew 
        dTew_t   = np.divide(Q_dot_ew, C_ew, out=np.zeros_like(C_ew), where=C_ew!=0) 

    
        Q_dot_h    = Q_dot_h_conv + Q_dot_h_rad.sum()
        Q_dot_v    = Q_dot_vent 
        Q_dot_t    = Q_dot_tr_walls.sum() + Q_dot_tr_wd + Q_dot_wl_ir[ind, :].sum() + Q_dot_wd_ir[ind]
        Q_dot_m    = C_in * dTin_t + C_wd * dTwd_t + (dTiw_t * C_iw).sum() + (dTmw_t * C_mw).sum() + (dTew_t * C_ew).sum() 
    
        # kWh per sec that are further multiplied by time_step (in sec) to be integrated as kWh
    
        dQh_t_kWh_per_s = Q_dot_h /1000 /3600 
        dQv_t_kWh_per_s = Q_dot_v /1000 /3600
        dQt_t_kWh_per_s = Q_dot_t /1000 /3600
        dQm_t_kWh_per_s = Q_dot_m /1000 /3600
    
        dTi_t = [dTin_t]
        dTi_t.extend([dTwd_t])
        dTi_t.extend(dTiw_t)
        dTi_t.extend(dTmw_t)
        dTi_t.extend(dTew_t)
        
        dTi_t.extend([dQh_t_kWh_per_s, dQv_t_kWh_per_s, dQt_t_kWh_per_s, dQm_t_kWh_per_s])

        return dTi_t

        
    method = 'LSODA'
    # method = 'BDF'
    # method = 'RK23'
    # method = 'Radau'
    # method = 'RK45'

    # ---- Time vector ----

    ind_start = 0
    ind_stop = len(df) - 1
    t_eval = np.unique(ts[ind_start:ind_stop + 1])
    t_span = (t_eval[0], t_eval[-1])

    # Intermittent heating
    
    R_array = solve_ivp(lambda t, y: model_ode(t, y, ind_start, ind_stop),
                        t_span, Ti0, t_eval=ts, method=method, max_step = time_step)
    Ti  = R_array.y.T
    
    Twi = Ti[:, 2 : 2 + n_op_walls]
    Twm = Ti[:, 2 + n_op_walls: 2 + 2 * n_op_walls]
    Twe = Ti[:, 2 + 2 * n_op_walls: 2 + 3 * n_op_walls]
    
    Qi  = Ti[:, -n_flows:]
    
    ind_start = df.index[0]
    ind_stop  = df.index[-1]
    
    d = pd.date_range(start = ind_start , end = ind_stop, freq='15min')

    dfr = pd.DataFrame(
        np.column_stack((Ti[:, 0], Ti[:, 1],  Qi)),
        columns=['t_in', 't_wd', \
                 'Qh_kWh',  'Qv_kWh', 'Qt_kWh', 'Qm_kWh'],
        index=d
    )
    
    dfr["Q_dot_h"] = dfr["Qh_kWh"].diff().fillna(0) * 1000 * 3600 / time_step
    dfr["Q_dot_v"] = dfr["Qv_kWh"].diff().fillna(0) * 1000 * 3600 / time_step
    dfr["Q_dot_t"] = dfr["Qt_kWh"].diff().fillna(0) * 1000 * 3600 / time_step
    dfr["Q_dot_m"] = dfr["Qm_kWh"].diff().fillna(0) * 1000 * 3600 / time_step

    dfr['Q_dot_h_Wm2'] = dfr['Q_dot_h'] / A_floor
    dfr['Qh_kWhm2']    = dfr['Qh_kWh'] / A_floor

    dfi = pd.concat([dfr, df], axis=1).copy()

    dfi['overcool_step'] = np.where((dfi['f_occ'] > 0) & (dfi['t_in'] < dfi['t_set_h'] - 0.5), 1, 0)
    dfi['fr_overcool']   = np.cumsum(dfi['overcool_step']) / (dfi['f_occ'] > 0).sum()
    

    # Continuous heating

    ind_start = 0
    ind_stop = len(df) - 1
    t_eval = np.unique(ts[ind_start:ind_stop + 1])
    t_span = (t_eval[0], t_eval[-1])

    t_set_min = t_set_heating
    t_set_h = np.ones(len(df)) * t_set_heating

    R_array = solve_ivp(lambda t, y: model_ode(t, y, ind_start, ind_stop),
                        t_span, Ti0, t_eval=ts, method=method, max_step = time_step)
    Ti  = R_array.y.T
    
    Twi = Ti[:, 3  : 2 + n_op_walls]
    Twm = Ti[:, 3 + n_op_walls: 3 + 2 * n_op_walls]
    Twe = Ti[:, 3 + 2 * n_op_walls: 3 + 3 * n_op_walls]
    
    Qi  = Ti[:, -n_flows:]
    
    ind_start = df.index[0]
    ind_stop  = df.index[-1]
    
    d = pd.date_range(start = ind_start , end = ind_stop, freq='15min')

    dfr = pd.DataFrame(
        np.column_stack((Ti[:, 0], Ti[:, 1],  Qi)),
        columns=['t_in', 't_wd', \
                 'Qh_kWh',  'Qv_kWh', 'Qt_kWh', 'Qm_kWh'],
        index=d
    )
    
    dfr["Q_dot_h"] = dfr["Qh_kWh"].diff().fillna(0) * 1000 * 3600 / time_step
    dfr["Q_dot_v"] = dfr["Qv_kWh"].diff().fillna(0) * 1000 * 3600 / time_step
    dfr["Q_dot_t"] = dfr["Qt_kWh"].diff().fillna(0) * 1000 * 3600 / time_step
    dfr["Q_dot_m"] = dfr["Qm_kWh"].diff().fillna(0) * 1000 * 3600 / time_step

    dfr['Q_dot_h_Wm2'] = dfr['Q_dot_h'] / A_floor
    dfr['Qh_kWhm2']    = dfr['Qh_kWh'] / A_floor
    
    dfc = pd.concat([dfr, df], axis=1).copy()

    dfc['overcool_step'] = np.where((dfc['f_occ'] > 0) & (dfc['t_in'] < dfc['t_set_h'] - 0.5), 1, 0)
    dfc['fr_overcool']   = np.cumsum(dfc['overcool_step']) / (dfc['f_occ'] > 0).sum()

    # remove timezone information
    df.index  = df.index.tz_localize(None)
    dfi.index = dfi.index.tz_localize(None)
    dfc.index = dfc.index.tz_localize(None)

    return df, dfi, dfc


def check_balance(dfr, time_step) :
    

    dfr["Qh_kWh"] = dfr["Q_dot_h"].cumsum() * time_step / 3600 / 1000
    dfr["Qv_kWh"] = dfr["Q_dot_v"].cumsum() * time_step / 3600 / 1000
    dfr["Qt_kWh"] = dfr["Q_dot_t"].cumsum() * time_step / 3600 / 1000
    dfr["Qm_kWh"] = dfr["Q_dot_m"].cumsum() * time_step / 3600 / 1000
    

    balance = pd.DataFrame({
        "heating": pd.Series(dtype='int'),
        "ventilation": pd.Series(dtype='int'),
        "transmission": pd.Series(dtype='int'),
        "thermal mass": pd.Series(dtype='int'),
        "unbalance": pd.Series(dtype='int'),
    })
    
    
    dfrb = dfr.copy()
    QH_kWh = dfrb["Qh_kWh"].values[-1] - dfrb["Qh_kWh"].values[0]
    QV_kWh = dfrb["Qv_kWh"].values[-1] - dfrb["Qv_kWh"].values[0]
    QT_kWh = dfrb["Qt_kWh"].values[-1] - dfrb["Qt_kWh"].values[0]
    QM_kWh = dfrb["Qm_kWh"].values[-1] - dfrb["Qm_kWh"].values[0]

    QU_kWh = QH_kWh  + QV_kWh + QT_kWh  - QM_kWh

    balance = pd.concat([balance, \
               pd.DataFrame({ \
                    "heating": QH_kWh,  \
                    "ventilation": QV_kWh, "transmission": QT_kWh, \
                    "thermal mass": - QM_kWh,  \
                    "unbalance": QU_kWh}, index=[0])])
    balance['unit'] = 'kWh'
    balance = balance.set_index('unit')

    return balance








    




