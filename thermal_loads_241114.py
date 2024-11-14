import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interpn
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

from math import sqrt, atan, log, exp, sin, cos, tan


pi = np.pi
rho_a = 1.2 #[kg/m^3]
c_p_a = 1000  #[J/kg-K] specific heat capacity of humid air per kg of dry air
sigma_b = 5.67 * 10**-8
T_ref = 273.15
t0 = 0
time_step = 600

hp_name = 'heat_production'

def composite_material(lstm, dfm, name12, name1, name2, L1, L2):
    f1 = L1 / (L1 + L2)
    f2 = 1 - f1
    lambda1, rho1, c1 = dfm.loc[dfm['material'] == name1, ['lambda', 'rho', 'c']].iloc[0]
    lambda2, rho2, c2 = dfm.loc[dfm['material'] == name2, ['lambda', 'rho', 'c']].iloc[0]
    lambda12 = f1 * lambda1 + f2 * lambda2
    rho12    = f1 * rho1 + f2 * rho2
    c12      = f1 * c1 + f2 * c2
    lstm.append({"material": name12, "lambda":lambda12, "rho":rho12, "c":c12})
    return lstm


def mysumdown(series):
    mysum = 0
    stop = 0
    for s in series:
        if s==0: 
            stop = 1
            break
        else:
            mysum = mysum + s
    if stop == 0:
        mysum = mysum/2
    return mysum

def mysumup(series):
    mysum = 0
    stop = 0
    for s in series[::-1]:
        if s==0:
            stop = 1
            break
        else:
            mysum = mysum + s
    if stop == 0:
        mysum = mysum/2
    return mysum


def VF_h_to_h(r,h) :
    if r >= 0: 
        VF = 0.5 * ( h**2 + 2*r**2 - h * (h**2 + 4 * r**2)**0.5 )/ r**2
    else :
        VF = 0
    return VF
    
    
def add_copy(name_orig, name_copy, walls_types) :
    wtl = walls_types + [(name_copy, t[1], t[2], t[3]) for t in walls_types    if t[0] == name_orig]
    return wtl


def add_inverse_copy(name_orig, name_copy, walls_types) :
    wtl = walls_types + [(name_copy,t[1],t[2],t[3]) for t in walls_types[::-1] if t[0] == name_orig]
    return wtl


def RC_calc(dfz, dfo, dfwle, dfwli, U_wd, U_dr) :
    
    dfRC = dfz[['zone', 't_in_star', 'vol_int', 'n_fl', 'area_fl', \
           'f_wd_0', 'f_dr_0', 'f_wd_90', 'f_dr_90', 'f_wd_180', 'f_dr_180', \
           'H_T_wd', 'H_T_dr', 'H_V_su', 'H_T', 'H_V', 'PHI_T', 'PHI_V']].copy()

    dfRC['area_h'] = dfRC['area_fl'].divide(dfRC['n_fl'])   .replace((np.inf, -np.inf, np.nan), (0, 0, 0)) # for each floor
    dfRC['h_ceil'] = dfRC['vol_int'].divide(dfRC['area_fl']).replace((np.inf, -np.inf, np.nan), (0, 0, 0)) # for each floor
    dfRC['r_fl']   = (dfRC['area_h']/pi)**0.5                    # for each floor
    dfRC['area_v'] = dfRC['h_ceil'] * 2 * np.pi * dfRC['r_fl']   # for each floor
    
    dfRC['VF_h_to_h'] = [VF_h_to_h(r, h) for r,h in zip(dfRC['r_fl'].values, dfRC['h_ceil'].values)] # for each floor
    dfRC['VF_h_to_v'] = (1 - dfRC['VF_h_to_h']).where(dfRC['VF_h_to_h'] <= 1, 0)
    dfRC['VF_v_to_h'] = (dfRC['VF_h_to_v'] * dfRC['area_h']).divide(dfRC['area_v']).replace((np.inf, -np.inf, np.nan), (0, 0, 0))
    dfRC['VF_v_to_v'] = (1 - 2 * dfRC['VF_v_to_h']).where(dfRC['VF_v_to_h'] <= 0.5, 0)
    
    dfRC = dfRC.drop(columns=[ 'h_ceil', 'r_fl'])
    
    dfRC = pd.merge(dfRC, dfo, on = "zone",  how='left')
    
    dfRC['H_V_sb'] = dfRC['H_V'] - dfRC['H_V_su'] + dfRC['f_fan_sb'] * dfRC['H_V_su'] 
    
    dfRC['C2']   = dfRC['vol_int'] * rho_a * c_p_a * 5
    dfRC['U0wd'] = dfRC['H_T_wd'] / (1-U_wd/8+U_wd/3)
    dfRC['fUwd'] = 1 / (3/U_wd-3/8+1)
    dfRC['U0dr'] = dfRC['H_T_dr'] / (1-U_dr/8+U_dr/3)
    dfRC['fUdr'] = 1 / (3/U_dr-3/8+1)
    dfRC['U0vo'] = dfRC['H_V'] 
    dfRC['U0vs'] = dfRC['H_V_sb']
    
    dfRC = dfRC.drop(columns=['f_fan_sb', 'H_T_wd', 'H_T_dr', 'H_V_su'])
    
    lstcol     = ['zone', 'area_wl', 'Ahi', 'Ahc', 'ACi', 'ACe', 'H_T_wl', 'f_wl_0', 'f_wl_90', 'f_wl_180', 'PHI_rad']
    lstdropcol = ['area_wl', 'Ahi', 'Ahc', 'ACi', 'ACe', 'H_T_wl', 'f_wl_0', 'f_wl_90', 'f_wl_180', 'PHI_rad']
    
    dfze  = dfwle[lstcol].groupby(by=['zone'],as_index=False).agg('sum')
    dfzi  = dfwli[lstcol].groupby(by=['zone'],as_index=False).agg('sum')
    
    dfRC = pd.merge(dfRC, dfze[lstcol], on = "zone",  how='left').replace(np.nan, 0)
    dfRC['U2']      = dfRC['Ahc'] 
    dfRC['C1']      = dfRC['ACi']
    dfRC['U1'] = (dfRC['H_T_wl']*dfRC['Ahi']).divide(dfRC['Ahi']-dfRC['H_T_wl']).replace((np.inf, -np.inf, np.nan), (0, 0, 0))
    dfRC['fT1_0']   = dfRC['f_wl_0']
    dfRC['fT1_90']  = dfRC['f_wl_90']
    dfRC['fT1_180'] = dfRC['f_wl_180']
    dfRC['Q1_rad']  = 0
    dfRC = dfRC.drop(columns=lstdropcol)
    
    dfRC = pd.merge(dfRC, dfzi[lstcol], on = "zone",  how='left').replace(np.nan, 0)
    dfRC['U3']      = dfRC['Ahc'] 
    dfRC['C3']      = dfRC['ACi']
    dfRC['fT3_0']   = dfRC['f_wl_0']
    dfRC['fT3_90']  = dfRC['f_wl_90']
    dfRC['fT3_180'] = dfRC['f_wl_180']
    dfRC['Q3_rad']  = dfRC['PHI_rad']
    dfRC = dfRC.drop(columns=lstdropcol)
    
    dfRC['c_eff_Whm3K'] = (dfRC['C1'] + dfRC['C3']).divide(3600 *  dfRC['vol_int']).replace((np.inf, -np.inf, np.nan), (0, 0, 0))
    dfRC['tau_0_h']     = (dfRC['C1'] + dfRC['C3']).divide(3600 * (dfRC['H_T'] + dfRC['H_V_sb'])).replace((np.inf, -np.inf, np.nan), (0, 0, 0))
    
    return dfRC

def Qrh_calc(i, dfRC, dfh, dfd, dfw, rh_type, t_ext) : 
    
    area_fl     = dfRC.iloc[i]['area_fl']
    V_int       = dfRC.iloc[i]['vol_int'] 
    t_in_star   = dfRC.iloc[i]['t_in_star'] 
    t_set_min   = dfRC.iloc[i]['t_set_min'] 
    n_sb        = dfRC.iloc[i]['H_V_sb'] * 3600 / rho_a / c_p_a / V_int
    tau_0_h     = dfRC.iloc[i]['tau_0_h']
    c_eff_Whm3K = max(15, min(dfRC.iloc[i]['c_eff_Whm3K'], 50))
    time_ni_rh_h = dfRC.iloc[i]['time_night_rh_h']
    time_we_rh_h = dfRC.iloc[i]['time_we_rh_h']  
    time_hl_rh_h = dfRC.iloc[i]['time_hol_rh_h']
    sum_f_on_h  = np.asarray(dfh.iloc[i]['f_occ_h']).sum()
    sum_f_on_d  = np.asarray(dfd.iloc[i]['f_occ_d']).sum()
    sum_f_on_w  = np.asarray(dfw.iloc[i]['f_occ_w']).sum()
    time_ni_sb_h = max(0, 24 - sum_f_on_h)
    time_we_sb_h = max(0, 24*(7-sum_f_on_d) + 24 - sum_f_on_h)  
    time_hl_sb_h = max(0, 7 * 24 * (9-sum_f_on_w) + 24 - sum_f_on_h)
    Qht, Qhv     = dfRC.iloc[i][['PHI_T', 'PHI_V']]

    if rh_type == 0: 
        f_over_max  = dfRC.iloc[i]['f_oversize_imposed']
        Qrh      = max(0, (f_over_max - 1)) * (Qht + Qhv)

    elif rh_type == 1: 

        n_inf_sb    = max(0.1, min(n_sb, 0.5 ))
        time_sb_h_1 = max(  8, min(time_ni_sb_h , 168 ))
        time_sb_h_2 = max(  8, min(time_we_sb_h , 168 ))
        time_sb_h_3 = max(  8, min(time_hl_sb_h , 168 ))
        time_rh_h_1 = max(0.5, min(time_ni_rh_h , 12  ))
        time_rh_h_2 = max(0.5, min(time_we_rh_h , 12  ))
        time_rh_h_3 = max(0.5, min(time_hl_rh_h , 12  ))
        
        c,d,n,t  = ([15, 50], [8, 14, 62, 168], [0.1, 0.5], [0.5, 1, 2, 3, 4, 6, 12])
        points = (c,d,n,t)
        values = [[[[sum(x for tpl, x in frh_data_F1 if tpl==(cv, dv, nv, tv)) for tv in t] for nv in n] for dv in d] for cv in c]
        frh1 = interpn(points, values, np.array([c_eff_Whm3K, time_sb_h_1, n_inf_sb, time_rh_h_1])) if time_ni_sb_h > 0 else 0
        frh2 = interpn(points, values, np.array([c_eff_Whm3K, time_sb_h_2, n_inf_sb, time_rh_h_2])) if time_we_sb_h > 0 else 0
        frh3 = interpn(points, values, np.array([c_eff_Whm3K, time_sb_h_3, n_inf_sb, time_rh_h_3])) if time_hl_sb_h > 0 else 0
        Qrh = np.max([frh1, frh2, frh3]) * area_fl
        
        # print(frh1, frh2, frh3)

    elif rh_type == 2:

        dt_sb_max = max(0, t_in_star - t_set_min)
        n_inf_sb  = max(0.1, min(n_sb, 0.5 )) 
        time_h_sb_h = max(0, time_ni_sb_h - time_ni_rh_h)
        time_d_sb_h = max(0, time_we_sb_h - time_we_rh_h)
        time_w_sb_h = max(0, time_hl_sb_h - time_hl_rh_h)
        dt_sb_1   = (t_in_star - t_ext) * (1 - exp(-time_h_sb_h / tau_0_h))
        dt_sb_2   = (t_in_star - t_ext) * (1 - exp(-time_d_sb_h / tau_0_h))
        dt_sb_3   = (t_in_star - t_ext) * (1 - exp(-time_w_sb_h / tau_0_h))
        dt_sb_1   = min(dt_sb_1, dt_sb_max)
        dt_sb_2   = min(dt_sb_2, dt_sb_max)
        dt_sb_3   = min(dt_sb_3, dt_sb_max)
        dtemp_sb_1  = max(  1, min(dt_sb_1 , 5 ))
        dtemp_sb_2  = max(  1, min(dt_sb_2 , 5 ))
        dtemp_sb_3  = max(  1, min(dt_sb_3 , 5 ))
        time_rh_h_1 = max(0.5, min(time_ni_rh_h , 4 ))
        time_rh_h_2 = max(0.5, min(time_we_rh_h , 4 ))
        time_rh_h_3 = max(0.5, min(time_hl_rh_h , 4 ))
        
        c,d,n,t  = ([15, 50], [1, 2, 3, 4, 5], [0.1, 0.5], [0.5, 1, 2, 3, 4, 6, 12])
        points = (c,d,n,t)
        values = [[[[sum(x for tpl, x in frh_data_F3 if tpl==(cv, dv, nv, tv)) for tv in t] for nv in n] for dv in d] for cv in c]
        frh1 = interpn(points, values, np.array([c_eff_Whm3K, dtemp_sb_1, n_inf_sb , time_rh_h_1])) if time_ni_sb_h > 0 else 0
        frh2 = interpn(points, values, np.array([c_eff_Whm3K, dtemp_sb_2, n_inf_sb , time_rh_h_2])) if time_we_sb_h > 0 else 0
        frh3 = interpn(points, values, np.array([c_eff_Whm3K, dtemp_sb_3, n_inf_sb , time_rh_h_3])) if time_hl_sb_h > 0 else 0
        Qrh = np.max([frh1, frh2, frh3]) * area_fl

        # print(frh1, frh2, frh3)

    return Qrh


def Qrh_hp_calc(dfRC, time_sb_hp_h, time_rh_hp_h,  f_oversize_hp, rh_type, t_ext, t_in) : 

    i = 0
    
    area_fl     = dfRC.iloc[i]['area_fl']
    V_int       = dfRC.iloc[i]['vol_int'] 
    n_sb        = dfRC.iloc[i]['H_V_sb'] * 3600 / rho_a / c_p_a / V_int
    tau_0_h     = dfRC.iloc[i]['tau_0_h']
    c_eff_Whm3K = max(15, min(dfRC.iloc[i]['c_eff_Whm3K'], 50))

    if rh_type == 0: 
        Qht, Qhv   = dfRC.iloc[i][['PHI_T', 'PHI_V']]  
        Qrh        = max(0, (f_oversize_hp - 1)) * (Qht + Qhv)

    elif rh_type == 1: 

        n_inf_sb    = max(0.1, min(n_sb, 0.5 ))
        time_sb_h = max(  8, min(time_sb_hp_h , 168 ))
        time_rh_h = max(0.5, min(time_rh_hp_h , 12  ))
        
        c,d,n,t  = ([15, 50], [8, 14, 62, 168], [0.1, 0.5], [0.5, 1, 2, 3, 4, 6, 12])
        points = (c,d,n,t)
        values = [[[[sum(x for tpl, x in frh_data_F1 if tpl==(cv, dv, nv, tv)) for tv in t] for nv in n] for dv in d] for cv in c]
        frh = interpn(points, values, np.array([c_eff_Whm3K, time_sb_h, n_inf_sb, time_rh_h])) if time_sb_hp_h > 0 else 0
        Qrh = frh * area_fl
        
#         print(frh, Qrh, type(Qrh))

    elif rh_type == 2:

        n_inf_sb  = max(0.1, min(n_sb, 0.5 )) 
        time_sb_h = max(0, time_sb_hp_h - time_rh_hp_h)
        dt_sb     = (t_in - t_ext) * (1 - exp(-time_sb_h / tau_0_h))
        dtemp_sb  = max(  1, min(dt_sb , 5 ))
        time_rh_h = max(0.5, min(time_sb_h , 4 ))
        
        c,d,n,t  = ([15, 50], [1, 2, 3, 4, 5], [0.1, 0.5], [0.5, 1, 2, 3, 4, 6, 12])
        points = (c,d,n,t)
        values = [[[[sum(x for tpl, x in frh_data_F3 if tpl==(cv, dv, nv, tv)) for tv in t] for nv in n] for dv in d] for cv in c]
        frh = interpn(points, values, np.array([c_eff_Whm3K, dtemp_sb, n_inf_sb , time_rh_h])) if time_sb_hp_h > 0 else 0
        Qrh = frh * area_fl

#         print(frh, Qrh, type(Qrh))

    return Qrh
    

def facade_calc(dftraz, dfsortz, dfv, dfz, DUwalls, t_in, t_ext) :

    dftraz['H_T'] = (dftraz['H_T_wl'] + dftraz['H_T_wd'] + dftraz['H_T_dr']).where((dftraz['PHI_rad'] == 0), 0)

    dftraz  = dftraz.drop(columns=[ 'H_T_wl', 'H_T_wd', 'H_T_dr'])
    dftraz  = dftraz.groupby    (by=['zone', 'azimuth', 'slope', 'wall_type'],as_index=False).agg('sum')
    dftraz  = dftraz.sort_values(by=['zone', 'azimuth', 'slope', 'wall_type'])
    
    dftraz['area'] = dftraz['area_wl'] + dftraz['area_wd'] + dftraz['area_dr']
    
    # Facade walls belonging to all zones 
    
    dftrazv  = dftraz.loc[(dftraz['slope'] == 90) & (dftraz['wall_type'].isin(DUwalls))].copy()
    dftrazvz = dftrazv.drop(columns=['azimuth','slope','wall_type']).groupby(by=['zone'],as_index=False).agg('sum') 
    
    list_facade_z   = dftrazvz['zone'].tolist()
    dfsortfacz      = dfsortz.loc[(dfsortz['zone'].isin(list_facade_z))].copy()
    
    area_fac_tot      = dftrazvz['area'].sum() 
    dftrazvz['f_fac'] = dftrazvz['area'].divide(area_fac_tot).replace((np.inf, -np.inf, np.nan), (0, 0, 0))
    
    dftrazv           = pd.merge(dftrazv, dftrazvz[['zone', 'area', 'f_fac']], on = "zone",  how='left')
    dftrazv['f_az']   = dftrazv['area_x'].divide(dftrazv['area_y']).replace((np.inf, -np.inf, np.nan), (0, 0, 0))
    dftrazv           = dftrazv.drop(columns=['area_x','area_y'])
    
    # Not facade walls belonging to zones that do not include facade walls
    
    dftrazc     = dftraz.loc[((dftraz['slope'] != 90) | (~dftraz['wall_type'].isin(DUwalls))) & (~dftraz['zone'].isin(list_facade_z))].copy()
    dftrazcz    = dftrazc.drop(columns=['azimuth','slope','wall_type','area']).groupby(by=['zone'],as_index=False).agg('sum')
    area_wl_c   = dftrazcz['area_wl'].sum()
    area_wd_c   = dftrazcz['area_wd'].sum()
    area_dr_c   = dftrazcz['area_dr'].sum()
    H_T_c       = dftrazcz['H_T'].sum()
    
    dftrazcz    = pd.merge(dftrazcz, dfv[['zone', 'area_fl', 'H_V_su', 'H_V_ATD', 'H_V_ia', 'H_V_ie']], on = "zone",  how='left')
    dftrazcz    = pd.merge(dftrazcz, dfz[['zone', 'PHI_R']], on = "zone",  how='left')
    area_fl_c   = dftrazcz['area_fl'].sum()
    H_V_su_c    = dftrazcz['H_V_su'].sum()
    H_V_ie_c    = dftrazcz['H_V_ie'].sum()
    PHI_R_c     = dftrazcz['PHI_R'].sum()
    
    # Not facade walls belonging to zones that include facade walls
    
    dftrazh  = dftraz.loc[((dftraz['slope'] != 90) | (~dftraz['wall_type'].isin(DUwalls))) & (dftraz['zone'].isin(list_facade_z))].copy()
    dftrazhz = dftrazh.drop(columns=['azimuth','slope','wall_type','area']).groupby(by=['zone'],as_index=False).agg('sum')
    dftrazv  = pd.merge(dftrazv, dftrazhz, on = "zone",  how='left')
    
    dftrazv['area_wl'] = dftrazv['area_wl_x'] + dftrazv['f_az'] * dftrazv['area_wl_y'] + dftrazv['f_fac'] * dftrazv['f_az'] * area_wl_c
    dftrazv['area_wd'] = dftrazv['area_wd_x'] + dftrazv['f_az'] * dftrazv['area_wd_y'] + dftrazv['f_fac'] * dftrazv['f_az'] * area_wd_c
    dftrazv['area_dr'] = dftrazv['area_dr_x'] + dftrazv['f_az'] * dftrazv['area_dr_y'] + dftrazv['f_fac'] * dftrazv['f_az'] * area_dr_c
    dftrazv['H_T']     = dftrazv['H_T_x']     + dftrazv['f_az'] * dftrazv['H_T_y']     + dftrazv['f_fac'] * dftrazv['f_az'] * H_T_c
    
    
    dftrazv = dftrazv[['zone', 'azimuth', 'area_wl', 'area_wd', 'area_dr', 'H_T', 'f_fac', 'f_az']]
    dftrazv = pd.merge(dftrazv, dfv[['zone', 'area_fl', 'H_V_su', 'H_V_ATD', 'H_V_ia', 'H_V_ie']], on = "zone",  how='left')
    dftrazv = pd.merge(dftrazv, dfz[['zone', 'PHI_R']], on = "zone",  how='left')
    
    dftrazv['area_fl'] = dftrazv['f_az'] * dftrazv['area_fl'] + dftrazv['f_fac'] * dftrazv['f_az'] * area_fl_c
    dftrazv['H_V_su']  = dftrazv['f_az'] * dftrazv['H_V_su']  + dftrazv['f_fac'] * dftrazv['f_az'] * H_V_su_c
    dftrazv['H_V_ie']  = dftrazv['f_az'] * dftrazv['H_V_ie']  + dftrazv['f_fac'] * dftrazv['f_az'] * H_V_ie_c
    dftrazv['PHI_R']   = dftrazv['f_az'] * dftrazv['PHI_R']   + dftrazv['f_fac'] * dftrazv['f_az'] * PHI_R_c
    
    dftrazv = dftrazv.drop(columns = ['f_fac', 'f_az'])
    
    dfresa =  pd.merge(dfsortfacz, dftrazv, on = 'zone', how='left')
    
    dfresa['H_V'] = dfresa['H_V_su'] + dfresa['H_V_ATD'] + dfresa['H_V_ia'] + dfresa['H_V_ie']
    
    DELTAT             = (t_in - t_ext)
    dfresa['PHI_T (W)'] = dfresa['H_T'] * DELTAT
    dfresa['PHI_V (W)'] = dfresa['H_V'] * DELTAT
    
    dfresa = dfresa.rename(columns={'PHI_R':'PHI_R (W)'})
    dfresa = dfresa[['zone', 'azimuth', 'PHI_T (W)', 'PHI_V (W)', 'PHI_R (W)']]
    dfresa['PHI_total (W)'] = dfresa[['PHI_T (W)', 'PHI_V (W)', 'PHI_R (W)']].sum(axis=1)
    
    dfresa = dfresa.groupby(by=['zone','azimuth']).agg('sum')
    
    
    dfresa = dfresa.rename(columns={ 'PHI_T (W)':'Transmission [W]','PHI_V (W)':'Ventilation [W]','PHI_R (W)':'Reheating [W]',\
                             'PHI_total (W)':'Total [W]'}).astype(int)
    
    dfresa = dfresa.reset_index(level='azimuth')
    
    dfresa =  pd.merge(dfsortfacz, dfresa, on = 'zone', how='left')
    dfresa = dfresa.set_index('zone')

    return dfresa



def thermal_loads(gen, vent, setback, hourly, daily, weekly, \
                  walls, windows, doors, materials, walls_types,  \
                  GRwalls, DUwalls, BEwalls, INTwalls):
    
    [t_ext, t_in, t_min, t_avg, t_in, DU, U_wd, U_dr, n_50, rh_type, time_setback_hp_h, time_reheat_hp_h, f_oversize_hp] = gen

    n_min = 0.5
    
    dfv = pd.DataFrame(vent, columns=('zone', 't_in_star', 'n_fl', 'area_fl', 'vol_int',  \
                                  'q_su_m3h', 'q_ex_m3h', 'epsilon_rec', 'q_ATD_m3h', 'q_ia_m3h', 't_ia' ))
    
    dfv['Vti']          = dfv['t_in_star'] * dfv['vol_int']
    dfv['q_leak_m3h']   = 0.1 * n_50  * dfv['vol_int']
    dfv['q_min_m3h']    = n_min * dfv['vol_int']
    dfv['q_ihalf_m3h']  = 0.5 * (dfv['q_ex_m3h'] - dfv['q_su_m3h']  - dfv['q_ATD_m3h'])
    dfv['q_infadd_m3h'] = dfv['q_ihalf_m3h'].where(dfv['q_ihalf_m3h'] > 0, 0)
    dfv['q_lack_m3h']   = dfv['q_min_m3h'] - dfv['q_su_m3h']   - dfv['q_ATD_m3h'] \
                        - dfv['q_ia_m3h']  - dfv['q_leak_m3h'] - dfv['q_infadd_m3h']
    dfv['q_gap_m3h']    = dfv['q_lack_m3h'].where(dfv['q_lack_m3h'] > 0, 0)
    dfv['t_rec']        = t_ext + dfv['epsilon_rec'] * (dfv['t_in_star'] - t_ext)
    dfv['H_V_su']       = c_p_a * rho_a * dfv['q_su_m3h'] /3600 * (dfv['t_in_star']-dfv['t_rec'])/(t_in - t_ext)
    dfv['H_V_ATD']      = c_p_a * rho_a * dfv['q_ATD_m3h']/3600 * (dfv['t_in_star']-t_ext)/(t_in - t_ext)
    dfv['H_V_ia']       = c_p_a * rho_a * dfv['q_ia_m3h']/3600 * (dfv['t_in_star']-dfv['t_ia'])/(t_in - t_ext)
    dfv['H_V_ie']       = c_p_a * rho_a * (dfv['q_leak_m3h'] + dfv['q_infadd_m3h'] + dfv['q_gap_m3h']) /3600 \
                        * (dfv['t_in_star'] - t_ext)/(t_in - t_ext)
    
    dfv['H_V']          =  dfv['H_V_su'] + dfv['H_V_ATD'] + dfv['H_V_ia'] + dfv['H_V_ie']
    dfv['PHI_V']        =  dfv['H_V']   * (t_in - t_ext)
    
    dfsortz = dfv[['zone']]
    
    dfv['t_in_star'] = dfv['Vti'].divide(dfv['vol_int']).replace((np.inf, -np.inf, np.nan), (0, 0, 0))
    dfv              = dfv[['zone', 't_in_star', 'n_fl', 'area_fl',  'vol_int', 'H_V_su', 'H_V_ATD', 'H_V_ia', 'H_V_ie', 'H_V', 'PHI_V']]




    walls_types = pd.DataFrame(walls_types, columns=('wall_type', 'layer', 'thickness', 'material'))

    dfwl        = pd.merge(walls_types, materials, on = 'material', how='left')
    dfwl['R']   = dfwl['thickness'] / dfwl['lambda']
    dfwl['C']   = dfwl['thickness'] * dfwl['rho'] * dfwl['c']
    dfwl['M']   = dfwl['thickness'] * dfwl['rho']
    dfwl['Ci']  = dfwl['C'] .where((dfwl['lambda'] >= 0.1) & (dfwl['rho'] >= 2) , 0)
    dfwl['Ce']  = dfwl['Ci'].copy()
    
    dfwl = dfwl.groupby(by='wall_type',as_index=False).agg({'thickness':'sum','R':'sum', 'M':'sum', 'Ci':mysumdown,'Ce':mysumup})
    


    wl = pd.DataFrame(walls, columns=('zone', 'azimuth', 'slope', 'wall_type', 'dim1', 'dim2', 'h_in', 'h_out', 't_out', 'f_rad'))

    wl['gross_area'] = wl['dim1'] * wl['dim2']
    wl = wl.drop(columns = ['dim1', 'dim2'])
    wl = pd.merge(wl, dfwl, on = "wall_type",  how='left')
    wl = pd.merge(wl, dfv[['zone', 't_in_star']], on = "zone",  how='left')
    
    wl['U_wl']    = 1 / (1/wl['h_in'] + wl['R'] + 1/wl['h_out'])
    wl.loc[ wl['f_rad']  > 0, 'f_temp' ] = 0
    wl.loc[ wl['f_rad'] <= 0, 'f_temp' ] = (wl['t_in_star'] - wl['t_out']) / (t_in - t_ext)
    
    wl['Ahi']     = wl['h_in']   * wl['gross_area']
    wl['AU_wl']   = wl['U_wl']   * wl['gross_area']
    wl['Af_temp'] = wl['f_temp'] * wl['gross_area']
    
    wlres = wl[['wall_type', 'thickness', 'U_wl', 'M', 'Ci', 'Ce']]
    
    wl = wl[['zone', 'azimuth', 'slope', 'wall_type', 'gross_area', 'Ci', 'Ce', 'AU_wl', 'Af_temp', 'Ahi', 'f_rad']]
    
    wl = \
    wl.groupby(by=['zone','azimuth','slope','wall_type'],as_index = False).agg({'gross_area':'sum', 'Ci':'mean', 'Ce':'mean',\
                                                  'Ahi':'sum', 'AU_wl':'sum', 'Af_temp':'sum', 'f_rad':'sum'})
    
    wl['h_in']    =  wl['Ahi']    .divide(wl['gross_area']).replace((np.inf, -np.inf, np.nan), (0, 0, 0))
    wl['U_wl']    =  wl['AU_wl']  .divide(wl['gross_area']).replace((np.inf, -np.inf, np.nan), (0, 0, 0))
    wl['f_temp']  =  wl['Af_temp'].divide(wl['gross_area']).replace((np.inf, -np.inf, np.nan), (0, 0, 0))
    
    wl = wl[['zone', 'azimuth', 'slope', 'wall_type',  'gross_area', 'h_in', 'Ci', 'Ce', 'U_wl', 'f_temp', 'f_rad']]
    
    
    wl.loc[ wl['wall_type'].isin(DUwalls), 'DU' ] = DU
    wl.loc[~wl['wall_type'].isin(DUwalls), 'DU' ] = 0
    wl.loc[ wl['wall_type'].isin(GRwalls), 'fgr'] = 1.15 * 1.45
    wl.loc[~wl['wall_type'].isin(GRwalls), 'fgr'] = 1
    
    
    wd = pd.DataFrame(windows, columns=('zone', 'azimuth', 'slope', 'wall_type', 'number', 'breadth', 'height'))
    dr = pd.DataFrame(doors,   columns=('zone', 'azimuth', 'slope', 'wall_type', 'number', 'breadth', 'height'))
    
    wd['area_wd'] = wd['number'] * wd['breadth'] * wd['height']
    dr['area_dr'] = dr['number'] * dr['breadth'] * dr['height']
    
    wd = wd.groupby(by=['zone', 'azimuth', 'slope','wall_type'],as_index = False).agg({'area_wd':'sum'})
    dr = dr.groupby(by=['zone', 'azimuth', 'slope','wall_type'],as_index = False).agg({'area_dr':'sum'})


    dftr = pd.merge(wl, wd, how="outer")
    dftr = pd.merge(dftr, dr, how="outer")
    dftr = dftr.fillna(0)
    
    dftr['area_wl'] =  (dftr['gross_area'] - dftr['area_wd'] - dftr['area_dr']).clip(lower = 0)
    dftr            =  dftr.drop(columns=['gross_area'])
    dftr['U_wl+DU'] =  dftr['U_wl'] + dftr['DU']
    dftr['H_T_wl']  =  dftr['area_wl'] * (dftr['U_wl'] + dftr['DU']) * dftr['f_temp'] * dftr['fgr']
    dftr['H_T_wd']  =  dftr['area_wd'] * U_wd
    dftr['H_T_dr']  =  dftr['area_dr'] * U_dr
    dftr['ACi']     =  dftr['area_wl'] * dftr['Ci']
    dftr['ACe']     =  dftr['area_wl'] * dftr['Ce']
    dftr['Ahi']     =  dftr['area_wl'] * dftr['h_in']
    dftr['Ahc']     = (dftr['area_wl'] * 3).where(dftr['f_rad'] == 0 , dftr['area_wl'] * 5)
    dftr['PHI_rad'] =  dftr['f_rad'] * dftr['area_wl']
    
    dftr.loc[  dftr['slope'] <=  45, 'slope' ] = 0
    dftr.loc[ (dftr['slope'] >  45) & (dftr['slope'] <  135), 'slope' ] = 90
    dftr.loc[  dftr['slope'] >= 135, 'slope' ] = 180
    
    dftr   = dftr.sort_values(by=['zone', 'azimuth', 'slope', 'wall_type'])
    
    dftraz = dftr[['zone', 'azimuth', 'slope', 'wall_type', 'f_rad', 'PHI_rad',  'Ahi', 'Ahc', 'ACi', 'ACe', \
               'area_wl', 'area_wd', 'area_dr', 'H_T_wl', 'H_T_wd', 'H_T_dr']].copy()
    
    dftr   = dftr[['zone',            'slope', 'wall_type', 'f_rad', 'PHI_rad',  'Ahi', 'Ahc', 'ACi', 'ACe',  \
               'area_wl', 'area_wd', 'area_dr', 'H_T_wl', 'H_T_wd', 'H_T_dr']].copy()

    dftr   = dftr.groupby    (by=['zone', 'slope', 'wall_type'],as_index=False).agg('sum')
    dftr   = dftr.sort_values(by=['zone', 'slope', 'wall_type'])
    
    dftr['H_T'] = (dftr['H_T_wl'] + dftr['H_T_wd']  + dftr['H_T_dr']).where(dftr['f_rad'] == 0, 0)



    lstcol          = ['zone', 'slope', 'area_wl', 'area_wd', 'area_dr']
    dfzs            = dftr[lstcol].groupby(by=['zone', 'slope'], as_index=False).agg('sum')
    dfzs['area_sl'] = dfzs['area_wl'] + dfzs['area_wd'] + dfzs['area_dr']
    dfzs            = dfzs.drop(columns=['area_wl', 'area_wd', 'area_dr'])
    
    listsl = [0, 90, 180]
    for i in listsl:
        dftr    = pd.merge(dftr, dfzs[dfzs['slope'] == i], on = ['zone', 'slope'],  how='left').fillna(0)
        dftr['f_wl_'+ str(i)]    =  dftr['area_wl'].divide(dftr['area_sl']).replace((np.inf, -np.inf, np.nan), (0, 0, 0))
        dftr['f_wd_'+ str(i)]    =  dftr['area_wd'].divide(dftr['area_sl']).replace((np.inf, -np.inf, np.nan), (0, 0, 0))
        dftr['f_dr_'+ str(i)]    =  dftr['area_dr'].divide(dftr['area_sl']).replace((np.inf, -np.inf, np.nan), (0, 0, 0))
        dftr['area_sl_'+ str(i)] =  dftr['area_sl']
        dftr                     =  dftr.drop(columns=['area_sl'])
    
    conde = (~dftr['wall_type'].isin(INTwalls)) & (dftr['PHI_rad'] == 0)
    condi = ( dftr['wall_type'].isin(INTwalls)) | (dftr['PHI_rad'] >  0)
    
    dfwle = dftr.loc[conde]
    dfwli = dftr.loc[condi]

    # Heat production hp

    dftraz_hp = dftraz.copy()
    dftraz_hp.loc[dftraz_hp['wall_type'].isin(BEwalls), 'H_T_wl' ] = 0
    dftraz_hp.loc[dftraz_hp['wall_type'].isin(BEwalls), 'H_T_wd' ] = 0
    dftraz_hp.loc[dftraz_hp['wall_type'].isin(BEwalls), 'H_T_dr' ] = 0
    dftraz_hp.loc[dftraz_hp['wall_type'].isin(BEwalls), 'H_T' ] = 0

    dftr_hp = dftr.copy()
    dftr_hp.loc[dftr_hp['wall_type'].isin(BEwalls), 'H_T_wl' ] = 0
    dftr_hp.loc[dftr_hp['wall_type'].isin(BEwalls), 'H_T_wd' ] = 0
    dftr_hp.loc[dftr_hp['wall_type'].isin(BEwalls), 'H_T_dr' ] = 0
    dftr_hp.loc[dftr_hp['wall_type'].isin(BEwalls), 'H_T' ] = 0

    
    conde_hp = (~dftr_hp['wall_type'].isin(INTwalls)) & (dftr_hp['PHI_rad'] == 0) & (~dftr_hp['wall_type'].isin(BEwalls))
    condi_hp = ( dftr_hp['wall_type'].isin(INTwalls)) | (dftr_hp['PHI_rad'] >  0) | ( dftr_hp['wall_type'].isin(BEwalls))
    
    dfwle_hp = dftr_hp.loc[conde_hp]
    dfwli_hp = dftr_hp.loc[condi_hp]


    lstcol_tr = ['zone', 'PHI_rad', 'area_wl', 'area_wd', 'area_dr', 'H_T_wl', 'H_T_wd', 'H_T_dr', 'H_T', \
             'f_wl_0', 'f_wd_0', 'f_dr_0', 'f_wl_90', 'f_wd_90', 'f_dr_90', 'f_wl_180', 'f_wd_180', 'f_dr_180']
    
    lstcol_v = ['zone', 't_in_star', 'vol_int', 'n_fl', 'area_fl', 'H_V_su', 'H_V_ATD', 'H_V_ia', 'H_V_ie', 'H_V', 'PHI_V']
    
    dfz = dftr[lstcol_tr].groupby(by=['zone'],as_index=False).agg('sum').copy()
    dfz = pd.merge(dfsortz, dfz, on = 'zone', how='left')
    dfz = pd.merge(dfz, dfv[lstcol_v], on = 'zone',  how='left')

    dfz['PHI_T']   =  dfz['H_T'] * (t_in - t_ext)

    dfz_hp = dftr_hp[lstcol_tr].groupby(by=['zone'],as_index=False).agg('sum').copy()
    dfz_hp = pd.merge(dfsortz, dfz_hp, on = 'zone', how='left')
    dfz_hp = pd.merge(dfz_hp, dfv[lstcol_v], on = 'zone',  how='left')
    
    dfz_hp['PHI_T'] =  dfz_hp['H_T']   * (t_in - t_ext)

    dfz_hp_row = dfz_hp.drop(columns=['zone']).agg({
        'PHI_rad': 'sum', 'area_wl': 'sum', 'area_wd': 'sum',  'area_dr': 'sum', 
        'H_T_wl': 'sum', 'H_T_wd': 'sum', 'H_T_dr': 'sum', 'H_T': 'sum',
        'f_wl_0': 'mean', 'f_wd_0': 'mean', 'f_dr_0': 'mean', 'f_wl_90': 'mean', 'f_wd_90': 'mean', 'f_dr_90': 'mean', 
        'f_wl_180': 'mean', 'f_wd_180': 'mean', 'f_dr_180': 'mean', 
        't_in_star': 'mean', 'vol_int': 'sum', 'n_fl': 'sum', 'area_fl': 'sum', 
        'H_V_su': 'sum', 'H_V_ATD': 'sum', 'H_V_ia': 'sum', 'H_V_ie': 'sum',
        'H_V': 'sum', 'PHI_V': 'sum', 'PHI_T': 'sum'
          })
    dfz_hp_row = pd.DataFrame([dfz_hp_row])
    dfz_hp_row.insert(loc=0, column='zone', value=[hp_name])
    

    dfzg = dfz[['zone', 'H_T_wl', 'H_T_wd', 'H_T_dr', 'H_V_su', 'H_V_ATD', 'H_V_ia', 'H_V_ie']] # graph
    dfzg = dfzg.set_index('zone')
    
    
    dfh = pd.DataFrame(hourly , columns=('zone', 'f_occ_h'))
    dfd = pd.DataFrame(daily  , columns=('zone', 'f_occ_d'))
    dfw = pd.DataFrame(weekly , columns=('zone', 'f_occ_w'))
    
    dfh = pd.merge(dfsortz, dfh, on = 'zone', how='left')
    dfd = pd.merge(dfsortz, dfd, on = 'zone', how='left')
    dfw = pd.merge(dfsortz, dfw, on = 'zone', how='left')
    
    dfo = pd.DataFrame(setback, columns=('zone', 't_set_min', 'time_night_rh_h', 'time_we_rh_h', 'time_hol_rh_h', 'f_oversize_imposed', 'f_fan_sb'))

    dfRC    = RC_calc(dfz, dfo, dfwle, dfwli, U_wd, U_dr)
    dfRC_hp = RC_calc(dfz_hp, dfo, dfwle_hp, dfwli_hp, U_wd, U_dr)
    
    dfRC_hp_row = dfRC_hp[['zone', 'area_fl', 'vol_int', 'C1', 'C3', 'H_T', 'H_V_sb', 'PHI_T', 'PHI_V' ]]
    dfRC_hp_row = dfRC_hp_row.drop(columns=['zone']).sum()
    dfRC_hp_row = pd.DataFrame([dfRC_hp_row])
    dfRC_hp_row.insert(loc=0, column='zone', value=[hp_name])
    
    dfRC_hp_row['c_eff_Whm3K'] = (dfRC_hp_row['C1'] + dfRC_hp_row['C3']).divide(3600 *  dfRC_hp_row['vol_int']).replace((np.inf, -np.inf, np.nan), (0, 0, 0))
    dfRC_hp_row['tau_0_h'] = (dfRC_hp_row['C1'] + dfRC_hp_row['C3']).divide(3600 * (dfRC_hp_row['H_T'] + dfRC_hp_row['H_V_sb'])).replace((np.inf, -np.inf, np.nan), (0, 0, 0))


    
    zname_list = [] 
    Qrh   = np.zeros(len(dfRC))
    
    for i in range(0, len(dfRC)):
        zname       = dfRC.iloc[i]['zone']
        zname_list.append(zname)
        
        Qrh[i] = Qrh_calc(i, dfRC, dfh, dfd, dfw, rh_type, t_ext)
        
        dfRC.loc[ (dfRC['zone']== zname), 'PHI_R' ] = Qrh[i]
        dfz .loc[ (dfz ['zone']== zname), 'PHI_R' ] = Qrh[i]

    Qrh_hp = Qrh_hp_calc(dfRC_hp_row, time_setback_hp_h , time_reheat_hp_h,  f_oversize_hp, rh_type, t_ext, t_in)

    dfRC_hp_row.loc[ (dfRC_hp_row['zone']== hp_name), 'PHI_R' ] = Qrh_hp
    dfz_hp_row.loc[ (dfz_hp_row ['zone']== hp_name),  'PHI_R' ] = Qrh_hp
    
    Qht_hp = dfRC_hp_row.loc[ (dfRC_hp_row['zone']== hp_name), 'PHI_T'].values[0]
    Qhv_hp = dfRC_hp_row.loc[ (dfRC_hp_row['zone']== hp_name), 'PHI_V'].values[0]

    for i in range(0, len(dfRC_hp)):
        zname     = dfRC_hp.iloc[i]['zone']
        Qht, Qhv  = dfRC_hp.iloc[i][['PHI_T', 'PHI_V']]
        
        dfRC_hp.loc[ (dfRC_hp['zone']== zname), 'PHI_R' ] = Qrh_hp * (Qht + Qhv) / (Qht_hp + Qhv_hp) if  (Qht_hp + Qhv_hp) > 0 else 0
        dfz_hp .loc[ (dfz_hp ['zone']== zname), 'PHI_R' ] = Qrh_hp * (Qht + Qhv) / (Qht_hp + Qhv_hp) if  (Qht_hp + Qhv_hp) > 0 else 0
         
        
    # Wall layers
    dfwc = pd.merge(walls_types[['wall_type', 'layer', 'material', 'thickness']], materials[['material','lambda']], \
            on = 'material', how='left') \
            .round({'thickness': 3, 'lambda': 3}) \
            .rename(columns={'thickness': 'e [m]',"lambda": "lambda [W/mK]"})\
            .set_index('wall_type')
    
    
    dfwp = wlres.groupby(by='wall_type',as_index=False).agg("mean") \
              .round({'thickness': 3, 'U_wl': 2, 'M': 0, 'Ci': 0, 'Ce': 0}) \
              .rename(columns={'thickness': 'e [m]', 'U_wl': 'U [W/m²K]', 'M': 'M [kg/m²]',  \
              'Ci': 'Ci [J/kgK]', 'Ce': 'Ce [J/kgK]'}).set_index('wall_type')
    
    # Wall areas by zone and azimuth
    dfar = dftraz[['zone', 'azimuth', 'slope', 'wall_type', 'area_wl', 'area_wd', 'area_dr']]\
            .rename(columns={'area_wl':'area walls [m²]', 'area_wd':'area windows [m²]', 'area_dr':'area doors [m²]'})\
            .set_index('zone').round(2)
    
    # Areas by zone
    dfat = dftr[['zone', 'wall_type', 'area_wl', 'area_wd', 'area_dr']]\
        .rename(columns={'area_wl':'area walls [m²]', 'area_wd':'area windows [m²]', 'area_dr':'area doors [m²]'})\
        .set_index('zone').round(2)
    
    # Zone Heat loss Coefficients by zone
    dfhl = dfz[['zone', 'H_T_wl', 'H_T_wd', 'H_T_dr', 'H_V_su', 'H_V_ATD', 'H_V_ia', 'H_V_ie']]\
            .rename(columns={'H_T_wl':'H_T walls [W/K]','H_T_wd':'H_T windows [W/K]','H_T_dr':'H_T doors [W/K]',\
             'H_V_su':'H_V_su [W/K]','H_V_ATD':'H_V_ATD [W/K]', 'H_V_ia': 'H_V_ia [W/K]', 'H_V_ie':'H_V_ie [W/K]',}) \
            .set_index('zone').round(2)
    
    # Heating powers by zone

    dfz    = pd.concat([dfz, dfz_hp_row])
    dfresz = dfz[['zone', 'PHI_T', 'PHI_V', 'PHI_R', 'PHI_rad']]
    dfresz = dfresz.set_index('zone')
    
    dfresz['PHI_total']   = dfresz[['PHI_T', 'PHI_V', 'PHI_R']].sum(axis=1)
    dfresz['f_reheat']    = (dfresz['PHI_total'] / (dfresz['PHI_T'] + dfresz['PHI_V'])).where((dfresz['PHI_T'] + dfresz['PHI_V']) > 0, 0)
    dfresz['f_radiative'] = (dfresz['PHI_rad']   / dfresz['PHI_total']).where(dfresz['PHI_total'] > 0, 0)
    dfresz = dfresz.drop(columns=['PHI_rad'])
    
    dfresz[['PHI_T', 'PHI_V', 'PHI_R', 'PHI_total']] = dfresz[['PHI_T', 'PHI_V', 'PHI_R', 'PHI_total']].astype(int)
    dfresz[['f_reheat', 'f_radiative']] = dfresz[['f_reheat', 'f_radiative']].astype(float).round(2)
    
    dfresz = dfresz.rename(columns={'PHI_T':'Transmission [W]','PHI_V':'Ventilation [W]','PHI_R':'Reheating [W]',\
                             'PHI_total':'Total [W]', 'f_reheat':'Reheat oversizing factor', 'f_radiative':'Radiative fraction'})
    
    # Heating powers by zone and by facade
    
    dftraz  = dftraz.groupby    (by=['zone', 'azimuth', 'slope', 'wall_type'],as_index=False).agg('sum')
    dftraz  = dftraz.sort_values(by=['zone', 'azimuth', 'slope', 'wall_type'])
    dfresa = facade_calc(dftraz, dfsortz, dfv, dfz, DUwalls, t_in, t_ext)
    
    dftraz_hp  = dftraz_hp.groupby    (by=['zone', 'azimuth', 'slope', 'wall_type'],as_index=False).agg('sum')
    dftraz_hp  = dftraz_hp.sort_values(by=['zone', 'azimuth', 'slope', 'wall_type'])
    dfresa_hp = facade_calc(dftraz_hp, dfsortz, dfv, dfz_hp, DUwalls, t_in, t_ext)

    dfresa_hp = dfresa_hp.groupby(by=['azimuth'],as_index=False).agg('sum')
    dfresa_hp['zone'] = hp_name
    dfresa_hp = dfresa_hp.set_index('zone')

    dfresa = pd.concat([dfresa, dfresa_hp])
    
    return  dfRC, dfRC_hp, dfh, dfd, dfw, dfwc, dfwp, dfar, dfat, dfhl, dfzg, dfresz, dfresa, zname_list


def simulation(dfRC, dfh, dfd, dfw):

    def find_first_non_zero(vector):
        for i, value in enumerate(vector):
            if value != 0:
                return i
        return 0  
    
    def Qh(Ti, t):
        ind = int((t - t0) / time_step) 
        if ind>len(t_set_sbs)-1: ind=len(t_set_sbs)-1
        Tset  = t_set_sbs[ind]
        Qhmax = Qhmax_sbs[ind]
        Xh = 1/(1 + np.exp( (Ti - Tset)/ 0.125) )
        Qh = Xh * Qhmax
        return Qh
    
    def dT_t(t, T_array):
        T1, T2, T3 = T_array
        Ti = T2
        Q  = Qh(Ti, t)
            
        i1 = f1_rad * Q
        i3 = f3_rad * Q
        i2 = (1 - f1_rad - f3_rad) * Q

        ind = int((t - t0) / time_step) 
        if ind>len(t_ext_sbs)-1: ind=len(t_ext_sbs)-1
        T0 = t_ext_sbs[ind]
        U0 = U0v_sbs[ind] + U0wd + U0dr

        Twd = T2 + (T0-T2)*fUwd
        Tdr = T2 + (T0-T2)*fUdr
        Tsl = fT @ np.array([Twd, Tdr, T1, T3])

        Ebi = sigma_b * (Tsl+T_ref)**4
        Ji  = np.linalg.inv(mij) @ Ebi
        qi  = (Ebi - Ji)*eps/(1-eps)
        Qsl = qi * Asl
        Qnet = n_fl * fT.T @ Qsl
    #     if (Qnet.sum() > 10) or (Qnet.sum() < -10):
    #         print('unbalance')

        dT1_t = - 1/C1 * U1 * (T1 - T0) + 1/C1 * U2   * (T2 - T1)  + (i1-Qnet[2])/C1     if C1 > 0 else 0
        dT2_t = - 1/C2 * U2 * (T2 - T1) + 1/C2 * U0   * (T0 - T2)   \
                + 1/C2 * U3 * (T3 - T2) + (i2-Qnet[0]-Qnet[1])/C2                        if C2 > 0 else 0
        dT3_t = - 1/C3 * U3 * (T3 - T2) + (i3-Qnet[3])/C3                                if C3 > 0 else 0

        return [dT1_t, dT2_t, dT3_t]

    
    t_ext_sbs = np.repeat(t_ext_hbh, 6)
    ts  = np.arange(0, time_step * len(t_ext_sbs), time_step)
    t0  = ts[0]
    hr  = ts /3600
    day = hr /24

    zname_list    = []

    Qhiv      = np.zeros((len(dfRC), len(t_ext_sbs)))
    Qhcv      = np.zeros((len(dfRC), len(t_ext_sbs)))
    Tiiv      = np.zeros((len(dfRC), len(t_ext_sbs)))
    Ticv      = np.zeros((len(dfRC), len(t_ext_sbs)))

    Qhiz_kWh      = np.zeros((len(dfRC), len(t_ext_sbs)))
    Qhcz_kWh      = np.zeros((len(dfRC), len(t_ext_sbs)))
    fr_overcooliz = np.zeros((len(dfRC), len(t_ext_sbs)))
    fr_overcoolcz = np.zeros((len(dfRC), len(t_ext_sbs)))

    DQh           = np.zeros(len(dfRC))

    dfsim_list= []

    for i in range(0, len(dfRC)):

        zname                    = dfRC.iloc[i]['zone']
        n_fl                     = dfRC.iloc[i]['n_fl']
        t_in_star                = dfRC.iloc[i]['t_in_star']
        Q1_rad, Q3_rad           = dfRC.iloc[i][['Q1_rad', 'Q3_rad']]
        Qht, Qhv, Qrh            = dfRC.iloc[i][['PHI_T', 'PHI_V', 'PHI_R']]
        fUwd, U0wd, fUdr, U0dr   = dfRC.iloc[i][['fUwd', 'U0wd', 'fUdr', 'U0dr']]
        U0vo, U0vs               = dfRC.iloc[i][['U0vo', 'U0vs']]
        C1, C2, C3               = dfRC.iloc[i][['C1', 'C2', 'C3']]   
        U1, U2, U3               = dfRC.iloc[i][['U1', 'U2', 'U3']]

        zname_list.append(zname)
        
        t_set_min, f_over_max, time_h_rh_h, time_d_rh_h, time_w_rh_h = \
            dfRC.iloc[i][['t_set_min', 'f_oversize_imposed', 'time_night_rh_h', 'time_we_rh_h', 'time_hol_rh_h']]

        Tinit  = np.array([t_set_min, t_set_min, t_set_min])
        Qhnom  = Qht + Qhv + Qrh

        if (Q1_rad + Q3_rad) < Qhnom :
            f1_rad = Q1_rad / Qhnom if Qhnom > 0 else 0
            f3_rad = Q3_rad / Qhnom if Qhnom > 0 else 0
        else :
            f1_rad = Q1_rad / (Q1_rad + Q3_rad) if (Q1_rad + Q3_rad) > 0 else 0
            f3_rad = Q3_rad / (Q1_rad + Q3_rad) if (Q1_rad + Q3_rad) > 0 else 0   

        fT  = np.array([dfRC.iloc[i][['f_wd_0',   'f_dr_0',   'fT1_0',   'fT3_0'  ]].values, \
                        dfRC.iloc[i][['f_wd_90',  'f_dr_90',  'fT1_90',  'fT3_90' ]].values, \
                        dfRC.iloc[i][['f_wd_180', 'f_dr_180', 'fT1_180', 'fT3_180']].values])
        Asl = dfRC.iloc[i][['area_h', 'area_v', 'area_h']].values

        eps = 0.9
        VFhtov = dfRC.iloc[i][['VF_h_to_v']].values[0]
        VFhtoh = dfRC.iloc[i][['VF_h_to_h']].values[0]
        VFvtoh = dfRC.iloc[i][['VF_v_to_h']].values[0]
        VFvtov = dfRC.iloc[i][['VF_v_to_v']].values[0]
        mij = np.array([[             1,   (eps-1)*VFhtov, (eps-1)*VFhtoh], \
                        [(eps-1)*VFvtoh, 1-(1-eps)*VFvtov, (eps-1)*VFvtoh], \
                        [(eps-1)*VFhtoh,   (eps-1)*VFhtov,              1]]) / eps

        nbhours = len(t_ext_hbh)
        f_oc_h = np.asarray(dfh.iloc[i]['f_occ_h'])
        h_start_d = find_first_non_zero(f_oc_h)
        f_oc_h = np.tile(f_oc_h, int(nbhours/24) + 1)
        f_oc_h = f_oc_h[0:nbhours]
        f_oc_d = np.asarray(dfd.iloc[i]['f_occ_d'])
        f_oc_d = np.repeat(f_oc_d, 24)
        f_oc_d = np.tile(f_oc_d, int(nbhours/24/7) + 1)
        f_oc_d = f_oc_d[0:nbhours]
        f_oc_w = np.asarray(dfw.iloc[i]['f_occ_w'])
        f_oc_w = np.repeat(f_oc_w, 24*7)
        f_oc_w = np.tile(f_oc_w, int(nbhours/24/7/9) + 1)
        f_oc_w = f_oc_w[0:nbhours]
        f_oc   = f_oc_h * f_oc_d * f_oc_w

        nbstph = int(3600 / time_step)
        f_oc = np.repeat(f_oc, nbstph)

        nbstrh_h = int(time_h_rh_h * nbstph)
        nbstrh_d = int(time_d_rh_h * nbstph)
        nbstrh_w = int(time_w_rh_h * nbstph)

        f_oc_h = np.repeat(f_oc_h, nbstph)
        f_oc_d = np.repeat(f_oc_d, nbstph)
        f_oc_w = np.repeat(f_oc_w, nbstph)
        
        f_on_h = f_oc
        f_on_d = f_oc_d
        # f_on_w = f_oc_w
        
        for hrh in range(nbstrh_h): f_on_h  = np.maximum(f_on_h, np.roll(f_on_h, -1)) 
        for hrh in range(nbstrh_d): f_on_d  = np.maximum(f_on_d, np.roll(f_on_d, -1)) 
        # for hrh in range(nbstrh_w): f_on_w  = np.maximum(f_on_w, np.roll(f_on_w, -1)) 

        df_on_d = np.roll(f_on_d - f_oc_d, (h_start_d - time_h_rh_h)  * nbstph)

        f_on = np.maximum(f_on_h, df_on_d)

        f_set = f_on
        
        # f_hs  = f_oc
        # for hrh in range( nbstrh ):
        #     f_hs  = np.maximum(f_hs, (1 - hrh/nbstrh) * np.roll(f_hs, -1))  # progressive restart of the heating system
            
        f_vs  = f_oc
        f_resartv = np.maximum(np.diff(f_vs, prepend=0), np.zeros(len(f_vs)))
        f_resartv = np.roll(f_resartv, -nbstph)
        for n in range(nbstph) :
            f_vs = np.maximum(f_vs, n/nbstph * np.roll(f_resartv, n)) # progressive restart of the ventilation system

        U0v_sbs  = U0vo * f_vs + U0vs * (1-f_vs)

        occupancy_sbs = np.where(f_oc > 0, 1, 0)

        # Intermittent heating
        t_set_sbs = t_in_star * f_set + t_set_min * (1-f_set)
        Qhmax_sbs = Qhnom     * np.ones(len(f_on))

    #     T_list    = odeint(dT_t, Tinit, ts)
    #     Tii = T_list[:,1] 

        T_array = solve_ivp(dT_t, (ts[0],ts[-1]), Tinit, t_eval=ts, method="LSODA", min_step=1, max_step = 600)
        Tii = T_array.y[1]

        Qhivar = np.array([Qh(xTi, xt) for (xTi, xt) in zip(Tii,ts)])
        Qhi_kWh       = np.cumsum(Qhivar)    /1000 * time_step /3600 
        overcooli_ts  = np.where((f_oc > 0) & (Tii < t_in_star - 0.5), 1, 0)
        fr_overcooli  = np.cumsum(overcooli_ts) / occupancy_sbs.sum()
        Tiiv[i,:]     = Tii
        Qhiv[i,:]     = Qhivar
        Qhiz_kWh[i,:] = Qhi_kWh 
        fr_overcooliz[i,:] = fr_overcooli

        # Continuous heating
        t_set_sbs = t_in_star * np.ones(len(f_set))
        Qhmax_sbs = Qhnom     * np.ones(len(f_on))

    #     T_list    = odeint(dT_t, Tinit, ts)
    #     Tic = T_list[:,1] 

        T_array = solve_ivp(dT_t, (ts[0],ts[-1]), Tinit, t_eval=ts, method="LSODA", min_step=1, max_step = 600)
        Tic = T_array.y[1]

        Qhcvar = [Qh(xTi, xt) for (xTi, xt) in zip(Tic,ts)]
        Qhc_kWh       = np.cumsum(Qhcvar)    /1000 * time_step /3600 
        overcoolc_ts  = np.where((f_oc > 0) & (Tic < t_in_star - 0.5), 1, 0)
        fr_overcoolc  = np.cumsum(overcoolc_ts) / occupancy_sbs.sum()
        Ticv[i,:]     = Tic
        Qhcv[i,:]     = Qhcvar
        Qhcz_kWh[i,:] = Qhc_kWh 
        fr_overcoolcz[i,:] = fr_overcoolc

        # Energy savings
        DQh[i] = int(100 * (Qhc_kWh[-1] - Qhi_kWh[-1]) / (Qhc_kWh[-1]+0.0000001))

        # Define the date range for a non-leap year
        start_date = '2023-10-01 00:00'  # October 1st at 0 AM (any non-leap year)
        end_date = '2024-06-08 23:50'    # June 8th at 11 PM (any non-leap year)
        
        # Generate the date range for every hour between the start and end dates
        date_range = pd.date_range(start=start_date, end=end_date, freq='10T')
        
        # Format the dates as required: day/month hour
        formatted_dates = date_range.strftime('%d/%m %-Hh%M')  # %-Hh for hour without leading zero

        dfsim = pd.DataFrame({'date': formatted_dates,'t_ext (°C)': t_ext_sbs, \
                't_in interm (°C)': Tiiv[i,:], 'cooling discomfort time fraction interm (-)': fr_overcooliz[i,:],  \
                'heating power interm (W)': Qhiv[i,:],  'heating consumption interm (kWh)': Qhiz_kWh[i,:], \
                't_in continuous (°C)': Ticv[i,:], 'cooling discomfort time fraction continuous (-)': fr_overcoolcz[i,:],  \
                'heating power continuous (W)': Qhcv[i,:],  'heating consumption continuous (kWh)': Qhcz_kWh[i,:]})

        dfsim_list.append(dfsim)


    return zname_list, day, t_ext_sbs, Tiiv, fr_overcooliz, Qhiv, Qhiz_kWh, Ticv, fr_overcoolcz, Qhcv, Qhcz_kWh, DQh, dfsim_list 


# EN12831 Annex F Table F1
# c_eff (Wh/m³K), heating setback time (h), infiltration rate during interruption time (1/h), reheat time (h)
frh_data_F1 = \
[((15,  8, 0.1, 0.5), 63),
 ((15,  8, 0.1, 1.0), 34),
 ((15,  8, 0.1, 2.0), 14),
 ((15,  8, 0.1, 3.0), 5),
 ((15,  8, 0.1, 4.0), 0), 
 ((15,  8, 0.1, 6.0), 0),
 ((15,  8, 0.1, 12.), 0),
 
 ((15,  8, 0.5, 0.5), 74),
 ((15,  8, 0.5, 1.0), 43),
 ((15,  8, 0.5, 2.0), 21),
 ((15,  8, 0.5, 3.0), 10),
 ((15,  8, 0.5, 4.0), 3), 
 ((15,  8, 0.5, 6.0), 0),
 ((15,  8, 0.5, 12.), 0),
 
 ((15, 14, 0.1, 0.5), 88),
 ((15, 14, 0.1, 1.0), 50),
 ((15, 14, 0.1, 2.0), 28),
 ((15, 14, 0.1, 3.0), 17),
 ((15, 14, 0.1, 4.0), 11), 
 ((15, 14, 0.1, 6.0), 3),
 ((15, 14, 0.1, 12.), 0),
 
 ((15, 14, 0.5, 0.5), 91),
 ((15, 14, 0.5, 1.0), 50),
 ((15, 14, 0.5, 2.0), 28),
 ((15, 14, 0.5, 3.0), 18),
 ((15, 14, 0.5, 4.0), 12), 
 ((15, 14, 0.5, 6.0), 5),
 ((15, 14, 0.5, 12.), 0),
 
 ((15, 62, 0.1, 0.5), 92),
 ((15, 62, 0.1, 1.0), 55),
 ((15, 62, 0.1, 2.0), 32),
 ((15, 62, 0.1, 3.0), 23),
 ((15, 62, 0.1, 4.0), 17), 
 ((15, 62, 0.1, 6.0), 10),
 ((15, 62, 0.1, 12.), 2),
 
 ((15, 62, 0.5, 0.5), 92),
 ((15, 62, 0.5, 1.0), 55),
 ((15, 62, 0.5, 2.0), 32),
 ((15, 62, 0.5, 3.0), 22),
 ((15, 62, 0.5, 4.0), 17), 
 ((15, 62, 0.5, 6.0), 10),
 ((15, 62, 0.5, 12.), 2),
 
 ((15, 168, 0.1, 0.5), 92),
 ((15, 168, 0.1, 1.0), 55),
 ((15, 168, 0.1, 2.0), 32),
 ((15, 168, 0.1, 3.0), 23),
 ((15, 168, 0.1, 4.0), 17), 
 ((15, 168, 0.1, 6.0), 10),
 ((15, 168, 0.1, 12.), 2),
 
 ((15, 168, 0.5, 0.5), 92),
 ((15, 168, 0.5, 1.0), 55),
 ((15, 168, 0.5, 2.0), 32),
 ((15, 168, 0.5, 3.0), 23),
 ((15, 168, 0.5, 4.0), 17), 
 ((15, 168, 0.5, 6.0), 10),
 ((15, 168, 0.5, 12.), 2),
 
 
 ((50,  8, 0.1, 0.5), 16),
 ((50,  8, 0.1, 1.0), 10),
 ((50,  8, 0.1, 2.0), 3),
 ((50,  8, 0.1, 3.0), 0),
 ((50,  8, 0.1, 4.0), 0), 
 ((50,  8, 0.1, 6.0), 0),
 ((50,  8, 0.1, 12.), 0),
 
 ((50,  8, 0.5, 0.5), 26),
 ((50,  8, 0.5, 1.0), 16),
 ((50,  8, 0.5, 2.0), 8),
 ((50,  8, 0.5, 3.0), 2),
 ((50,  8, 0.5, 4.0), 0), 
 ((50,  8, 0.5, 6.0), 0),
 ((50,  8, 0.5, 12.), 0),
 
 ((50, 14, 0.1, 0.5), 38),
 ((50, 14, 0.1, 1.0), 29),
 ((50, 14, 0.1, 2.0), 18),
 ((50, 14, 0.1, 3.0), 12),
 ((50, 14, 0.1, 4.0), 7), 
 ((50, 14, 0.1, 6.0), 1),
 ((50, 14, 0.1, 12.), 0),
 
 ((50, 14, 0.5, 0.5), 56),
 ((50, 14, 0.5, 1.0), 43),
 ((50, 14, 0.5, 2.0), 29),
 ((50, 14, 0.5, 3.0), 21),
 ((50, 14, 0.5, 4.0), 15), 
 ((50, 14, 0.5, 6.0), 5),
 ((50, 14, 0.5, 12.), 0),
 
 ((50, 62, 0.1, 0.5), 100),
 ((50, 62, 0.1, 1.0), 100),
 ((50, 62, 0.1, 2.0), 86),
 ((50, 62, 0.1, 3.0), 73),
 ((50, 62, 0.1, 4.0), 64), 
 ((50, 62, 0.1, 6.0), 52),
 ((50, 62, 0.1, 12.), 31),
 
 ((50, 62, 0.5, 0.5), 100),
 ((50, 62, 0.5, 1.0), 100),
 ((50, 62, 0.5, 2.0), 100),
 ((50, 62, 0.5, 3.0), 94),
 ((50, 62, 0.5, 4.0), 84), 
 ((50, 62, 0.5, 6.0), 70),
 ((50, 62, 0.5, 12.), 45),
 
 ((50, 168, 0.1, 0.5), 100),
 ((50, 168, 0.1, 1.0), 100),
 ((50, 168, 0.1, 2.0), 100),
 ((50, 168, 0.1, 3.0), 100),
 ((50, 168, 0.1, 4.0), 95), 
 ((50, 168, 0.1, 6.0), 81),
 ((50, 168, 0.1, 12.), 57),
 
 ((50, 168, 0.5, 0.5), 100),
 ((50, 168, 0.5, 1.0), 100),
 ((50, 168, 0.5, 2.0), 100),
 ((50, 168, 0.5, 3.0), 100),
 ((50, 168, 0.5, 4.0), 95), 
 ((50, 168, 0.5, 6.0), 81),
 ((50, 168, 0.5, 12.), 57) ]

# EN12831 Annex F Table F3
# c_eff (Wh/m³K), temperature drop (K), infiltration rate during interruption time (1/h), reheat time (h)
frh_data_F3 = \
[((15, 1, 0.1, 0.5), 12),
 ((15, 1, 0.1, 1.0), 8), 
 ((15, 1, 0.1, 2.0), 5), 
 ((15, 1, 0.1, 3.0), 3), 
 ((15, 1, 0.1, 4.0), 2), 

 ((15, 1, 0.5, 0.5), 14),
 ((15, 1, 0.5, 1.0), 10), 
 ((15, 1, 0.5, 2.0), 7), 
 ((15, 1, 0.5, 3.0), 5), 
 ((15, 1, 0.5, 4.0), 4),

 ((15, 2, 0.1, 0.5), 27),
 ((15, 2, 0.1, 1.0), 18), 
 ((15, 2, 0.1, 2.0), 10), 
 ((15, 2, 0.1, 3.0), 7), 
 ((15, 2, 0.1, 4.0), 5), 

 ((15, 2, 0.5, 0.5), 29),
 ((15, 2, 0.5, 1.0), 21), 
 ((15, 2, 0.5, 2.0), 13), 
 ((15, 2, 0.5, 3.0), 10), 
 ((15, 2, 0.5, 4.0), 8),

 ((15, 3, 0.1, 0.5), 39),
 ((15, 3, 0.1, 1.0), 26), 
 ((15, 3, 0.1, 2.0), 15), 
 ((15, 3, 0.1, 3.0), 9), 
 ((15, 3, 0.1, 4.0), 7), 

 ((15, 3, 0.5, 0.5), 44),
 ((15, 3, 0.5, 1.0), 32), 
 ((15, 3, 0.5, 2.0), 21), 
 ((15, 3, 0.5, 3.0), 15), 
 ((15, 3, 0.5, 4.0), 13),

 ((15, 4, 0.1, 0.5), 50),
 ((15, 4, 0.1, 1.0), 33), 
 ((15, 4, 0.1, 2.0), 20), 
 ((15, 4, 0.1, 3.0), 14), 
 ((15, 4, 0.1, 4.0), 10), 

 ((15, 4, 0.5, 0.5), 58),
 ((15, 4, 0.5, 1.0), 41), 
 ((15, 4, 0.5, 2.0), 28), 
 ((15, 4, 0.5, 3.0), 21), 
 ((15, 4, 0.5, 4.0), 17),

 ((15, 5, 0.1, 0.5), 100),
 ((15, 5, 0.1, 1.0), 100), 
 ((15, 5, 0.1, 2.0), 43), 
 ((15, 5, 0.1, 3.0), 33), 
 ((15, 5, 0.1, 4.0), 28), 

 ((15, 5, 0.5, 0.5), 100),
 ((15, 5, 0.5, 1.0), 100), 
 ((15, 5, 0.5, 2.0), 47), 
 ((15, 5, 0.5, 3.0), 37), 
 ((15, 5, 0.5, 4.0), 31),

 ((50, 1, 0.1, 0.5), 12),
 ((50, 1, 0.1, 1.0), 8), 
 ((50, 1, 0.1, 2.0), 5), 
 ((50, 1, 0.1, 3.0), 3), 
 ((50, 1, 0.1, 4.0), 2), 

 ((50, 1, 0.5, 0.5), 18),
 ((50, 1, 0.5, 1.0), 14), 
 ((50, 1, 0.5, 2.0), 11), 
 ((50, 1, 0.5, 3.0), 10), 
 ((50, 1, 0.5, 4.0), 9),

 ((50, 2, 0.1, 0.5), 28),
 ((50, 2, 0.1, 1.0), 21), 
 ((50, 2, 0.1, 2.0), 15), 
 ((50, 2, 0.1, 3.0), 12), 
 ((50, 2, 0.1, 4.0), 10), 

 ((50, 2, 0.5, 0.5), 35),
 ((50, 2, 0.5, 1.0), 28), 
 ((50, 2, 0.5, 2.0), 22), 
 ((50, 2, 0.5, 3.0), 19), 
 ((50, 2, 0.5, 4.0), 17),

 ((50, 3, 0.1, 0.5), 44),
 ((50, 3, 0.1, 1.0), 34), 
 ((50, 3, 0.1, 2.0), 25), 
 ((50, 3, 0.1, 3.0), 20), 
 ((50, 3, 0.1, 4.0), 18), 

 ((50, 3, 0.5, 0.5), 53),
 ((50, 3, 0.5, 1.0), 43), 
 ((50, 3, 0.5, 2.0), 33), 
 ((50, 3, 0.5, 3.0), 27), 
 ((50, 3, 0.5, 4.0), 25),

 ((50, 4, 0.1, 0.5), 60),
 ((50, 4, 0.1, 1.0), 48), 
 ((50, 4, 0.1, 2.0), 35), 
 ((50, 4, 0.1, 3.0), 29), 
 ((50, 4, 0.1, 4.0), 26), 

 ((50, 4, 0.5, 0.5), 69),
 ((50, 4, 0.5, 1.0), 56), 
 ((50, 4, 0.5, 2.0), 43), 
 ((50, 4, 0.5, 3.0), 37), 
 ((50, 4, 0.5, 4.0), 34),

 ((50, 5, 0.1, 0.5), 100),
 ((50, 5, 0.1, 1.0), 100), 
 ((50, 5, 0.1, 2.0), 85), 
 ((50, 5, 0.1, 3.0), 75), 
 ((50, 5, 0.1, 4.0), 72), 

 ((50, 5, 0.5, 0.5), 100),
 ((50, 5, 0.5, 1.0), 100), 
 ((50, 5, 0.5, 2.0), 94), 
 ((50, 5, 0.5, 3.0), 84), 
 ((50, 5, 0.5, 4.0), 76)]




we_dict = {'CP': {0: 9300, 1: 9880, 2: 3200, 3: 2630, 4: 1790, 5: 6250, 6: 3570, 7: 8690, 8: 4540, 9: 4770, 10: 5300, 11: 1070, 12: 6150, 13: 5537, 14: 4430, 15: 4160, 16: 7640, 17: 2000, 18: 8570, 19: 8850, 20: 2370, 21: 6700, 22: 3665, 23: 1730, 24: 9960, 25: 5330, 26: 7800, 27: 6717, 28: 6790, 29: 4880, 30: 1160, 31: 8580, 32: 4340, 33: 4920, 34: 2387, 35: 4837, 36: 2490, 37: 4690, 38: 6600, 39: 6500, 40: 5570, 41: 1320, 42: 8730, 43: 2340, 44: 1650, 45: 3130, 46: 3460, 47: 7970, 48: 1082, 49: 3580, 50: 2590, 51: 9290, 52: 4257, 53: 7320, 54: 3060, 55: 6687, 56: 6880, 57: 1547, 58: 9120, 59: 4610, 60: 3360, 61: 5555, 62: 3740, 63: 7130, 64: 8370, 65: 4670, 66: 3950, 67: 2530, 68: 2820, 69: 2850, 70: 3190, 71: 3840, 72: 2880, 73: 2150, 74: 6830, 75: 7300, 76: 3370, 77: 1420, 78: 1440, 79: 7090, 80: 4260, 81: 9660, 82: 2930, 83: 2960, 84: 8450, 85: 3960, 86: 7940, 87: 8000, 88: 7620, 89: 1000, 90: 9255, 91: 4760, 92: 4210, 93: 4790, 94: 4750, 95: 7760, 96: 5630, 97: 7160, 98: 6000, 99: 1450, 100: 6200, 101: 4050, 102: 1325, 103: 7950, 104: 6460, 105: 6810, 106: 5590, 107: 4560, 108: 7340, 109: 4170, 110: 7780, 111: 6180, 112: 1490, 113: 5660, 114: 4367, 115: 4607, 116: 8340, 117: 6929, 118: 8420, 119: 8660, 120: 9840, 121: 8540, 122: 9800, 123: 9470, 124: 9200, 125: 8720, 126: 2480, 127: 9070, 128: 3590, 129: 3290, 130: 8600, 131: 1700, 132: 3650, 133: 5500, 134: 4820, 135: 5680, 136: 4357, 137: 7370, 138: 1620, 139: 2570, 140: 6940, 141: 7190, 142: 2650, 143: 9900, 144: 5310, 145: 7890, 146: 1050, 147: 7850, 148: 4480, 149: 6997, 150: 9420, 151: 6560, 152: 4130, 153: 2910, 154: 7730, 155: 7120, 156: 6740, 157: 1040, 158: 4700, 159: 1140, 160: 9940, 161: 4317, 162: 6240, 163: 6637, 164: 5380, 165: 4190, 166: 4347, 167: 4400, 168: 4620, 169: 6220, 170: 7880, 171: 5150, 172: 5620, 173: 6820, 174: 6140, 175: 1190, 176: 5070, 177: 7080, 178: 7910, 179: 6440, 180: 1570, 181: 1083, 182: 9890, 183: 5575, 184: 2440, 185: 4250, 186: 3450, 187: 5030, 188: 1470, 189: 3600, 190: 9000, 191: 9500, 192: 6280, 193: 5340, 194: 3890, 195: 8470, 196: 3380, 197: 1755, 198: 6670, 199: 4460, 200: 1390, 201: 1850, 202: 2280, 203: 3150, 204: 9450, 205: 6720, 206: 3545, 207: 1500, 208: 3945, 209: 9220, 210: 4180, 211: 5360, 212: 3930, 213: 6120, 214: 4280, 215: 8530, 216: 3500, 217: 5540, 218: 5370, 219: 3940, 220: 3870, 221: 2220, 222: 1357, 223: 2620, 224: 7350, 225: 6887, 226: 3020, 227: 2200, 228: 2270, 229: 3540, 230: 1540, 231: 4217, 232: 2230, 233: 4040, 234: 3717, 235: 4650, 236: 9550, 237: 3550, 238: 8950, 239: 3320, 240: 1560, 241: 3730, 242: 3220, 243: 7387, 244: 8830, 245: 2320, 246: 9667, 247: 6990, 248: 6660, 249: 3530, 250: 8650, 251: 5560, 252: 2540, 253: 3040, 254: 2235, 255: 4500, 256: 8480, 257: 8900, 258: 1315, 259: 8770, 260: 1460, 261: 1050, 262: 8870, 263: 8490, 264: 4845, 265: 5190, 266: 1090, 267: 1370, 268: 4450, 269: 7050, 270: 2920, 271: 1910, 272: 2950, 273: 1880, 274: 9970, 275: 2460, 276: 3140, 277: 4720, 278: 3640, 279: 9690, 280: 9910, 281: 8300, 282: 8680, 283: 1081, 284: 8670, 285: 2550, 286: 8610, 287: 3470, 288: 3070, 289: 3720, 290: 8500, 291: 1950, 292: 9150, 293: 9770, 294: 8520, 295: 5080, 296: 1310, 297: 7100, 298: 6980, 299: 2430, 300: 9270, 301: 3620, 302: 3400, 303: 8920, 304: 1380, 305: 7070, 306: 9280, 307: 9340, 308: 8880, 309: 6860, 310: 8860, 311: 1750, 312: 7870, 313: 3970, 314: 6210, 315: 7860, 316: 3000, 317: 7900, 318: 6890, 319: 6800, 320: 8810, 321: 1770, 322: 4000, 323: 2500, 324: 9570, 325: 4990, 326: 2275, 327: 4830, 328: 4287, 329: 1630, 330: 2547, 331: 3350, 332: 6540, 333: 9080, 334: 9160, 335: 3920, 336: 1840, 337: 4710, 338: 8647, 339: 9920, 340: 3210, 341: 3560, 342: 9680, 343: 3680, 344: 3630, 345: 1830, 346: 9990, 347: 2390, 348: 4960, 349: 7170, 350: 6960, 351: 6900, 352: 4570, 353: 6630, 354: 2800, 355: 2450, 356: 3670, 357: 1860, 358: 6769, 359: 9090, 360: 8930, 361: 6567, 362: 1785, 363: 9820, 364: 2330, 365: 8957, 366: 6780, 367: 5640, 368: 8760, 369: 8430, 370: 4577, 371: 9500, 372: 2400, 373: 1080, 374: 6590, 375: 7000, 376: 7750, 377: 6110, 378: 1435, 379: 8890, 380: 7140, 381: 2640, 382: 7700, 383: 6750, 384: 5000, 385: 4550, 386: 6950, 387: 9810, 388: 3910, 389: 6840, 390: 4120, 391: 9850, 392: 2845, 393: 3850, 394: 8620, 395: 2560, 396: 9400, 397: 1400, 398: 5350, 399: 2250, 400: 4877, 401: 5520, 402: 8400, 403: 9860, 404: 8020, 405: 8780, 406: 3660, 407: 1745, 408: 4360, 409: 1350, 410: 1340, 411: 9700, 412: 8460, 413: 1160, 414: 3050, 415: 2360, 416: 4590, 417: 4680, 418: 3090, 419: 3900, 420: 6850, 421: 7740, 422: 3990, 423: 1670, 424: 4860, 425: 7600, 426: 1360, 427: 5600, 428: 8740, 429: 4850, 430: 6230, 431: 8970, 432: 5170, 433: 2580, 434: 2870, 435: 7390, 436: 7040, 437: 7380, 438: 4730, 439: 1367, 440: 2520, 441: 2380, 442: 1430, 443: 4350, 444: 6987, 445: 2470, 446: 3770, 447: 2310, 448: 1330, 449: 5580, 450: 8800, 451: 9600, 452: 1760, 453: 3110, 454: 6767, 455: 8755, 456: 7610, 457: 2840, 458: 6680, 459: 4470, 460: 7330, 461: 1060, 462: 6870, 463: 1210, 464: 6747, 465: 4420, 466: 5060, 467: 4780, 468: 1030, 469: 1030, 470: 2627, 471: 3270, 472: 2970, 473: 2900, 474: 7180, 475: 4100, 476: 7830, 477: 1082, 478: 2890, 479: 1640, 480: 1060, 481: 9170, 482: 1080, 483: 1210, 484: 2860, 485: 1200, 486: 9980, 487: 9520, 488: 9830, 489: 9100, 490: 1600, 491: 1150, 492: 3800, 493: 6470, 494: 7060, 495: 5140, 496: 5377, 497: 4630, 498: 4900, 499: 8587, 500: 4140, 501: 2940, 502: 8840, 503: 4970, 504: 1820, 505: 9190, 506: 4987, 507: 6927, 508: 9140, 509: 6970, 510: 1740, 511: 3080, 512: 3980, 513: 4910, 514: 4890, 515: 6530, 516: 8700, 517: 3390, 518: 3300, 519: 4557, 520: 6730, 521: 3700, 522: 8820, 523: 7500, 524: 3120, 525: 4980, 526: 4870, 527: 1480, 528: 2300, 529: 1180, 530: 1180, 531: 6640, 532: 4537, 533: 4800, 534: 8630, 535: 6690, 536: 1495, 537: 4530, 538: 1800, 539: 5670, 540: 6760, 541: 4600, 542: 8640, 543: 3790, 544: 2290, 545: 1190, 546: 2350, 547: 5550, 548: 9950, 549: 9250, 550: 9185, 551: 4950, 552: 5650, 553: 1457, 554: 4520, 555: 8790, 556: 4300, 557: 4219, 558: 1410, 559: 1170, 560: 1170, 561: 1300, 562: 4840, 563: 3830, 564: 6920, 565: 1780, 566: 8940, 567: 2260, 568: 9230, 569: 8560, 570: 1970, 571: 9260, 572: 8710, 573: 2110, 574: 2830, 575: 8750, 576: 1200, 577: 1150, 578: 2160, 579: 9790, 580: 2990, 581: 5530, 582: 2240, 583: 1930, 584: 8210, 585: 8380, 586: 9240, 587: 9060, 588: 1980, 589: 9750, 590: 2980, 591: 9930, 592: 3520, 593: 8980, 594: 9620, 595: 3440, 596: 8377, 597: 9870, 598: 3690, 599: 9630, 600: 8550, 601: 2070}, 'Localité': {0: 'Aalst', 1: 'Aalter', 2: 'Aarschot', 3: 'Aartselaar', 4: 'Affligem', 5: 'Aiseau-Presles', 6: 'Alken', 7: 'Alveringem', 8: 'Amay', 9: 'Amel', 10: 'Andenne', 11: 'Anderlecht', 12: 'Anderlues', 13: 'Anhée', 14: 'Ans', 15: 'Anthisnes', 16: 'Antoing', 17: 'Antwerpen', 18: 'Anzegem', 19: 'Ardooie', 20: 'Arendonk', 21: 'Arlon', 22: 'As', 23: 'Asse', 24: 'Assenede', 25: 'Assesse', 26: 'Ath', 27: 'Attert', 28: 'Aubange', 29: 'Aubel', 30: 'Auderghem', 31: 'Avelgem', 32: 'Awans', 33: 'Aywaille', 34: 'Baarle-Hertog', 35: 'Baelen', 36: 'Balen', 37: 'Bassenge', 38: 'Bastogne', 39: 'Beaumont', 40: 'Beauraing', 41: 'Beauvechain', 42: 'Beernem', 43: 'Beerse', 44: 'Beersel', 45: 'Begijnendijk', 46: 'Bekkevoort', 47: 'Beloeil', 48: 'Berchem-Sainte-Agathe', 49: 'Beringen', 50: 'Berlaar', 51: 'Berlare', 52: 'Berloz', 53: 'Bernissart', 54: 'Bertem', 55: 'Bertogne', 56: 'Bertrix', 57: 'Bever', 58: 'Beveren', 59: 'Beyne-Heusay', 60: 'Bierbeek', 61: 'Bièvre', 62: 'Bilzen', 63: 'Binche', 64: 'Blankenberge', 65: 'Blegny', 66: 'Bocholt', 67: 'Boechout', 68: 'Bonheiden', 69: 'Boom', 70: 'Boortmeerbeek', 71: 'Borgloon', 72: 'Bornem', 73: 'Borsbeek', 74: 'Bouillon', 75: 'Boussu', 76: 'Boutersem', 77: "Braine-l'Alleud", 78: 'Braine-le-Château', 79: 'Braine-le-Comte', 80: 'Braives', 81: 'Brakel', 82: 'Brasschaat', 83: 'Brecht', 84: 'Bredene', 85: 'Bree', 86: 'Brugelette', 87: 'Brugge (beh./sauf Zeebrugge)', 88: 'Brunehaut', 89: 'Bruxelles / Brussel', 90: 'Buggenhout', 91: 'Büllingen', 92: 'Burdinne', 93: 'Burg-Reuland', 94: 'Bütgenbach', 95: 'Celles', 96: 'Cerfontaine', 97: 'Chapelle-lez-Herlaimont', 98: 'Charleroi', 99: 'Chastre', 100: 'Châtelet', 101: 'Chaudfontaine', 102: 'Chaumont-Gistoux', 103: 'Chièvres', 104: 'Chimay', 105: 'Chiny', 106: 'Ciney', 107: 'Clavier', 108: 'Colfontaine', 109: 'Comblain-au-Pont', 110: 'Comines-Warneton', 111: 'Courcelles', 112: 'Court-Saint-Etienne', 113: 'Couvin', 114: 'Crisnée', 115: 'Dalhem', 116: 'Damme', 117: 'Daverdisse', 118: 'De Haan', 119: 'De Panne', 120: 'De Pinte', 121: 'Deerlijk', 122: 'Deinze', 123: 'Denderleeuw', 124: 'Dendermonde', 125: 'Dentergem', 126: 'Dessel', 127: 'Destelbergen', 128: 'Diepenbeek', 129: 'Diest', 130: 'Diksmuide', 131: 'Dilbeek', 132: 'Dilsen-Stokkem', 133: 'Dinant', 134: 'Dison', 135: 'Doische', 136: 'Donceel', 137: 'Dour', 138: 'Drogenbos', 139: 'Duffel', 140: 'Durbuy', 141: 'Ecaussinnes', 142: 'Edegem', 143: 'Eeklo', 144: 'Eghezée', 145: 'Ellezelles', 146: 'Elsene', 147: 'Enghien', 148: 'Engis', 149: 'Erezée', 150: 'Erpe-Mere', 151: 'Erquelinnes', 152: 'Esneux', 153: 'Essen', 154: 'Estaimpuis', 155: 'Estinnes', 156: 'Etalle', 157: 'Etterbeek', 158: 'Eupen', 159: 'Evere', 160: 'Evergem', 161: 'Faimes', 162: 'Farciennes', 163: 'Fauvillers', 164: 'Fernelmont', 165: 'Ferrières', 166: 'Fexhe-le-Haut-Clocher', 167: 'Flémalle', 168: 'Fléron', 169: 'Fleurus', 170: 'Flobecq', 171: 'Floreffe', 172: 'Florennes', 173: 'Florenville', 174: "Fontaine-l'Evêque", 175: 'Forest', 176: 'Fosses-la-Ville', 177: 'Frameries', 178: 'Frasnes-lez-Anvaing', 179: 'Froidchapelle', 180: 'Galmaarden', 181: 'Ganshoren', 182: 'Gavere', 183: 'Gedinne', 184: 'Geel', 185: 'Geer', 186: 'Geetbets', 187: 'Gembloux', 188: 'Genappe', 189: 'Genk', 190: 'Gent', 191: 'Geraardsbergen', 192: 'Gerpinnes', 193: 'Gesves', 194: 'Gingelom', 195: 'Gistel', 196: 'Glabbeek', 197: 'Gooik', 198: 'Gouvy', 199: 'Grâce-Hollogne', 200: 'Grez-Doiceau', 201: 'Grimbergen', 202: 'Grobbendonk', 203: 'Haacht', 204: 'Haaltert', 205: 'Habay', 206: 'Halen', 207: 'Halle', 208: 'Ham', 209: 'Hamme', 210: 'Hamoir', 211: 'Hamois', 212: 'Hamont-Achel', 213: 'Ham-sur-Heure-Nalinnes', 214: 'Hannut', 215: 'Harelbeke', 216: 'Hasselt', 217: 'Hastière', 218: 'Havelange', 219: 'Hechtel-Eksel', 220: 'Heers', 221: 'Heist-op-den-Berg', 222: 'Hélécine', 223: 'Hemiksem', 224: 'Hensies', 225: 'Herbeumont', 226: 'Herent', 227: 'Herentals', 228: 'Herenthout', 229: 'Herk-de-Stad', 230: 'Herne', 231: 'Héron', 232: 'Herselt', 233: 'Herstal', 234: 'Herstappe', 235: 'Herve', 236: 'Herzele', 237: 'Heusden-Zolder', 238: 'Heuvelland', 239: 'Hoegaarden', 240: 'Hoeilaart', 241: 'Hoeselt', 242: 'Holsbeek', 243: 'Honnelles', 244: 'Hooglede', 245: 'Hoogstraten', 246: 'Horebeke', 247: 'Hotton', 248: 'Houffalize', 249: 'Houthalen-Helchteren', 250: 'Houthulst', 251: 'Houyet', 252: 'Hove', 253: 'Huldenberg', 254: 'Hulshout', 255: 'Huy', 256: 'Ichtegem', 257: 'Ieper', 258: 'Incourt', 259: 'Ingelmunster', 260: 'Ittre', 261: 'Ixelles', 262: 'Izegem', 263: 'Jabbeke', 264: 'Jalhay', 265: 'Jemeppe-sur-Sambre', 266: 'Jette', 267: 'Jodoigne', 268: 'Juprelle', 269: 'Jurbise', 270: 'Kalmthout', 271: 'Kampenhout', 272: 'Kapellen', 273: 'Kapelle-op-den-Bos', 274: 'Kaprijke', 275: 'Kasterlee', 276: 'Keerbergen', 277: 'Kelmis', 278: 'Kinrooi', 279: 'Kluisbergen', 280: 'Knesselare', 281: 'Knokke-Heist', 282: 'Koekelare', 283: 'Koekelberg', 284: 'Koksijde', 285: 'Kontich', 286: 'Kortemark', 287: 'Kortenaken', 288: 'Kortenberg', 289: 'Kortessem', 290: 'Kortrijk', 291: 'Kraainem', 292: 'Kruibeke', 293: 'Kruishoutem', 294: 'Kuurne', 295: 'La Bruyère', 296: 'La Hulpe', 297: 'La Louvière', 298: 'La Roche-en-Ardenne', 299: 'Laakdal', 300: 'Laarne', 301: 'Lanaken', 302: 'Landen', 303: 'Langemark-Poelkapelle', 304: 'Lasne', 305: 'Le Roeulx', 306: 'Lebbeke', 307: 'Lede', 308: 'Ledegem', 309: 'Léglise', 310: 'Lendelede', 311: 'Lennik', 312: 'Lens', 313: 'Leopoldsburg', 314: 'Les Bons Villers', 315: 'Lessines', 316: 'Leuven', 317: 'Leuze-en-Hainaut', 318: 'Libin', 319: 'Libramont-Chevigny', 320: 'Lichtervelde', 321: 'Liedekerke', 322: 'Liège', 323: 'Lier', 324: 'Lierde', 325: 'Lierneux', 326: 'Lille', 327: 'Limbourg', 328: 'Lincent', 329: 'Linkebeek', 330: 'Lint', 331: 'Linter', 332: 'Lobbes', 333: 'Lochristi', 334: 'Lokeren', 335: 'Lommel', 336: 'Londerzeel', 337: 'Lontzen', 338: 'Lo-Reninge', 339: 'Lovendegem', 340: 'Lubbeek', 341: 'Lummen', 342: 'Maarkedal', 343: 'Maaseik', 344: 'Maasmechelen', 345: 'Machelen', 346: 'Maldegem', 347: 'Malle', 348: 'Malmedy', 349: 'Manage', 350: 'Manhay', 351: 'Marche-en-Famenne', 352: 'Marchin', 353: 'Martelange', 354: 'Mechelen', 355: 'Meerhout', 356: 'Meeuwen-Gruitrode', 357: 'Meise', 358: 'Meix-devant-Virton', 359: 'Melle', 360: 'Menen', 361: 'Merbes-le-Château', 362: 'Merchtem', 363: 'Merelbeke', 364: 'Merksplas', 365: 'Mesen', 366: 'Messancy', 367: 'Mettet', 368: 'Meulebeke', 369: 'Middelkerke', 370: 'Modave', 371: 'Moerbeke', 372: 'Mol', 373: 'Molenbeek-Saint-Jean', 374: 'Momignies', 375: 'Mons', 376: "Mont-de-l'Enclus", 377: 'Montigny-le-Tilleul', 378: 'Mont-Saint-Guibert', 379: 'Moorslede', 380: 'Morlanwelz', 381: 'Mortsel', 382: 'Mouscron', 383: 'Musson', 384: 'Namur', 385: 'Nandrin', 386: 'Nassogne', 387: 'Nazareth', 388: 'Neerpelt', 389: 'Neufchâteau', 390: 'Neupré', 391: 'Nevele', 392: 'Niel', 393: 'Nieuwerkerken', 394: 'Nieuwpoort', 395: 'Nijlen', 396: 'Ninove', 397: 'Nivelles', 398: 'Ohey', 399: 'Olen', 400: 'Olne', 401: 'Onhaye', 402: 'Oostende', 403: 'Oosterzele', 404: 'Oostkamp', 405: 'Oostrozebeke', 406: 'Opglabbeek', 407: 'Opwijk', 408: 'Oreye', 409: 'Orp-Jauche', 410: 'Ottignies-Louvain-la-Neuve', 411: 'Oudenaarde', 412: 'Oudenburg', 413: 'Oudergem', 414: 'Oud-Heverlee', 415: 'Oud-Turnhout', 416: 'Ouffet', 417: 'Oupeye', 418: 'Overijse', 419: 'Overpelt', 420: 'Paliseul', 421: 'Pecq', 422: 'Peer', 423: 'Pepingen', 424: 'Pepinster', 425: 'Peruwelz', 426: 'Perwez', 427: 'Philippeville', 428: 'Pittem', 429: 'Plombières', 430: 'Pont-à-Celles', 431: 'Poperinge', 432: 'Profondeville', 433: 'Putte', 434: 'Puurs', 435: 'Quaregnon', 436: 'Quévy', 437: 'Quiévrain', 438: 'Raeren', 439: 'Ramillies', 440: 'Ranst', 441: 'Ravels', 442: 'Rebecq', 443: 'Remicourt', 444: 'Rendeux', 445: 'Retie', 446: 'Riemst', 447: 'Rijkevorsel', 448: 'Rixensart', 449: 'Rochefort', 450: 'Roeselare', 451: 'Ronse', 452: 'Roosdaal', 453: 'Rotselaar', 454: 'Rouvroy', 455: 'Ruiselede', 456: 'Rumes', 457: 'Rumst', 458: 'Sainte-Ode', 459: 'Saint-Georges-sur-Meuse', 460: 'Saint-Ghislain', 461: 'Saint-Gilles', 462: 'Saint-Hubert', 463: 'Saint-Josse-ten-Noode', 464: 'Saint-Léger', 465: 'Saint-Nicolas', 466: 'Sambreville', 467: 'Sankt Vith', 468: 'Schaarbeek', 469: 'Schaerbeek', 470: 'Schelle', 471: 'Scherpenheuvel-Zichem', 472: 'Schilde', 473: 'Schoten', 474: 'Seneffe', 475: 'Seraing', 476: 'Silly', 477: 'Sint-Agatha-Berchem', 478: 'Sint-Amands', 479: 'Sint-Genesius-Rode', 480: 'Sint-Gillis', 481: 'Sint-Gillis-Waas', 482: 'Sint-Jans-Molenbeek', 483: 'Sint-Joost-ten-Node', 484: 'Sint-Katelijne-Waver', 485: 'Sint-Lambrechts-Woluwe', 486: 'Sint-Laureins', 487: 'Sint-Lievens-Houtem', 488: 'Sint-Martens-Latem', 489: 'Sint-Niklaas', 490: 'Sint-Pieters-Leeuw', 491: 'Sint-Pieters-Woluwe', 492: 'Sint-Truiden', 493: 'Sivry-Rance', 494: 'Soignies', 495: 'Sombreffe', 496: 'Somme-Leuze', 497: 'Soumagne', 498: 'Spa', 499: 'Spiere-Helkijn', 500: 'Sprimont', 501: 'Stabroek', 502: 'Staden', 503: 'Stavelot', 504: 'Steenokkerzeel', 505: 'Stekene', 506: 'Stoumont', 507: 'Tellin', 508: 'Temse', 509: 'Tenneville', 510: 'Ternat', 511: 'Tervuren', 512: 'Tessenderlo', 513: 'Theux', 514: 'Thimister-Clermont', 515: 'Thuin', 516: 'Tielt', 517: 'Tielt-Winge', 518: 'Tienen', 519: 'Tinlot', 520: 'Tintigny', 521: 'Tongeren', 522: 'Torhout', 523: 'Tournai', 524: 'Tremelo', 525: 'Trois-Ponts', 526: 'Trooz', 527: 'Tubize', 528: 'Turnhout', 529: 'Uccle', 530: 'Ukkel', 531: 'Vaux-sur-Sûre', 532: 'Verlaine', 533: 'Verviers', 534: 'Veurne', 535: 'Vielsalm', 536: 'Villers-la-Ville', 537: 'Villers-le-Bouillet', 538: 'Vilvoorde', 539: 'Viroinval', 540: 'Virton', 541: 'Visé', 542: 'Vleteren', 543: 'Voeren', 544: 'Vorselaar', 545: 'Vorst', 546: 'Vosselaar', 547: 'Vresse-sur-Semois', 548: 'Waarschoot', 549: 'Waasmunster', 550: 'Wachtebeke', 551: 'Waimes', 552: 'Walcourt', 553: 'Walhain', 554: 'Wanze', 555: 'Waregem', 556: 'Waremme', 557: 'Wasseiges', 558: 'Waterloo', 559: 'Watermaal-Bosvoorde', 560: 'Watermael-Boitsfort', 561: 'Wavre', 562: 'Welkenraedt', 563: 'Wellen', 564: 'Wellin', 565: 'Wemmel', 566: 'Wervik', 567: 'Westerlo', 568: 'Wetteren', 569: 'Wevelgem', 570: 'Wezembeek-Oppem', 571: 'Wichelen', 572: 'Wielsbeke', 573: 'Wijnegem', 574: 'Willebroek', 575: 'Wingene', 576: 'Woluwé-Saint-Lambert', 577: 'Woluwe-Saint-Pierre', 578: 'Wommelgem', 579: 'Wortegem-Petegem', 580: 'Wuustwezel', 581: 'Yvoir', 582: 'Zandhoven', 583: 'Zaventem', 584: 'Zedelgem', 585: 'Zeebrugge', 586: 'Zele', 587: 'Zelzate', 588: 'Zemst', 589: 'Zingem', 590: 'Zoersel', 591: 'Zomergem', 592: 'Zonhoven', 593: 'Zonnebeke', 594: 'Zottegem', 595: 'Zoutleeuw', 596: 'Zuienkerke', 597: 'Zulte', 598: 'Zutendaal', 599: 'Zwalm', 600: 'Zwevegem', 601: 'Zwijndrecht'}, 'theta_e_base': {0: -7, 1: -7, 2: -7, 3: -7, 4: -7, 5: -8, 6: -7, 7: -7, 8: -8, 9: -10, 10: -8, 11: -7, 12: -8, 13: -9, 14: -8, 15: -9, 16: -7, 17: -7, 18: -7, 19: -7, 20: -8, 21: -10, 22: -8, 23: -7, 24: -7, 25: -9, 26: -7, 27: -10, 28: -9, 29: -9, 30: -7, 31: -7, 32: -8, 33: -9, 34: -8, 35: -9, 36: -8, 37: -7, 38: -10, 39: -8, 40: -9, 41: -7, 42: -7, 43: -8, 44: -7, 45: -7, 46: -8, 47: -7, 48: -7, 49: -8, 50: -7, 51: -7, 52: -8, 53: -7, 54: -7, 55: -10, 56: -10, 57: -7, 58: -7, 59: -9, 60: -7, 61: -10, 62: -7, 63: -8, 64: -6, 65: -8, 66: -8, 67: -7, 68: -7, 69: -7, 70: -7, 71: -7, 72: -7, 73: -7, 74: -9, 75: -7, 76: -7, 77: -8, 78: -7, 79: -7, 80: -8, 81: -7, 82: -8, 83: -8, 84: -6, 85: -8, 86: -7, 87: -7, 88: -7, 89: -7, 90: -7, 91: -11, 92: -8, 93: -10, 94: -11, 95: -7, 96: -9, 97: -8, 98: -8, 99: -8, 100: -8, 101: -8, 102: -8, 103: -7, 104: -9, 105: -9, 106: -9, 107: -9, 108: -7, 109: -8, 110: -7, 111: -8, 112: -7, 113: -9, 114: -8, 115: -8, 116: -7, 117: -10, 118: -6, 119: -6, 120: -7, 121: -7, 122: -7, 123: -7, 124: -7, 125: -7, 126: -8, 127: -7, 128: -7, 129: -8, 130: -7, 131: -7, 132: -8, 133: -7, 134: -9, 135: -9, 136: -9, 137: -7, 138: -7, 139: -7, 140: -9, 141: -8, 142: -7, 143: -7, 144: -8, 145: -7, 146: -7, 147: -7, 148: -8, 149: -9, 150: -7, 151: -8, 152: -8, 153: -8, 154: -7, 155: -8, 156: -9, 157: -7, 158: -9, 159: -7, 160: -7, 161: -8, 162: -8, 163: -10, 164: -8, 165: -9, 166: -8, 167: -8, 168: -9, 169: -8, 170: -7, 171: -8, 172: -9, 173: -9, 174: -8, 175: -7, 176: -8, 177: -8, 178: -7, 179: -9, 180: -7, 181: -7, 182: -7, 183: -10, 184: -8, 185: -8, 186: -7, 187: -8, 188: -8, 189: -8, 190: -7, 191: -7, 192: -8, 193: -9, 194: -8, 195: -7, 196: -7, 197: -7, 198: -10, 199: -8, 200: -7, 201: -7, 202: -8, 203: -7, 204: -7, 205: -10, 206: -8, 207: -7, 208: -8, 209: -7, 210: -9, 211: -9, 212: -8, 213: -8, 214: -8, 215: -7, 216: -7, 217: -9, 218: -9, 219: -8, 220: -7, 221: -7, 222: -7, 223: -7, 224: -7, 225: -10, 226: -7, 227: -8, 228: -8, 229: -7, 230: -7, 231: -8, 232: -8, 233: -8, 234: -8, 235: -9, 236: -7, 237: -8, 238: -7, 239: -7, 240: -7, 241: -7, 242: -7, 243: -7, 244: -7, 245: -8, 246: -7, 247: -9, 248: -10, 249: -8, 250: -7, 251: -9, 252: -7, 253: -7, 254: -8, 255: -8, 256: -7, 257: -7, 258: -8, 259: -7, 260: -8, 261: -7, 262: -7, 263: -7, 264: -10, 265: -8, 266: -7, 267: -7, 268: -8, 269: -7, 270: -8, 271: -7, 272: -8, 273: -7, 274: -7, 275: -8, 276: -7, 277: -9, 278: -8, 279: -7, 280: -7, 281: -6, 282: -7, 283: -7, 284: -6, 285: -7, 286: -7, 287: -7, 288: -7, 289: -7, 290: -7, 291: -7, 292: -7, 293: -7, 294: -7, 295: -8, 296: -7, 297: -7, 298: -10, 299: -8, 300: -7, 301: -7, 302: -7, 303: -7, 304: -8, 305: -7, 306: -7, 307: -7, 308: -7, 309: -10, 310: -7, 311: -7, 312: -7, 313: -8, 314: -8, 315: -7, 316: -7, 317: -7, 318: -10, 319: -10, 320: -7, 321: -7, 322: -8, 323: -7, 324: -7, 325: -10, 326: -8, 327: -9, 328: -7, 329: -7, 330: -7, 331: -7, 332: -8, 333: -7, 334: -7, 335: -8, 336: -7, 337: -9, 338: -7, 339: -7, 340: -7, 341: -8, 342: -7, 343: -8, 344: -7, 345: -7, 346: -7, 347: -8, 348: -10, 349: -8, 350: -10, 351: -9, 352: -9, 353: -10, 354: -7, 355: -8, 356: -8, 357: -7, 358: -9, 359: -7, 360: -7, 361: -8, 362: -7, 363: -7, 364: -8, 365: -7, 366: -9, 367: -9, 368: -7, 369: -6, 370: -9, 371: -7, 372: -8, 373: -7, 374: -9, 375: -7, 376: -7, 377: -8, 378: -8, 379: -7, 380: -8, 381: -7, 382: -7, 383: -9, 384: -8, 385: -9, 386: -9, 387: -7, 388: -8, 389: -10, 390: -9, 391: -7, 392: -7, 393: -7, 394: -6, 395: -7, 396: -7, 397: -8, 398: -9, 399: -8, 400: -9, 401: -9, 402: -6, 403: -7, 404: -7, 405: -7, 406: -8, 407: -7, 408: -8, 409: -7, 410: -7, 411: -7, 412: -7, 413: -7, 414: -7, 415: -8, 416: -9, 417: -7, 418: -7, 419: -8, 420: -10, 421: -7, 422: -8, 423: -7, 424: -9, 425: -7, 426: -8, 427: -9, 428: -7, 429: -9, 430: -8, 431: -7, 432: -8, 433: -7, 434: -7, 435: -7, 436: -8, 437: -7, 438: -9, 439: -8, 440: -7, 441: -8, 442: -7, 443: -8, 444: -9, 445: -8, 446: -7, 447: -8, 448: -7, 449: -9, 450: -7, 451: -7, 452: -7, 453: -7, 454: -9, 455: -7, 456: -7, 457: -7, 458: -10, 459: -8, 460: -7, 461: -7, 462: -10, 463: -7, 464: -9, 465: -8, 466: -8, 467: -10, 468: -7, 469: -7, 470: -7, 471: -8, 472: -8, 473: -8, 474: -8, 475: -8, 476: -7, 477: -7, 478: -7, 479: -7, 480: -7, 481: -7, 482: -7, 483: -7, 484: -7, 485: -7, 486: -7, 487: -7, 488: -7, 489: -7, 490: -7, 491: -7, 492: -7, 493: -9, 494: -7, 495: -8, 496: -9, 497: -9, 498: -9, 499: -7, 500: -9, 501: -7, 502: -7, 503: -10, 504: -7, 505: -7, 506: -9, 507: -9, 508: -7, 509: -10, 510: -7, 511: -7, 512: -8, 513: -9, 514: -9, 515: -8, 516: -7, 517: -7, 518: -7, 519: -9, 520: -10, 521: -7, 522: -7, 523: -7, 524: -7, 525: -10, 526: -8, 527: -7, 528: -8, 529: -7, 530: -7, 531: -10, 532: -8, 533: -9, 534: -7, 535: -10, 536: -8, 537: -8, 538: -7, 539: -9, 540: -9, 541: -7, 542: -7, 543: -8, 544: -8, 545: -7, 546: -8, 547: -10, 548: -7, 549: -7, 550: -7, 551: -11, 552: -8, 553: -8, 554: -8, 555: -7, 556: -8, 557: -8, 558: -8, 559: -7, 560: -7, 561: -7, 562: -9, 563: -7, 564: -9, 565: -7, 566: -7, 567: -8, 568: -7, 569: -7, 570: -7, 571: -7, 572: -7, 573: -7, 574: -7, 575: -7, 576: -7, 577: -7, 578: -7, 579: -7, 580: -8, 581: -9, 582: -8, 583: -7, 584: -7, 585: -6, 586: -7, 587: -7, 588: -7, 589: -7, 590: -8, 591: -7, 592: -8, 593: -7, 594: -7, 595: -7, 596: -7, 597: -7, 598: -8, 599: -7, 600: -7, 601: -7}, 'theta_e_min': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: -1, 6: 0, 7: 0, 8: -1, 9: -3, 10: -1, 11: 0, 12: -1, 13: -2, 14: -1, 15: -2, 16: 0, 17: 0, 18: 0, 19: 0, 20: -1, 21: -3, 22: -1, 23: 0, 24: 0, 25: -2, 26: 0, 27: -3, 28: -2, 29: -2, 30: 0, 31: 0, 32: -1, 33: -2, 34: -1, 35: -2, 36: -1, 37: 0, 38: -3, 39: -1, 40: -2, 41: 0, 42: 0, 43: -1, 44: 0, 45: 0, 46: -1, 47: 0, 48: 0, 49: -1, 50: 0, 51: 0, 52: -1, 53: 0, 54: 0, 55: -3, 56: -3, 57: 0, 58: 0, 59: -2, 60: 0, 61: -3, 62: 0, 63: -1, 64: 0, 65: -1, 66: -1, 67: 0, 68: 0, 69: 0, 70: 0, 71: 0, 72: 0, 73: 0, 74: -2, 75: 0, 76: 0, 77: -1, 78: 0, 79: 0, 80: -1, 81: 0, 82: -1, 83: -1, 84: 0, 85: -1, 86: 0, 87: 0, 88: 0, 89: 0, 90: 0, 91: -3, 92: -1, 93: -3, 94: -3, 95: 0, 96: -2, 97: -1, 98: -1, 99: -1, 100: -1, 101: -1, 102: -1, 103: 0, 104: -2, 105: -2, 106: -2, 107: -2, 108: 0, 109: -1, 110: 0, 111: -1, 112: 0, 113: -2, 114: -1, 115: -1, 116: 0, 117: -3, 118: 0, 119: 0, 120: 0, 121: 0, 122: 0, 123: 0, 124: 0, 125: 0, 126: -1, 127: 0, 128: 0, 129: -1, 130: 0, 131: 0, 132: -1, 133: 0, 134: -2, 135: -2, 136: -2, 137: 0, 138: 0, 139: 0, 140: -2, 141: -1, 142: 0, 143: 0, 144: -1, 145: 0, 146: 0, 147: 0, 148: -1, 149: -2, 150: 0, 151: -1, 152: -1, 153: -1, 154: 0, 155: -1, 156: -2, 157: 0, 158: -2, 159: 0, 160: 0, 161: -1, 162: -1, 163: -3, 164: -1, 165: -2, 166: -1, 167: -1, 168: -2, 169: -1, 170: 0, 171: -1, 172: -2, 173: -2, 174: -1, 175: 0, 176: -1, 177: -1, 178: 0, 179: -2, 180: 0, 181: 0, 182: 0, 183: -3, 184: -1, 185: -1, 186: 0, 187: -1, 188: -1, 189: -1, 190: 0, 191: 0, 192: -1, 193: -2, 194: -1, 195: 0, 196: 0, 197: 0, 198: -3, 199: -1, 200: 0, 201: 0, 202: -1, 203: 0, 204: 0, 205: -3, 206: -1, 207: 0, 208: -1, 209: 0, 210: -2, 211: -2, 212: -1, 213: -1, 214: -1, 215: 0, 216: 0, 217: -2, 218: -2, 219: -1, 220: 0, 221: 0, 222: 0, 223: 0, 224: 0, 225: -3, 226: 0, 227: -1, 228: -1, 229: 0, 230: 0, 231: -1, 232: -1, 233: -1, 234: -1, 235: -2, 236: 0, 237: -1, 238: 0, 239: 0, 240: 0, 241: 0, 242: 0, 243: 0, 244: 0, 245: -1, 246: 0, 247: -2, 248: -3, 249: -1, 250: 0, 251: -2, 252: 0, 253: 0, 254: -1, 255: -1, 256: 0, 257: 0, 258: -1, 259: 0, 260: -1, 261: 0, 262: 0, 263: 0, 264: -3, 265: -1, 266: 0, 267: 0, 268: -1, 269: 0, 270: -1, 271: 0, 272: -1, 273: 0, 274: 0, 275: -1, 276: 0, 277: -2, 278: -1, 279: 0, 280: 0, 281: 0, 282: 0, 283: 0, 284: 0, 285: 0, 286: 0, 287: 0, 288: 0, 289: 0, 290: 0, 291: 0, 292: 0, 293: 0, 294: 0, 295: -1, 296: 0, 297: 0, 298: -3, 299: -1, 300: 0, 301: 0, 302: 0, 303: 0, 304: -1, 305: 0, 306: 0, 307: 0, 308: 0, 309: -3, 310: 0, 311: 0, 312: 0, 313: -1, 314: -1, 315: 0, 316: 0, 317: 0, 318: -3, 319: -3, 320: 0, 321: 0, 322: -1, 323: 0, 324: 0, 325: -3, 326: -1, 327: -2, 328: 0, 329: 0, 330: 0, 331: 0, 332: -1, 333: 0, 334: 0, 335: -1, 336: 0, 337: -2, 338: 0, 339: 0, 340: 0, 341: -1, 342: 0, 343: -1, 344: 0, 345: 0, 346: 0, 347: -1, 348: -3, 349: -1, 350: -3, 351: -2, 352: -2, 353: -3, 354: 0, 355: -1, 356: -1, 357: 0, 358: -2, 359: 0, 360: 0, 361: -1, 362: 0, 363: 0, 364: -1, 365: 0, 366: -2, 367: -2, 368: 0, 369: 0, 370: -2, 371: 0, 372: -1, 373: 0, 374: -2, 375: 0, 376: 0, 377: -1, 378: -1, 379: 0, 380: -1, 381: 0, 382: 0, 383: -2, 384: -1, 385: -2, 386: -2, 387: 0, 388: -1, 389: -3, 390: -2, 391: 0, 392: 0, 393: 0, 394: 0, 395: 0, 396: 0, 397: -1, 398: -2, 399: -1, 400: -2, 401: -2, 402: 0, 403: 0, 404: 0, 405: 0, 406: -1, 407: 0, 408: -1, 409: 0, 410: 0, 411: 0, 412: 0, 413: 0, 414: 0, 415: -1, 416: -2, 417: 0, 418: 0, 419: -1, 420: -3, 421: 0, 422: -1, 423: 0, 424: -2, 425: 0, 426: -1, 427: -2, 428: 0, 429: -2, 430: -1, 431: 0, 432: -1, 433: 0, 434: 0, 435: 0, 436: -1, 437: 0, 438: -2, 439: -1, 440: 0, 441: -1, 442: 0, 443: -1, 444: -2, 445: -1, 446: 0, 447: -1, 448: 0, 449: -2, 450: 0, 451: 0, 452: 0, 453: 0, 454: -2, 455: 0, 456: 0, 457: 0, 458: -3, 459: -1, 460: 0, 461: 0, 462: -3, 463: 0, 464: -2, 465: -1, 466: -1, 467: -3, 468: 0, 469: 0, 470: 0, 471: -1, 472: -1, 473: -1, 474: -1, 475: -1, 476: 0, 477: 0, 478: 0, 479: 0, 480: 0, 481: 0, 482: 0, 483: 0, 484: 0, 485: 0, 486: 0, 487: 0, 488: 0, 489: 0, 490: 0, 491: 0, 492: 0, 493: -2, 494: 0, 495: -1, 496: -2, 497: -2, 498: -2, 499: 0, 500: -2, 501: 0, 502: 0, 503: -3, 504: 0, 505: 0, 506: -2, 507: -2, 508: 0, 509: -3, 510: 0, 511: 0, 512: -1, 513: -2, 514: -2, 515: -1, 516: 0, 517: 0, 518: 0, 519: -2, 520: -3, 521: 0, 522: 0, 523: 0, 524: 0, 525: -3, 526: -1, 527: 0, 528: -1, 529: 0, 530: 0, 531: -3, 532: -1, 533: -2, 534: 0, 535: -3, 536: -1, 537: -1, 538: 0, 539: -2, 540: -2, 541: 0, 542: 0, 543: -1, 544: -1, 545: 0, 546: -1, 547: -3, 548: 0, 549: 0, 550: 0, 551: -3, 552: -1, 553: -1, 554: -1, 555: 0, 556: -1, 557: -1, 558: -1, 559: 0, 560: 0, 561: 0, 562: -2, 563: 0, 564: -2, 565: 0, 566: 0, 567: -1, 568: 0, 569: 0, 570: 0, 571: 0, 572: 0, 573: 0, 574: 0, 575: 0, 576: 0, 577: 0, 578: 0, 579: 0, 580: -1, 581: -2, 582: -1, 583: 0, 584: 0, 585: 0, 586: 0, 587: 0, 588: 0, 589: 0, 590: -1, 591: 0, 592: -1, 593: 0, 594: 0, 595: 0, 596: 0, 597: 0, 598: -1, 599: 0, 600: 0, 601: 0}, 'theta_e_avg': {0: 10, 1: 10, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10, 9: 8, 10: 10, 11: 10, 12: 10, 13: 9, 14: 10, 15: 9, 16: 10, 17: 10, 18: 10, 19: 10, 20: 10, 21: 8, 22: 10, 23: 10, 24: 10, 25: 9, 26: 10, 27: 8, 28: 9, 29: 9, 30: 10, 31: 10, 32: 10, 33: 9, 34: 10, 35: 9, 36: 10, 37: 10, 38: 8, 39: 10, 40: 9, 41: 10, 42: 10, 43: 10, 44: 10, 45: 10, 46: 10, 47: 10, 48: 10, 49: 10, 50: 10, 51: 10, 52: 10, 53: 10, 54: 10, 55: 8, 56: 8, 57: 10, 58: 10, 59: 9, 60: 10, 61: 8, 62: 10, 63: 10, 64: 10, 65: 10, 66: 10, 67: 10, 68: 10, 69: 10, 70: 10, 71: 10, 72: 10, 73: 10, 74: 9, 75: 10, 76: 10, 77: 10, 78: 10, 79: 10, 80: 10, 81: 10, 82: 10, 83: 10, 84: 10, 85: 10, 86: 10, 87: 10, 88: 10, 89: 10, 90: 10, 91: 7, 92: 10, 93: 8, 94: 7, 95: 10, 96: 9, 97: 10, 98: 10, 99: 10, 100: 10, 101: 10, 102: 10, 103: 10, 104: 9, 105: 9, 106: 9, 107: 9, 108: 10, 109: 10, 110: 10, 111: 10, 112: 10, 113: 9, 114: 10, 115: 10, 116: 10, 117: 8, 118: 10, 119: 10, 120: 10, 121: 10, 122: 10, 123: 10, 124: 10, 125: 10, 126: 10, 127: 10, 128: 10, 129: 10, 130: 10, 131: 10, 132: 10, 133: 10, 134: 9, 135: 9, 136: 9, 137: 10, 138: 10, 139: 10, 140: 9, 141: 10, 142: 10, 143: 10, 144: 10, 145: 10, 146: 10, 147: 10, 148: 10, 149: 9, 150: 10, 151: 10, 152: 10, 153: 10, 154: 10, 155: 10, 156: 9, 157: 10, 158: 9, 159: 10, 160: 10, 161: 10, 162: 10, 163: 8, 164: 10, 165: 9, 166: 10, 167: 10, 168: 9, 169: 10, 170: 10, 171: 10, 172: 9, 173: 9, 174: 10, 175: 10, 176: 10, 177: 10, 178: 10, 179: 9, 180: 10, 181: 10, 182: 10, 183: 8, 184: 10, 185: 10, 186: 10, 187: 10, 188: 10, 189: 10, 190: 10, 191: 10, 192: 10, 193: 9, 194: 10, 195: 10, 196: 10, 197: 10, 198: 8, 199: 10, 200: 10, 201: 10, 202: 10, 203: 10, 204: 10, 205: 8, 206: 10, 207: 10, 208: 10, 209: 10, 210: 9, 211: 9, 212: 10, 213: 10, 214: 10, 215: 10, 216: 10, 217: 9, 218: 9, 219: 10, 220: 10, 221: 10, 222: 10, 223: 10, 224: 10, 225: 8, 226: 10, 227: 10, 228: 10, 229: 10, 230: 10, 231: 10, 232: 10, 233: 10, 234: 10, 235: 9, 236: 10, 237: 10, 238: 10, 239: 10, 240: 10, 241: 10, 242: 10, 243: 10, 244: 10, 245: 10, 246: 10, 247: 9, 248: 8, 249: 10, 250: 10, 251: 9, 252: 10, 253: 10, 254: 10, 255: 10, 256: 10, 257: 10, 258: 10, 259: 10, 260: 10, 261: 10, 262: 10, 263: 10, 264: 8, 265: 10, 266: 10, 267: 10, 268: 10, 269: 10, 270: 10, 271: 10, 272: 10, 273: 10, 274: 10, 275: 10, 276: 10, 277: 9, 278: 10, 279: 10, 280: 10, 281: 10, 282: 10, 283: 10, 284: 10, 285: 10, 286: 10, 287: 10, 288: 10, 289: 10, 290: 10, 291: 10, 292: 10, 293: 10, 294: 10, 295: 10, 296: 10, 297: 10, 298: 8, 299: 10, 300: 10, 301: 10, 302: 10, 303: 10, 304: 10, 305: 10, 306: 10, 307: 10, 308: 10, 309: 8, 310: 10, 311: 10, 312: 10, 313: 10, 314: 10, 315: 10, 316: 10, 317: 10, 318: 8, 319: 8, 320: 10, 321: 10, 322: 10, 323: 10, 324: 10, 325: 8, 326: 10, 327: 9, 328: 10, 329: 10, 330: 10, 331: 10, 332: 10, 333: 10, 334: 10, 335: 10, 336: 10, 337: 9, 338: 10, 339: 10, 340: 10, 341: 10, 342: 10, 343: 10, 344: 10, 345: 10, 346: 10, 347: 10, 348: 8, 349: 10, 350: 8, 351: 9, 352: 9, 353: 8, 354: 10, 355: 10, 356: 10, 357: 10, 358: 9, 359: 10, 360: 10, 361: 10, 362: 10, 363: 10, 364: 10, 365: 10, 366: 9, 367: 9, 368: 10, 369: 10, 370: 9, 371: 10, 372: 10, 373: 10, 374: 9, 375: 10, 376: 10, 377: 10, 378: 10, 379: 10, 380: 10, 381: 10, 382: 10, 383: 9, 384: 10, 385: 9, 386: 9, 387: 10, 388: 10, 389: 8, 390: 9, 391: 10, 392: 10, 393: 10, 394: 10, 395: 10, 396: 10, 397: 10, 398: 9, 399: 10, 400: 9, 401: 9, 402: 10, 403: 10, 404: 10, 405: 10, 406: 10, 407: 10, 408: 10, 409: 10, 410: 10, 411: 10, 412: 10, 413: 10, 414: 10, 415: 10, 416: 9, 417: 10, 418: 10, 419: 10, 420: 8, 421: 10, 422: 10, 423: 10, 424: 9, 425: 10, 426: 10, 427: 9, 428: 10, 429: 9, 430: 10, 431: 10, 432: 10, 433: 10, 434: 10, 435: 10, 436: 10, 437: 10, 438: 9, 439: 10, 440: 10, 441: 10, 442: 10, 443: 10, 444: 9, 445: 10, 446: 10, 447: 10, 448: 10, 449: 9, 450: 10, 451: 10, 452: 10, 453: 10, 454: 9, 455: 10, 456: 10, 457: 10, 458: 8, 459: 10, 460: 10, 461: 10, 462: 8, 463: 10, 464: 9, 465: 10, 466: 10, 467: 8, 468: 10, 469: 10, 470: 10, 471: 10, 472: 10, 473: 10, 474: 10, 475: 10, 476: 10, 477: 10, 478: 10, 479: 10, 480: 10, 481: 10, 482: 10, 483: 10, 484: 10, 485: 10, 486: 10, 487: 10, 488: 10, 489: 10, 490: 10, 491: 10, 492: 10, 493: 9, 494: 10, 495: 10, 496: 9, 497: 9, 498: 9, 499: 10, 500: 9, 501: 10, 502: 10, 503: 8, 504: 10, 505: 10, 506: 9, 507: 9, 508: 10, 509: 8, 510: 10, 511: 10, 512: 10, 513: 9, 514: 9, 515: 10, 516: 10, 517: 10, 518: 10, 519: 9, 520: 8, 521: 10, 522: 10, 523: 10, 524: 10, 525: 8, 526: 10, 527: 10, 528: 10, 529: 10, 530: 10, 531: 8, 532: 10, 533: 9, 534: 10, 535: 8, 536: 10, 537: 10, 538: 10, 539: 9, 540: 9, 541: 10, 542: 10, 543: 10, 544: 10, 545: 10, 546: 10, 547: 8, 548: 10, 549: 10, 550: 10, 551: 7, 552: 10, 553: 10, 554: 10, 555: 10, 556: 10, 557: 10, 558: 10, 559: 10, 560: 10, 561: 10, 562: 9, 563: 10, 564: 9, 565: 10, 566: 10, 567: 10, 568: 10, 569: 10, 570: 10, 571: 10, 572: 10, 573: 10, 574: 10, 575: 10, 576: 10, 577: 10, 578: 10, 579: 10, 580: 10, 581: 9, 582: 10, 583: 10, 584: 10, 585: 10, 586: 10, 587: 10, 588: 10, 589: 10, 590: 10, 591: 10, 592: 10, 593: 10, 594: 10, 595: 10, 596: 10, 597: 10, 598: 10, 599: 10, 600: 10, 601: 10}}

we = pd.DataFrame.from_dict(we_dict)

# from October 1st to June 9th
t_ext_hbh = np.array([12.9, 11.9, 11. , 10. ,  9. ,  7.9,  6.6,  5.3,  3.9,  4.4,  6.4,  8.1,  9.7, 11.1, 12.3, 13.1, 13.3, 13.1, 12.3, 11.1, 10.1,  9.4,  8.7,  8. ,  7.3,  6.6,  6. ,  5.6,
        5.2,  4.8,  4.4,  4. ,  3.6,  4.1,  5.3,  6.7,  8.6, 10.4, 11.3, 11.7, 12.4, 12.7, 12.3, 11.3, 10.5, 10.2,  9.9,  9.6,  9.3,  9. ,  8.7,  8.4,  8.2,  8. ,  7.8,  7.6,
        7.3,  7.6,  8.9,  9.5,  9.4,  9.9, 10.4, 10.7, 11. , 11.2, 11.3, 11.2, 10.9, 10.4,  9.9,  9.4,  8.9,  8.4,  7.9,  7.4,  6.9,  6.4,  5.9,  5.5,  5. ,  5.6,  7.7,  9.8,
       11.5, 13.2, 14.5, 15.4, 15.8, 15.7, 14.9, 13.6, 12.6, 12. , 11.3, 10.7, 10. ,  9.4,  8.7,  8.1,  7.4,  6.8,  6.1,  5.4,  4.8,  5.4,  7.7, 10.1, 12.5, 14.6, 16.1, 17. ,
       17.2, 17. , 16.2, 14.9, 13.8, 13.1, 12.5, 11.8, 11.1, 10.4,  9.9,  9.3,  8.7,  8.2,  7.6,  7.1,  6.5,  7.2,  9.1, 11.2, 13.3, 14.4, 14.9, 15.6, 15.9, 15.5, 14.7, 13.6,
       12.8, 12.5, 12.2, 11.9, 11.6, 11.3, 11.1, 11. , 10.9, 10.8, 10.8, 10.7, 10.6, 10.8, 11.8, 13.1, 13.9, 14.2, 14.9, 15.6, 15.4, 14.7, 14.2, 13.8, 13.5, 13.1, 12.8, 12.4,
       12.1, 11.7, 11.3, 10.7, 10.1,  9.5,  8.9,  8.3,  7.7,  7.6,  8.4,  9.8, 10.9, 11.5, 12.2, 13.1, 13.8, 13.7, 13.1, 12.4, 12. , 11.9, 11.8, 11.7, 11.6, 11.5, 11.4, 11.4,
       11.4, 11.4, 11.4, 11.4, 11.4, 11.6, 11.7, 11.8, 12.3, 13. , 13.6, 14.1, 14.5, 14.6, 14.5, 14.2, 14. , 13.9, 13.9, 13.9, 13.9, 13.8, 13.8, 13.7, 13.7, 13.6, 13.5, 13.4,
       13.4, 13.5, 13.9, 14.2, 14.7, 16. , 17.3, 17.5, 17.2, 17.5, 17.6, 17.4, 17. , 16.7, 16.3, 16. , 15.7, 15.4, 15. , 14.6, 14.2, 13.8, 13.4, 13. , 12.6, 12.8, 13.5, 14.4,
       15.4, 16.6, 17.8, 19. , 19.9, 20.2, 20. , 19.1, 18.3, 17.9, 17.4, 17. , 16.5, 16.1, 15.7, 15.1, 14.6, 14.1, 13.6, 13.1, 12.6, 12.4, 12.9, 13.7, 14.3, 14.9, 15.8, 16.6,
       16.9, 16.7, 16.1, 15.5, 15.1, 14.9, 14.6, 14.3, 14.1, 13.8, 13.5, 13.2, 13. , 12.7, 12.4, 12.1, 11.9, 11.8, 12.1, 12.8, 13.9, 15.3, 16.3, 16.6, 16.8, 16.9, 16.6, 16. ,
       15.5, 15. , 14.5, 14.1, 13.6, 13.1, 12.6, 12.1, 11.6, 11. , 10.5,  9.9,  9.4,  9.5, 10.4, 11.7, 13.3, 15.1, 16.6, 17.5, 17.9, 17.9, 17.5, 16.5, 15.6, 15.2, 14.8, 14.3,
       13.9, 13.4, 13. , 12.7, 12.3, 12. , 11.6, 11.3, 10.9, 10.9, 11.8, 13.2, 14.4, 15.1, 15.6, 16.1, 15.8, 15.3, 14.9, 14.3, 13.8, 13.4, 12.9, 12.5, 12. , 11.6, 11.1, 10.5,
        9.9,  9.3,  8.7,  8.1,  7.5,  7.6,  8.3,  8.9,  9.9, 11.2, 11.8, 12.2, 13.2, 13.8, 13.6, 12.9, 12.3, 12. , 11.7, 11.4, 11. , 10.7, 10.4, 10.2, 10. ,  9.8,  9.5,  9.3,
        9.1,  9.1,  9.4, 10.2, 11.5, 12.7, 14. , 14.8, 15.2, 15.3, 15. , 14.3, 13.7, 13.3, 12.9, 12.5, 12. , 11.6, 11.2, 10.6, 10. ,  9.5,  8.9,  8.4,  7.8,  7.9,  8.8, 10.6,
       12.7, 14.1, 14.4, 14.6, 15.1, 15.2, 14.8, 14. , 13.2, 12.8, 12.4, 12. , 11.5, 11.1, 10.8, 10.6, 10.3, 10. ,  9.7,  9.4,  9.1,  9. ,  9.2,  9.5,  9.7,  9.9, 10.3, 10.6,
       10.7, 10.6, 10.4, 10.2, 10.1, 10.1, 10.2, 10.2, 10.3, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11. , 11.2, 11.7, 12.5, 13.1, 13.7, 14.4, 14.6, 14.2, 14.1, 14. , 13.6,
       13.3, 13.1, 12.8, 12.5, 12.3, 12. , 11.7, 11.2, 10.7, 10.3,  9.8,  9.4,  8.9,  8.9,  9.9, 11.4, 13.2, 14.7, 15.3, 15.6, 15.6, 15.2, 14.5, 13.7, 13.1, 12.7, 12.4, 12. ,
       11.7, 11.3, 11. , 10.6, 10.3, 10. ,  9.6,  9.3,  9. ,  8.9,  9. ,  9.2,  9.4,  9.5,  9.8, 10.2, 10.5, 10.6, 10.4, 10.2,  9.9,  9.7,  9.5,  9.3,  9.1,  8.8,  8.5,  8.1,
        7.6,  7.1,  6.6,  6.1,  5.6,  5.5,  5.8,  6.3,  7.2,  8.3,  9.1,  9.5,  9.6,  9.5,  9.1,  8.7,  8.4,  8.2,  7.9,  7.6,  7.4,  7.1,  6.8,  6.7,  6.5,  6.3,  6.1,  6. ,
        5.8,  5.9,  6.6,  7.8,  9.2, 10.4, 11.3, 11.7, 12. , 11.8, 11.1, 10.5, 10.1,  9.7,  9.3,  8.9,  8.5,  8.1,  7.7,  7.2,  6.7,  6.2,  5.7,  5.3,  4.8,  4.5,  4.8,  5.6,
        6.4,  7.1,  7.8,  8.3,  8.2,  7.8,  7.5,  7.2,  6.8,  6.5,  6.1,  5.7,  5.4,  5. ,  4.6,  4.2,  3.7,  3.2,  2.7,  2.3,  1.8,  1.5,  2.2,  3.5,  4.9,  6.1,  6.9,  7.6,
        8.2,  8.4,  8.1,  7.6,  7.3,  6.9,  6.6,  6.3,  6. ,  5.7,  5.4,  5.2,  5.1,  4.9,  4.7,  4.6,  4.4,  4.2,  5.2,  6.7,  7.2,  7.2,  7.3,  7.6,  7.9,  7.7,  7.1,  6.7,
        6.4,  6. ,  5.7,  5.3,  5. ,  4.7,  4.3,  3.9,  3.4,  3. ,  2.6,  2.1,  1.7,  1.3,  2.3,  4.3,  5.5,  6.2,  6.7,  7.1,  7.3,  7.2,  6.6,  6. ,  5.7,  5.4,  5.1,  4.9,
        4.6,  4.3,  4. ,  3.9,  3.8,  3.7,  3.6,  3.5,  3.4,  3.3,  4.2,  6.2,  7.7,  8.9, 10.3, 11.2, 11.5, 11.4, 10.6,  9.9,  9.5,  9.2,  8.9,  8.6,  8.3,  8. ,  7.6,  7.3,
        6.9,  6.6,  6.2,  5.9,  5.5,  5.2,  5.8,  6.4,  7. ,  8.6,  9.9, 10.9, 11.3, 10.9, 10.1,  9.5,  9.2,  8.8,  8.5,  8.2,  7.9,  7.6,  7.2,  6.8,  6.4,  6.1,  5.7,  5.3,
        5. ,  4.6,  4.9,  6.2,  7.7,  8.7,  9.5,  9.7,  9.4,  8.9,  8.3,  7.7,  7.3,  6.8,  6.4,  5.9,  5.5,  5. ,  4.4,  3.4,  2.3,  1.2,  0.2, -1. , -2. , -3.1, -2.8, -1.4,
        0.1,  1.4,  2.2,  2.6,  2.5,  1.7,  0.8,  0.1, -0.3, -0.7, -1.1, -1.5, -1.9, -2.3, -2.4, -2.2, -2. , -1.8, -1.6, -1.4, -1.2, -1. , -0.8, -0.3,  1.2,  2.7,  3.9,  5. ,
        5.6,  5.8,  5.5,  5. ,  4.8,  4.6,  4.4,  4.2,  4. ,  3.8,  3.5,  3.2,  2.8,  2.5,  2.1,  1.7,  1.4,  1. ,  1.4,  2.3,  3.2,  3.9,  4. ,  3.7,  3.4,  3.3,  3.2,  3. ,
        2.9,  2.8,  2.7,  2.6,  2.5,  2.4,  2.3,  2.3,  2.3,  2.3,  2.3,  2.3,  2.2,  2.2,  2.5,  3.1,  3.7,  3.9,  4. ,  4. ,  4. ,  3.8,  3.5,  3.4,  3.4,  3.3,  3.3,  3.2,
        3.2,  3.1,  3.1,  3. ,  3. ,  2.9,  2.9,  2.9,  2.8,  2.8,  3. ,  3.7,  4.3,  4.6,  5.1,  5.1,  4.9,  4.9,  4.7,  4.5,  4.3,  4.2,  4.1,  4. ,  3.9,  3.8,  3.6,  3.4,
        3.2,  3. ,  2.8,  2.6,  2.4,  2.2,  2.4,  3.1,  4.1,  4.9,  5.4,  6.2,  6.7,  6.8,  6.7,  6.4,  6.2,  6. ,  5.9,  5.7,  5.5,  5.3,  5.2,  5. ,  4.9,  4.7,  4.6,  4.4,
        4.2,  4.1,  4.5,  5.1,  5.6,  5.9,  5.9,  6. ,  6.1,  6. ,  5.6,  5.4,  5.6,  5.8,  6. ,  6.2,  6.3,  6.5,  6.7,  6.8,  6.9,  7.1,  7.2,  7.3,  7.4,  7.5,  7.9,  8.5,
        9.1,  9.4,  9.1,  8.8,  8.9,  9. ,  8.9,  8.8,  8.8,  8.8,  8.8,  8.8,  8.8,  8.8,  8.8,  8.7,  8.7,  8.6,  8.5,  8.4,  8.3,  8.3,  8.2,  8.8,  9.7, 10.1, 10.6, 10.9,
       10.8, 10.6, 10.2, 10. ,  9.9,  9.9,  9.9,  9.8,  9.8,  9.8,  9.7,  9.6,  9.6,  9.5,  9.5,  9.4,  9.4,  9.3,  9.7, 10.5, 11.1, 11.5, 11.8, 12. , 11.7, 11.3, 11.1, 10.9,
       10.8, 10.7, 10.7, 10.6, 10.6, 10.5, 10.4, 10.3, 10.3, 10.2, 10.1, 10. , 10. ,  9.9, 10.8, 12.1, 13.3, 14.5, 15. , 15.4, 15.7, 15.4, 14.6, 14. , 13.4, 12.9, 12.4, 11.9,
       11.4, 10.8, 10.1,  9.3,  8.4,  7.6,  6.7,  5.8,  4.9,  4.1,  4.3,  5.6,  6.4,  6.9,  7.9,  8.9,  9.8, 10.2,  9.7,  9. ,  8.4,  7.9,  7.3,  6.8,  6.2,  5.7,  5.2,  4.8,
        4.3,  3.9,  3.5,  3.1,  2.6,  2.2,  3. ,  5.1,  6.9,  8.2,  9.2,  9.9, 10.3, 10.1,  9.2,  8.4,  8. ,  7.5,  7. ,  6.6,  6.1,  5.6,  5.2,  4.7,  4.3,  3.9,  3.5,  3.1,
        2.7,  2.3,  2.9,  4.5,  6.3,  8.3, 10.2, 11.3, 11.7, 11.2,  9.9,  9. ,  8.5,  8. ,  7.4,  6.9,  6.4,  5.9,  5.3,  4.6,  3.9,  3.2,  2.6,  1.9,  1.2,  0.5,  0.8,  2.2,
        3.8,  5.7,  7. ,  7.4,  7.4,  6.9,  5.8,  5.2,  5. ,  4.9,  4.8,  4.6,  4.5,  4.4,  4.3,  4.4,  4.5,  4.6,  4.7,  4.8,  4.8,  4.9,  5.4,  6.5,  7.1,  7.2,  7.8,  8.6,
        9. ,  8.8,  8.4,  8. ,  7.8,  7.5,  7.3,  7.1,  6.8,  6.6,  6.2,  5.8,  5.4,  5. ,  4.6,  4.2,  3.8,  3.4,  3.9,  4.9,  6.5,  8.7,  9.9, 10.3, 10.5, 10.2,  9.4,  8.7,
        8.4,  8. ,  7.7,  7.4,  7. ,  6.7,  6.4,  6. ,  5.7,  5.3,  5. ,  4.6,  4.2,  3.9,  3.9,  4. ,  4.7,  5.8,  6.5,  6.8,  6.7,  6.3,  5.7,  5.6,  6. ,  6.4,  6.8,  7.2,
        7.6,  8. ,  8.3,  8.5,  8.7,  8.9,  9.2,  9.4,  9.6,  9.8, 10.1, 10.6, 11.5, 12.5, 13.1, 13.6, 14. , 14. , 13.6, 13.2, 12.9, 12.5, 12.2, 11.9, 11.6, 11.3, 10.9, 10.3,
        9.8,  9.2,  8.7,  8.1,  7.6,  7. ,  7.3,  8.8, 10.5, 11.8, 12.3, 12.8, 13. , 12.4, 11.5, 10.9, 10.5, 10.2,  9.9,  9.6,  9.2,  8.9,  8.6,  8.3,  8. ,  7.7,  7.4,  7.1,
        6.8,  6.4,  6.5,  7.1,  7.7,  8.2,  8.8,  9.2,  9.4,  9.3,  9. ,  8.6,  8.5,  8.3,  8.1,  7.9,  7.8,  7.6,  7.5,  7.4,  7.3,  7.2,  7.1,  7. ,  6.9,  6.9,  7. ,  7.6,
        8.5,  9.5, 10.5, 11.2, 10.9, 10.2,  9.8,  9.5,  9.4,  9.3,  9.2,  9.1,  9. ,  8.9,  8.7,  8.5,  8.2,  8. ,  7.8,  7.6,  7.4,  7.2,  7.2,  7.5,  8. ,  9.2, 10.6, 11.4,
       11.8, 11.6, 11. , 10.4, 10. ,  9.6,  9.2,  8.8,  8.4,  8. ,  7.2,  6.2,  5. ,  3.9,  2.8,  1.6,  0.5, -0.7, -0.8,  0.3,  1.8,  3.4,  5. ,  5.8,  5.9,  5.3,  4.2,  3.6,
        3.7,  3.9,  4.1,  4.2,  4.4,  4.6,  4.8,  5.1,  5.5,  5.8,  6.2,  6.5,  6.9,  7.2,  7.4,  7.6,  7.9,  8.2,  8.5,  8.7,  8.8,  8.7,  8.6,  8.5,  8.4,  8.4,  8.4,  8.3,
        8.3,  8.2,  8. ,  7.4,  6.9,  6.4,  5.9,  5.3,  4.8,  4.3,  4.3,  5.2,  6.3,  7.3,  8.1,  8.3,  8.1,  7.7,  7.1,  6.6,  6.3,  6. ,  5.7,  5.4,  5.1,  4.7,  4.4,  4.1,
        3.9,  3.6,  3.4,  3.1,  2.8,  2.6,  2.7,  3.6,  4.6,  5.3,  6.3,  7.5,  8.3,  8.2,  7.4,  6.8,  6.4,  5.9,  5.5,  5.1,  4.7,  4.2,  3.7,  3.1,  2.5,  2. ,  1.4,  0.8,
        0.2, -0.4, -0.6,  0.2,  1.6,  3. ,  3.3,  2.8,  2.7,  2.5,  2. ,  1.6,  1.3,  1.1,  0.8,  0.5,  0.2, -0.1, -0.4, -0.7, -1. , -1.3, -1.6, -1.9, -2.1, -2.4, -2.5, -2.4,
       -1.7, -0.9, -0.4,  0.1,  0.6,  0.5, -0.1, -0.4, -0.3, -0.3, -0.3, -0.3, -0.3, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2,  0.1,  0.9,  1.4,  1.3,  1. ,
        0.9,  0.7,  0.5,  0.6,  1. ,  1.4,  1.7,  2.1,  2.5,  2.8,  3. ,  3.1,  3.2,  3.2,  3.3,  3.3,  3.4,  3.4,  3.4,  3.8,  4.4,  5. ,  5.3,  5.5,  5.6,  5.6,  5.5,  5.2,
        5. ,  4.8,  4.6,  4.4,  4.2,  3.9,  3.6,  3. ,  2.4,  1.8,  1.2,  0.6, -0.1, -0.7, -1.3, -1.3, -0.9, -0.6, -0.4, -0.4, -0.5, -0.8, -1.2, -1.2, -0.9, -0.6, -0.3,  0.1,
        0.4,  0.7,  1. ,  1.3,  1.6,  1.9,  2.2,  2.5,  2.7,  3. ,  3.3,  3.8,  4.9,  5.7,  6. ,  6.7,  7.1,  7.1,  6.7,  6.3,  6.3,  6.2,  6.1,  6.1,  6. ,  5.9,  5.8,  5.5,
        5.3,  5.1,  4.8,  4.6,  4.4,  4.1,  3.9,  4.3,  5.2,  5.7,  6. ,  6.5,  6.6,  6.4,  6. ,  5.6,  5.3,  5.1,  4.8,  4.6,  4.3,  4.1,  3.8,  3.4,  3.1,  2.7,  2.4,  2. ,
        1.6,  1.3,  0.9,  1.2,  1.5,  1.5,  2. ,  2.5,  2.9,  2.7,  2.2,  1.9,  1.8,  1.6,  1.5,  1.3,  1.1,  1. ,  0.9,  0.7,  0.6,  0.5,  0.4,  0.2,  0.1, -0.1, -0.2,  0.2,
        0.8,  1.3,  1.9,  2.4,  2.6,  2.6,  2.2,  1.9,  1.8,  1.7,  1.6,  1.5,  1.3,  1.2,  1.1,  1. ,  0.9,  0.8,  0.7,  0.5,  0.4,  0.3,  0.2,  0.8,  2.2,  3.6,  4.4,  4.7,
        4.9,  4.8,  4.2,  3.6,  3.3,  3.1,  2.8,  2.6,  2.3,  2. ,  1.7,  1.4,  1.2,  0.9,  0.6,  0.3,  0.1, -0.3, -0.5,  0.4,  2.4,  4.5,  6.1,  7.1,  6.9,  5.8,  4.6,  3.9,
        3.6,  3.3,  3. ,  2.7,  2.4,  2.1,  1.6,  1.1,  0.5, -0.1, -0.6, -1.2, -1.7, -2.3, -2.9, -2.2,  0.2,  2.6,  4.4,  5.6,  5.9,  5.2,  3.6,  2.7,  2.7,  2.6,  2.5,  2.4,
        2.3,  2.2,  2.2,  2.1,  2.1,  2.1,  2.1,  2.1,  2.2,  2.2,  2.2,  2.7,  3.2,  3.7,  4.3,  4.7,  4.5,  4.2,  3.9,  3.6,  3.5,  3.4,  3.2,  3.1,  2.9,  2.8,  2.5,  1.9,
        1.4,  0.8,  0.2, -0.4, -0.9, -1.5, -2.1, -2.2, -1.8, -1.3, -1. , -0.8, -0.9, -1.1, -1.5, -1.7, -1.6, -1.5, -1.4, -1.3, -1.3, -1.2, -1. , -0.8, -0.5, -0.3,  0.1,  0.4,
        0.7,  0.9,  1.2,  2.7,  4.1,  4.5,  5.6,  6.9,  7.7,  7.5,  6.6,  5.9,  5.5,  5.1,  4.7,  4.3,  3.9,  3.5,  2.9,  2.1,  1.2,  0.4, -0.6, -1.5, -2.3, -3.2, -4.1, -3.2,
       -0.7,  1.2,  2.4,  3.1,  3.2,  2.3,  0.8,  0.2,  0.5,  0.7,  1. ,  1.3,  1.6,  1.9,  2.2,  2.4,  2.7,  3. ,  3.3,  3.6,  3.9,  4.2,  4.5,  4.9,  5.4,  5.8,  6.4,  7.2,
        7.6,  7.5,  7.1,  6.9,  6.9,  6.9,  6.9,  6.9,  7. ,  7. ,  6.9,  6.7,  6.5,  6.3,  6.1,  6. ,  5.8,  5.6,  5.4,  5.9,  7.3,  8.7,  9.4,  9.7,  9.7,  9.3,  8.7,  8.3,
        8.3,  8.3,  8.4,  8.4,  8.5,  8.5,  8.5,  8.6,  8.6,  8.7,  8.7,  8.8,  8.8,  8.9,  8.9,  9.8, 11.4, 12.5, 12.7, 12.9, 13.1, 12.8, 12.4, 11.9, 11.5, 11.2, 10.9, 10.5,
       10.2,  9.9,  9.4,  8.7,  8. ,  7.2,  6.5,  5.7,  5. ,  4.2,  3.5,  4.5,  6.6,  7.5,  8. ,  8.3,  8. ,  7.2,  6.2,  5.9,  6.3,  6.8,  7.3,  7.7,  8.2,  8.7,  9. ,  9.3,
        9.6,  9.9, 10.2, 10.5, 10.8, 11.1, 11.4, 11.7, 11.9, 11.9, 12. , 12.1, 12.2, 12.2, 12.2, 12.1, 12.1, 12.1, 12.1, 12.1, 12.1, 12.1, 11.9, 11.5, 11.1, 10.7, 10.3,  9.9,
        9.5,  9.1,  8.7,  8.8,  9.4,  9.9, 10.3, 10.5, 10.4, 10. ,  9.6,  9.4,  9.2,  9. ,  8.9,  8.7,  8.5,  8.3,  8.1,  7.8,  7.6,  7.3,  7.1,  6.8,  6.6,  6.3,  6.1,  7.2,
        9. , 10.4, 11.8, 12.8, 13.2, 12.9, 11.9, 10.9, 10.2,  9.6,  9. ,  8.3,  7.7,  7.1,  6.3,  5.1,  3.8,  2.6,  1.4,  0.1, -1.2, -2.4, -3.6, -4.1, -4.1, -4.1, -3.1, -1.6,
       -0.5, -0.4, -1.1, -1.5, -0.9, -0.3,  0.3,  0.9,  1.5,  2.1,  2.6,  3.1,  3.6,  4.1,  4.7,  5.2,  5.7,  6.2,  6.7,  7.1,  7.2,  7.6,  8.4,  9.2,  9.5,  9.4,  9.1,  8.9,
        8.8,  8.7,  8.7,  8.6,  8.5,  8.4,  7.9,  7.1,  6.2,  5.3,  4.5,  3.6,  2.7,  1.8,  1. ,  1.6,  2.9,  3.6,  5.1,  6.5,  7. ,  6.5,  5.3,  4.3,  3.8,  3.2,  2.7,  2.1,
        1.5,  1. ,  0.4, -0.2, -0.7, -1.3, -1.9, -2.5, -3. , -3.6, -4.2, -4. , -3.4, -3.3, -3.1, -3.1, -3.2, -3.3, -3.6, -3.8, -4.1, -4.3, -4.5, -4.7, -4.9, -5.1, -5.3, -5.3,
       -5.4, -5.5, -5.6, -5.6, -5.7, -5.8, -5.8, -5.5, -5.1, -4.6, -3.7, -3.1, -2.9, -3.1, -3.7, -4.1, -4.2, -4.3, -4.5, -4.6, -4.7, -4.8, -4.9, -4.9, -5. , -5. , -5. , -5.1,
       -5.1, -5.2, -5.2, -3.9, -1.8,  0. ,  1.5,  2.2,  2.4,  1.7,  0.5, -0.2, -0.3, -0.5, -0.7, -0.8, -1. , -1.1, -1.4, -1.7, -2.1, -2.4, -2.8, -3.1, -3.5, -3.8, -4.2, -3. ,
       -1. ,  0.2,  0.7,  0.8,  0.8,  0.5, -0.3, -0.8, -0.8, -0.9, -1. , -1. , -1.1, -1.1, -1.1, -1.1, -1. , -1. , -1. , -0.9, -0.9, -0.9, -0.8, -0.4,  0.4,  1. ,  1.4,  1.6,
        1.6,  1.5,  1.1,  1.1,  1.5,  2. ,  2.5,  2.9,  3.4,  3.8,  4.1,  4.2,  4.3,  4.4,  4.6,  4.7,  4.8,  4.9,  5. ,  5.6,  6. ,  6.1,  6.6,  7.4,  8.1,  8.3,  8.1,  7.8,
        7.8,  7.8,  7.8,  7.7,  7.7,  7.7,  7.6,  7.5,  7.4,  7.3,  7.2,  7.2,  7.1,  7. ,  6.9,  7.2,  8.3,  9.5, 10. , 10.3, 10.2,  9.9,  9.5,  9.1,  8.9,  8.7,  8.4,  8.2,
        8. ,  7.8,  7.5,  7.2,  6.9,  6.6,  6.2,  5.9,  5.6,  5.2,  4.9,  5.1,  5.9,  6.3,  6.7,  6.9,  6.9,  6.8,  6.5,  6.2,  6. ,  5.7,  5.4,  5.2,  4.9,  4.5,  7.4, 10.2,
        9.9,  9.7,  9.4,  9.2,  8.9,  8.7,  8.4,  8.4,  8.5,  8.5,  8.5,  8.6,  8.7,  8.7,  8.6,  8.5,  8.5,  8.4,  8.3,  8.2,  8.2,  8.1,  7.9,  7.8,  7.6,  7.4,  7.3,  7.1,
        6.9,  6.7,  6.6,  6.7,  6.9,  7.1,  7.2,  7.2,  7.3,  7.3,  7.1,  7. ,  7. ,  7. ,  6.9,  6.9,  6.9,  6.8,  6.8,  6.8,  6.9,  6.9,  7. ,  7. ,  7. ,  7.1,  7.1,  7.3,
        7.5,  7.7,  7.9,  8.1,  8.1,  8.1,  8. ,  7.9,  7.8,  7.7,  7.6,  7.5,  7.5,  7.4,  6.9,  6.1,  5.3,  4.5,  3.7,  2.9,  2.1,  1.3,  0.5,  1.1,  2.4,  3.7,  5.1,  6.1,
        6.7,  6.5,  5.5,  4.7,  4.5,  4.2,  3.9,  3.7,  3.4,  3.1,  2.9,  2.8,  2.8,  2.7,  2.6,  2.5,  2.4,  2.4,  2.3,  2.5,  3. ,  3.6,  4.4,  4.8,  4.8,  4.6,  4.3,  4. ,
        3.8,  3.6,  3.4,  3.2,  3. ,  2.7,  2.5,  2.1,  1.8,  1.4,  1. ,  0.7,  0.3, -0.1, -0.4, -0.2,  0.1,  0.6,  2.5,  3.7,  3.9,  3.9,  3.4,  2.9,  2.7,  2.5,  2.3,  2.1,
        1.9,  1.7,  1.5,  1.4,  1.2,  1.1,  0.9,  0.7,  0.6,  0.4,  0.3,  1.1,  2.4,  3.1,  3.5,  4. ,  4.4,  4.1,  3.4,  3. ,  2.8,  2.7,  2.5,  2.4,  2.2,  2.1,  1.9,  1.7,
        1.5,  1.3,  1.1,  0.9,  0.6,  0.4,  0.2,  0.6,  1.3,  1.7,  1.7,  1.5,  1.4,  1.4,  1.2,  1. ,  0.9,  0.8,  0.6,  0.5,  0.4,  0.3, -0.1, -0.6, -1. , -1.5, -2. , -2.5,
       -2.9, -3.4, -3.9, -3.6, -2.3, -1. ,  0.5,  1.8,  2.4,  2.4,  1.7,  1. ,  0.8,  0.6,  0.3,  0.1, -0.1, -0.3, -0.5, -0.7, -0.8, -1. , -1.2, -1.3, -1.5, -1.6, -1.8, -1.1,
       -0.1, -0.1,  0.2,  1.1,  1.9,  2.3,  2. ,  1.6,  1.5,  1.4,  1.2,  1.1,  1. ,  0.8,  0.7,  0.6,  0.5,  0.4,  0.3,  0.2,  0. , -0.2, -0.3,  0.1,  0.5,  1. ,  1.9,  2.7,
        3.5,  3.9,  3.7,  3.2,  2.8,  2.4,  1.9,  1.5,  1.1,  0.6,  0. , -0.9, -1.9, -2.9, -4. , -5. , -6. , -7. , -7.3, -7.1, -7.6, -7.8, -7.2, -6.9, -6.8, -7. , -7.5, -7.9,
       -7.6, -7.1, -7.1, -7.1, -7.1, -7.1, -7. , -6.8, -6.6, -6.3, -6.1, -5.8, -5.6, -5.3, -5.1, -4. , -2.5, -1.6, -1.2, -1.1, -1.1, -1.2, -1.6, -2. , -2.2, -2.4, -2.6, -2.8,
       -3. , -3.2, -3.5, -3.9, -4.3, -4.7, -5.1, -5.5, -5.9, -6.3, -6.7, -5.6, -3.8, -2.1, -0.3,  0.8,  1.3,  1.2,  0.2, -0.7, -1.1, -1.5, -1.9, -2.3, -2.7, -3. , -3.4, -3.9,
       -4.4, -4.9, -5.4, -5.8, -6.3, -6.8, -7.3, -6.8, -5.8, -5.1, -4.6, -4.6, -5.2, -5.5, -5.8, -5.9, -5.7, -5.4, -5.1, -4.8, -4.6, -4.3, -4.1, -3.9, -3.7, -3.6, -3.4, -3.2,
       -3. , -2.9, -2.7, -2.5, -2.2, -1.8, -1.2, -0.6, -0.4, -0.6, -1. , -1. , -0.8, -0.6, -0.3, -0.1,  0.2,  0.4,  0.5,  0.4,  0.4,  0.3,  0.3,  0.2,  0.2,  0.1,  0. ,  0.3,
        0.6,  0.7,  0.7,  0.8,  0.9,  1. ,  1. ,  0.9,  0.9,  0.9,  0.9,  0.9,  0.9,  0.9,  0.9,  1. ,  1. ,  1. ,  1. ,  1.1,  1.1,  1.1,  1.2,  1.5,  2.3,  3.3,  3.8,  3.7,
        3.6,  3.9,  3.9,  3.7,  3.7,  3.6,  3.6,  3.5,  3.4,  3.4,  3.3,  3.2,  3.2,  3.2,  3.2,  3.2,  3.2,  3.2,  3.2,  3.7,  5. ,  6.4,  7.6,  7.6,  7.1,  6.7,  6.2,  5.9,
        5.8,  5.7,  5.6,  5.5,  5.4,  5.3,  5.1,  4.7,  4.4,  4.1,  3.8,  3.5,  3.2,  2.8,  2.5,  2.7,  3.4,  4.3,  5.2,  6. ,  6.3,  6.1,  5.5,  5. ,  4.8,  4.6,  4.4,  4.2,
        3.9,  3.7,  3.5,  3.3,  3.1,  2.9,  2.7,  2.4,  2.2,  2. ,  1.8,  3. ,  5.4,  7.7,  9.5, 10.7, 11.3, 11.2, 10.2,  9.4,  9. ,  8.6,  8.2,  7.8,  7.4,  7. ,  6.6,  6.2,
        5.7,  5.3,  4.8,  4.4,  3.9,  3.5,  3. ,  4.6,  6.9,  7.6,  8.9, 10.4, 10.6, 10.3,  9.5,  8.7,  8.2,  7.7,  7.1,  6.6,  6.1,  5.6,  5. ,  4.5,  4. ,  3.4,  2.9,  2.4,
        1.8,  1.3,  0.8,  1.7,  3.6,  4.8,  5.8,  6.6,  6.8,  6.3,  5.4,  4.6,  4.1,  3.6,  3.2,  2.7,  2.2,  1.7,  1.2,  0.7,  0.2, -0.4, -1. , -1.5, -2. , -2.6, -3.1, -2.8,
       -1.6, -0.5,  0.3,  0.8,  0.8,  0.2, -0.6, -0.9, -1.1, -1.2, -1.4, -1.5, -1.7, -1.9, -2. , -2.1, -2.3, -2.4, -2.6, -2.7, -2.9, -3. , -3.2, -2.2, -0.7,  0.3,  0.9,  1.4,
        1.6,  1.5,  1.1,  0.8,  1.1,  1.5,  1.8,  2.2,  2.5,  2.9,  3.1,  3.3,  3.4,  3.6,  3.8,  4. ,  4.1,  4.3,  4.5,  4.9,  5.7,  6.8,  7.6,  8. ,  8.2,  8.2,  8. ,  7.7,
        7.5,  7.4,  7.2,  7. ,  6.8,  6.7,  6.4,  6.1,  5.9,  5.6,  5.3,  5. ,  4.7,  4.4,  4.1,  4.5,  5.5,  6.5,  7.2,  7.6,  7.6,  7.2,  6.7,  6.3,  6. ,  5.7,  5.3,  5. ,
        4.6,  4.3,  3.8,  3.2,  2.6,  2. ,  1.4,  0.8,  0.2, -0.5, -1.1, -0.7,  1.2,  3.8,  5.7,  6.5,  6.4,  5.6,  4.6,  4. ,  3.9,  3.8,  3.7,  3.6,  3.5,  3.4,  3.3,  3.3,
        3.3,  3.3,  3.3,  3.3,  3.3,  3.3,  3.3,  3.8,  4.7,  5.1,  5.3,  5.7,  6.1,  6.3,  6.1,  5.9,  5.7,  5.5,  5.4,  5.2,  5. ,  4.9,  4.7,  4.4,  4.2,  3.9,  3.7,  3.4,
        3.2,  2.9,  2.8,  3.1,  4.5,  6.3,  7.2,  7.3,  7.3,  7.3,  7.1,  6.8,  6.7,  6.5,  6.3,  6.2,  6. ,  5.9,  5.8,  5.7,  5.6,  5.6,  5.5,  5.5,  5.4,  5.4,  5.3,  6.2,
        8. ,  9.7, 10.6, 10.9, 11. , 11.1, 10.8, 10.3, 10. ,  9.6,  9.3,  9. ,  8.6,  8.3,  7.9,  7.5,  7.2,  6.8,  6.4,  6. ,  5.6,  5.2,  5.2,  5.4,  6.1,  7. ,  7.8,  8.7,
        9.4,  9.7,  9.2,  8.7,  8.4,  8.1,  7.8,  7.5,  7.2,  6.9,  6.5,  6.2,  5.9,  5.5,  5.2,  4.9,  4.5,  4.2,  4.2,  4.6,  5.2,  6.2,  7.2,  7.9,  8.6,  9. ,  8.9,  8.5,
        8. ,  7.6,  7.1,  6.6,  6.1,  5.6,  5.1,  4.5,  4. ,  3.5,  3. ,  2.4,  1.9,  1.4,  1.4,  2. ,  2.9,  3.9,  4.9,  5.8,  6.1,  6. ,  5.7,  5. ,  4.3,  3.8,  3.2,  2.7,
        2.1,  1.5,  0.9,  0.3, -0.3, -0.9, -1.5, -2.1, -2.7, -3.3, -3.5, -2.9, -2.1, -1.3, -0.6, -0.3, -0.4, -0.7, -1.2, -1.7, -2. , -2.1, -2.2, -2.3, -2.4, -2.5, -2.5, -2.5,
       -2.4, -2.3, -2.3, -2.2, -2.1, -2. , -1.6, -0.6,  1.1,  2.9,  4.3,  5.2,  5.8,  5.9,  5.4,  4.5,  3.7,  3.3,  2.8,  2.3,  1.9,  1.4,  0.8,  0.1, -0.7, -1.5, -2.2, -3. ,
       -3.8, -4.5, -4.5, -3.3, -1.6, -0.1,  1.3,  2.4,  3. ,  3.1,  2.5,  1.4,  0.7,  0.6,  0.5,  0.4,  0.4,  0.3,  0.3,  0.4,  0.5,  0.6,  0.8,  0.9,  1. ,  1.2,  1.3,  1.6,
        1.7,  2.3,  3.2,  3.9,  4.3,  4.2,  4. ,  3.7,  3.4,  3.1,  2.9,  2.7,  2.4,  2.2,  1.7,  1.1,  0.4, -0.4, -1.1, -1.8, -2.5, -3.2, -3.1, -2.3, -2. , -1.7, -1. , -0.3,
        0.4,  0.6,  0.4,  0. , -0.5, -0.8, -1.1, -1.4, -1.7, -2.1, -2.3, -2.5, -2.6, -2.8, -3. , -3.2, -3.4, -3.6, -3.5, -3.3, -3. , -2.7, -2.1, -1.6, -1.1, -1. , -1.2, -1.5,
       -1.8, -1.8, -1.8, -1.8, -1.8, -1.9, -1.9, -1.9, -1.9, -1.9, -1.9, -1.9, -1.9, -2. , -1.8, -1.4, -0.8, -0.1,  0.7,  1.4,  2.2,  2.6,  2.5,  2. ,  1.6,  1.5,  1.4,  1.3,
        1.2,  1.1,  0.9,  0.7,  0.5,  0.2,  0. , -0.2, -0.4, -0.6, -0.6,  0.1,  0.9,  1.5,  2.1,  2.5,  2.7,  2.6,  2.1,  1.7,  1.4,  1.2,  1. ,  0.8,  0.6,  0.4,  0.2, -0.1,
       -0.3, -0.6, -0.8, -1.1, -1.3, -1.6, -1.4, -1. , -0.4,  0.7,  1.6,  2.2,  2.4,  2.4,  2.2,  1.7,  1.3,  1.3,  1.2,  1.2,  1.1,  1.1,  1.1,  1.1,  1.2,  1.2,  1.3,  1.4,
        1.4,  1.5,  1.9,  2.6,  3.8,  5.5,  6.8,  7.2,  7.1,  7.3,  7.3,  6.7,  6. ,  5.3,  4.7,  4. ,  3.3,  2.7,  1.8,  0.6, -0.7, -2. , -3.3, -4.5, -5.8, -6.5, -7. , -6.8,
       -5.6, -4.8, -3.9, -3.3, -3.2, -3.3, -3.7, -4.5, -5.1, -5.5, -5.9, -6.3, -6.6, -7. , -7.1, -6.9, -6.7, -6.5, -6.3, -6.1, -5.9, -5.6, -5.2, -4.2, -3.1, -2.8, -2.8, -2.3,
       -1.8, -1.7, -2. , -2.5, -2.6, -2. , -1.5, -0.9, -0.4,  0.2,  0.5,  0.6,  0.6,  0.7,  0.7,  0.8,  0.8,  0.9,  1. ,  2.1,  3.6,  4.3,  4.5,  4.8,  4.7,  4.2,  3.8,  3.6,
        3.4,  3.4,  3.4,  3.4,  3.4,  3.4,  3.4,  3.3,  3.2,  3.2,  3.1,  3.1,  3. ,  3. ,  3.3,  4.2,  5.5,  6.5,  7. ,  7.2,  6.9,  6.2,  5.7,  5.3,  5. ,  4.8,  4.6,  4.3,
        4.1,  3.9,  3.5,  2.8,  2. ,  1.3,  0.6, -0.2, -1. , -1.7, -1.7, -0.8,  0.5,  2. ,  3.1,  3.9,  4.5,  4.3,  3.8,  3. ,  2.4,  2.4,  2.4,  2.4,  2.4,  2.4,  2.5,  2.6,
        2.8,  2.9,  3.1,  3.3,  3.4,  3.6,  3.8,  4.1,  4.5,  4.9,  5.2,  5.4,  5.4,  5.2,  5. ,  5. ,  4.9,  4.8,  4.8,  4.7,  4.6,  4.6,  4.4,  4.2,  4.1,  3.9,  3.7,  3.5,
        3.4,  3.2,  3.5,  4.4,  5.5,  6.2,  6.5,  6.9,  7.7,  8.2,  8.1,  7.6,  7.2,  7. ,  6.8,  6.6,  6.4,  6.1,  5.9,  5.7,  5.5,  5.3,  5.1,  5. ,  4.8,  4.6,  5.1,  6.3,
        7.8,  9.6, 11.2, 12.2, 12.6, 12.6, 12.1, 11.2, 10.6, 10.5, 10.4, 10.3, 10.2, 10.1,  9.9,  9.6,  9.4,  9.1,  8.9,  8.6,  8.4,  8.1,  8.1,  8.4,  9. ,  9.8, 10.5, 10.8,
       11. , 11.1, 10.9, 10.6, 10.2,  9.8,  9.4,  9.1,  8.7,  8.3,  7.9,  7.5,  7.1,  6.7,  6.2,  5.8,  5.4,  4.9,  4.9,  5.3,  6. ,  6.6,  6.9,  7. ,  7. ,  6.8,  6.5,  6.2,
        5.9,  5.7,  5.5,  5.2,  5. ,  4.7,  4.5,  4.4,  4.2,  4.1,  3.9,  3.7,  3.6,  3.4,  3.8,  4.6,  5. ,  5.8,  7.2,  7.8,  7.7,  7.3,  6.9,  6.4,  6. ,  5.8,  5.6,  5.4,
        5.2,  4.9,  4.7,  4.4,  4.1,  3.8,  3.5,  3.3,  3. ,  2.7,  3. ,  3.7,  4.2,  5. ,  5.6,  6.3,  7.5,  8.1,  7.7,  7.1,  6.6,  6.1,  5.6,  5.1,  4.6,  4.1,  3.5,  2.9,
        2.2,  1.5,  0.8,  0.2, -0.6, -1.3, -0.9,  0.5,  1.8,  2.9,  4. ,  5.2,  6.2,  6.7,  6.5,  5.7,  5.1,  4.9,  4.7,  4.4,  4.2,  4. ,  3.9,  3.9,  3.9,  3.9,  3.9,  3.9,
        3.9,  3.9,  4.6,  6. ,  7.3,  8.1,  8.9,  9.6, 10. , 10.2, 10. ,  9.3,  8.6,  8.1,  7.7,  7.3,  6.9,  6.5,  6. ,  5.4,  4.9,  4.4,  3.9,  3.4,  2.8,  2.3,  3. ,  5.1,
        7.3,  9.2, 10.6, 11.8, 12.5, 12.7, 12.3, 11.3, 10.5, 10.1,  9.8,  9.5,  9.1,  8.8,  8.4,  7.9,  7.4,  6.9,  6.4,  5.9,  5.4,  4.8,  4.7,  5.1,  5.7,  6.1,  6.2,  6.3,
        6.4,  6.4,  6.1,  5.9,  5.7,  5.4,  5.1,  4.9,  4.6,  4.3,  3.9,  3.5,  3.1,  2.7,  2.2,  1.8,  1.4,  0.9,  1.7,  3.6,  5.3,  6.9,  8.2,  8.7,  8.6,  8.3,  7.8,  6.8,
        6. ,  5.4,  4.8,  4.2,  3.7,  3.1,  2.5,  1.8,  1.1,  0.4, -0.4, -1.1, -1.8, -2.1, -1.4,  0. ,  1.3,  2.7,  4.3,  5.7,  6.7,  7.1,  6.9,  6.1,  5.3,  4.7,  4.1,  3.5,
        2.9,  2.3,  1.6,  1. ,  0.4, -0.2, -0.9, -1.5, -2.1, -2.5, -2. , -0.6,  1. ,  2.5,  4.2,  5.3,  5.8,  6.1,  6. ,  5.1,  4.4,  4.2,  3.9,  3.7,  3.5,  3.2,  3.1,  3. ,
        2.8,  2.7,  2.6,  2.5,  2.4,  2.3,  2.4,  2.5,  2.7,  3.1,  3.3,  3.5,  3.5,  3.5,  3.5,  3.5,  3.4,  3.4,  3.3,  3.2,  3.1,  3. ,  2.9,  2.9,  2.8,  2.8,  2.8,  2.7,
        2.7,  2.7,  3.1,  3.6,  4.1,  4.6,  4.9,  5.4,  5.7,  5.9,  5.8,  5.4,  5.1,  4.9,  4.7,  4.4,  4.2,  3.9,  3.7,  3.3,  3. ,  2.7,  2.4,  2.1,  1.8,  1.7,  2.2,  3. ,
        3.8,  4.3,  4.4,  4.9,  5.5,  5.4,  5.1,  4.9,  4.5,  4.2,  3.8,  3.4,  3.1,  2.7,  2.3,  1.9,  1.5,  1.1,  0.7,  0.2, -0.2, -0.3, -0.1,  0.5,  0.9,  1. ,  1.2,  1.5,
        1.8,  1.9,  2. ,  1.9,  1.6,  1.2,  0.9,  0.5,  0.1, -0.3, -0.6, -1. , -1.4, -1.7, -2.1, -2.5, -2.9, -2.9, -2.2, -1.2, -0.3,  0.8,  1.6,  2.5,  3.2,  3.5,  3.4,  2.7,
        1.9,  1.2,  0.8,  0.3, -0.1, -0.5, -0.9, -1.1, -1.3, -1.5, -1.7, -1.9, -2.1, -2.1, -1.6, -0.4,  1.3,  3.1,  4.8,  5.7,  6. ,  6.1,  5.7,  4.8,  3.9,  3.3,  3.1,  2.8,
        2.5,  2.3,  1.9,  1.6,  1.2,  0.8,  0.5,  0.1, -0.3,  0.1,  0.9,  2.2,  3.8,  5.7,  7. ,  7.5,  7.9,  8. ,  7.9,  7.4,  6.5,  5.6,  4.9,  4.2,  3.5,  2.9,  2.2,  1.5,
        0.8,  0.1, -0.6, -1.3, -2. , -1.9, -0.9,  0.8,  2.9,  4.7,  6.1,  7.2,  8. ,  8.4,  8.2,  7.4,  6.1,  5.2,  4.6,  4.1,  3.5,  3. ,  2.5,  2. ,  1.5,  1. ,  0.5, -0.1,
       -0.6, -0.4,  1. ,  2.9,  4.9,  6.8,  8.1,  8.7,  9.1,  9.2,  8.5,  7.3,  6. ,  5.3,  5.2,  5.1,  4.9,  4.8,  4.7,  4.6,  4.5,  4.4,  4.3,  4.2,  4.1,  4.1,  4.5,  5.1,
        5.7,  6. ,  5.8,  5.6,  5.6,  5.6,  5.5,  5.3,  5.1,  5. ,  5. ,  4.9,  4.8,  4.7,  4.6,  4.5,  4.4,  4.3,  4.2,  4.1,  4. ,  4.1,  4.9,  6.2,  7. ,  7.4,  8. ,  8.6,
        8.2,  7.5,  7.2,  7.1,  6.9,  6.6,  6.5,  6.4,  6.3,  6.2,  6. ,  5.8,  5.5,  5.3,  5.1,  4.9,  4.6,  4.7,  5. ,  5.6,  6.2,  6.9,  7.7,  8.1,  8. ,  7.8,  7.6,  7.2,
        6.7,  6.7,  7.3,  7.8,  8.4,  9. ,  9.3,  9.4,  9.5,  9.7,  9.8,  9.9, 10.1, 10.3, 10.7, 10.9, 11. , 11.4, 11.9, 12.3, 12.5, 12.7, 12.6, 12.4, 12.2, 11.9, 11.6, 11.4,
       11.1, 10.8, 10.3,  9.7,  9. ,  8.3,  7.7,  7. ,  6.3,  6.2,  7. ,  8.6,  9.5,  9.6,  9.7, 10. , 10.1, 10.2, 10.2,  9.9,  9.3,  8.8,  8.4,  8. ,  7.6,  7.2,  6.9,  6.7,
        6.5,  6.3,  6.1,  5.9,  5.7,  5.9,  6.5,  6.8,  7.5,  8.9,  9.8, 10.6, 11.1, 11.3, 11.2, 10.8, 10.3,  9.7,  9.3,  8.8,  8.4,  8. ,  7.5,  7. ,  6.4,  5.9,  5.4,  4.8,
        4.3,  4.4,  5. ,  5.4,  5.5,  5.7,  6. ,  6.2,  6.2,  6.2,  6.3,  6.3,  6. ,  5.6,  5.2,  4.8,  4.4,  4. ,  3.7,  3.4,  3.2,  2.9,  2.6,  2.4,  2.1,  2.8,  4.5,  5.8,
        6.7,  6.8,  6.7,  7.1,  7.6,  8.1,  7.8,  7. ,  6.3,  5.9,  5.7,  5.6,  5.4,  5.3,  5.2,  5.1,  5. ,  4.9,  4.9,  4.8,  4.7,  5.4,  6.9,  8.4,  9.6, 10.2, 10.8, 11.1,
       11. , 10.7, 10. ,  9.3,  8.8,  8.4,  8. ,  7.6,  7.3,  6.9,  6.5,  6. ,  5.5,  5. ,  4.5,  4. ,  3.5,  4.2,  5.8,  7.1,  8. ,  8.9,  9.7,  9.9,  9.9,  9.7,  9.2,  8.4,
        7.5,  7. ,  6.8,  6.6,  6.4,  6.2,  6.1,  6. ,  5.9,  5.9,  5.8,  5.7,  5.7,  6.1,  7. ,  8.1,  8.7,  8.7,  8.6,  8.7,  9.2,  9.4,  9.4,  9.1,  8.8,  8.4,  7.8,  7.3,
        6.7,  6.2,  5.6,  5. ,  4.4,  3.8,  3.1,  2.5,  1.9,  2.8,  4.6,  5.6,  7. ,  8.9, 10.4, 11.1, 11.5, 12.1, 12. , 11.3, 10.2,  9.2,  8.5,  7.9,  7.2,  6.5,  5.9,  5.3,
        4.7,  4.1,  3.4,  2.8,  2.2,  2.3,  3.7,  5.6,  7.2,  8.8,  9.7,  9.8,  9.8,  9.7,  9.4,  8.7,  7.9,  7.2,  6.8,  6.4,  6. ,  5.6,  5.3,  5.1,  4.9,  4.6,  4.4,  4.2,
        4. ,  5.1,  6.9,  8.1,  9.4, 10.9, 12.6, 13.8, 14.1, 14.2, 14.3, 13.8, 12.7, 11.8, 11.2, 10.6, 10. ,  9.4,  8.9,  8.3,  7.8,  7.3,  6.8,  6.3,  5.7,  6.3,  8.1,  8.7,
        8.5,  9.5, 10.9, 11.7, 12.1, 12.3, 12.1, 11.8, 11.3, 10.5,  9.7,  8.9,  8. ,  7.2,  6.4,  5.6,  4.7,  3.8,  3. ,  2.1,  1.2,  2.1,  4.6,  6.9,  8.9, 10.7, 12.3, 13.5,
       14.3, 14.5, 14. , 12.9, 11.3, 10. ,  9.2,  8.3,  7.5,  6.6,  6. ,  5.5,  5. ,  4.5,  3.9,  3.4,  3.1,  3.6,  5. ,  6.5,  7.9,  9.5, 10.9, 12.1, 13.1, 13.6, 13.9, 13.6,
       12.6, 11.9, 11.6, 11.3, 11.1, 10.8, 10.6, 10.4, 10.3, 10.1, 10. ,  9.9,  9.8,  9.9, 10.1, 10.2, 10.4, 10.4, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.4, 10.3, 10.1, 10. ,
        9.6,  8.8,  8.1,  7.3,  6.6,  5.6,  4.4,  3.2,  2.7,  3. ,  3.6,  4.3,  4.9,  5.2,  5.5,  5.9,  6.1,  6.5,  6.8,  6.6,  6.1,  5.5,  4.9,  4.3,  3.7,  3.1,  2.6,  2.2,
        1.8,  1.4,  1. ,  0.6,  0.6,  1.2,  2.2,  3.8,  5.6,  7.1,  8.4,  9.5, 10.2, 10.5, 10.5, 10. ,  9.2,  8.2,  7.4,  6.5,  5.6,  4.7,  3.9,  2.9,  2. ,  1. ,  0. , -1. ,
       -1.3, -0.9,  0.3,  2. ,  3.7,  5. ,  5.8,  6.2,  6.4,  6.1,  5.5,  4.8,  4.1,  3.5,  3.2,  2.9,  2.7,  2.4,  2.4,  2.5,  2.6,  2.8,  2.9,  3. ,  3.4,  4. ,  5.2,  6.7,
        8. ,  9. , 10.5, 11.7, 12.1, 12.3, 12.4, 11.9, 10.9, 10.3, 10.3, 10.2, 10.1, 10. , 10. , 10. , 10. ,  9.9,  9.9,  9.9, 10. , 10.6, 11.1, 11.4, 12.1, 12.7, 13. , 13.4,
       13.6, 13.8, 14.1, 14.3, 14. , 13.7, 13.3, 13. , 12.6, 12.3, 11.9, 11.5, 11.1, 10.7, 10.3,  9.9,  9.8, 10.1, 10.8, 11.5, 12. , 12.4, 12.7, 12.7, 12.7, 12.5, 12.3, 12.2,
       11.8, 11.6, 11.5, 11.5, 11.4, 11.3, 11.3, 11.4, 11.5, 11.6, 11.6, 11.7, 12. , 12.5, 12.7, 12.8, 13.4, 14.2, 14.6, 14.8, 15.1, 15.2, 15.1, 14.8, 14.5, 14.1, 13.8, 13.5,
       13.1, 12.8, 12.4, 11.9, 11.5, 11.1, 10.6, 10.2, 10.2, 10.9, 11.4, 11.8, 12.5, 13.5, 14.7, 15.3, 15.7, 15.6, 15.2, 14.7, 14.1, 13.5, 13. , 12.4, 11.8, 11.3, 10.7, 10. ,
        9.3,  8.6,  7.9,  7.2,  7. ,  7.8,  9. , 10.6, 12.2, 13.4, 14.5, 14.9, 14.6, 14.6, 14.3, 13.4, 12.4, 11.8, 11.4, 11. , 10.5, 10.1,  9.8,  9.4,  9.1,  8.7,  8.3,  8. ,
        8.2,  9.6, 11.5, 13.4, 15.2, 16.9, 18.2, 19.3, 20.1, 20.4, 20.1, 19.3, 18.1, 17.1, 16.5, 15.9, 15.3, 14.7, 14.2, 13.8, 13.5, 13.1, 12.7, 12.4, 12.3, 12.3, 12.8, 13.9,
       15.3, 16.7, 17.4, 17.6, 18.3, 19.1, 19.4, 19.3, 18.8, 17.9, 16.8, 15.8, 14.7, 13.6, 12.5, 11.2,  9.8,  8.5,  7.2,  5.8,  5.6,  6.4,  7.5,  9.3, 11. , 12.2, 13. , 13.5,
       14. , 14.3, 14.4, 14.1, 13.3, 12.5, 11.8, 11.2, 10.5,  9.9,  9.4,  9.2,  9. ,  8.8,  8.5,  8.3,  8.5,  8.9,  9.3, 10. , 10.6, 11.2, 11.5, 11.4, 11.2, 11. , 11. , 11. ,
       10.8, 10.5, 10.1,  9.8,  9.5,  9.1,  8.8,  8.5,  8.2,  7.9,  7.6,  7.3,  7.4,  8.1,  8.7,  9.6, 10.7, 11.2, 11.6, 12.1, 12.1, 11.6, 11. , 10.6, 10.1,  9.7,  9.5,  9.2,
        8.9,  8.7,  8.3,  7.9,  7.5,  7.2,  6.8,  6.4,  6.8,  7.7,  8.4,  8.6,  8.6,  9. ,  9.7, 10.2, 10.5, 10.8, 10.5, 10.1,  9.8,  9.3,  8.8,  8.2,  7.7,  7.1,  6.6,  6.1,
        5.5,  4.9,  4.4,  3.8,  3.8,  4.7,  6.3,  7.9,  9. ,  9.5, 10.2, 10.8, 10.8, 10.6, 10.2,  9.7,  9.2,  8.4,  7.8,  7.4,  6.9,  6.5,  6.1,  5.7,  5.4,  5.1,  4.7,  4.4,
        5. ,  6.2,  7.2,  8.5, 10. , 11.4, 12.4, 13.1, 13.7, 13.6, 13.2, 12.7, 12. , 11.1, 10.3,  9.7,  9. ,  8.4,  7.8,  7.2,  6.6,  6.1,  5.5,  4.9,  4.8,  4.9,  4.9,  5.2,
        5.6,  5.9,  6.4,  7.2,  7.9,  8.2,  8.4,  8.3,  8. ,  7.6,  7.2,  6.7,  6.3,  5.8,  5.3,  4.8,  4.3,  3.8,  3.3,  2.9,  3. ,  3.4,  3.6,  4. ,  4.4,  4.8,  5.2,  5.3,
        5.5,  5.5,  5.5,  5.4,  5.1,  4.7,  4.3,  3.9,  3.4,  3. ,  2.6,  2.3,  1.9,  1.6,  1.2,  0.9,  1.4,  2.9,  4.1,  5. ,  6. ,  7.7,  9.1, 10.1, 10.9, 11.2, 11.3, 11.1,
       10.3,  9.2,  8.1,  7.1,  6.1,  5.1,  4.2,  3.2,  2.1,  1.1,  0.1, -1. , -0.7,  1.2,  3.2,  5.2,  7.1,  8.7, 10.1, 11.1, 11.7, 11.9, 11.7, 10.9,  9.6,  8.1,  6.8,  5.9,
        5. ,  4. ,  3.3,  2.8,  2.3,  1.8,  1.2,  0.7,  1.5,  3.5,  5.3,  7. ,  8.7, 10.3, 11.6, 12.6, 13. , 13. , 12.7, 11.9, 10.8,  9.4,  8.3,  7.5,  6.7,  5.9,  5.2,  4.6,
        4. ,  3.3,  2.7,  2.1,  2.5,  4.3,  6.2,  8.2, 10.1, 11.7, 13.1, 14.1, 14.8, 15.2, 15.1, 14.5, 13.5, 12.1, 10.7,  9.5,  8.3,  7.1,  6. ,  5. ,  3.9,  2.8,  1.6,  0.5,
        1.1,  3.3,  5.1,  6.4,  7.9,  9.5, 10.5, 11.1, 11.6, 11.8, 11.7, 11.2, 10.2,  8.8,  7.7,  6.9,  6.2,  5.5,  5. ,  4.7,  4.3,  4. ,  3.7,  3.4,  4.2,  6.1,  7.5,  8.5,
        9.8, 11.2, 12.2, 12.9, 13.5, 13.6, 13.3, 12.8, 12. , 10.9, 10. ,  9.3,  8.7,  8.1,  7.5,  7. ,  6.5,  5.9,  5.4,  4.9,  5.1,  5.8,  6.2,  7.4,  8.9,  9.8, 10.4, 11. ,
       11.4, 11.4, 11.2, 10.9, 10.6, 10. ,  9.4,  8.9,  8.4,  7.9,  7.4,  7. ,  6.6,  6.2,  5.9,  5.5,  6.5,  8.5, 10.4, 12.4, 13.8, 14.3, 14.4, 14.4, 14.2, 14.1, 13.9, 13.4,
       12.5, 11.4, 10.4,  9.4,  8.4,  7.4,  6.5,  5.6,  4.7,  3.8,  2.8,  2.3,  3. ,  4.7,  6.5,  7.7,  8.5,  9.2, 10. , 11.2, 12.1, 12.8, 13.2, 13. , 12.4, 11.4, 10.4,  9.4,
        8.5,  7.6,  6.8,  6. ,  5.3,  4.5,  3.8,  3.4,  3.6,  3.9,  4.1,  4.4,  5. ,  5.3,  5.4,  5.4,  5.4,  5.3,  5.3,  5.3,  5.1,  4.9,  4.6,  4.3,  3.9,  3.6,  3.3,  3.1,
        2.9,  2.7,  2.4,  2.3,  2.5,  2.7,  2.7,  2.9,  3.2,  3.4,  3.6,  3.8,  4. ,  4.2,  4.2,  4. ,  3.9,  3.7,  3.9,  4.5,  5.2,  5.8,  6.2,  6.4,  6.6,  6.7,  6.9,  7.1,
        7.7,  8.5,  9.5, 10.6, 11.4, 12.2, 12.9, 13.2, 12.9, 12.3, 12.1, 11.8, 11.3, 10.9, 10.4, 10. ,  9.7,  9.3,  8.9,  8.3,  7.8,  7.2,  6.6,  6.5,  7.3,  8.5, 10. , 11.6,
       12.9, 14. , 15. , 15.9, 16.4, 16.4, 16.3, 16. , 15.2, 14.2, 13.2, 12.4, 11.6, 10.8, 10.1,  9.5,  8.9,  8.2,  7.6,  7.4,  7.9,  8.8, 10.3, 12. , 13.4, 14.6, 15.6, 15.9,
       15.8, 15.8, 15.9, 15.7, 15. , 14.1, 13.3, 12.5, 11.7, 11. , 10.4,  9.9,  9.4,  9. ,  8.5,  8.3,  8.8,  9.4,  9.8, 10.6, 11.8, 13. , 13.9, 14.6, 14.8, 14.4, 13.8, 13.2,
       12.5, 12. , 11.7, 11.7, 11.7, 11.7, 11.7, 11.7, 11.6, 11.6, 11.6, 11.6, 11.6, 11.9, 12.5, 13.2, 13.6, 13.7, 14.1, 14.4, 14.5, 14.5, 14.6, 14.6, 14.4, 14.2, 13.9, 13.6,
       13.2, 12.9, 12.6, 12.3, 12. , 11.7, 11.4, 11.4, 12.1, 13.3, 14.1, 14.6, 14.9, 15.2, 15.6, 15.6, 15.3, 15.3, 15.2, 14.7, 14.3, 14. , 13.5, 12.8, 12.2, 11.5, 10.6,  9.5,
        8.3,  7.1,  6. ,  5.5,  5.8,  6.7,  7.7,  8.8, 10.2, 11.5, 12.8, 14.2, 15.1, 15.4, 15.3, 14.7, 13.9, 13. , 12. , 11.2, 10.3,  9.4,  8.5,  7.7,  6.8,  5.9,  5.1,  4.8,
        5.3,  6.1,  6.7,  7.3,  8.7, 10.1, 11.2, 11.8, 12. , 11.9, 11.6, 11.6, 11.3, 10.6, 10. ,  9.5,  8.9,  8.4,  8. ,  7.8,  7.5,  7.2,  7. ,  6.9,  7. ,  7.3,  7.7,  8.1,
        8.4,  8.6,  8.6,  8.5,  8.3,  8.2,  8.2,  8. ,  7.9,  7.7,  7.5,  7.3,  7.1,  6.9,  6.6,  6.1,  5.6,  5.1,  4.6,  4.5,  5. ,  5.8,  6.6,  7.5,  8.2,  8.6,  9. ,  8.9,
        8.4,  8.2,  8.2,  8.4,  8.4,  8.1,  7.7,  7.2,  6.6,  6.1,  5.7,  5.4,  5.1,  4.8,  4.5,  4.5,  4.9,  6. ,  7.7,  9.1, 10.2, 11.5, 12.8, 13.7, 14.4, 14.8, 14.6, 14.1,
       13.5, 12.7, 11.7, 10.6,  9.5,  8.4,  7.3,  6.2,  5.1,  4. ,  2.9,  2.5,  3. ,  4.1,  5.8,  7.4,  8.5,  9. ,  9.2,  9.2,  8.6,  8. ,  7.8,  7.6,  7.2,  6.6,  6.1,  5.9,
        5.7,  5.4,  5.3,  5.2,  5.1,  5.1,  5. ,  5. ,  5.6,  6.6,  8. ,  9.6, 10.9, 12. , 12.9, 13.7, 13.9, 13.5, 12.9, 12.4, 11.7, 10.9, 10.2,  9.6,  8.9,  8.3,  7.7,  6.9,
        6.1,  5.3,  4.6,  4.6,  5.5,  6.5,  7.2,  7.6,  8.8, 10.1, 10.5, 10.6, 10.4, 10.5, 10.8, 10.9, 10.5,  9.8,  9.2,  8.7,  8.3,  7.8,  7.5,  7.4,  7.3,  7.1,  7. ,  6.9,
        7.7,  9.1, 10.9, 12.8, 14.2, 15. , 15.6, 16.3, 16.7, 16.9, 17.2, 17.2, 16.7, 15.8, 14.7, 13.5, 12.3, 11.1, 10. ,  9. ,  8. ,  7. ,  5.9,  5.8,  6.9,  8.1,  8.9,  9.5,
       10.3, 11.5, 12.9, 14.4, 15.5, 15.9, 16. , 15.9, 15.3, 14.4, 13.4, 12.4, 11.5, 10.5,  9.7,  9. ,  8.3,  7.6,  6.9,  6.7,  7. ,  7.9,  9.3, 10.7, 11.7, 12.6, 13.5, 14.3,
       14.5, 14.5, 14.2, 13.8, 13.4, 12.7, 11.9, 11.2, 10.5,  9.8,  9.2,  8.7,  8.2,  7.7,  7.2,  7.1,  7.5,  7.9,  8.3,  9.6, 10.8, 11. , 11.2, 11.6, 12.1, 12.8, 13.3, 13.4,
       13. , 12.5, 11.9, 11.2, 10.5,  9.8,  9.1,  8.4,  7.7,  6.9,  6.2,  6.2,  7.2,  8.8, 10.5, 12.2, 13.8, 15.2, 16.5, 17.4, 17.9, 18.1, 18. , 17.4, 16.6, 15.5, 14.6, 14.3,
       13.9, 13.6, 13.5, 13.6, 13.8, 13.9, 14.1, 14.7, 15.8, 17.2, 19. , 20.6, 22.2, 23.7, 24.9, 25.9, 26.7, 27. , 26.9, 26.5, 25.8, 24.8, 23.8, 22.9, 22. , 21.1, 20.1, 18.9,
       17.7, 16.6, 15.4, 15.4, 16.4, 17.5, 18.2, 18.8, 19.2, 19.1, 19.3, 19.9, 20.4, 20.7, 20.7, 20.4, 20. , 19.5, 18.7, 17.7, 16.7, 15.7, 14.9, 14.2, 13.5, 12.8, 12.1, 12.3,
       13.5, 14.8, 15.9, 16.8, 17.6, 18. , 18.5, 19.2, 19.6, 19.5, 18.9, 18.3, 17.9, 17.3, 16.6, 15.9, 15.2, 14.6, 14. , 13.6, 13.1, 12.7, 12.3, 12.4, 12.9, 13.5, 14.6, 16.1,
       17.6, 18.7, 19.5, 20.4, 20.9, 21.1, 21.2, 21.1, 20.5, 19.7, 18.7, 17.6, 16.6, 15.6, 14.6, 13.6, 12.6, 11.6, 10.5, 10.5, 11.4, 12.4, 13.6, 14.8, 15.9, 16.9, 17.5, 17.5,
       17.4, 17.3, 17.2, 17.3, 17. , 16.3, 15.5, 14.8, 14. , 13.3, 12.8, 12.6, 12.3, 12.1, 11.9, 12. , 12.7, 13.6, 14.3, 14.5, 14.9, 15.4, 15.8, 16.4, 17.2, 17.6, 17.8, 17.9,
       17.6, 17.1, 16.5, 16. , 15.5, 15. , 14.5, 14.1, 13.7, 13.3, 12.8, 12.9, 13.5, 14.8, 16.3, 17.1, 17.7, 17.8, 18. , 18.3, 18.2, 17.9, 17.6, 17.2, 16.7, 16.2, 15.6, 15.1,
       14.8, 14.4, 14.1, 13.9, 13.7, 13.4, 13.2, 13.6, 14.6, 15.5, 16.1, 16.5, 17.5, 18.8, 20. , 20.9, 21.5, 21.8, 21.7, 21.3, 20.8, 20. , 19.2, 18.4, 17.6, 16.7, 15.9, 15.1,
       14.2, 13.3, 12.5, 12.7, 14.1, 15.5, 17.2, 19.1, 20.8, 22.2, 23.5, 24.6, 25.4, 25.8, 25.9, 25.5, 24.7, 23.6, 22.3, 20.8, 19.3, 17.7, 16.3, 14.9, 13.5, 12.1, 10.7, 10.5,
       11.8, 13. , 13.9, 15.3, 16.6, 17.5, 18.2, 18.9, 19.3, 19.4, 19.1, 18.6, 18. , 17.2, 16.2, 15.1, 13.8, 12.6, 11.5, 10.6,  9.7,  8.8,  7.9,  7.6,  8.2,  9.5, 11.4, 13.1,
       14.6, 16. , 17.3, 18.3, 18.9, 19.1, 19.3, 19.1, 18.4, 17.3, 16.1, 15. , 13.8, 12.7, 11.7, 10.8,  9.9,  9. ,  8.1,  7.9,  9. , 10.6, 12.1, 13.7, 15.3, 16.8, 18. , 18.9,
       19.4, 19.7, 19.8, 19.4, 18.6, 17.5, 16.3, 15. , 13.7, 12.5, 11.3, 10.1,  8.8,  7.7,  6.5,  6.6,  8.1,  9.7, 11.5, 13.3, 14.9, 16.2, 17.2, 17.8, 18.2, 18.4, 18.1, 17.4,
       16.4, 15.3, 14.1, 12.9, 11.7, 10.5,  9.6,  8.8,  8.1,  7.3,  6.5,  6.7,  8. ,  9.3, 10.8, 12.4, 13.9, 15.4, 16.7, 17.7, 18.5, 18.9, 19. , 18.8, 18. , 16.9, 15.6, 14.4,
       13.5, 12.6, 11.8, 11.2, 10.5,  9.9,  9.3,  9.5, 10.4, 11.4, 12.9, 14.1, 15. , 15.8, 16.4, 16.5, 16.5, 16.7, 16.8, 16.6, 16.2, 15.4, 14.6, 13.6, 12.6, 11.6, 10.8, 10.1,
        9.4,  8.6,  7.9,  7.9,  9. , 10.4, 11.7, 12.3, 12.3, 12.2, 12.7, 13.2, 13.2, 13.1, 12.9, 12.6, 12. , 11.3, 10.8, 10.1,  9.3,  8.5,  7.9,  7.3,  6.7,  6.2,  5.6,  6.2,
        7.7,  9.3, 11.3, 13.2, 14.6, 15.8, 16.7, 17.4, 18. , 18.3, 18.2, 17.7, 17. , 15.9, 14.6, 13.5, 12.6, 11.7, 11. , 10.6, 10.1,  9.7,  9.3,  9.4, 10.1, 10.7, 11. , 11.1,
       11.7, 12.7, 13.5, 14.2, 14.5, 14.6, 14.5, 14.4, 14.4, 14. , 13.4, 12.6, 11.5, 10.4,  9.3,  8.3,  7.2,  6.1,  5.1,  5.4,  6.8,  8. ,  9.3, 10.7, 12.3, 13.8, 15.1, 16.2,
       17. , 17.3, 17.2, 16.8, 16. , 14.9, 13.7, 12.5, 11.4, 10.3,  9.4,  8.8,  8.1,  7.5,  6.8,  7. ,  8.3,  9.2, 10. , 11.2, 12.1, 12.7, 13. , 13.2, 13.3, 13.3, 12.8, 12. ,
       11.6, 11.1, 10.4,  9.9,  9.4,  9. ,  8.7,  8.5,  8.4,  8.2,  8. ,  8.8,  9.4,  9.5, 10.2, 11. , 11.8, 12.6, 13.3, 13.7, 13.8, 14.1, 14.3, 14.1, 13.6, 13. , 12.4, 11.8])

