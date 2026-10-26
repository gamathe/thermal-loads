import pandas as pd                 # Data tables
import os                           # Returns current directory, create files
import numpy as np                  # Arrays
import matplotlib.pyplot as plt	    # Graphs

from math import sqrt, atan, log, exp, sin, cos, tan, trunc

import ipywidgets as widgets
from pathlib import Path


def _lbl(text, width='100px'):
    return widgets.Label(str(text), layout=widgets.Layout(width=width))

def env_form(df, key_cols, save_file, options_w, options_t):
    """Show one row of dropdowns + a heating value per unique key, autosave to CSV, return a getter."""
    save_file = Path(save_file)
    num = [c for c in key_cols if pd.api.types.is_numeric_dtype(df[c])]
    RAD = 'rad heating [W/m2]'

    def norm(values):  # stable key: numbers rounded, everything else as str
        return tuple(f'{float(v):.3f}' if c in num else str(v)
                     for c, v in zip(key_cols, values))

    keys = (df[key_cols].dropna().drop_duplicates().sort_values(key_cols)
            .itertuples(index=False, name=None))

    saved = {}
    if save_file.exists():
        s = pd.read_csv(save_file, dtype=str).fillna('')
        saved = {norm(r[:len(key_cols)]): (r[-3], r[-2], float(r[-1] or 0))
                 for r in s.itertuples(index=False, name=None)}

    controls, out = {}, widgets.Output()
    rows = [widgets.HBox([_lbl(c) for c in key_cols] +
                         [_lbl('wall_type', '200px'), _lbl('t_env', '150px'), _lbl(RAD, '150px')])]

    def get_df():
        return pd.DataFrame([(*k, w.value, t.value, r.value) for k, (w, t, r) in controls.items()],
                            columns=key_cols + ['wall_type', 't_env', RAD])

    def save(_=None):
        get_df().to_csv(save_file, index=False)
        with out:
            out.clear_output()
            print(f'Saved to {save_file.resolve()}')

    for key in keys:
        w0, t0, r0 = saved.get(norm(key), ('', '', 0.0))
        dw = widgets.Dropdown(options=list(dict.fromkeys([*options_w, w0])), value=w0,
                              layout=widgets.Layout(width='200px'))
        dt = widgets.Dropdown(options=list(dict.fromkeys([*options_t, t0])), value=t0,
                              layout=widgets.Layout(width='150px'))
        dr = widgets.FloatText(value=r0, layout=widgets.Layout(width='150px'))
        for c in (dw, dt, dr):
            c.observe(save, names='value')
        controls[key] = (dw, dt, dr)
        rows.append(widgets.HBox([_lbl(v) for v in key] + [dw, dt, dr]))

    display(widgets.VBox(rows), out)
    return get_df


def read_dataframe(df):

    mapping = {
        -135: "NE",
        -90:  "E",
        -45:  "SE",
        0:    "S",
        45:   "SW",
        90:   "W",
        135:  "NW",
        180:  "N",
    }
    
    df["azimuth_deg"] = df["azimuth_deg"].fillna(0)

    wrapped = (df["azimuth_deg"] + 180) % 360 - 180
    df["azimuth_deg"] = (np.round(wrapped / 45) * 45).astype(int)
    df.loc[df["azimuth_deg"] == -180, "azimuth_deg"] = 180

    df["orientation"] = df["azimuth_deg"].map(mapping)

    col = "orientation"
    iaz = df.columns.get_loc("azimuth_deg")

    df = df[
        [c for c in df.columns[:iaz] if c != col]
        + [col]
        + [c for c in df.columns[iaz:] if c != col]
    ].copy()
    
    wl = df.loc[df['type'] != 'window'].copy()
    wd = df.loc[df['type'] == 'window'].copy()

    return wl, wd
    

def azimuth_norm(az):
    # azimuth comprised between -360° and 360°
    az = az - 360 * trunc(az/360)
    # azimuth comprised between -180° and 180°"
    if (abs(az) > 180) :
        if (az > 180) : 
            az = az - 360
        else:
            az = az + 360
    return az


def orilist(azimuth_0h_deg) :
    
    az_num = round(((azimuth_norm(azimuth_0h_deg) + 180) / 45))
    if   (az_num == 0) : 
        lst_ori = ['N', 'E', 'S', 'W', 'floor', 'roof']
    elif (az_num == 1) :
        lst_ori = ['NE', 'SE', 'SW', 'NW', 'floor', 'roof']
    elif (az_num == 2) :
        lst_ori = ['E', 'S', 'W', 'N', 'floor', 'roof']
    elif (az_num == 3) :
        lst_ori = ['SE', 'SW', 'NW', 'NE', 'floor', 'roof']
    elif (az_num == 4) :
        lst_ori = [ 'S', 'W', 'N', 'E', 'floor', 'roof']
    elif (az_num == 5) :
        lst_ori = ['SW', 'NW', 'NE', 'SE', 'floor', 'roof']
    elif (az_num == 6) :
        lst_ori = [ 'W', 'N', 'E', 'S', 'floor', 'roof']
    elif (az_num == 7) :
        lst_ori = ['NW', 'NE', 'SE', 'SW', 'floor', 'roof']
    elif (az_num == 8) :
        lst_ori = [ 'N', 'E', 'S', 'W', 'floor', 'roof']

    return lst_ori

    

def dfwl_final(wl0, azimuth_0h_deg):

    dfwl = wl0[['azimuth', 'slope_deg', 'wall_type', 'net_area_m2', 't_env', 'rad heating [W/m2]']].copy()
    dfwl = dfwl.rename(columns={"t_env": "t_env_name", 'slope_deg': 'slope'})

    lst_ori = orilist(azimuth_0h_deg)

    data_ori = {\
            'azimuth': ['az0', 'az0+90', 'az0+180', 'az0-90', 'floor', 'roof'], \
            'facades': ['0h', '3h', '6h', '9h', 'floor', 'roof'], \
            'orientation': orilist(azimuth_0h_deg)[:4] + ['floor', 'roof']
           }

    
    df_ori = pd.DataFrame(data=data_ori)
    
    dfwl['azimuth'] = dfwl['azimuth'].astype(str)
    df_ori['azimuth'] = df_ori['azimuth'].astype(str)
    
    dfwl = pd.merge(dfwl, df_ori, left_on='azimuth', right_on='azimuth', how='left').set_index('facades')

    dfwl.loc[dfwl['wall_type'] == 'no', ["f_temp"]] = 0
    dfwl.loc[dfwl['wall_type'] == 'no', ["rad heating [W/m2]"]] = 0

    dfwl.loc[dfwl['t_env_name'] == 't_ext', ["T"]] = 'E'
    dfwl.loc[dfwl['t_env_name'] == 't_min', ["T"]] = 'U'
    dfwl.loc[dfwl['t_env_name'] == 't_avg', ["T"]] = 'B'
    dfwl.loc[(dfwl['t_env_name'] == 't_in') & (dfwl['wall_type'] != 'no'), ["T"]] = 'C'
    dfwl.loc[(dfwl['t_env_name'] == 't_in') & (dfwl['wall_type'] == 'no'), ["T"]] = 'N'
    
    dfwl = dfwl[['azimuth', 'slope', 'orientation', 'wall_type', 'net_area_m2', 'T', 'rad heating [W/m2]']]

    return dfwl


def dfwi_final(wi0, azimuth_0h_deg, dfwl, L_rm, B_rm, H_rm):

    dfwi = wi0[['azimuth', 'wall_type', \
                           'width_m', 'height_m', 'sill_height_m', 'dist_corner_m', \
                           't_env', 'rad heating [W/m2]']].copy()

    dfwi = dfwi.rename(columns={
        "width_m": "breadth [m]",
        "height_m": "height [m]",
        "sill_height_m": "sill height [m]",
        "dist_corner_m": "dist corner [m]",
        "t_env": "t_env_name"})

    if len(dfwi) == 0 :
        datawi = [('az0', 'no' , 0.001, 0.001,  0.0, 0.0, 't_in', 0)]
        dfwi = pd.DataFrame(datawi, columns=('azimuth',  'wall_type', \
                                   'breadth [m]', 'height [m]', 'sill height [m]', 'dist corner [m]', \
                                   't_env_name', 'rad heating [W/m2]')).set_index('azimuth')


    data_ori = {\
        'azimuth': ['az0', 'az0+90', 'az0+180', 'az0-90'], \
        'facades': ['0h', '3h', '6h', '9h'], \
        'orientation': orilist(azimuth_0h_deg)[:4], \
        'host' : [0, 1, 2, 3]
           }
    
    df_ori = pd.DataFrame(data=data_ori)

    dfwi['azimuth'] = dfwi['azimuth'].astype(str)
    df_ori['azimuth'] = df_ori['azimuth'].astype(str)

    dfwi = pd.merge(dfwi, df_ori, left_on='azimuth', right_on='azimuth', how='left').set_index('facades')

    dfwi.loc[dfwi['wall_type'] == 'no', ["f_temp"]] = 0
    dfwi.loc[dfwi['wall_type'] == 'no', ["rad heating [W/m2]"]] = 0

    dfwi.loc[dfwi['t_env_name'] == 't_ext', ["T"]] = 'E'
    dfwi.loc[dfwi['t_env_name'] == 't_min', ["T"]] = 'U'
    dfwi.loc[dfwi['t_env_name'] == 't_avg', ["T"]] = 'B'
    dfwi.loc[(dfwi['t_env_name'] == 't_in') & (dfwi['wall_type'] != 'no'), ["T"]] = 'C'
    dfwi.loc[(dfwi['t_env_name'] == 't_in') & (dfwi['wall_type'] == 'no'), ["T"]] = 'N'
 
    dfwi.loc[(dfwi['host'] == 0) | (dfwi['host'] == 2), 'b_facade'] = B_rm
    dfwi.loc[(dfwi['host'] == 1) | (dfwi['host'] == 3), 'b_facade'] = L_rm

    dfwi['sill height [m]'] = dfwi['sill height [m]'].clip(0, H_rm)
    dfwi['dist corner [m]'] = dfwi['dist corner [m]'].clip(0, dfwi['b_facade'])
    dfwi['height [m]']      = dfwi['height [m]'].clip(0, H_rm - dfwi['sill height [m]'] )
    dfwi['breadth [m]']     = dfwi['breadth [m]'].clip(0, dfwi['b_facade'])
    
    dfwi['area'] = dfwi['height [m]'] * dfwi['breadth [m]'] 

    dfwi = dfwi[['azimuth', 'orientation', 'wall_type', 'T', \
                 'breadth [m]', 'height [m]', 'sill height [m]', 'dist corner [m]', \
                 "rad heating [W/m2]", 'area', 'host']]

    return dfwi


def dfwd_final(wd0, azimuth_0h_deg, dfwl, L_rm, B_rm, H_rm):

    dfwd = wd0[['azimuth', 'width_m', 'height_m', 'sill_height_m', 'dist_corner_m']].copy()

    dfwd = dfwd.rename(columns={
        "width_m": "breadth [m]",
        "height_m": "height [m]",
        "sill_height_m": "sill height [m]",
        "dist_corner_m": "dist corner [m]"})

    data_ori = { \
            'azimuth': ['az0', 'az0+90', 'az0+180', 'az0-90'], \
            'facades': ['0h', '3h', '6h', '9h'], \
            'orientation': orilist(azimuth_0h_deg)[:4], \
            'azimuth_deg': [azimuth_norm(az) for az in azimuth_0h_deg + np.array([0, 90, 180, -90])], \
             'host' : [0, 1, 2, 3]
           }

    df_ori = pd.DataFrame(data=data_ori)

    dfwd['azimuth'] = dfwd['azimuth'].astype(str)
    df_ori['azimuth'] = df_ori['azimuth'].astype(str)

    dfwd = pd.merge(dfwd, df_ori, left_on='azimuth', right_on='azimuth', how='left').set_index('facades')
    
    # Keep only windows defined on external walls
    dfwle = dfwl.loc[dfwl['T'] == 'E'].copy()
    dfwd = dfwd.join(dfwle[['T']], how='left')

    dfwd['area'] = dfwd['height [m]'] * dfwd['breadth [m]'] 

    dfwd.loc[(dfwd['host'] == 0) | (dfwd['host'] == 2), 'b_facade'] = B_rm
    dfwd.loc[(dfwd['host'] == 1) | (dfwd['host'] == 3), 'b_facade'] = L_rm

    dfwd['sill height [m]'] = dfwd['sill height [m]'].clip(0, H_rm)
    dfwd['height [m]']      = dfwd['height [m]'].clip(0, H_rm - dfwd['sill height [m]'] )

    dfwd = dfwd[['orientation', 'azimuth_deg', \
                 'breadth [m]', 'height [m]', 'sill height [m]', 'dist corner [m]', \
                 'area', 'host']]

    return dfwd



