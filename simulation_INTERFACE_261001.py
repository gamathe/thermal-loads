import pandas as pd                 # Data tables
import os                           # Returns current directory, create files
import numpy as np                  # Arrays
import matplotlib.pyplot as plt	    # Graphs

from math import sqrt, atan, log, exp, sin, cos, tan, trunc

import plotly.graph_objects as go


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
        lst_ori = ['N', 'E', 'S', 'W', 'floor', 'ceiling/roof']
    elif (az_num == 1) :
        lst_ori = ['NE', 'SE', 'SW', 'NW', 'floor', 'ceiling/roof']
    elif (az_num == 2) :
        lst_ori = ['E', 'S', 'W', 'N', 'floor', 'ceiling/roof']
    elif (az_num == 3) :
        lst_ori = ['SE', 'SW', 'NW', 'NE', 'floor', 'ceiling/roof']
    elif (az_num == 4) :
        lst_ori = [ 'S', 'W', 'N', 'E', 'floor', 'ceiling/roof']
    elif (az_num == 5) :
        lst_ori = ['SW', 'NW', 'NE', 'SE', 'floor', 'ceiling/roof']
    elif (az_num == 6) :
        lst_ori = [ 'W', 'N', 'E', 'S', 'floor', 'ceiling/roof']
    elif (az_num == 7) :
        lst_ori = ['NW', 'NE', 'SE', 'SW', 'floor', 'ceiling/roof']
    elif (az_num == 8) :
        lst_ori = [ 'N', 'E', 'S', 'W', 'floor', 'ceiling/roof']

    return lst_ori

def temperature_factors(CP, we):
    
    # Environmental temperatures
    t_ext = we.loc[we['CP'] == CP, 'theta_e_base'].iloc[0]
    t_min = we.loc[we['CP'] == CP, 'theta_e_min'].iloc[0] 
    t_avg = we.loc[we['CP'] == CP, 'theta_e_avg'].iloc[0]

    f_temp_min = (20 - t_min) / (20 - t_ext)
    f_temp_avg = (20 - t_avg) / (20 - t_ext)

    return f_temp_min, f_temp_avg
    

def dfwl_final(wl0, azimuth_0h_deg):

    dfwl = wl0.copy()

    lst_ori = orilist(azimuth_0h_deg)

    data_ori = {\
            'azimuth': ['az0', 'az0+90', 'az0+180', 'az0-90', 'floor', 'ceiling/roof'], \
            'facades': ['0h', '3h', '6h', '9h', 'floor', 'ceiling/roof'], \
            'orientation': orilist(azimuth_0h_deg)[:4] + ['floor', 'ceiling/roof'], \
            'slope'  : [90, 90, 90, 90, 0, 180]
           }

    
    df_ori = pd.DataFrame(data=data_ori)
    dfwl = pd.merge(dfwl, df_ori, left_index=True, right_on='azimuth', how='left').set_index('facades')

    dfwl.loc[dfwl['wall_type'] == 'no', ["f_temp"]] = 0
    dfwl.loc[dfwl['wall_type'] == 'no', ["rad heating [W/m2]"]] = 0

    dfwl.loc[dfwl['t_env_name'] == 't_ext', ["T"]] = 'E'
    dfwl.loc[dfwl['t_env_name'] == 't_min', ["T"]] = 'U'
    dfwl.loc[dfwl['t_env_name'] == 't_avg', ["T"]] = 'B'
    dfwl.loc[(dfwl['t_env_name'] == 't_int') & (dfwl['wall_type'] != 'no'), ["T"]] = 'C'
    dfwl.loc[(dfwl['t_env_name'] == 't_int') & (dfwl['wall_type'] == 'no'), ["T"]] = 'N'
    
    dfwl = dfwl[['azimuth', 'orientation', 'slope', 'wall_type', 'T', 'rad heating [W/m2]']]

    return dfwl


def dfwi_final(wi0, azimuth_0h_deg, dfwl, L_rm, B_rm, H_rm):

    dfwi = wi0.copy()

    if len(dfwi) == 0 :
        datawi = [('az0', 'no' , 0.001, 0.001,  0.0, 0.0, 't_int', 0)]
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
    dfwi = pd.merge(dfwi, df_ori, left_index=True, right_on='azimuth', how='left').set_index('facades')

    dfwi.loc[dfwi['wall_type'] == 'no', ["f_temp"]] = 0
    dfwi.loc[dfwi['wall_type'] == 'no', ["rad heating [W/m2]"]] = 0

    dfwi.loc[dfwi['t_env_name'] == 't_ext', ["T"]] = 'E'
    dfwi.loc[dfwi['t_env_name'] == 't_min', ["T"]] = 'U'
    dfwi.loc[dfwi['t_env_name'] == 't_avg', ["T"]] = 'B'
    dfwi.loc[(dfwi['t_env_name'] == 't_int') & (dfwi['wall_type'] != 'no'), ["T"]] = 'C'
    dfwi.loc[(dfwi['t_env_name'] == 't_int') & (dfwi['wall_type'] == 'no'), ["T"]] = 'N'
 
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

    dfwd = wd0.copy()

    data_ori = { \
            'azimuth': ['az0', 'az0+90', 'az0+180', 'az0-90'], \
            'facades': ['0h', '3h', '6h', '9h'], \
            'orientation': orilist(azimuth_0h_deg)[:4], \
            'azimuth_deg': [azimuth_norm(az) for az in azimuth_0h_deg + np.array([0, 90, 180, -90])], \
             'host' : [0, 1, 2, 3]
           }

    df_ori = pd.DataFrame(data=data_ori).set_index('azimuth')
    
    dfwd = dfwd.join(df_ori, how='left')
    dfwd = dfwd.set_index('facades')
    
    # Keep only windows defined on external walls
    dfwle = dfwl.loc[dfwl['T'] == 'E'].copy()
    dfwd = dfwd.join(dfwle[['T']], how='left')

    dfwd['area'] = dfwd['height [m]'] * dfwd['breadth [m]'] 

    dfwd.loc[(dfwd['host'] == 0) | (dfwd['host'] == 2), 'b_facade'] = B_rm
    dfwd.loc[(dfwd['host'] == 1) | (dfwd['host'] == 3), 'b_facade'] = L_rm

    dfwd['sill height [m]'] = dfwd['sill height [m]'].clip(0, H_rm)
    dfwd['height [m]']      = dfwd['height [m]'].clip(0, H_rm - dfwd['sill height [m]'] )
    
    dfwd = dfwd.loc[dfwd.index.repeat(dfwd['number'])].copy()
    dfwd['i'] = dfwd.groupby(level=0).cumcount()
    cond = dfwd['i'] > 0 
    dfwd.loc[cond, 'dist corner [m]'] = dfwd.loc[cond, 'dist corner [m]'] \
                    + dfwd.loc[cond, 'i'] * (dfwd.loc[cond, 'breadth [m]'] + dfwd.loc[cond, 'dist windows [m]'])

    dfwd['dist corner [m]'] = dfwd['dist corner [m]'].clip(0, dfwd['b_facade'] )
    dfwd['breadth [m]']     = dfwd['breadth [m]'].clip(0, dfwd['b_facade'] - dfwd['dist corner [m]'])

    dfwd = dfwd[['orientation', 'azimuth_deg', \
                 'breadth [m]', 'height [m]', 'sill height [m]', 'dist corner [m]', \
                 'area', 'host']]

    return dfwd



def create_box(x0, y0, z0, dx, dy, dz, color):
    """Creates a box (room) as a semi-transparent mesh."""
    # Define the 8 vertices of the box
    x = [x0, x0+dx, x0+dx, x0,    x0,    x0+dx, x0+dx, x0]
    y = [y0, y0,    y0+dy, y0+dy, y0,    y0,    y0+dy, y0+dy]
    z = [z0, z0,    z0,    z0,    z0+dz, z0+dz, z0+dz, z0+dz]

    # Define the 12 triangles (2 per face × 6 faces)
    faces = [
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [2, 3, 7], [2, 7, 6],  # back
        [1, 2, 6], [1, 6, 5],  # right
        [0, 3, 7], [0, 7, 4],  # left
    ]

    i, j, k = zip(*faces)

    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color=color,
        opacity=0.3,
        name="Room"
    )


def create_rectangle_on_wall(origin, width, height, normal_axis, color, name="Window"):
    """Creates a rectangular mesh (window) on a given wall."""
    x0, y0, z0 = origin
    if normal_axis == 'x':  # wall perpendicular to x-axis
        x = [x0]*4
        y = [y0, y0+width, y0+width, y0]
        z = [z0, z0,       z0+height, z0+height]
    elif normal_axis == 'y':  # wall perpendicular to y-axis
        x = [x0, x0+width, x0+width, x0]
        y = [y0]*4
        z = [z0, z0,       z0+height, z0+height]
    elif normal_axis == 'z':  # floor or ceiling
        x = [x0, x0+width, x0+width, x0]
        y = [y0, y0,       y0+height, y0+height]
        z = [z0]*4
    else:
        raise ValueError("normal_axis must be 'x', 'y', or 'z'")

    faces = [[0, 1, 2], [0, 2, 3]]
    i, j, k = zip(*faces)

    return go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color=color,
        opacity=1.0,
        name=name
    )

def create_room_windows(L_rm, B_rm, H_rm, dfwi, dfwd) :
    
    # Combine room and windows
    
    # Create the room 
    room = create_box(x0=0, y0=0, z0=0, dx=L_rm, dy=B_rm, dz=H_rm, color='lightblue')

    # Create the additional internal walls
    wallsint = []
    for index, row in dfwi.iterrows():
        host = row['host']
        bwd  = row['breadth [m]']
        hwd  = row['height [m]']
        zwd  = row['sill height [m]'] 
        dcr  = row['dist corner [m]']
        
        if host == 0 : # window carved in 0h wall
            naxis = 'x'
            xorig = L_rm
            yorig = dcr
            zorig = zwd
            wallsint.append(create_rectangle_on_wall(origin=(xorig, yorig, zorig), width=bwd, height=hwd, normal_axis=naxis, color='red', \
                                                    name="Window " + str(index)))
    
        elif host == 1 : # window carved in 3h wall
            naxis = 'y'
            xorig = dcr
            yorig = 0
            zorig = zwd
            wallsint.append(create_rectangle_on_wall(origin=(xorig, yorig, zorig), width=bwd, height=hwd, normal_axis=naxis, color='red', \
                                                    name="Window " + str(index)))
            
        elif host == 2 : # window carved in 6h wall
            naxis = 'x'
            xorig = 0
            yorig = B_rm - bwd - dcr
            zorig = zwd
            wallsint.append(create_rectangle_on_wall(origin=(xorig, yorig, zorig), width=bwd, height=hwd, normal_axis=naxis, color='red', \
                                                    name="Window " + str(index)))
    
        else :  # host == 3 : # window carved in 9h wall
            naxis = 'y'
            xorig = L_rm - bwd - dcr 
            yorig = B_rm
            zorig = zwd
            wallsint.append(create_rectangle_on_wall(origin=(xorig, yorig, zorig), width=bwd, height=hwd, normal_axis=naxis, color='red', \
                                                    name="Window " + str(index)))
    
    # Create the windows
    windows = []
    for index, row in dfwd.iterrows():
        host = row['host']
        bwd  = row['breadth [m]']
        hwd  = row['height [m]']
        zwd  = row['sill height [m]'] 
        dcr  = row['dist corner [m]']
        
        if host == 0 : # window carved in 0h wall
            naxis = 'x'
            xorig = L_rm
            yorig = dcr
            zorig = zwd
            windows.append(create_rectangle_on_wall(origin=(xorig, yorig, zorig), width=bwd, height=hwd, normal_axis=naxis, color='blue', \
                                                    name="Window " + str(index)))
    
        elif host == 1 : # window carved in 3h wall
            naxis = 'y'
            xorig = dcr
            yorig = 0
            zorig = zwd
            windows.append(create_rectangle_on_wall(origin=(xorig, yorig, zorig), width=bwd, height=hwd, normal_axis=naxis, color='blue', \
                                                    name="Window " + str(index)))
            
        elif host == 2 : # window carved in 6h wall
            naxis = 'x'
            xorig = 0
            yorig = B_rm - bwd - dcr
            zorig = zwd
            windows.append(create_rectangle_on_wall(origin=(xorig, yorig, zorig), width=bwd, height=hwd, normal_axis=naxis, color='blue', \
                                                    name="Window " + str(index)))
    
        else :  # host == 3 : # window carved in 9h wall
            naxis = 'y'
            xorig = L_rm - bwd - dcr 
            yorig = B_rm
            zorig = zwd
            windows.append(create_rectangle_on_wall(origin=(xorig, yorig, zorig), width=bwd, height=hwd, normal_axis=naxis, color='blue', \
                                                    name="Window " + str(index)))

    return room, wallsint, windows



