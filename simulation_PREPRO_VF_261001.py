import pandas as pd                 # Data tables
import os                           # Returns current directory, create files
import numpy as np                  # Arrays
import matplotlib.pyplot as plt	    # Graphs

from math import sqrt, atan, log, exp, sin, cos, tan, trunc

from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.optimize import *

pi = np.pi

# Infinitely long, directly opposed parallel plates of the same finite width.

def VF_para_inf(h,w):
    # h : Distance between plates
    # w : Width of the plates
    if w == 0:
        VF = 0
    else:
        HW = h/w
        VF = sqrt(1 + HW**2 ) - HW
    return VF

# Infinitely long, perpendicular plates, from plate 1 to plate 2.

def VF_per_inf(a1,a2):
    # a1 : Width of plate 1
    # a2 : Width of plate 2
    if a1 == 0:
        VF = 0
    else:
        HW = a2/a1
        VF = 1/2*(1 + HW - sqrt (1 + HW**2 ))
    return VF

# View factor parallel plates from 1 to 2 
#(Ref. Isidoro Martinez,"Radiative view Factors", p. 23 )

def BB(x,y,eta,xi,z):
    BB = ((y-eta)*sqrt((x-xi)**2+z**2)*atan((y-eta)/sqrt((x-xi)**2+z**2)) \
          +(x-xi)*sqrt((y-eta)**2+z**2)*atan((x-xi)/sqrt((y-eta)**2+z**2))-z**2/2*log((x-xi)**2+(y-eta)**2+z**2))
    return BB

# This VF is checked and it is OK
# a1, b1, a2, b2 : half breadths (along x) and half lenghts (along y) of surfaces 1 and 2
# a0, b0 : coordinates of the center of surface 1 projected on surface 2, refering to surface 2 center
def VF_para(a1, b1, a2, b2, a0, b0, d):
        eps = 0.0000001
        if (d > eps) and (a1 > eps) and  (b1 > eps) and (a2 > eps) and (b2 > eps):
            z = d
            x = np.array([a0 - a1, a0 + a1])
            y = np.array([b0 - b1, b0 + b1])
            xi = np.array([- a2, a2])
            eta = np.array([- b2, b2])
            F = 0
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        for l in range(2):
                            F = F + ((-1)**(4+i+j+k+l)) * BB(x[i],y[j],eta[k],xi[l],z)
            A=(x[1]-x[0])*(y[1]-y[0])
            VF = F / (2 * pi * A)
        else:
            VF =0
        return VF

# View factor finite parallel plates same size
def VF_para_id(a,b,c):
    X = a/c
    Y = b/c
    VF = VF_para(a/2, b/2, a/2, b/2, 0, 0, c)
    return VF

# View factor perpendicular plates from 1 to 2 without offsets
def VF_per(a,b,c):
    if a != 0 and b != 0 and c != 0 :
        H=b/c
        W=a/c
        A=((1+W**2)*(1+H**2))/(1+W**2+H**2)
        B=((W**2*(1+W**2+H**2))/((1+W**2)*(W**2+H**2)))**(W**2)
        C=((H**2*(1+W**2+H**2))/((1+H**2)*(W**2+H**2)))**(H**2)
        VF=1/(pi*W)*(W*atan(1/W)+H*atan(1/H)-sqrt(H**2+W**2)*atan(1/sqrt(H**2+W**2))+ (1/4)* log(A*B*C))
    else:
        VF =0
    return VF

# View factor perpendicular plates from 1 to 2 with offsets
def VF_perp(a_1,a_2,b_1,b_2,c):
    A_1 = a_2*c
    A_13= (a_1+a_2)*c
    A_3 = a_1*c
    A_2 = b_2*c
    A_24= (b_1+b_2)*c
    A_4 = b_1*c
    if a_1 == 0 and  b_1 == 0:
        VF = VF_per(a_2,b_2,c)
    elif a_1 == 0:
        VF = VF_per(a_2,b_1+b_2,c) - VF_per(a_2,b_1,c)
    elif b_1 == 0:
        VF = (A_13*VF_per(a_1+a_2,b_2,c)-A_3*VF_per(a_1,b_2,c))/A_1
    else:
        VF=(A_13*VF_per(a_1+a_2,b_1+b_2,c)+A_3*VF_per(a_1,b_1,c) \
            -A_3*VF_per(a_1,b_1+b_2,c)-A_13*VF_per(a_1+a_2,b_1,c))/A_1
    return VF

# View factor between two not touching perpendicular rectangles from 1 to 2
def VF_perp_nt(a1, a2, b1, b2, c1, c2, c3):
# a1, a2, b1, b2, c1, c2, and  c3:  dimensions as shown in the figure see notebooks view_factors.
    
# Rectangle 1
# a1: dist measured in the plane of rectangle 1 perpendicular to the common edge of the two rectangles planes
#     distance from this common edge to the closest edge of rectangle 1 that is parallel to the common edge
# a2: height of rectangle 1 measured in the plane of rectangle 1 perpendicular to the common edge of the two rectangles planes
    
# Rectangle 2
# b1: dist measured in the plane of rectangle 2 perpendicular to the common edge of the two rectangles planes
#     distance from this common edge to the closest edge of rectangle 2 that is parallel to the common edge
# b2: height of rectangle 2 measured in the plane of rectangle 2 perpendicular to the common edge of the two rectangles planes
    
# Dimensions measured parallel to the common edge of the two rectangles
# c1: width of rectangle 2 parallel to the common edge of the two rectangles
# c2: distance from the two nearest edges of both rectangles parallel to the common edge of the two rectangles planes
# c3: width of rectangle 1 parallel to the common edge of the two rectangles

    if c2 == 0: # two rectangles sharing the same edge without spacing
        # Reduce to touching perpendicular rectangles
        BB012345 = VF_per((a1+a2),(b1+b2),(c1+c3))*(a1+a2)*(c1+c3)
        BB1234   = VF_per((a1+a2),(b1+b2), c1 )*(a1+a2)*c1
        BB0145   = VF_per((a1+a2),(b1+b2), c3 )*(a1+a2)*c3
        BB345    = VF_per(a1,b1,(c1+c3))*a1*(c1+c3)
    
        B345_012345 = VF_per(a1,(b1+b2),(c1+c3))*a1*(c1+c3)
        B012345_345 = VF_per((a1+a2),b1,(c1+c3))*(a1+a2)*(c1+c3)
    
        B0145_45 = VF_per((a1+a2),b1,c3)*(a1+a2)*c3
        B1234_34 = VF_per((a1+a2),b1,c1)*(a1+a2)*c1
    
        B45_0145 = VF_per(a1,(b1+b2),c3)*a1*c3
        B34_1234 = VF_per(a1,(b1+b2),c1)*a1*c1
    
        BB45 = VF_per(a1,b1,c3)*a1*c3
        BB34 = VF_per(a1,b1,c1)*a1*c1
    
        VF = (BB012345 - BB1234 - BB0145 + BB345
              - B345_012345 - B012345_345 + B0145_45 + B1234_34
              + B45_0145 + B34_1234 - BB45 - BB34) / (2*a2*c3)
        
        return VF
    
    BB012345=VF_per((a1+a2),(b1+b2),(c1+c2+c3))*(a1+a2)*(c1+c2+c3)
    BB1234=VF_per((a1+a2),(b1+b2),(c1+c2))*(a1+a2)*(c1+c2)
    BB0145=VF_per((a1+a2),(b1+b2),(c2+c3))*(a1+a2)*(c2+c3)
    BB345=VF_per(a1,b1,(c1+c2+c3))*a1*(c1+c2+c3)
    
    B345_012345=VF_per(a1,(b1+b2),(c1+c2+c3))*a1*(c1+c2+c3)
    B012345_345=VF_per((a1+a2),b1,(c1+c2+c3))*(a1+a2)*(c1+c2+c3)
    
    B0145_45=VF_per((a1+a2),b1,(c2+c3))*(a1+a2)*(c2+c3)
    B1234_34=VF_per((a1+a2),b1,(c1+c2))*(a1+a2)*(c1+c2)
    B45_0145=VF_per(a1,(b1+b2),(c2+c3))*a1*(c2+c3)
    B34_1234=VF_per(a1,(b1+b2),(c1+c2))*a1*(c1+c2)
    
    BB14=VF_per((a1+a2),(b1+b2),c2)*(a1+a2)*c2
    B14_4=VF_per((a1+a2),b1,c2)*(a1+a2)*c2
    BB45=VF_per(a1,b1,(c2+c3))*a1*(c2+c3)
    BB34=VF_per(a1,b1,(c1+c2))*a1*(c1+c2)
    B4_14=VF_per(a1,(b1+b2),c2)*a1*c2
    BB4=VF_per(a1,b1,c2)*a1*c2
    
    VF=(BB012345-BB1234-BB0145+BB345-B345_012345-B012345_345+B0145_45+B1234_34+B45_0145+B34_1234+BB14-B14_4 \
        -BB45-BB34-B4_14+BB4)/(2*a2*c3)
    
    return VF
    

def View_factors_room(length_6h_0h, length_9h_3h, Height_room):

    # View factors between entire (not carved) walls
    
    area_wall_0h = length_9h_3h * Height_room
    area_wall_3h = length_6h_0h * Height_room
    area_wall_6h = length_9h_3h * Height_room
    area_wall_9h = length_6h_0h * Height_room
    area_floor   = length_6h_0h * length_9h_3h
    area_ceiling = length_6h_0h * length_9h_3h

    areas_wall = np.array([area_wall_0h, area_wall_3h, area_wall_6h, area_wall_9h, area_floor, area_ceiling])
 
    # "!Compute view factors from wall 1 to wall 2"
    # "Two opposite identical planes"
    FV_floor_to_ceiling=VF_para_id(length_9h_3h,length_6h_0h,Height_room)
    FV_ceiling_to_floor=FV_floor_to_ceiling
 
    # "Floor to vertical walls"
    FV_floor_to_3h=VF_per(length_9h_3h,Height_room,length_6h_0h)
    FV_3h_to_floor=VF_per(Height_room,length_9h_3h,length_6h_0h)
 
    FV_floor_to_6h=VF_per(length_6h_0h,Height_room,length_9h_3h)
    FV_6h_to_floor=VF_per(Height_room,length_6h_0h,length_9h_3h)

    FV_floor_to_9h=FV_floor_to_3h
    FV_9h_to_floor=FV_3h_to_floor

    FV_floor_to_0h=FV_floor_to_6h
    FV_0h_to_floor=FV_6h_to_floor
 
    # "Ceiling and floor: identical FV for walls"
    FV_ceiling_to_3h=FV_floor_to_3h
    FV_3h_to_ceiling=FV_3h_to_floor

    FV_ceiling_to_6h=FV_floor_to_6h
    FV_6h_to_ceiling=FV_6h_to_floor

    FV_ceiling_to_9h=FV_ceiling_to_3h
    FV_9h_to_ceiling=FV_3h_to_ceiling

    FV_ceiling_to_0h=FV_ceiling_to_6h
    FV_0h_to_ceiling=FV_6h_to_ceiling
 
    # "Wall to wall: facing walls"
    FV_3h_to_9h=VF_para_id(length_6h_0h,Height_room,length_9h_3h)
    FV_9h_to_3h=FV_3h_to_9h

    FV_0h_to_6h=VF_para_id(length_9h_3h,Height_room,length_6h_0h)
    FV_6h_to_0h=FV_0h_to_6h
 
    # "Wall to wall: corners"
    FV_6h_to_9h=VF_per(length_9h_3h,length_6h_0h,Height_room)
    FV_9h_to_6h=VF_per(length_6h_0h,length_9h_3h,Height_room)

    FV_0h_to_9h=FV_6h_to_9h
    FV_9h_to_0h=FV_9h_to_6h

    FV_6h_to_3h=FV_6h_to_9h
    FV_3h_to_6h=FV_9h_to_6h

    FV_0h_to_3h=FV_0h_to_9h
    FV_3h_to_0h=FV_9h_to_0h

 
    # "!FV_wall(i,j) : View factor from the wall i to the wall j"
    FV_wall = np.array([[0             , FV_0h_to_3h     , FV_0h_to_6h     , FV_0h_to_9h     , FV_0h_to_floor     , FV_0h_to_ceiling   ], \
                      [FV_3h_to_0h     , 0               , FV_3h_to_6h     , FV_3h_to_9h     , FV_3h_to_floor     , FV_3h_to_ceiling   ], \
                      [FV_6h_to_0h     , FV_6h_to_3h     , 0               , FV_6h_to_9h     , FV_6h_to_floor     , FV_6h_to_ceiling   ], \
                      [FV_9h_to_0h     , FV_9h_to_3h     , FV_9h_to_6h     , 0               , FV_9h_to_floor     , FV_9h_to_ceiling   ], \
                      [FV_floor_to_0h  , FV_floor_to_3h  , FV_floor_to_6h  , FV_floor_to_9h  , 0                  , FV_floor_to_ceiling], \
                      [FV_ceiling_to_0h, FV_ceiling_to_3h, FV_ceiling_to_6h, FV_ceiling_to_9h, FV_ceiling_to_floor, 0                  ]])

    return FV_wall, areas_wall
    


def View_factors_windows(L_rm, B_rm, H_rm, dfwd):

    # View factors between windows

    # length on 6h-0h direction
    length_6h_0h = L_rm
    # length on 9h-3h direction
    length_9h_3h = B_rm
    # Room height
    Height_room  = H_rm

    # View factors from windows i to windows j
    # iterate over the list of dict
    VF_wd = np.zeros([len(dfwd), len(dfwd)])
    areas_wd = np.zeros(len(dfwd))
    
    for i, row in dfwd.iterrows():
    
        Height_wd_i      = row['height [m]']
        Breadth_wd_i     = row['breadth [m]']
        Height_wd_sill_i = row['sill height [m]']
        D_corner_i       = row['dist corner [m]']
        host_i           = row['host']
        area_i           = row['area']

        if area_i == 0 : continue

        areas_wd[i] = area_i
     
        for j, row in dfwd.iloc[i + 1:].iterrows():
            
            Height_wd_j      = row['height [m]']
            Breadth_wd_j     = row['breadth [m]']
            Height_wd_sill_j = row['sill height [m]']
            D_corner_j       = row['dist corner [m]']
            host_j           = row['host']
            area_j           = row['area']

            if area_j == 0 : continue
            if host_i == host_j : continue

            if (abs(host_j - host_i)) % 2 == 0: # even number -> parallel windows

                if host_i == 0 or host_i == 2 :
                    d = length_6h_0h
                    B = length_9h_3h
                else :
                    d = length_9h_3h
                    B = length_6h_0h
                    
                VF_i_to_j = View_factors_para_windows(Height_wd_i, Breadth_wd_i, Height_wd_sill_i, D_corner_i, host_i, area_i, \
                              Height_wd_j, Breadth_wd_j, Height_wd_sill_j, D_corner_j, host_j, area_j, d, B)
                
            else : # -> perpendicular windows
                
                VF_i_to_j = View_factors_perp_windows(length_6h_0h, length_9h_3h, Height_room, \
                              Height_wd_i, Breadth_wd_i, Height_wd_sill_i, D_corner_i, host_i, area_i, \
                              Height_wd_j, Breadth_wd_j, Height_wd_sill_j, D_corner_j, host_j, area_j)
                
            VF_wd[i, j] = VF_i_to_j
            VF_wd[j, i] = VF_i_to_j * area_i / area_j if area_j > 0 else 0
            
    return VF_wd
    

def View_factors_para_windows(Height_wd_i, Breadth_wd_i, Height_wd_sill_i, D_corner_i, host_i, area_i, \
                              Height_wd_j, Breadth_wd_j, Height_wd_sill_j, D_corner_j, host_j, area_j, d, B):

    sill_i = Height_wd_sill_i
    top_i = Height_wd_sill_i + Height_wd_i

    sill_j = Height_wd_sill_j 
    top_j  = Height_wd_sill_j + Height_wd_j

    a1 = Breadth_wd_i / 2
    b1 = Height_wd_i / 2
    a2 = Breadth_wd_j / 2
    b2 = Height_wd_j / 2

    a0 = abs(B - D_corner_j - Breadth_wd_j / 2 - D_corner_i - Breadth_wd_i / 2)
    b0 = abs(top_j - Height_wd_j / 2 - sill_i - Height_wd_i / 2)

    VF_i_to_j = VF_para(a1, b1, a2, b2, a0, b0, d)

    return VF_i_to_j
    

def View_factors_perp_windows(length_6h_0h, length_9h_3h, Height_room, \
                              Height_wd_i, Breadth_wd_i, Height_wd_sill_i, D_corner_i, host_i, area_i, \
                              Height_wd_j, Breadth_wd_j, Height_wd_sill_j, D_corner_j, host_j, area_j):


    sill_i = Height_wd_sill_i
    top_i = Height_wd_sill_i + Height_wd_i

    sill_j = Height_wd_sill_j 
    top_j  = Height_wd_sill_j + Height_wd_j
    
    if host_i == 0 and host_j == 1 :
        a1 = D_corner_i
        a2 = Breadth_wd_i
        b1 = length_6h_0h - D_corner_j - Breadth_wd_j
        b2 = Breadth_wd_j
    elif host_j == 0 and host_i == 1 :
        b1 = D_corner_j
        b2 = Breadth_wd_j
        a1 = length_6h_0h - D_corner_i - Breadth_wd_i
        a2 = Breadth_wd_i
        
    elif host_i == 1 and host_j == 2 :
        a1 = D_corner_i
        a2 = Breadth_wd_i
        b1 = length_9h_3h - D_corner_j - Breadth_wd_j
        b2 = Breadth_wd_j
    elif host_j == 1 and host_i == 2 :
        b1 = D_corner_j
        b2 = Breadth_wd_j
        a1 = length_9h_3h - D_corner_i - Breadth_wd_i
        a2 = Breadth_wd_i
        
    elif host_i == 2 and host_j == 3 :
        a1 = D_corner_i
        a2 = Breadth_wd_i
        b1 = length_6h_0h - D_corner_j - Breadth_wd_j
        b2 = Breadth_wd_j
    elif host_j == 2 and host_i == 3 :
        b1 = D_corner_j
        b2 = Breadth_wd_j
        a1 = length_6h_0h - D_corner_i - Breadth_wd_i
        a2 = Breadth_wd_i
        
    elif host_i == 3 and host_j == 0 :
        a1 = D_corner_i
        a2 = Breadth_wd_i
        b1 = length_9h_3h - D_corner_j - Breadth_wd_j
        b2 = Breadth_wd_j
    elif host_j == 3 and host_i == 0 :
        b1 = D_corner_j
        b2 = Breadth_wd_j
        a1 = length_9h_3h - D_corner_i - Breadth_wd_i
        a2 = Breadth_wd_i
        

    if sill_i == sill_j and top_i == top_j : # Same sill height, same window height
        c = Height_wd_i
        VF_i_to_j = VF_perp(a1, a2, b1, b2, c)
        
        
    elif top_i <= sill_j or sill_i >= top_j: # No intersection of projections along Z
        c1 = Height_wd_j
        c2 = sill_j - top_i if top_i <= sill_j else sill_i - top_j
        c3 = Height_wd_i
        VF_i_to_j = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)
        

    elif sill_i >= sill_j and top_i <= top_j : # Z projection of j includes Z projection of i
        c1 = top_j - top_i
        c2 = 0
        c3 = Height_wd_i
        VF_i_to_j1 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3) if c1 > 0 else 0
        
        c1 = sill_i - sill_j
        c2 = 0
        c3 = Height_wd_i
        VF_i_to_j2 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3) if c1 > 0 else 0

        c = Height_wd_i
        VF_i_to_j3 = VF_perp(a1, a2, b1, b2, c)

        VF_i_to_j = VF_i_to_j1 + VF_i_to_j2 + VF_i_to_j3
        
        
    elif sill_i <= sill_j and top_i >= top_j : # Z projection of i includes Z projection of j
        c1 = top_i - top_j
        c2 = 0
        c3 = Height_wd_j
        VF_j_to_i1 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3) if c1 > 0 else 0
        
        c1 = sill_j - sill_i
        c2 = 0
        c3 = Height_wd_j
        VF_j_to_i2 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3) if c1 > 0 else 0

        c = Height_wd_j
        VF_j_to_i3 = VF_perp(a1, a2, b1, b2, c)

        VF_j_to_i = VF_j_to_i1 + VF_j_to_i2 + VF_j_to_i3
        VF_i_to_j = area_j * VF_j_to_i / area_i if area_i > 0 else 0


    elif sill_i <= sill_j and sill_j <= top_i and top_i <= top_j : # Intersection of projections along Z, j higher than i
        c1 = top_j - top_i
        c2 = 0
        c3 = Height_wd_i
        VF_i_to_j1 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3) if c1 > 0 else 0

        c = top_i - sill_j
        VF_j2_to_i1 = VF_perp(a1, a2, b1, b2, c)

        c1 = sill_j - sill_i
        c2 = 0
        c3 = top_i - sill_j
        VF_j2_to_i2 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3) if c1 > 0 else 0

        VF_j2_to_i = VF_j2_to_i1 + VF_j2_to_i2
        area_j2 = area_j * (top_i - sill_j) / Height_wd_j if Height_wd_j > 0 else 0
        VF_i_to_j2 = area_j2 * VF_j2_to_i / area_i if area_i > 0 else 0

        VF_i_to_j = VF_i_to_j1 + VF_i_to_j2
        
    
    elif sill_j <= sill_i and sill_i <= top_j and top_i >= top_j : # Intersection of projections along Z, i higher than j
        c1 = sill_i - sill_j
        c2 = 0
        c3 = Height_wd_i
        VF_i_to_j2 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3) if c1 > 0 else 0

        c = top_j - sill_i
        VF_j1_to_i2 = VF_perp(a1, a2, b1, b2, c)

        c1 = top_i - top_j
        c2 = 0
        c3 = top_j - sill_i
        VF_j1_to_i1 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3) if c1 > 0 else 0

        VF_j1_to_i = VF_j1_to_i1 + VF_j1_to_i2
        area_j1 = area_j * (top_j - sill_i) / Height_wd_j if Height_wd_j > 0 else 0
        VF_i_to_j1 = area_j1 * VF_j1_to_i / area_i if area_i > 0 else 0

        VF_i_to_j = VF_i_to_j1 + VF_i_to_j2

    return VF_i_to_j
    


def View_window_0h_to_walls(L_rm, B_rm, H_rm, Height_wd, Breadth_wd, Height_wd_sill, D_corner_wd):

    # View factors between one window to all walls

    # length on 6h-0h direction
    length_6h_0h = L_rm
    # length on 9h-3h direction
    length_9h_3h = B_rm
    # Room height
    Height_room  = H_rm
    
    VF_wd_to_walls = np.zeros(6)
    

    # To wall 3h
    a1 = D_corner_wd
    a2 = Breadth_wd
    b1 = 0
    b2 = length_6h_0h 
    c  = Height_wd
    VF_wd_to_3h_2 = VF_perp(a1, a2, b1, b2, c)

    c1 = Height_wd_sill
    c2 = 0
    c3 = Height_wd
    VF_wd_to_3h_1 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    c1 = Height_room - Height_wd_sill - Height_wd
    c2 = 0
    c3 = Height_wd
    VF_wd_to_3h_3 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    VF_wd_to_walls[1] = VF_wd_to_3h_1 + VF_wd_to_3h_2 + VF_wd_to_3h_3
    

    # To wall 9h
    a1 = length_9h_3h - D_corner_wd - Breadth_wd
    a2 = Breadth_wd
    b1 = 0
    b2 = length_6h_0h 
    c  = Height_wd
    VF_wd_to_9h_2 = VF_perp(a1, a2, b1, b2, c)

    c1 = Height_wd_sill
    c2 = 0
    c3 = Height_wd
    VF_wd_to_9h_1 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    c1 = Height_room - Height_wd_sill - Height_wd
    c2 = 0
    c3 = Height_wd
    VF_wd_to_9h_3 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    VF_wd_to_walls[3] = VF_wd_to_9h_1 + VF_wd_to_9h_2 + VF_wd_to_9h_3
    

    # To floor
    a1 = Height_wd_sill
    a2 = Height_wd
    b1 = 0
    b2 = length_6h_0h 
    c  = Breadth_wd
    VF_wd_to_fl_2 = VF_perp(a1, a2, b1, b2, c)

    c1 = D_corner_wd
    c2 = 0
    c3 = Breadth_wd
    VF_wd_to_fl_1 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    c1 = length_9h_3h - Breadth_wd - D_corner_wd
    c2 = 0
    c3 = Breadth_wd
    VF_wd_to_fl_3 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    VF_wd_to_walls[4] = VF_wd_to_fl_1 + VF_wd_to_fl_2 + VF_wd_to_fl_3
    

    # To ceiling
    a1 = Height_room - Height_wd_sill - Height_wd 
    a2 = Height_wd
    b1 = 0
    b2 = length_6h_0h 
    c  = Breadth_wd
    VF_wd_to_ce_2 = VF_perp(a1, a2, b1, b2, c)

    c1 = D_corner_wd
    c2 = 0
    c3 = Breadth_wd
    VF_wd_to_ce_1 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    c1 = length_9h_3h - Breadth_wd - D_corner_wd
    c2 = 0
    c3 = Breadth_wd
    VF_wd_to_ce_3 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    VF_wd_to_walls[5] = VF_wd_to_ce_1 + VF_wd_to_ce_2 + VF_wd_to_ce_3

    VF_wd_to_walls[2] = 1 - VF_wd_to_walls[1] - VF_wd_to_walls[3] - VF_wd_to_walls[4] - VF_wd_to_walls[5]
    

    # To wall 6h
    a1 = Breadth_wd / 2
    b1 = Height_wd / 2
    a2 = length_9h_3h / 2
    b2 = Height_room / 2 
    a0 = D_corner_wd + Breadth_wd / 2 - length_9h_3h / 2
    b0 = Height_wd_sill + Height_wd / 2 - Height_room / 2
    d  = length_6h_0h

    VF_wd_to_6h = VF_para(a1, b1, a2, b2, a0, b0, d)

    # Check results

    # print()
    # print('wd_0h_to_6h ', VF_wd_to_6h - VF_wd_to_walls[2], ' = 0 ?')
    # print()
 
    return VF_wd_to_walls
    

def View_window_6h_to_walls(L_rm, B_rm, H_rm, Height_wd, Breadth_wd, Height_wd_sill, D_corner_wd):

    # View factors between one window to all walls

    # length on 6h-0h direction
    length_6h_0h = L_rm
    # length on 9h-3h direction
    length_9h_3h = B_rm
    # Room height
    Height_room  = H_rm
    
    VF_wd_to_walls = np.zeros(6)

    
    # To wall 9h
    a1 = D_corner_wd
    a2 = Breadth_wd
    b1 = 0
    b2 = length_6h_0h 
    c  = Height_wd
    VF_wd_to_9h_2 = VF_perp(a1, a2, b1, b2, c)

    c1 = Height_wd_sill
    c2 = 0
    c3 = Height_wd
    VF_wd_to_9h_1 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    c1 = Height_room - Height_wd_sill - Height_wd
    c2 = 0
    c3 = Height_wd
    VF_wd_to_9h_3 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    VF_wd_to_walls[3] = VF_wd_to_9h_1 + VF_wd_to_9h_2 + VF_wd_to_9h_3
    

    # To wall 3h
    a1 = length_9h_3h - D_corner_wd - Breadth_wd
    a2 = Breadth_wd
    b1 = 0
    b2 = length_6h_0h 
    c  = Height_wd
    VF_wd_to_3h_2 = VF_perp(a1, a2, b1, b2, c)

    c1 = Height_wd_sill
    c2 = 0
    c3 = Height_wd
    VF_wd_to_3h_1 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    c1 = Height_room - Height_wd_sill - Height_wd
    c2 = 0
    c3 = Height_wd
    VF_wd_to_3h_3 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    VF_wd_to_walls[1] = VF_wd_to_3h_1 + VF_wd_to_3h_2 + VF_wd_to_3h_3
    

    # To floor
    a1 = Height_wd_sill
    a2 = Height_wd
    b1 = 0
    b2 = length_6h_0h 
    c  = Breadth_wd
    VF_wd_to_fl_2 = VF_perp(a1, a2, b1, b2, c)

    c1 = D_corner_wd
    c2 = 0
    c3 = Breadth_wd
    VF_wd_to_fl_1 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    c1 = length_9h_3h - Breadth_wd - D_corner_wd
    c2 = 0
    c3 = Breadth_wd
    VF_wd_to_fl_3 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    VF_wd_to_walls[4] = VF_wd_to_fl_1 + VF_wd_to_fl_2 + VF_wd_to_fl_3
    

    # To ceiling
    a1 = Height_room - Height_wd_sill - Height_wd 
    a2 = Height_wd
    b1 = 0
    b2 = length_6h_0h 
    c  = Breadth_wd
    VF_wd_to_ce_2 = VF_perp(a1, a2, b1, b2, c)

    c1 = D_corner_wd
    c2 = 0
    c3 = Breadth_wd
    VF_wd_to_ce_1 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    c1 = length_9h_3h - Breadth_wd - D_corner_wd
    c2 = 0
    c3 = Breadth_wd
    VF_wd_to_ce_3 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    VF_wd_to_walls[5] = VF_wd_to_ce_1 + VF_wd_to_ce_2 + VF_wd_to_ce_3

    VF_wd_to_walls[0] = 1 - VF_wd_to_walls[1] - VF_wd_to_walls[3] - VF_wd_to_walls[4] - VF_wd_to_walls[5]
    

    # To wall 0h
    a1 = Breadth_wd / 2
    b1 = Height_wd / 2
    a2 = length_9h_3h / 2
    b2 = Height_room / 2 
    a0 = D_corner_wd + Breadth_wd / 2 - length_9h_3h / 2
    b0 = Height_wd_sill + Height_wd / 2 - Height_room / 2
    d  = length_6h_0h

    VF_wd_to_0h = VF_para(a1, b1, a2, b2, a0, b0, d)

    # Check results

    # print()
    # print('wd_6h_to_0h ', VF_wd_to_0h - VF_wd_to_walls[0], ' = 0 ?')
    # print()
 
    return VF_wd_to_walls
    
    
def View_window_3h_to_walls(L_rm, B_rm, H_rm, Height_wd, Breadth_wd, Height_wd_sill, D_corner_wd):

    # View factors between one window to all walls

    # length on 6h-0h direction
    length_6h_0h = L_rm
    # length on 9h-3h direction
    length_9h_3h = B_rm
    # Room height
    Height_room  = H_rm
    
    VF_wd_to_walls = np.zeros(6)
    

    # To wall 6h
    a1 = D_corner_wd
    a2 = Breadth_wd
    b1 = 0
    b2 = length_9h_3h 
    c  = Height_wd
    VF_wd_to_6h_2 = VF_perp(a1, a2, b1, b2, c)

    c1 = Height_wd_sill
    c2 = 0
    c3 = Height_wd
    VF_wd_to_6h_1 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    c1 = Height_room - Height_wd_sill - Height_wd
    c2 = 0
    c3 = Height_wd
    VF_wd_to_6h_3 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    VF_wd_to_walls[2] = VF_wd_to_6h_1 + VF_wd_to_6h_2 + VF_wd_to_6h_3
    

    # To wall 0h
    a1 = length_6h_0h - D_corner_wd - Breadth_wd
    a2 = Breadth_wd
    b1 = 0
    b2 = length_9h_3h  
    c  = Height_wd
    VF_wd_to_0h_2 = VF_perp(a1, a2, b1, b2, c)

    c1 = Height_wd_sill
    c2 = 0
    c3 = Height_wd
    VF_wd_to_0h_1 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    c1 = Height_room - Height_wd_sill - Height_wd
    c2 = 0
    c3 = Height_wd
    VF_wd_to_0h_3 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    VF_wd_to_walls[0] = VF_wd_to_0h_1 + VF_wd_to_0h_2 + VF_wd_to_0h_3
    

    # To floor
    a1 = Height_wd_sill
    a2 = Height_wd
    b1 = 0
    b2 = length_9h_3h 
    c  = Breadth_wd
    VF_wd_to_fl_2 = VF_perp(a1, a2, b1, b2, c)

    c1 = D_corner_wd
    c2 = 0
    c3 = Breadth_wd
    VF_wd_to_fl_1 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    c1 = length_6h_0h - Breadth_wd - D_corner_wd
    c2 = 0
    c3 = Breadth_wd
    VF_wd_to_fl_3 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    VF_wd_to_walls[4] = VF_wd_to_fl_1 + VF_wd_to_fl_2 + VF_wd_to_fl_3
    

    # To ceiling
    a1 = Height_room - Height_wd_sill - Height_wd 
    a2 = Height_wd
    b1 = 0
    b2 = length_9h_3h 
    c  = Breadth_wd
    VF_wd_to_ce_2 = VF_perp(a1, a2, b1, b2, c)

    c1 = D_corner_wd
    c2 = 0
    c3 = Breadth_wd
    VF_wd_to_ce_1 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    c1 = length_6h_0h - Breadth_wd - D_corner_wd
    c2 = 0
    c3 = Breadth_wd
    VF_wd_to_ce_3 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    VF_wd_to_walls[5] = VF_wd_to_ce_1 + VF_wd_to_ce_2 + VF_wd_to_ce_3

    VF_wd_to_walls[3] = 1 - VF_wd_to_walls[0] - VF_wd_to_walls[2] - VF_wd_to_walls[4] - VF_wd_to_walls[5]
    

    # To wall 9h
    a1 = Breadth_wd / 2
    b1 = Height_wd / 2
    a2 = length_6h_0h / 2
    b2 = Height_room / 2 
    a0 = D_corner_wd + Breadth_wd / 2 - length_6h_0h / 2
    b0 = Height_wd_sill + Height_wd / 2 - Height_room / 2
    d  = length_9h_3h

    VF_wd_to_9h = VF_para(a1, b1, a2, b2, a0, b0, d)

    # Check results

    # print()
    # print('wd_3h_to_9h ', VF_wd_to_9h - VF_wd_to_walls[3], ' = 0 ?')
    # print()
 
    return VF_wd_to_walls
    

def View_window_9h_to_walls(L_rm, B_rm, H_rm, Height_wd, Breadth_wd, Height_wd_sill, D_corner_wd):

    # View factors between one window to all walls

    # length on 6h-0h direction
    length_6h_0h = L_rm
    # length on 9h-3h direction
    length_9h_3h = B_rm
    # Room height
    Height_room  = H_rm
    
    VF_wd_to_walls = np.zeros(6)
    

    # To wall 0h
    a1 = D_corner_wd
    a2 = Breadth_wd
    b1 = 0
    b2 = length_9h_3h 
    c  = Height_wd
    VF_wd_to_0h_2 = VF_perp(a1, a2, b1, b2, c)

    c1 = Height_wd_sill
    c2 = 0
    c3 = Height_wd
    VF_wd_to_0h_1 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    c1 = Height_room - Height_wd_sill - Height_wd
    c2 = 0
    c3 = Height_wd
    VF_wd_to_0h_3 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    VF_wd_to_walls[0] = VF_wd_to_0h_1 + VF_wd_to_0h_2 + VF_wd_to_0h_3
    

    # To wall 6h
    a1 = length_6h_0h - D_corner_wd - Breadth_wd
    a2 = Breadth_wd
    b1 = 0
    b2 = length_9h_3h  
    c  = Height_wd
    VF_wd_to_6h_2 = VF_perp(a1, a2, b1, b2, c)

    c1 = Height_wd_sill
    c2 = 0
    c3 = Height_wd
    VF_wd_to_6h_1 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    c1 = Height_room - Height_wd_sill - Height_wd
    c2 = 0
    c3 = Height_wd
    VF_wd_to_6h_3 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    VF_wd_to_walls[2] = VF_wd_to_6h_1 + VF_wd_to_6h_2 + VF_wd_to_6h_3
    

    # To floor
    a1 = Height_wd_sill
    a2 = Height_wd
    b1 = 0
    b2 = length_9h_3h 
    c  = Breadth_wd
    VF_wd_to_fl_2 = VF_perp(a1, a2, b1, b2, c)

    c1 = D_corner_wd
    c2 = 0
    c3 = Breadth_wd
    VF_wd_to_fl_1 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    c1 = length_6h_0h - Breadth_wd - D_corner_wd
    c2 = 0
    c3 = Breadth_wd
    VF_wd_to_fl_3 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    VF_wd_to_walls[4] = VF_wd_to_fl_1 + VF_wd_to_fl_2 + VF_wd_to_fl_3
    

    # To ceiling
    a1 = Height_room - Height_wd_sill - Height_wd 
    a2 = Height_wd
    b1 = 0
    b2 = length_9h_3h 
    c  = Breadth_wd
    VF_wd_to_ce_2 = VF_perp(a1, a2, b1, b2, c)

    c1 = D_corner_wd
    c2 = 0
    c3 = Breadth_wd
    VF_wd_to_ce_1 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    c1 = length_6h_0h - Breadth_wd - D_corner_wd
    c2 = 0
    c3 = Breadth_wd
    VF_wd_to_ce_3 = VF_perp_nt(a1, a2, b1, b2, c1, c2, c3)

    VF_wd_to_walls[5] = VF_wd_to_ce_1 + VF_wd_to_ce_2 + VF_wd_to_ce_3

    VF_wd_to_walls[1] = 1 - VF_wd_to_walls[0] - VF_wd_to_walls[2] - VF_wd_to_walls[4] - VF_wd_to_walls[5]
    

    # To wall 3h
    a1 = Breadth_wd / 2
    b1 = Height_wd / 2
    a2 = length_6h_0h / 2
    b2 = Height_room / 2 
    a0 = D_corner_wd + Breadth_wd / 2 - length_6h_0h / 2
    b0 = Height_wd_sill + Height_wd / 2 - Height_room / 2
    d  = length_9h_3h

    VF_wd_to_3h = VF_para(a1, b1, a2, b2, a0, b0, d)

    # Check results

    # print()
    # print('wd_9h_to_3h ', VF_wd_to_3h - VF_wd_to_walls[1], ' = 0 ?')
    # print()
 
    return VF_wd_to_walls


def View_window_to_walls(host_wd, L_rm, B_rm, H_rm, Height_wd, Breadth_wd, Height_wd_sill, D_corner_wd):

    # View factors between one window to all walls
    
    VF_wd_to_walls = np.zeros(6)
    
    if host_wd == 0:
        VF_wd_to_walls = View_window_0h_to_walls(L_rm, B_rm, H_rm, Height_wd, Breadth_wd, Height_wd_sill, D_corner_wd)

    if host_wd == 1:
        VF_wd_to_walls = View_window_3h_to_walls(L_rm, B_rm, H_rm, Height_wd, Breadth_wd, Height_wd_sill, D_corner_wd)

    if host_wd == 2:
        VF_wd_to_walls = View_window_6h_to_walls(L_rm, B_rm, H_rm, Height_wd, Breadth_wd, Height_wd_sill, D_corner_wd)

    if host_wd == 3:
        VF_wd_to_walls = View_window_9h_to_walls(L_rm, B_rm, H_rm, Height_wd, Breadth_wd, Height_wd_sill, D_corner_wd) 
    
    return VF_wd_to_walls
    

def View_factors_multiple_wds(length_6h_0h, length_9h_3h, Height_room, dfwd, VF_wds):

    # Ui uncarved walls
    # Hi walls carved by windows 0 to k-1, not yet carved by window k --> values change at each addition of a window
    # Ri walls carved with windows --> values change at each addition of a window

    n_op_walls = 6
    n_wds = len(dfwd)

    areas_wds = dfwd['area'].values
    host_wds  = dfwd['host'].values

    # Initialisations
    
    # ComputeView factors for not yet carved walls
    VF_Hi_to_Hj, areas_Hi = View_factors_room(length_6h_0h, length_9h_3h, Height_room) 


    areas_Ri = areas_Hi
    
    n_walls = n_op_walls
    
    for k, row in dfwd.iterrows():
        
        Height_wd      = row['height [m]']
        Breadth_wd     = row['breadth [m]']
        Height_wd_sill = row['sill height [m]']
        D_corner_wd    = row['dist corner [m]']
        host_wd        = row['host']
        area_wd        = row['area']

        n_walls = n_walls + 1

        VF_wd_to_Rj = np.zeros(n_walls)
        VF_Ri_to_wd = np.zeros(n_walls)
        VF_Ri_to_Rj = np.zeros((n_walls, n_walls))

        areas_Ri = np.concatenate((areas_Ri, [area_wd]))

        areas_Ri[int(host_wd)] = areas_Ri[int(host_wd)] - area_wd 

        VF_Ri_to_Rj[n_op_walls : n_op_walls + k+1, n_op_walls : n_op_walls + k+1] = VF_wds[:k+1, :k+1]

        # Uj uncarved walls
        VF_wd_to_Uj = \
        View_window_to_walls(host_wd, length_6h_0h, length_9h_3h, Height_room, Height_wd, Breadth_wd, Height_wd_sill, D_corner_wd)
        
        for j in range(n_op_walls) :
            VF_wd_to_Rj[j] = VF_wd_to_Uj[j] - np.array([VF_wds[k, l] for l in range(k) if host_wds[l] == j ]).sum() 
            
        VF_wd_to_Rj[n_op_walls : n_op_walls + k] = VF_wds[k, :k ]

        VF_Ri_to_Rj[n_walls - 1, : n_walls - 1] = VF_wd_to_Rj[: n_walls - 1]

        for i in range(n_walls) :
            VF_Ri_to_wd[i] = area_wd * VF_wd_to_Rj[i] / areas_Ri[i] if areas_Ri[i] > 0 else 0
        VF_Ri_to_Rj[: n_walls - 1, n_walls - 1] = VF_Ri_to_wd[: n_walls - 1]

        for i in range(n_walls - 1) :
            for j in range(n_walls - 1) :
                if (i != host_wd) & (j != host_wd) :
                    VF_Ri_to_Rj[i, j] = VF_Hi_to_Hj[i, j]
    
        for i in range(n_walls - 1) :
            if (i != host_wd)  :
                VF_Ri_to_Rj[i, int(host_wd)] = 1 - np.array([VF_Ri_to_Rj[i, j] for j in range(n_walls) if host_wd != j]).sum()

        for j in range(n_walls - 1) :
            if (j != host_wd) :
                VF_Ri_to_Rj[int(host_wd), j] = areas_Ri[j] * VF_Ri_to_Rj[j, int(host_wd)] / areas_Ri[int(host_wd)] if areas_Ri[int(host_wd)] > 0 else 0

        VF_Hi_to_Hj = VF_Ri_to_Rj.copy()


    return VF_Ri_to_Rj, areas_Ri











