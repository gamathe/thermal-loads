import pandas as pd                 # Data tables
import os                           # Returns current directory, create files
import numpy as np                  # Arrays

from math import sqrt, atan, log, exp, sin, cos, tan, trunc

def composite_material(lstm, name12, name1, name2, L1, L2):
    materials = pd.DataFrame(lstm)
    f1 = L1 / (L1 + L2)
    f2 = 1 - f1
    lambda1, rho1, c1 = materials.loc[materials['material'] == name1, ['lambda', 'rho', 'c']].iloc[0]
    lambda2, rho2, c2 = materials.loc[materials['material'] == name2, ['lambda', 'rho', 'c']].iloc[0]
    lambda12 = f1 * lambda1 + f2 * lambda2
    rho12    = f1 * rho1 + f2 * rho2
    c12      = f1 * c1 + f2 * c2
    lstm = lstm.append({"material": name12, "lambda":lambda12, "rho":rho12, "c":c12})
    return lstm


def add_copy(name_orig, name_copy, walls_types) :
    wtl = walls_types + [(name_copy, t[1], t[2], t[3]) for t in walls_types    if t[0] == name_orig]
    return wtl


def add_inverse_copy(name_orig, name_copy, walls_types) :
    wtl = walls_types + [(name_copy,t[1],t[2],t[3]) for t in walls_types[::-1] if t[0] == name_orig]
    return wtl

# Dataframe of materials

# Define dtypes explicitly
materials = pd.DataFrame({
    "material": pd.Series(dtype='str'),
    "lambda": pd.Series(dtype='float'),
    "rho": pd.Series(dtype='float'),
    "c": pd.Series(dtype='float')
})

materials = pd.concat([materials, pd.DataFrame({"material":'concrete_bloc_19cm' ,"lambda":1.36,  "rho":2000, "c":1000}, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'concrete_bloc_14cm' ,"lambda":1.27,  "rho":2000, "c":1000}, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'insulation'         ,"lambda":0.026, "rho":80,   "c":840 }, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'reinforced_concrete',"lambda":2.30,  "rho":2400, "c":1000}, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'concrete'           ,"lambda":2.10,  "rho":2400, "c":1000}, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'brick'              ,"lambda":1.59,  "rho":2200, "c":1000}, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'screed'             ,"lambda":1.40,  "rho":2000, "c":1000}, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'foamglass'          ,"lambda":0.055, "rho":150,  "c":1000}, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'wood'               ,"lambda":0.13,  "rho":600,  "c":1000}, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'glass'              ,"lambda":1.00,  "rho":2500, "c":840 }, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'air_layer_60mm'     ,"lambda":0.35,  "rho":1.2,  "c":1000}, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'air_layer_275mm'    ,"lambda":1.62,  "rho":1.2,  "c":1000}, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'waterproof_membrane',"lambda":0.230, "rho":1300, "c":1000}, index=[0])])
materials = pd.concat([materials, pd.DataFrame({"material":'gravels'            ,"lambda":2.0,   "rho":2000, "c":1000}, index=[0])])

materials = materials.reset_index(drop=True)



# Waimes House

materials_list = []

materials_list.append({"material":'Plaster_Panel'       ,"lambda":0.32, "rho":1200,"c":1000})
materials_list.append({"material":'Dry_screed_Panel'    ,"lambda":0.38, "rho":1225,"c":1000})
materials_list.append({"material":'DF_panel'            ,"lambda":0.20, "rho":1200,"c":1000})
materials_list.append({"material":'Glass_wool'          ,"lambda":0.036,"rho":28,  "c":1030})
materials_list.append({"material":'Rock_wool'           ,"lambda":0.039,"rho":115, "c":825})
materials_list.append({"material":'OSB'                 ,"lambda":0.12, "rho":553, "c":1700})
materials_list.append({"material":'Wood_cement_Panel'   ,"lambda":0.35, "rho":1250,"c":1000})
materials_list.append({"material":'Wood'                ,"lambda":0.13, "rho":600, "c":1600})
materials_list.append({"material":'PIR'                 ,"lambda":0.025,"rho":40,  "c":840})
materials_list.append({"material":'EPS_board'           ,"lambda":0.034,"rho":25,  "c":1300})
materials_list.append({"material":'Expanded_foam'       ,"lambda":0.036,"rho":12,  "c":1300})
materials_list.append({"material":'Waterproof_membrane' ,"lambda":0.230,"rho":1300,"c":1000})
materials_list.append({"material":'Gravels'             ,"lambda":2.0,  "rho":2000,"c":1000})
materials_list.append({"material":'Air_30mm_horizontal' ,"lambda":0.180,"rho":1.2, "c":1000})
materials_list.append({"material":'Air_40mm_horizontal' ,"lambda":0.240,"rho":1.2, "c":1000})
materials_list.append({"material":'Air_80mm_upwards'    ,"lambda":0.500,"rho":1.2, "c":1000})
materials_list.append({"material":'Air_80mm_horizontal' ,"lambda":0.444,"rho":1.2, "c":1000})
materials_list.append({"material":'Air_80mm_downwards'  ,"lambda":0.372,"rho":1.2, "c":1000})
materials_list.append({"material":'Air_160mm_upwards'   ,"lambda":1.000,"rho":1.2, "c":1000})
materials_list.append({"material":'Air_160mm_horizontal',"lambda":0.889,"rho":1.2, "c":1000})
materials_list.append({"material":'Air_160mm_downwards' ,"lambda":0.711,"rho":1.2, "c":1000})
materials_list.append({"material":'Air_300mm_upwards'   ,"lambda":1.875,"rho":1.2, "c":1000})
materials_list.append({"material":'Air_300mm_horizontal',"lambda":1.667,"rho":1.2, "c":1000})
materials_list.append({"material":'Air_300mm_downwards' ,"lambda":1.304,"rho":1.2, "c":1000})



materials = pd.DataFrame(materials_list)

# Add a composite material
# Data = material list, names of the first and second materials, widths of the first and second materials
composite_material(materials_list, 'Insulation_layer', 'Glass_wool', 'Wood', 0.6, 0.06)

materials = pd.DataFrame(materials_list)
materials = materials.set_index('material')


# Holzkirchen

# # Dataframe of materials

# materials = pd.DataFrame(columns=["name","lambda","rho","c"])

# materials = materials.append({"name":'ext_wool_insulation',"lambda":0.040,"rho":80,"c":840}, ignore_index=True)
# materials = materials.append({"name":'PU_diffusion_resist',"lambda":0.024,"rho":32,"c":1300}, ignore_index=True)
# materials = materials.append({"name":'Soft_woodfiber_panel',"lambda":0.039,"rho":110,"c":2100}, ignore_index=True)
# materials = materials.append({"name":'int_wool_insulation',"lambda":0.040,"rho":30,"c":840}, ignore_index=True)
# materials = materials.append({"name":'EPS_insulation_board',"lambda":0.038,"rho":25,"c":1500}, ignore_index=True)
# materials = materials.append({"name":'mineral_wool'       ,"lambda":0.040,"rho":30,"c":840}, ignore_index=True)
# materials = materials.append({"name":'wood_insulation'    ,"lambda":0.045,"rho":81,"c":923}, ignore_index=True)

# materials = materials.append({"name":'ext_plaster'        ,"lambda":1.00,"rho":1200,"c":1000}, ignore_index=True)
# materials = materials.append({"name":'ext_brick'          ,"lambda":0.22,"rho": 800,"c":1000}, ignore_index=True)
# materials = materials.append({"name":'former_plaster'     ,"lambda":1.00,"rho":1200,"c":1000}, ignore_index=True)
# materials = materials.append({"name":'floating_Levell_Uni',"lambda":0.89,"rho":1400,"c":1000}, ignore_index=True)
# materials = materials.append({"name":'Sto_Armat_Classic'  ,"lambda":0.70,"rho":1500,"c":1000}, ignore_index=True)
# materials = materials.append({"name":'finishing_coat'     ,"lambda":0.70,"rho":1800,"c":1000}, ignore_index=True)
# materials = materials.append({"name":'new_ext_plaster'    ,"lambda":1.00,"rho":1200,"c":1000}, ignore_index=True)
# materials = materials.append({"name":'concrete'           ,"lambda":2.10,"rho":2400,"c":1000}, ignore_index=True)
# materials = materials.append({"name":'reinforced concrete',"lambda":2.00,"rho":2400,"c":1000}, ignore_index=True)
# materials = materials.append({"name":'int_brick'          ,"lambda":0.34,"rho":1000,"c":1000}, ignore_index=True)
# materials = materials.append({"name":'int_plaster'        ,"lambda":0.90,"rho":1200,"c":1000}, ignore_index=True)
# materials = materials.append({"name":'gypsum_board'       ,"lambda":0.25,"rho":900,"c":1000}, ignore_index=True)
# materials = materials.append({"name":'OSB'                ,"lambda":0.13,"rho":650,"c":1700}, ignore_index=True)
# materials = materials.append({"name":'floor_heating_panel',"lambda":0.035,"rho":168,"c":1000}, ignore_index=True)
# materials = materials.append({"name":'Fermacell'          ,"lambda":0.32,"rho":1150,"c":1100}, ignore_index=True)
# materials = materials.append({"name":'screed'             ,"lambda":1.40,"rho":2000,"c":1000}, ignore_index=True)
# materials = materials.append({"name":'PUR_high_insulation',"lambda":0.023,"rho":80,"c":840}, ignore_index=True)
# materials = materials.append({"name":'PUR_low_insulation' ,"lambda":0.025,"rho":80,"c":840}, ignore_index=True)
# materials = materials.append({"name":'Levelling_fill'     ,"lambda":0.060,"rho":80,"c":840}, ignore_index=True)
# materials = materials.append({"name":'roof_tiles'         ,"lambda":0.070,"rho":2000,"c":1000}, ignore_index=True)
# materials = materials.append({"name":'wood'               ,"lambda":0.130,"rho":500,"c":1600}, ignore_index=True)
# materials = materials.append({"name":'tile'               ,"lambda":0.700,"rho":2000,"c":1000}, ignore_index=True)

# # Bottom  up Heat flow, thickness from 50 to 300mm: R = 0.16 m2-K/W 
# # Horizontal Heat flow, thickness from 50 to 300mm: R = 0.18 m2-K/W 
# materials = materials.append({"name":'air_layer_70mm'     ,"lambda":0.07/0.16,"rho":1.2,"c":1000}, ignore_index=True)
# materials = materials.append({"name":'air_layer_80mm'     ,"lambda":0.08/0.16,"rho":1.2,"c":1000}, ignore_index=True)


