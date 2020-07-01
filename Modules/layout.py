# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:37:55 2020

@author: ledezmaluism
"""

# import nazca as nd
import numpy as np
from phidl import Device, Layer, LayerSet, make_device
import phidl.geometry as pg
import phidl.routing as pr
import phidl.utilities as pu

def setup_layers():
    ls = LayerSet() # Create a blank LayerSet
    ls.add_layer(name = 'Waveguides', gds_layer = 1, gds_datatype = 0,  description = 'Ridge waveguide', color = 'crimson', alpha = 0.7)
    ls.add_layer(name = 'Metal_1', gds_layer = 10, gds_datatype = 0,  description = 'Metal before waveguide etching (poling, markers)', color = 'goldenrod', alpha = 0.7)
    ls.add_layer(name = 'Marker Mask', gds_layer = 20, gds_datatype = 0,  description = 'Marker Protection', color = 'dodgerblue', alpha = 0.4)
    ls.add_layer(name = 'Topo Marker', gds_layer = 21, gds_datatype = 0,  description = 'Etched marker', color = 'lightslategray', alpha = 0.9)
    ls.add_layer(name = 'Chip boundary', gds_layer = 99, gds_datatype = 0,  description = 'Chip boundary', color = 'darkblue', alpha = 0.1)
    return ls

def waveguide(width = 1, length = 10, layer = 1):
    '''
    Parameters
    ----------
    width : FLOAT, optional
        WIDTH OF THE WAVEGUIDE. The default is 1.
    length : FLOAT, optional
        LENGTH OF THE WAVEGUIDE. The default is 10.
    layer : INT, optional
        LAYER. The default is 1.

    Returns
    -------
    WG : DEVICE (PHIDL)
        WAVEGUIDE OBJECT

    '''
    WG = Device('Waveguide')
    WG.add_polygon( [(0, 0), (length, 0), (length, width), (0, width)] , layer=layer)
    WG.add_port(name = 1, midpoint = [0,width/2], width = width, orientation = 180)
    WG.add_port(name = 2, midpoint = [length,width/2], width = width, orientation = 0)
    return WG

def global_markers(layer_marker=10, layer_mask=20):
    D = Device('Global Markers')
    R = pg.rectangle(size=(20,20), layer=layer_marker)
    a = D.add_array(R, columns = 6, rows = 6,  spacing = (100, 100))
    a.move([-260, -260]) #Center of the array
    
    #Add marker cover
    cover = pg.bbox(bbox = a.bbox, layer=layer_mask)
    D << pg.offset(cover, distance=100, layer=layer_mask)
    return D

def chip(size = (13000, 18000), keepout=2000, name='chip01', text_size=250,
         layer_text=10, layer=99):
    k = keepout
    DX = size[0]
    DY = size[1]
    OUT = pg.rectangle(size=size, layer=layer)
    IN = pg.rectangle(size=(DX-2*k, DY-2*k), layer=layer)
    IN.move((k,k))
    
    CHIP = pg.boolean(A = OUT, B = IN, operation = 'A-B', layer=layer)
    
    #Add name
    L = pg.text(text=name, size=text_size, layer=layer_text, justify='center')
    CHIP.add_ref(L).move((DX/2,k-text_size-200))
    
    #Add markers
    M = global_markers()
    offset = 110
    CHIP.add_ref(M).move([k-offset,k-offset])
    CHIP.add_ref(M).move([DX-k+offset,k-offset])
    CHIP.add_ref(M).move([k-offset,DY-k+offset])
    CHIP.add_ref(M).move([DX-k+offset,DY-k+offset])
    
    return CHIP
  
def poling_region(length=4000, period=5, dutycycle=0.4, gap=25,
                  Lfinger=50, layer=10):
    
    #Fixed parameters
    height = 50
    
    #Calculations
    Wfinger = period*dutycycle
    length = length - round(length%period,1)
    Nfinger = int(length/period)
    length =  length - (1-dutycycle)*period

    P = Device('Poling Electrodes')
    
    #Positive side
    R = pg.rectangle([length, height], layer=layer)
    F = pg.rectangle([Wfinger, Lfinger], layer=layer)
    P << R
    a = P.add_array(F, columns=Nfinger, rows=1, spacing=(period,0))
    a.move([0, height])
    
    #Negative side
    R2 = pg.rectangle([length, height], layer=layer)
    r2 = P.add_ref(R2)
    r2.move([0, height+Lfinger+gap])
    
    return P

def contact_pads(size = (150,150), label='', label_size = 50, layer=10):
    # P = Device('Pad')
    R = pg.rectangle(size, layer)
    if label != '':
        L = pg.text(label, label_size, layer=layer)
        L.move([10,10])
        P = pg.boolean(A = R, B = L, operation = 'A-B', layer=layer)
    else:
        P = R
    return P

def resonator_half(radius = 100, width=1.0, length = 4000, layer=1):
    D = Device()
    
    R1 = pg.arc(radius, width, theta=180, start_angle=90, layer=layer)
    R2 = pg.arc(radius, width, theta=180, start_angle=-90, layer=layer)
    L = waveguide(width, length, layer)
    
    r1 = D.add_ref(R1)
    r2 = D.add_ref(R2)
    l = D.add_ref(L)
    
    r1.connect(port=1, destination=l.ports[1])
    r2.connect(port=2, destination=l.ports[2])
    
    return  D

#OPO function
def OPO(name='OPO', length=4000, radius=100, width=1.0, Lc=200, Cgap=1.0, 
        Loc=50, Cogap=1.5, pp=5.0, dutycycle=0.4, pgap=25,
        Lout_signal = 2600, Lout_pump=2500, Lin_pump=2500,
        radius_cpout = 100):
    
    OPO =  Device('OPO')
    
    #Poling region - centered
    p = OPO.add_ref(poling_region(length, period=pp, dutycycle=dutycycle, gap=pgap))
    p.move([0,-100-pgap/2])
    
    #Waveguides and couplers
    wg_main = OPO.add_ref(waveguide(width, length))
    cp1_a = OPO.add_ref(waveguide(width, Lc))
    cp2_a = OPO.add_ref(waveguide(width, Lc))
    cp1_b = OPO.add_ref(waveguide(width, Lc))
    cp2_b = OPO.add_ref(waveguide(width, Lc))
    
    cp1_a.connect(port=2, destination=wg_main.ports[1])
    cp2_a.connect(port=1, destination=wg_main.ports[2])
    cp1_b.connect(port=2, destination=wg_main.ports[1])
    cp2_b.connect(port=1, destination=wg_main.ports[2])
        
    cp1_b.move([0, Cgap + width])
    cp2_b.move([0, Cgap + width])
    
    #Resonator
    r1 = OPO.add_ref(pg.arc(radius, width, theta=180, start_angle=90, layer=1))
    r2 = OPO.add_ref(pg.arc(radius, width, theta=180, start_angle=-90, layer=1))  
    r1.connect(port=2, destination=cp1_b.ports[1])
    r2.connect(port=1, destination=cp2_b.ports[2])
    wg_res = OPO.add_ref(waveguide(width, length+2*Lc))
    wg_res.connect(port=1, destination=r1.ports[1])
    
    #output coupler
    cpout = OPO.add_ref(waveguide(width, Loc))
    cpout.connect(port=2, destination=wg_res.ports[2])
    cpout.move([-Lc-Loc, Cogap + width])
    r3 = OPO.add_ref(pg.arc(radius_cpout, width, theta=180, start_angle=90, layer=1))
    r3.connect(port=2, destination=cpout.ports[2])
    
    #Output waveguides
    signal = OPO.add_ref(waveguide(width, Lout_signal))
    signal.connect(port=1, destination=r3.ports[1])
    
    pump_out = OPO.add_ref(waveguide(width, Lout_pump))
    pump_out.connect(port=1, destination=wg_main.ports[2])
    
    pump_in = OPO.add_ref(waveguide(width, Lin_pump))
    pump_in.connect(port=2, destination=wg_main.ports[1])
    
    return OPO