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
    ls.add_layer(name = 'Waveguides', gds_layer = 1, gds_datatype = 0,  description = 'Ridge waveguide', color = 'Crimson', alpha = 0.7)
    ls.add_layer(name = 'Metal_1', gds_layer = 10, gds_datatype = 0,  description = 'Metal before waveguide etching (poling, markers)', color = 'goldenrod', alpha = 0.7)
    ls.add_layer(name = 'Marker Mask', gds_layer = 20, gds_datatype = 0,  description = 'Marker Protection', color = 'DodgerBlue', alpha = 0.4)
    ls.add_layer(name = 'Topo Marker', gds_layer = 21, gds_datatype = 0,  description = 'Etched marker', color = 'LightSlateGray', alpha = 0.9)
    ls.add_layer(name = 'Chip boundary', gds_layer = 99, gds_datatype = 0,  description = 'Chip boundary', color = 'DarkBlue', alpha = 0.1)
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
    WG = Device('waveguide')
    WG.add_polygon( [(0, 0), (length, 0), (length, width), (0, width)] , layer=layer)
    WG.add_port(name = 'wgport1', midpoint = [0,width/2], width = width, orientation = 180)
    WG.add_port(name = 'wgport2', midpoint = [length,width/2], width = width, orientation = 0)
    return WG

def global_markers(layer_marker=10, layer_mask=20):
    D = Device()
    R = pg.rectangle(size=(20,20), layer=layer_marker)
    a = D.add_array(R, columns = 3, rows = 3,  spacing = (100, 100))
    a.move([-110, -110]) #Center of the array
    
    #Add marker cover
    cover = pg.bbox(bbox = a.bbox, layer=layer_mask)
    D << pg.offset(cover, distance=100, layer=layer_mask)
    return D

def chip(size = (13000, 18000), keepout=2000, name='chip01', text_size=500,
         text_location = 'SW', layer_text=10, layer=99):
    k = keepout
    DX = size[0]
    DY = size[1]
    OUT = pg.rectangle(size=size, layer=layer)
    IN = pg.rectangle(size=(DX-2*k, DY-2*k), layer=layer)
    IN.move((k,k))
    
    CHIP = pg.boolean(A = OUT, B = IN, operation = 'A-B', layer=layer)
    
    #Add name
    L = pg.text(text=name, size=text_size, layer=layer_text, justify='center')
    CHIP.add_ref(L).move((DX/2,k))
    
    #Add markers
    M = global_markers()
    CHIP.add_ref(M).move([k,k])
    CHIP.add_ref(M).move([DX-k,k])
    CHIP.add_ref(M).move([k,DY-k])
    CHIP.add_ref(M).move([DX-k,DY-k])
    
    return CHIP
  


def poling_region():
    pass

#OPO function
def OPO(name='OPO', length=4000, radius=200, width=1.0, Lc=200, Cgap=1.0, 
        Loc=50, Cogap=1.5, orientation=1):
    pass