# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 10:37:55 2020

@author: ledezmaluism
"""

import nazca as nd

#Resonator function
def racetrack_section(name='raceTrack', length=4000, radius=200, width=1.0):
    with nd.Cell(name=name) as rt:
        nd.bend(radius=radius, width=width, angle=180).put()
        nd.strt(length=length, width=width).put()
        nd.bend(radius=radius, width=width, angle=180).put()
    return rt

#Taper function
def taper(name='taper', length=50, width=1.5, width_out=0):
#     with nd.Cell(name=name) as tap:
#         marker_cover_pts = [(0,0), (100,0), (100,100), (0,100)]
#         marker_cover = nd.Polygon(points=marker_cover_pts, layer=2)
#     return tap
    pts = [(0,-width/2), (0,width/2), (length,width_out/2), (length,-width_out/2)]
    tap = nd.Polygon(points=pts)
    return tap

#OPO function
def OPO(name='OPO', length=4000, radius=200, width=1.0, Lc=200, Cgap=1.0, 
        Loc=50, Cogap=1.5, orientation=1):
    with nd.Cell(name=name) as opo:
        
        #Racetrack resonator side-length = OPO length + Couplers
        Lres = length + 2*Lc
        
        #Place input section
        Linput_1 =  600
        Ltaper = 200
        Linput_2 = 100
        wpump = 0.8
        nd.strt(length=Linput_1, width=wpump).put(-Linput_1 - Ltaper - Linput_2)
        taper(length=Ltaper, width=wpump, width_out=width).put()
        nd.strt(length=Linput_2, width=width).put(-Linput_2)
        
        #Main waveguide
        Ltaper = 250
        #nd.strt(length=Lres+Lxtra, width=width).put(-Lxtra)
        nd.strt(length=Lres, width=width).put()
        taper(length=Ltaper,).put()
        
        #Coupler offset
        yo = orientation*(width + Cgap)
        #First coupler
        nd.strt(length=Lc, width=width).put(0,yo)
        nd.bend(radius=100, width=width, angle=90*orientation).put()
        nd.bend(radius=100, width=width, angle=-90*orientation).put()
        taper(length=Ltaper,).put()
        #Second coupler
        nd.strt(length=Lc, width=width).put(length + Lc,yo)
        
        #Output coupler forward
        Lxtra = 1200
        Lxtra = Lxtra + Lc
        xo2 = length+Lc
        yo2 = orientation*(2*width + Cgap + Cogap + 2*radius)
        nd.strt(length=Loc, width=width).put(xo2,yo2)
        if orientation ==1:
            b = nd.bend(radius=radius, width=width, angle=180).put(xo2, yo2+2*radius, 180)
            nd.strt(length=Lxtra, width=width).put(b.pin['a0'])
        else:
            b = nd.bend(radius=radius, width=width, angle=180).put(xo2, yo2, 180)
            nd.strt(length=Lxtra, width=width).put(b.pin['b0'])
        
        #Draw racetrack
        if orientation==1:
            racetrack_section(name=name+'_rt', length=Lres, width=width).put(Lres, yo, 0)
        else:
            racetrack_section(name=name+'_rt', length=Lres, width=width).put(0, yo, 180)
    return opo