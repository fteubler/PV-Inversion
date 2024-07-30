import numpy as np
from basic_functions import PVonP, streamfunction, Theta, dFdpi
from scipy.interpolate import interp1d
import xarray

def prepare_PVI(u0,v0,T0,Phi0,grid,PVI='full',u1=None,v1=None,T1=None,Phi1=None,Tinter='dFIdpi',
                fCor= 'fPlane', redist='whole domain',psep = 600):
    '''
    This function prepares all the variables in order to put them directly into PVInversion.py
    Here, streamfunction is calculated, piecewise PV inversion prepared and various parameters set

    INPUT: u0,v0,T0,Phi0  zonal velocity, meridional velocity, Temperature and Geopotential
                          all input variables expected to be: (Np x Nlat x Nlon)
                          defined either as xarray.dataarray or numpy.array
                          These fields are used for full inverion

                    grid  is either xarray.DataArraay or dictionary with coordinate
                          information of p, lat and lon

           u1,v1,T1,Phi1 same as above
                         These fields are needed, if piecewise PV inversion is calculated
                         and refer to background values

    ARGUMENTS:  PVI       = {'full','up','low'}
                Tinter    = {'dFIdpi','linear','spline','log'}
                            defines how the Temperature field ad mid-levels are interpolated
                fCor      = {'fPlane','betaPlane'}
                            assumption made for streamfunction
                redist    = {'whole domain','each layer'}
                            also for PPVI: how missing PV values are redistrbuted to conserve
                            total PV
                psep      = 600 or any pressure level between upper and lower boundary
                           defines separatino level of upper und lower PV anomalies for piecewise
                           inversion as in Teubler and Riemer 2016
    '''

    if isinstance(u0,xarray.DataArray):
        u0 = np.asarray(u0.squeeze())
        v0 = np.asarray(v0.squeeze())
        T0 = np.asarray(T0.squeeze())
        Phi0 = np.asarray(Phi0.squeeze())

        lon = np.asarray(grid['lon'])
        lat = np.asarray(grid['lat'])
        p0  = np.asarray(grid['isobaricInhPa'])
    else:
        lon = np.asarray(grid['lon'])
        lat = np.asarray(grid['lat'])
        p0  = np.asarray(grid['p'])


    if isinstance(u1,xarray.DataArray):
        u1 = np.asarray(u1.squeeze())
        v1 = np.asarray(v1.squeeze())
        T1 = np.asarray(T1.squeeze())
        Phi1 = np.asarray(Phi1.squeeze())

    if u1 is not None:
        assert u1.shape == u0.shape, 'u1 and u0 have to be same shape'


    omega = 7.2921e-05
    f = 2*omega

    # define pressure levels for inversion
    p1 = 900
    p2 = 100
    dp = 50  # vertical scale
    p  = np.linspace(p1,p2,int((p1-p2)/dp+1))
    p12 = [875, 125]  #
    psep = 600  # separation level for piecewise inversion

    print('\n-----------------------------------------')
    print('*** prepare for PV inversion      : ', PVI)
    print('*** separation level for PPVI     : ', psep)
    print('*** Method for PPVI definition    : ', 'Substraction method from DE92')
    print('*** Theta interpolation           : ', Tinter)
    print('*** PV redistribution over        : ', redist)
    print('*** assumption for streamfunction : ', fCor)
    print('-----------------------------------------\n')

    if PVI == 'full':
        Th = Theta(T0, p0)

        # u,v,T and H interpolated on more pressure levels
        v = interp1d(p0, v0, 'cubic', axis=0)
        v = v(p)
        u = interp1d(p0, u0, 'cubic', axis=0)
        u = u(p)
        T = interp1d(p0, T0, 'cubic', axis=0)
        T = T(p)
        H = interp1d(p0, Phi0, 'cubic', axis=0)
        H = H(p)

        # boundary Theta at mid-level p12 (in Tinterp different methods
        # implemented)
        tht = np.zeros((2,len(lat),len(lon)))
        tht[0] = Tinterp(Th, H, p0, p, p12[0], Tinter)
        tht[1] = Tinterp(Th, H, p0, p, p12[1], Tinter)

        q = PVonP(v, u, T, p,lat,lon) * 1e+06
        S = streamfunction(v, u, lat,lon)

    elif PVI == 'up':
        Th = Theta(T0, p0)
        Thbg = Theta(T1, p0)

        v   = interp1d(p0, v0, 'cubic', axis=0)
        v = v(p)
        u   = interp1d(p0, u0, 'cubic', axis=0)
        u = u(p)
        T   = interp1d(p0, T0, 'cubic', axis=0)
        T = T(p)
        H   = interp1d(p0, Phi0, 'cubic', axis=0)
        H = H(p)

        vbg = interp1d(p0, v1, 'cubic', axis=0)
        vbg = vbg(p)
        ubg = interp1d(p0, u1, 'cubic', axis=0)
        ubg = ubg(p)
        Tbg = interp1d(p0, T1, 'cubic', axis=0)
        Tbg = Tbg(p)
        Hbg = interp1d(p0, Phi1, 'cubic', axis=0)
        Hbg = Hbg(p)


        Thano  = Tinterp(Th, H, p0, p, p12[1], Tinter) - Tinterp(Thbg, Hbg, p0, p, p12[1], Tinter)
        tht    = np.zeros((2,len(lat),len(lon)))
        tht[0] = Tinterp(Th, H, p0, p, p12[0], Tinter)
        tht[1] = Tinterp(Thbg, Hbg, p0, p, p12[1], Tinter)

        q_ana = PVonP(v, u, T, p,lat,lon) * 1e+06
        q_bg  = PVonP(vbg, ubg, Tbg, p,lat,lon) * 1e+06

        q          = np.zeros(q_ana.shape)
        q[p>=psep] = q_ana[p>=psep] # ana below(!) separation level!
        q[p< psep] = q_bg[ p< psep] # bg  above(!) separation level
        # redistribute anomalies over whole domain to conserve PV
        q, tht[1] = PVredist(q, q_ana-q, tht[1], Thano,lat,lon, redist)

        # first guess and boundary
        S = streamfunction(vbg, ubg, lat,lon)
        H = Hbg


    elif PVI == 'low':
        Th   = Theta(T0, p0)
        Thbg = Theta(T1, p0)

        v   = interp1d(p0, v0, 'cubic', axis=0)
        v   = v(p)
        u   = interp1d(p0, u0, 'cubic', axis=0)
        u   = u(p)
        T   = interp1d(p0, T0, 'cubic', axis=0)
        T   = T(p)
        H   = interp1d(p0, Phi0, 'cubic', axis=0)
        H   = H(p)
        vbg = interp1d(p0, v1, 'cubic', axis=0)
        vbg = vbg(p)
        ubg = interp1d(p0, u1, 'cubic', axis=0)
        ubg = ubg(p)
        Tbg = interp1d(p0, T1, 'cubic', axis=0)
        Tbg = Tbg(p)
        Hbg = interp1d(p0, Phi1, 'cubic', axis=0)
        Hbg = Hbg(p)

        Thano = Tinterp(Th, H, p0, p, p12[0], Tinter) - Tinterp(Thbg, Hbg, p0, p, p12[0], Tinter)
        tht = np.zeros((2,len(lat),len(lon)))
        tht[0] = Tinterp(Thbg, Hbg, p0, p, p12[0], Tinter)
        tht[1] = Tinterp(Th, H, p0, p, p12[1], Tinter)

        q_ana = PVonP(v, u, T, p,lat,lon) * 1e+06
        q_bg = PVonP(vbg, ubg, Tbg, p,lat,lon) * 1e+06

        q          = np.zeros(q_ana.shape)
        q[p>=psep] = q_bg[p>=psep]  # ana below(!) separation level!
        q[p< psep] = q_ana[p<psep]  # bg  above(!) separation level!

        # redistribute anomalies over whole domain to conserve PV
        q, tht[0] = PVredist(q, q_ana-q, tht[0], Thano, lat,lon,redist)

        # first guess and boundary
        S = streamfunction(v, u, lat,lon)
        H = H

    elif PVI == 'PVlow':
        Th   = Theta(T0, p0)
        Thbg = Theta(T1, p0)

        v   = interp1d(p0, v0, 'cubic', axis=0)
        v   = v(p)
        u   = interp1d(p0, u0, 'cubic', axis=0)
        u   = u(p)
        T   = interp1d(p0, T0, 'cubic', axis=0)
        T   = T(p)
        H   = interp1d(p0, Phi0, 'cubic', axis=0)
        H   = H(p)
        vbg = interp1d(p0, v1, 'cubic', axis=0)
        vbg = vbg(p)
        ubg = interp1d(p0, u1, 'cubic', axis=0)
        ubg = ubg(p)
        Tbg = interp1d(p0, T1, 'cubic', axis=0)
        Tbg = Tbg(p)
        Hbg = interp1d(p0, Phi1, 'cubic', axis=0)
        Hbg = Hbg(p)

        Thano  = Tinterp(Th, H, p0, p, p12[1], Tinter) - Tinterp(Thbg, Hbg, p0, p, p12[1], Tinter)
        tht = np.zeros((2,len(lat),len(lon)))
        tht[0] = Tinterp(Th, H, p0, p, p12[0], Tinter)
        tht[1] = Tinterp(Th, H, p0, p, p12[1], Tinter)

        q_ana = PVonP(v, u, T, p,lat,lon) * 1e+06
        q_bg = PVonP(vbg, ubg, Tbg, p,lat,lon) * 1e+06

        q          = np.zeros(q_ana.shape)
        q[p>=psep] = q_bg[p>=psep]  # ana below(!) separation level!
        q[p< psep] = q_ana[p<psep]  # bg  above(!) separation level!

        # redistribute anomalies over whole domain to conserve PV
        q, tht[0] = PVredist(q, q_ana-q, tht[0], Thano, lat,lon,redist)

        # first guess and boundary
        S = streamfunction(v, u, lat,lon)
        H = H

    elif PVI == 'Tlow':
        Th   = Theta(T0, p0)
        Thbg = Theta(T1, p0)

        v   = interp1d(p0, v0, 'cubic', axis=0)
        v   = v(p)
        u   = interp1d(p0, u0, 'cubic', axis=0)
        u   = u(p)
        T   = interp1d(p0, T0, 'cubic', axis=0)
        T   = T(p)
        H   = interp1d(p0, Phi0, 'cubic', axis=0)
        H   = H(p)
        Tbg = interp1d(p0, T1, 'cubic', axis=0)
        Tbg = Tbg(p)
        Hbg = interp1d(p0, Phi1, 'cubic', axis=0)
        Hbg = Hbg(p)

        Thano = Tinterp(Th, H, p0, p, p12[0], Tinter) - Tinterp(Thbg, Hbg, p0, p, p12[0], Tinter)
        tht = np.zeros((2,len(lat),len(lon)))
        tht[0] = Tinterp(Thbg, Hbg, p0, p, p12[0], Tinter)
        tht[1] = Tinterp(Th, H, p0, p, p12[1], Tinter)

        q = PVonP(v, u, T, p,lat,lon) * 1e+06
    
        # redistribute anomalies over whole domain to conserve PV
        q, tht[0] = PVredist(q, q-q, tht[0], Thano, lat,lon,redist)

        # first guess and boundary
        S = streamfunction(v, u, lat,lon)
        H = H


    if fCor == 'betaPlane':
        phi0 = np.deg2rad(45)
        thtS = tht * 1e-04 /(f* np.sin(phi0) + f * np.cos(phi0) *
                   np.tile( np.deg2rad(lat[None,:,None])-phi0, [2,1,len(lon)]))
        tht = [tht,thtS]
    #elif fCor == 'fPlane':
        # nothing to do

    return q, S, H, tht, p

def PVredist(q, q_ano, Th, Th_ano,lat,lon,redist):
    '''
    redistribution skips lowest and highest pressure level of potential vorticity because
    of Neumann boundaries
    q      defines the PV-field for piecewise inversion (PV without anomalies -> ST method)
    q_ano  defines all PV anomalies which are removed for inversion
    Th_ano defines all temperature anomalies which are removed for inversion (one level)
    '''
    Nz,Ny,Nx = q.shape

    # redistribute substracted PV and temperature anomalies for each layer
    Area   = intm(np.ones((Ny,Nx)), lat,lon)
    SumPVa = intm(q_ano[1:-1], lat,lon)
    SumTha = intm(Th_ano.squeeze(), lat,lon)

    Th = Th + SumTha / Area
    if redist   == 'each layer':
        q[1:-1] = q[1:-1] + np.tile(SumPVa[:,None,None] / Area, [1,Ny,Nx])
    elif redist == 'whole domain':
        q[1:-1] = q[1:-1] + np.sum(SumPVa) / Area / (Nz - 2)

    return q, Th


def Tinterp(Th, FI, p_ofTh, p_ofFI, p_new, Tinter):
    assert (Tinter == 'linear' or Tinter == 'cubic' or Tinter == 'log' or Tinter == 'dFIdpi'), \
        "choose one of the following: linear,cubic,log,dFIdpi"

    # boundary theta
    if Tinter == 'linear' or Tinter == 'cubic':
        Ti = interp1d(p_ofTh,Th,kind=Tinter,axis=0)
        Ti = Ti(p_new)

    elif Tinter == 'log':
        Ti = interp1d(np.log(p_ofTh), Th, 'cubic',axis=0)
        Ti = Ti(np.log(p_new))

    elif Tinter == 'dFIdpi':
        TFI, pfineu = dFdpi(FI, p_ofFI)
        Ti = -np.squeeze(TFI[pfineu == p_new])

    return Ti

def intm(F,lat,lon):
    '''
    2D integration of F at constant vertical level (r=const)
    Fint = int F dx dy
    Fint = int F r^2 cos(phi) dphi dlambda
    INPUT:     F        :(2D or 3D matrix)
              lat, lon : latitudes and longitudes respectively (1D array)

    returns   Fint

    F.Gierth 4.04.2013
    (translated to python by two HiWis)
    '''

    a = 6371000  # Earthradius in meter

    dlon = lon[1]-lon[0]
    dlat = lat[1]-lat[0]

    dx = np.deg2rad(abs(dlon))
    dy = np.deg2rad(abs(dlat))

    N     = F.shape

    # here the repetition of lat is variable and depends on input dimension of F
    lat_position_in_N = int(np.where(np.array(N) == len(lat))[0])
    index = tuple([None]*len(N[:lat_position_in_N]) + [...] + [None]*len(N[lat_position_in_N+1:]))
    # phi has same shape as F
    phi   = np.tile(np.deg2rad(lat[index]),N[:lat_position_in_N]+(1,)+N[lat_position_in_N+1:])

    Fint = np.sum( F * a**2 * dx * dy * np.cos(phi), axis=(lat_position_in_N,lat_position_in_N+1))

    return Fint
