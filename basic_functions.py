#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 13:56:39 2019

@author: gierth
"""
import numpy as np
from scipy.sparse import spdiags, identity, kron

def exner(p):
    '''
    calculates Exner function, which is used
    as vertical coordinate in pvinversion
    input: p in hPa
    '''

    Rd = 287.05
    cp = 1004.0
    kappa = Rd / cp  #
    po = 1000  # hPa

    PI = cp * (p / po) ** kappa
    return PI

def Theta(T,p):
    '''
    Calculates Potential Temperature.
    Pressure needs to be in hPa, T in K
    INPUT
            T can be of any dimension
              first dimension has to be pressure
    '''
    assert T.shape[0] == p.shape[0], 'first dimension has to be pressure'

    Rd = 287.05
    cp = 1004.0

    N  = T.shape

    # here the repetition of dp is variable and depends on input dimension of F
    index = tuple([...] + [None]*(len(N)-1))
    pres  = np.tile(p[index],(1,)+N[1:]) # pres has same shape as T
    assert pres.shape == T.shape, 'T and pres has to be same shape -> something wrong with tile'

    Th = T*(1000/pres)**(Rd/cp)

    return Th

def dFdp(F,p):
    """
    vertical derivation in pressure coordinates
    """
    assert F.shape[0] == len(p), 'First dimension of F must be pressure'

    N = F.shape
    dFdp = np.zeros(N)

    # dp is in general not constant -> calculate as difference
    dp   = np.zeros(len(p),)
    # interior
    dp[1:-1] = p[2:] - p[:-2]
    # edges
    dp[0]    = p[1]  - p[0]
    dp[-1]   = p[-1] - p[-2]

    dp = dp*100 # convert to pascal

    # here the repetition of dp is variable and depends on input dimension of F
    index = tuple([...] + [None]*(len(N)-1))
    dp = np.tile(dp[index],(1,)+N[1:]) # dp has same shape as F

    # edges
    dFdp[ 0,...] = (F[ 1,...] - F[ 0,...])/dp[0,...]
    dFdp[-1,...] = (F[-1,...] - F[-2,...])/dp[-1,...]

    # interior
    dFdp[1:-1,...] = (F[2:,...] - F[:-2,...])/dp[1:-1,...]

    return dFdp

def relvort(v, u, lat, lon):
    '''
    # **********************************************************
    # u(y=phi,x=lambda) zonal wind and v(y=phi,x=lambda) meridional wind
    # zeta in cartesian coordinates: v_x - u_y
    # zeta in spherical coordinates: v_lambda/a*cosphi - u_phi/a + u*tan(phi)/a
    #
    # F.Gierth
    #
    # last checked: 23.04.2012
    # last modified: 26.09.2012
    # translated to python 2018
    '''
    a = 6371220.0

    N = v.shape

    lat_position_in_N = int(np.where(np.array(N) == len(lat))[0])
    # here the repetition of lat is variable and depends on input dimension of F
    index = tuple([None]*len(N[:lat_position_in_N]) + [...] + [None]*len(N[lat_position_in_N+1:]))
    phi   = np.tile(lat[index],N[:lat_position_in_N]+(1,)+N[lat_position_in_N+1:])
    assert phi.shape == v.shape, 'phi has to be same shape as u and v'

    [dudy, _] = gradm(u, lat, lon)
    [_, dvdx] = gradm(v, lat, lon)

    zeta = dvdx - dudy + u * np.tan(np.deg2rad(phi)) / a

    return zeta

def PVonP(v,u,T,p,lat,lon):
    """
    calculates Ertels PV on pressure coordinates under assumption of hydrostatic equilibrium.

    INPUT:      u,v
                  T
        p, lat, lon
    """
    assert len(u.shape) == 3, 'all 3 dimensions needed to calculate PV'

    Nz,Ny,Nz = u.shape

    omega = 7.2921e-05
    f = 2*omega
    g = 9.80665

    Th = Theta(T,p)

    # horizontal components of relative vorticity
    dvdp = dFdp(v,p)
    dudp = dFdp(u,p)

    # gradient of Th
    [dThdy,dThdx] = gradm(Th,lat,lon)
    dThdp = dFdp(Th,p)

    N     = v.shape

    lat_position_in_N = int(np.where(np.array(N) == len(lat))[0])
    # here the repetition of lat is variable and depends on input dimension of F
    index = tuple([None]*len(N[:lat_position_in_N]) + [...] + [None]*len(N[lat_position_in_N+1:]))
    phi   = np.tile(lat[index],N[:lat_position_in_N]+(1,)+N[lat_position_in_N+1:])

    # vertical component of absolute vorticity
    absvort_p = relvort(v,u,lat,lon) + f*np.sin(np.deg2rad(phi))

    # potential vorticity equation in pressure coordinates
    PV = -g*(dudp*dThdy - dvdp*dThdx + absvort_p*dThdp)

    return PV

def streamfunction(v, u, lat, lon, tol = 1e-03):
    '''
    # calculates streamfunction psi from relative vorticity zeta
    # laplacian(psi) = - relative vorticity : Poisson equation
    # input horizontal velocity

    INPUT:    v,u  meridional and zonal wind component in ((p or Th), lat, lon)
          lat,lon
          solveEQ  solver for linear equation
              tol  tolerance for solver - the smaller the more accurate

    OUTPUT: Psi - streamfunction
    '''

    # define solver from PETSc: fgmres with sor precondition

    from petsc4py import PETSc
    # load and set solver
    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)

    ksp.setTolerances(rtol=tol, atol=1e-4, max_it = 100)

    Opt = PETSc.Options()
    # try to define matrix-free preconditioning of fgmres
    Opt.setValue('ksp_type','fgmres')
    Opt.setValue('ksp_fgmres_modifypcksp','')
    Opt.setValue('pc_type','sor')
    Opt.setValue('pc_sor_omega',1.) # one enough.. no sor needed
    Opt.setValue('pc_sor_its',5)   # 5 here fastest
    Opt.setValue('ksp_initial_guess_nonzero','true')

    a = 6371220.0
    periodic_boundaries_in_lon = False

    dlon = lon[1]-lon[0];
    dlat = lat[1]-lat[0];

    dx  = np.deg2rad(dlon)
    dy  = np.deg2rad(dlat)
    phi = np.deg2rad(lat)

    # expand third dimension in case u,v only 2D
    if np.ndim(v) == 2:
        v = v[np.newaxis,:,:]
        u = u[np.newaxis,:,:]

    Nz,Ny,Nx = v.shape
    psi      = np.zeros(v.shape)

    if (lon[-1]+dlon)%360 == lon[0]:
        periodic_boundaries_in_lon = True

    # get laplacian with 5-point star finite differences and Neumann boundaries
    Lh = getLh([Ny, Nx], lat,lon, 'NB')

    # modify zeta in order to add Neumann NBnd conditions: dPsi/dn = Vs
    # Neumann NBnd conditions have to fulfill the following condition:
    # -int_Domega dPsi/dn = int_omega zeta
    for k in range(Nz):
        if np.any(np.isnan(v[k,...])):
            print('level %s contains NaNs' % str(k))
            continue

        zeta = relvort(v[k,...],u[k,...],lat,lon)

        NBnd = np.zeros([Ny,Nx])
        if periodic_boundaries_in_lon == True:
            NBnd[ 0,:] =  2 * u[k, 0,:] / (a * dy) + u[k, 0,:] * np.tan(phi[ 0]) / a
            NBnd[-1,:] = -2 * u[k,-1,:] / (a * dy) + u[k,-1,:] * np.tan(phi[-1]) / a
        else:
            NBnd[0, :]  = 2 * u[ 0, :] / (a * dy) + u[k,0, :] * np.tan(phi[1]) / a
            NBnd[:,-1] += 2 * v[:, -1] / (a * dx * np.cos(phi))
            NBnd[-1,:] -= 2 * u[-1, :] / (a * dy) - u[-1, :] * np.tan(phi[-1]) / a
            NBnd[:, 0] -= 2 * v[ :, 0] / (a * dx * np.cos(phi))

        zeta_mod = -(zeta - NBnd)

        b  = np.reshape(zeta_mod, (Nx * Ny, 1), order='F')
        x0 = np.ones(b.shape)  # first guess

        # convert to PETSc.Vec and PETSc.Mat Type
        A = PETSc.Mat().createAIJ(size = Lh.shape,
                                            csr = (Lh.indptr, Lh.indices, Lh.data))
        x0 = PETSc.Vec().createWithArray(x0)
        b  = PETSc.Vec().createWithArray(b)

        # solve
        ksp.setOperators(A)
        ksp.setFromOptions() # load options defined above in Opt
        ksp.solve(b, x0)

        converged_reason = ksp.getConvergedReason()
        if converged_reason <= 0:
            print(converged_reason)

        y = x0.getArray()

        # at last level to solve, destroy solver environment
        if k == Nz: ksp.destroy()

        psi[k,...] = np.reshape(y, [Ny, Nx], order='F')

    return psi.squeeze()

def gradm(F,lat,lon):
    '''
    gradm calculates the gradients directly in spherical coordinates:
        -> f(y=lambda, x=phi) = 1/r*dFdphi and 1/r/cosphi*dFdlambda for constant grid spacing
    INPUT : F variable in any dimension unless lat and lon are last  (x,...,x,lat,lon)
            lat, lon 1D-array with grid or meshgrid of lat and lon

    function recognizes if input data is periodic and calculates it without longitudinal boundaries
    returns dFdlon, dFdlat
    '''
    periodic_boundaries_in_lon = False

    R = 6371220.0

    dlon = lon[1]-lon[0];
    dlat = lat[1]-lat[0];

    if (lon[-1]+dlon)%360 == lon[0]:
        periodic_boundaries_in_lon = True

    dx = np.deg2rad(dlon)
    dy = np.deg2rad(dlat)

    N  = F.shape
    dFdx = np.zeros(N)
    dFdy = np.zeros(N)

    lat_position_in_N = int(np.where(np.array(N) == len(lat))[0])
    # here the repetition of lat is variable and depends on input dimension of F
    index = tuple([None]*len(N[:lat_position_in_N]) + [...] + [None]*len(N[lat_position_in_N+1:]))
    phi   = np.tile(np.deg2rad(lat[index]),N[:lat_position_in_N]+(1,)+N[lat_position_in_N+1:])
    # phi has same shape as F

    assert phi.shape == F.shape, "something wrong with tile"

    # centered differences on interior grid points
    dFdy[...,1:-1,:] = (F[...,2:,:] - F[...,:-2,:])/(2*dy*R)
    dFdx[...,1:-1]   = (F[...,2:]   - F[...,:-2])  /(2*dx*R*np.cos(phi[...,1:-1]))

    # forward differences on northern and southern edges
    dFdy[..., 0,:]   = (F[..., 1,:] - F[..., 0,:])/(dy*R)
    dFdy[...,-1,:]   = (F[...,-1,:] - F[...,-2,:])/(dy*R)

    if periodic_boundaries_in_lon == True:
        # print('perdiodic boundaries in grad')
        # periodic differences similar to interior grid points
        dFdx[..., 0] = (F[...,1] - F[...,-1])/(2*dx*R*np.cos(phi[..., 0]))
        dFdx[...,-1] = (F[...,0] - F[...,-2])/(2*dx*R*np.cos(phi[...,-1]))
    else:
        # forward differences on left and right edges
        dFdx[..., 0] =   (F[..., 1] - F[..., 0])/(dx*R*np.cos(phi[..., 0]))
        dFdx[...,-1] =   (F[...,-1] - F[...,-2])/(dx*R*np.cos(phi[...,-1]))

    return dFdy,dFdx

def getLh(N, lat, lon, BB = 'DB'):
    """
    calculates laplacian of Poisson equation with Neumann or Dirichlet boundary
    conditions in spherical coordinates

    1,2 or 3d: uxx+uyy(+uzz)=f in spherical coordinates

                in 2D: 5-point-star
                in 3D: 7-point-star

    INPUT:
        N:      tuple or array with dimension size, for example N=[Nz Nphi Nlambda]
        phi     array with latitude degrees
        BB      boundary condition which is used for inversion purpose
                {'DB','NB' or None} Dirichlet or Neumann boundary condition
                Neumann condition will be replaced at the boundaries of f in
                function calling this routine
                if None is set the output matrix can be used to calculate derivation itself

    OUTPUT: sparse matrix with ((Nz) * Ny * Nx) x ((Nz) * Ny * Nx)

    F.Gierth
    last modified: 20.08.2012
    """
    periodic_boundaries_in_lon = False
    a = 6371220.0

    Ny = N[-2] # latitude
    Nx = N[-1] # longitude

    assert Ny == len(lat) and Nx == len(lon), "N not in correct shape: (Nz) x lat x lon"

    dlat = lat[1]-lat[0]
    dlon = lon[1]-lon[0]

    if (lon[-1] + dlon)%360 == lon[0]:
        periodic_boundaries_in_lon = True
        #print('perdiodic boundaries in getLh')

    dy = np.deg2rad(dlat)*a
    dx = np.deg2rad(dlon)*a

    if len(N) == 3:
        Nz = N[0]

    if len(N) >= 2:
        Nyphi = np.tile(lat[:,None],[1,Ny])
        assert Nyphi.shape == (Ny,Ny), "tile not working"

        e1 = np.ones(Nx)  # long
        e2 = np.ones(Ny)  # lat

        Ix = identity(Nx)
        Iy = identity(Ny)

        Dxx = spdiags([ e1, -2 * e1, e1], [-1, 0, 1], Nx, Nx).tolil()
        Dyy = spdiags([ e2, -2 * e2, e2], [-1, 0, 1], Ny, Ny).tolil()
        Dy  = spdiags([-e2,          e2], [-1,    1], Ny, Ny).tolil()

        if periodic_boundaries_in_lon:
            # matlab Dxx = Dxx + sparse([1 Nx],[Nx 1],[1 1],Nx,Nx);
            Dxx[ 0,-1] += 1
            Dxx[-1, 0] += 1

        if BB == 'NB':
            # matlab Dy = Dy - sparse([1 Ny], [2 Ny - 1], [1 -1], Ny, Ny);
            Dy[ 0, 1] -= 1
            Dy[-1,-2] += 1

            # matlab Dyy = Dyy + sparse([1, Ny], [2, Ny - 1], [1, 1], Ny, Ny)
            Dyy[ 0, 1] += 1
            Dyy[-1,-2] += 1

            if not periodic_boundaries_in_lon:
                print('no periodicity -> NB at boundary')
                Dxx[ 0, 1] = -2
                Dxx[-1,-2] = -2

        Dxx = Dxx.tocsr()
        Dyy = Dyy.tocsr()
        Dy  = Dy.tocsr()

        # now make sure spherical parameters are set
        Dxx = Dxx / dx ** 2
        Dyy = Dyy / dy ** 2

        # have to use multiply!!! * with sparse 2d arrays refers to matrix multiplication!!!
        Dy = Dy / 2 / dy / a
        Dy = Dy.multiply(np.tan(np.radians(Nyphi)))

        Axx = kron(Dxx, Iy/np.cos(np.deg2rad(Nyphi))**2)
        Ayy = kron(Ix, Dyy) - kron(Ix,Dy)

        A = -Axx - Ayy

    if len(N) == 3:  # for dim=3D

        # define in third dimension
        e3 = np.ones(Nz, 1)

        # define Lh for third dimension: has to be changed for each function
        Dzz = spdiags([-e3, 2 * e3, -e3], [-1, 0, 1], Nz, Nz)

        # define unit matrix
        Iz = identity(Nz)

        A = kron(Iz, A) + kron(kron(Dzz, Iy), Ix)

    return A

def dFdpi(F, p0):
    '''
    calculates derivative dphi/dexner
    -Th = dphi/dexner = - dphi/dp *p/exner/kappa
    INPUT
        F has to be X Dimesional with first dimension Pressure

    Output: -Th!
    '''
    p = np.copy(p0)

    N = F.shape

    Rd = 287.05
    cp = 1004.0
    kappa = Rd / cp

    Th = np.zeros(F.shape)
    dp = np.zeros(p.shape)

    dp[1:-1] = p[2:] - p[:-2]
    dp[0]    = p[1]  - p[0]
    dp[-1]   = p[-1] - p[-2]

    # Th will be defined at mid levels at edges
    p[-1] = np.mean(p[-2:])
    p[0]  = np.mean(p[:2])

    # at the edges
    Th[ 0, ...] = (F[ 1,...] - F[ 0,...]) / dp[ 0] * p[ 0] / kappa / exner(p[ 0])
    Th[-1, ...] = (F[-1,...] - F[-2,...]) / dp[-1] * p[-1] / kappa / exner(p[-1])

    # dp and p in 3D
    # here the repetition of dp is variable and depends on input dimension of F
    index = tuple([...] + [None]*(len(N)-1))
    dpnD = np.tile(dp[index],(1,)+N[1:]) # dp has same shape as F
    pnD  = np.tile( p[index],(1,)+N[1:]) # dp has same shape as F

    # interior grid points
    Th[1:-1] = (F[2:] - F[:-2])/ dpnD[1:-1] *pnD[1:-1] / kappa / exner(pnD[1:-1])

    return Th, p