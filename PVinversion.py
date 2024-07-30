import numpy as np
import copy
import scipy.sparse as sparse
from numba import jit
from petsc4py import PETSc


def PVinversion(q, S, H, tht, lat, lon, p,
                IITOT=49, tol=1e-3, TOTtol=1e-3, underrelax=0.5, Nsor=30,
                stbmin=0.001,vormin=0.001):
    """
    this PV inversion code solves the equations A3 and A4
    from Davis and Emanuel 1991 and can be applied to piecewise PV inversion
    from Davis 1992

    development of this python implementation:
    - original code in Fortran 70 was kindly provided by Chris Davis
    - translated into matlab and strongly modified by Franziska Teubler
    - translated into python by two HiWis

    This PV-Code was tested for data from IFS output with 1°x1° resolution on at least 17 pressure
    levels and is periodic in lon-direction

    INPUT: q is potential vorticity  (Nz,Ny,Nx)
           S is streamfunction       (Nz,Ny,Nx)
           H is Geopotential         (Nz,Ny,Nx)
         tht is potential temperature at upper and lower boundary used for Neumann boundary
                conditions           (2,Ny,Nx)
         lat is latitude with  len(lat) = Ny
         lon is longitude with len(lon) = Nx
           p is pressure (in hPa) with len(p) = Nz

    ARGUMENTS:
         IITOT      = 49    # maximal number of total iterations
         underrelax = 0.5   # underrelaxation parameter for total iterations
         tol        = 1e-3  # tolerance for single inversions (don't change!!)
         TOTtol     = 1e-3  # tolerance for total inversions
         Nsor       = 20    # number of sor-iterations for preconditioning
         stbmin     = 1e-3  # smallest value allowed to solve PDE (A4)
         vormin     = 1e-3  # smallest value allowed to solve PDE (A3)

    RECOMMENDATIONS to achieve convergence:
    I recommend not to change the tolerance levels. If no convergence is achieved,
    1. try to modify the under relaxation parameter from 0.5 to 0.2
    2. change stbmin to 1 (this helps to reduce potentially unstable regions)
    3. Another option is to vary the Nsor-parameter (between 15 and 30)
    4. If this does not help, the TOTtol can be reduced slightly, e.g. 0.005


    returns balanced fields q,H,S and flag err
        err flag for conversions
        err = 0 converged
        err = 1 diverged
        err = 2 converged to NaN

    """

    # define parameters for iterative solver (matlab parameters:)
    #omega3D = 1.85  # for 3D(H) SOR preconditioning
    #omega2D = 1.94  # for 2D(S) SOR preconditioning
    omega3D = 1.    # interestingly, here (python with petsc4py) omega = 1 most stable)
    omega2D = 1.

    print('  ')
    print('*** PV inversion *** ')
    print('*** under relaxation parameter      : ', underrelax)
    print('*** tolerance single inversions     : ', tol)
    print('*** total tolerance                 : ', TOTtol)
    print('*** omega3D (H iteration)           : ', omega3D)
    print('*** omega2D (S iteration)           : ', omega2D)
    print('*** Number of sor iterations        : ', Nsor)
    print('*** min rel vorticity value allowed : ', vormin)
    print('*** min stability value allowed     : ', stbmin)
    print('  ')


    # load and set solver for system of linear equations of single inversions
    ksp = PETSc.KSP()
    ksp.create(PETSc.COMM_WORLD)

    ksp.setTolerances(rtol=tol, atol=1e-4, max_it = 100)

    Opt = PETSc.Options()
    # define matrix-free preconditioning of fgmres
    Opt.setValue('ksp_type','fgmres')
    Opt.setValue('ksp_fgmres_modifypcksp','')
    Opt.setValue('pc_type','sor') # set omega below before solving
    Opt.setValue('pc_sor_its',Nsor) # more stable with more sor iterations
    Opt.setValue('ksp_initial_guess_nonzero','true')

    #Opt.setValue('ksp_view','')
    #Opt.setValue('ksp_monitor_true_residual','')

    # grid size
    Nz,Ny,Nx = q.shape
    dx = lon[1]-lon[0]
    dy = lat[1]-lat[0]
    dp = np.abs(p[1]-p[0])


    # -----------------------------------------------------------------------------
    # no changes below
    # -----------------------------------------------------------------------------

    # define physical constants
    g     = 9.81
    omega = 7.2921e-05
    a     = 6371220.0
    cp    = 1004.0
    Rl    = 287.05
    kappa = Rl / cp

    # -------------------------------------------------------------------------
    # remove arbitrary constant from S integration
    dS = S[-1][0][0] - H[-1][0][0]
    S = S - dS

    # q in terms of PVU and streamfunction
    q[ 0] = np.zeros((Ny, Nx))
    q[-1] = np.zeros((Ny, Nx))

    tmp  = np.tile(lat[:,None], (1, Nx))
    fcor = 2 * omega * np.sin(np.radians(tmp))
    PI   = exner(p)  # hier muss p in hPa gegeben sein!

    qmin = 0.01  # PVU
    p0   = 1e+5  # in Pascal
    ll   = a * np.radians(dx)  # m: order of hor.scale
    ff   = 1e-04  # 1/s: order of Coriolisparameter

    # ------------------------
    # Nondimensionalize
    # ------------------------
    fcor   = fcor / ff
    fcor3D = np.tile(fcor[None,:,:],(Nz,1,1))

    tho    = ff * ff * ll * ll / dp  # m ** 2 / s ** 2 / hPa? static stability?
    frc    = dp * tho / (ff * ff * ll * ll)  # = 1, warum dann?
    qconst = 1.e6 * kappa * g * cp * ff * tho / (p0 * dp)  #
    pif    = (PI / cp) ** 2.5

    # min vorticity value
    vormin = (vormin-fcor3D)/frc

    PI  = PI / dp
    dPI = np.zeros(len(PI))

    for i in range(1, len(PI) - 1):
        dPI[i] = (PI[i+1] - PI[i-1]) / 2
    dPI3D  = np.tile(dPI[:,None,None],(1,Ny,Nx))

    # boundary Theta
    tht /= tho

    # streamfunction and geopotential
    S = S / tho / dp * ff
    H = H / tho / dp

    # potential vorticity
    qmin = qmin * pif / qconst

    pif = pif.reshape(len(PI), 1, 1)
    q   = q * np.tile(pif, (1, Ny, Nx)) / qconst

    # get rid of neg. and small PV values
    q = qltqmin(q, qmin)

    # ---------------------------
    # definition for horizontal and vertical derivations
    # ---------------------------
    ac   = np.cos(np.radians(lat))
    at   = np.tan(np.radians(lat))
    ac3D = np.tile(ac[None,:,None],(Nz,1,Nx))
    # coefficients for vertical second derivation Hzz

    bh = np.zeros(len(PI))
    bb = np.zeros(len(PI))
    bl = np.zeros(len(PI))

    for i in range(1, len(PI) - 1):
        bh[i] =  2 / ((PI[i+1] - PI[i]) * (PI[i+1] - PI[i-1]))
        bb[i] = -2 / ((PI[i+1] - PI[i]) * (PI[i] - PI[i-1]))
        bl[i] =  2 / ((PI[i+1] - PI[i-1]) * (PI[i] - PI[i-1]))


    # define differential operators for later use
    Lh_xx = -A_xx(Nz,Ny,Nx,axis=2)
    Lh_yy = -A_xx(Nz,Ny,Nx,axis=1)
    Lh_zz = -A_xx(Nz,Ny,Nx,B0=(bl,bb,bh),axis=0)

    Lh = laplacian(Nz,Ny,Nx,ac,at,dy)

    Lh_xy = -A_xy(Nz,Ny,Nx,axis=(1,2))
    Lh_xz = -A_xy(Nz,Ny,Nx,axis=(0,2))
    Lh_yz = -A_xy(Nz,Ny,Nx,axis=(0,1))
    Lh_y  = -A_xy(Nz,Ny,Nx,axis=1)

    it = 1
    while it <= IITOT:

        # save for underrelaxation
        HOLD = copy.deepcopy(H)

        # calculate right - hand side of A3 (rhs)
        # 1. calcualate derivations with differential operator
        # ----------------------------------------------------
        y   = Lh * np.reshape(S, (Nz*Ny*Nx, 1), order='F')
        vor = np.reshape(y, (Nz,Ny,Nx), order='F')

        y   = Lh_zz * np.reshape(H, (Nz*Ny*Nx, 1), order='F')
        stb = np.reshape(y, (Nz,Ny,Nx), order='F')

        y   = Lh_xx * np.reshape(S, (Nz*Ny*Nx, 1), order='F')
        Sxx = np.reshape(y, (Nz,Ny,Nx), order='F')

        y   = Lh_yy * np.reshape(S, (Nz*Ny*Nx, 1), order='F')
        Syy = np.reshape(y, (Nz,Ny,Nx), order='F')

        y   = Lh_xy * np.reshape(S, (Nz*Ny*Nx, 1), order='F')
        Sxy = -np.reshape(y, (Nz,Ny,Nx), order='F')/4

        y   = Lh_xz * np.reshape(S, (Nz*Ny*Nx, 1), order='F')
        Sxz = np.reshape(y, (Nz,Ny,Nx), order='F')/4

        y   = Lh_xz * np.reshape(H, (Nz*Ny*Nx, 1), order='F')
        Hxz = np.reshape(y, (Nz,Ny,Nx), order='F')/4

        y   = Lh_yz * np.reshape(H, (Nz*Ny*Nx, 1), order='F')
        Hyz = -np.reshape(y, (Nz,Ny,Nx), order='F')/4

        y   = Lh_yz * np.reshape(S, (Nz*Ny*Nx, 1), order='F')
        Syz = -np.reshape(y, (Nz,Ny,Nx), order='F')/4

        y1  = Lh_y * np.reshape(fcor3D, (Nz*Ny*Nx, 1), order='F')
        y2  = Lh_y * np.reshape(S     , (Nz*Ny*Nx, 1), order='F')
        betaS = np.reshape(y1*y2, (Nz,Ny,Nx), order='F')/4


        # 2. Increase PV where absolute vorticity and static stability are too small to
        #    fullfill conditions on partial elliptic differential equations
        # -----------------------------------------------------------------
        Mvor   = vor <= vormin
        vordif = vor - vormin
        vor[Mvor] = vormin[Mvor]
        stb[stb <= stbmin] = stbmin # due to stability reasons
        q[Mvor]   = q[Mvor] - vordif[Mvor]*stb[Mvor]

        # now go on with modified values
        asi = fcor3D + frc*vor

        # 3. now calculate right-hand side
        # ------------------------------------------------------------------
        rhs = fcor3D*vor + 2*frc/ac3D**2 * ( Sxx * Syy - Sxy * Sxy ) \
                + betaS \
                + q \
                + frc/ac3D**2 * Hxz * np.divide(Sxz,dPI3D**2,where=dPI3D!=0) \
                + frc * Hyz * np.divide(Syz,dPI3D**2,where=dPI3D!=0)


        # add Neumann - boundary conditions to rhs for horizontal boundaries
        rhs[ 1] = rhs[ 1] - tht[ 0] / dPI[ 1] * asi[ 1]
        rhs[-2] = rhs[-2] + tht[-1] / dPI[-2] * asi[-2]
        # add Dirichlet - boundary conditions to rhs for meridional boundaries
        for i in range(len(rhs)):
            rhs[i][ 1] = rhs[i][ 1] - H[i][ 0] * (1 + at[ 1] * np.radians(dy) / 2)
            rhs[i][-2] = rhs[i][-2] - H[i][-1] * (1 - at[-2] * np.radians(dy) / 2)

        # inversion matrix with vertical Neumann boundaries,
        # Dirichlet boundaries in lat direction and
        # cyclic in lon direction
        A = A3D(Nz-2, Ny-2, Nx, ac[1:-1], at[1:-1],
                bl[1:-1], bb[1:-1], bh[1:-1], vor[1:-1, 1:-1], fcor[1:-1],dy)

        # solve for H
        # -----------------------------------------
        x0 = np.reshape(   H[1:-1, 1:-1], ((Ny-2) * Nx * (Nz-2), 1), order='F')  # first guess
        b  = np.reshape(-rhs[1:-1, 1:-1], ((Ny-2) * Nx * (Nz-2), 1), order='F')


        RES_criterium = np.linalg.norm(A.dot(x0) - b) / np.linalg.norm(b)
        print("%s < TOTtol (%s)?" % (str(RES_criterium),str(TOTtol)))
        # check for convergence
        if RES_criterium > TOTtol:
            print('No, go to iteration: ', it)
            if it == IITOT:  print('NO CONVERGENCE -> Maximal iteration number reached')
            if RES_criterium > 1:
                err = 1
                print(' ')
                print('STARTED DIVERGING')
                print('----------------------')
                break
            # here the solver is called
            Opt.setValue('pc_sor_omega',omega3D)
            y = linsolv(ksp, A, b, x0)
            H[1:-1, 1:-1] = np.reshape(y, (Nz-2, Ny-2, Nx), order='F')

        elif np.isnan(RES_criterium):
            err = 2
            print(' ')
            print('No, CONVERGENCED to NAN')
            print('----------------------')
            break
        else:
            err = 0
            print(' ')
            print('Yes: TOTAL CONVERGENCE')
            print('----------------------')
            break


        H[ 0] = H[ 1] + tht[ 0] * (PI[ 1] - PI[ 0])
        H[-1] = H[-2] - tht[-1] * (PI[-1] - PI[-2])

        # underrelax solution
        # -------------------
        H = underrelax * H + (1 - underrelax) * HOLD

        SOLD = copy.deepcopy(S)

        # calculate right - hand side of A3
        # 1. calculate derivations with differentil operators
        # --------------------------------
        y    = Lh * np.reshape(H, (Nz*Ny*Nx, 1), order='F')
        delH = np.reshape(y, (Nz,Ny,Nx), order='F')

        y   = Lh_zz * np.reshape(H, (Nz*Ny*Nx, 1), order='F')
        stb = np.reshape(y, (Nz,Ny,Nx), order='F')

        y   = Lh_xx * np.reshape(S, (Nz*Ny*Nx, 1), order='F')
        Sxx = np.reshape(y, (Nz,Ny,Nx), order='F')

        y   = Lh_yy * np.reshape(S, (Nz*Ny*Nx, 1), order='F')
        Syy = np.reshape(y, (Nz,Ny,Nx), order='F')

        y   = Lh_xy * np.reshape(S, (Nz*Ny*Nx, 1), order='F')
        Sxy = -np.reshape(y, (Nz,Ny,Nx), order='F')/4

        y   = Lh_xz * np.reshape(S, (Nz*Ny*Nx, 1), order='F')
        Sxz = np.reshape(y, (Nz,Ny,Nx), order='F')/4

        y   = Lh_xz * np.reshape(H, (Nz*Ny*Nx, 1), order='F')
        Hxz = np.reshape(y, (Nz,Ny,Nx), order='F')/4

        y   = Lh_yz * np.reshape(H, (Nz*Ny*Nx, 1), order='F')
        Hyz = -np.reshape(y, (Nz,Ny,Nx), order='F')/4

        y   = Lh_yz * np.reshape(S, (Nz*Ny*Nx, 1), order='F')
        Syz = -np.reshape(y, (Nz,Ny,Nx), order='F')/4

        y1  = Lh_y * np.reshape(fcor3D, (Nz*Ny*Nx, 1), order='F')
        y2  = Lh_y * np.reshape(S     , (Nz*Ny*Nx, 1), order='F')
        betaS = np.reshape(y1*y2, (Nz,Ny,Nx), order='F')/4


        # 2. Increase PV where stability is too small.
        # -------------------------------------------------------------
        Mstb      = stb <= stbmin
        stbdif    = stb - stbmin
        stb[Mstb] = stbmin
        Mvor      = vor <= vormin
        vor[Mvor] = vormin[Mvor]

        # stbdif negativ
        q[Mstb]   = q[Mstb] - (fcor3D[Mstb] + vor[Mstb])*stbdif[Mstb]

        rhs = 1/( fcor3D + frc*stb ) \
             * ( q - fcor3D * stb + delH - ( 2*frc/ac3D**2 * ( Sxx * Syy - Sxy * Sxy ) + betaS ) \
                 + frc/ac3D**2 * Hxz * np.divide(Sxz,dPI3D**2,where=dPI3D!=0) \
                 + frc* Hyz * np.divide(Syz,dPI3D**2,where=dPI3D!=0) )

        # add Dirichlet - boundary conditions to rhs for meridional boundaries
        for i in range(len(rhs)):
            rhs[i][ 1] = rhs[i][ 1] - S[i][ 0] * (1 + at[ 1] * np.radians(dy) / 2)
            rhs[i][-2] = rhs[i][-2] - S[i][-1] * (1 - at[-2] * np.radians(dy) / 2)

        # inversion matrix cyclonic in lon direction
        A = A3D_S(Nz-2, Ny-2, Nx, ac[1:-1], at[1:-1], dy)

        # solve for S on each level
        # -------------------------
        x0 = np.reshape(   S[1:-1, 1:-1], ((Nz-2) * (Ny-2) * Nx, 1), order='F')  # first guess
        b  = np.reshape(-rhs[1:-1, 1:-1], ((Nz-2) * (Ny-2) * Nx, 1), order='F')  # rhs

        #y  = linsolv_old(A, b, x0, omega2D, tol)
        Opt.setValue('pc_sor_omega',omega2D)
        y  = linsolv(ksp, A, b, x0)

        S[1:-1, 1:-1] = np.reshape(y, (Nz-2, Ny-2, Nx), order='F')

        S[ 0] = S[ 1] + tht[ 0] * (PI[ 1] - PI[ 0])
        S[-1] = S[-2] - tht[-1] * (PI[-1] - PI[-2])

        # underrelax solution
        # -------------------
        S = underrelax * S + (1 - underrelax) * SOLD
        it = it + 1  # total iteration counter


    # ------------------------
    # Dimensionalizeagain
    # ------------------------
    # boundaryTheta
    tht *= tho

    # streamfunction and geopotential
    S = S * tho * dp / ff
    H = H * tho * dp

    # potentialvorticity
    q = q / np.tile(pif[:,None,None], (1, Ny, Nx)) * qconst

    return S, H, q, err


def exner(p):
    # calculates Exner function, which is used
    # as vertical coordinate in pvinversion
    # input: p in hPa

    cp = 1004  # m ** 2 / (K * s ** 2)
    Rd = 287  # m ** 2 / (K * s ** 2)
    kappa = Rd / cp  #
    po = 1000  # hPa

    PI = cp * (p / po) ** kappa
    return PI

def A_xy(Nz,Ny,Nx,axis=0):
    '''
    performs one or more derivation in any direction
    number and direction of derivations given by axis
    axis = 0: Nz (number of vertical grid points)
         = 1: Ny (number of meridional grid points)
         = 2: Nx (number of zonal grid points)
    or: axis = (0,1) : derivation in z- and y- direction

    returns A (Nz*Ny*Nx x Nz*Ny*Nx)
    '''
    if isinstance(axis,int): axis = (axis,None)

    if 0 in axis:
        ez = np.full(Nz, 1)  # Unit vector
        Dz = sparse.diags([-ez[:-1], ez[1:]], [-1, 1])
    else:
        Dz = sparse.identity(Nz, format="csr") # Iz

    if 1 in axis:
        ey = np.full(Ny, 1)  # Unit vector
        Dy = sparse.diags([-ey[:-1], ey[1:]], [-1, 1])
    else:
        Dy = sparse.identity(Ny, format="csr") # Iy

    if 2 in axis:
        ex = np.full(Nx, 1)  # Unit vector
        Dx = sparse.diags([-ex[:-1], ex[1:]], [-1, 1])
        # cyclic in lon-direction
        Dx = Dx.tolil() # has to be a lil-matrix for modification
        Dx[0,Nx-1] = -1
        Dx[Nx-1,0] =  1
        Dx = Dx.tocsr()
    else:
        Dx = sparse.identity(Nx, format="csr") # Ix

    A = -sparse.kron(Dx,sparse.kron(Dy,Dz))

    return A

def A_xx(Nz,Ny,Nx,B0=None,axis=0):
    '''
    second derivative in any direction, similar to A_xy
    axis = 0: Nz with vec B0
         = 1: Ny with vec B1
         = 2: Nx with vec B2

    returns A (Nz*Ny*Nx x Nz*Ny*Nx)
    '''
    if isinstance(axis,int): axis = (axis,None)

    if 0 in axis:
        bl = B0[0]
        bb = B0[1]
        bh = B0[2]

        Dzz = sparse.diags([bl[1:], bb, bh[:-1]], [-1, 0, 1])
    else:
        Dzz = sparse.identity(Nz, format="csr") # Iz


    if 1 in axis:
        ey = np.full(Ny, 1)  # Unit vector
        Dyy = sparse.diags([ey[:-1], -2 * ey, ey[:-1]], [-1, 0, 1])
    else:
        Dyy = sparse.identity(Ny, format="csr") # Iy


    if 2 in axis:
        ex = np.full(Nx, 1)  # Unit vector
        Dxx = sparse.diags([ex[:-1], -2 * ex, ex[:-1]], [-1, 0, 1])
        # cyclic in lon-direction
        Dxx = Dxx.tolil() # has to be a lil-matrix for modification
        Dxx[0, Nx-1] = 1
        Dxx[Nx-1, 0] = 1
        Dxx = Dxx.tocsr()
    else:
        Dxx = sparse.identity(Nx, format="csr") # Iy

    A = -sparse.kron(Dxx,sparse.kron(Dyy,Dzz))

    return A

def A3D(Nz, Ny, Nx, ac, at, bl, bb, bh, vor, fcor, dy):

    # Einheitsmatrix und Einheitsvektor
    Ix = sparse.identity(Nx, format="csr")  # UnitMatrix
    ex = np.full(Nx, 1)  # Unit vector
    Iy = sparse.identity(Ny, format="csr")
    ey = np.full(Ny, 1)
    Iz = sparse.identity(Nz, format="csr")
    # ez = np.full(Nz, 1)
    # Einheitsmatrix und Einheitsvektor

    # inversion operator for each dimension
    Dxx = sparse.diags([ex[:-1], -2 * ex, ex[:-1]], [-1, 0, 1]).tolil()  # has to be a lil-matrix
    Dyy = sparse.diags([ey[:-1], -2 * ey, ey[:-1]], [-1, 0, 1])
    Dy = sparse.diags([-ey[:-1], ey[:-1]], [-1, 1])
    Dzz = sparse.diags([bl[1:], bb, bh[:-1]], [-1, 0, 1]).tolil()  # has to be a lil-matrix

    # boundary conditions:
    # --------------------
    # Neumann boundary condition in z - direction -> modify Dzz
    Dzz[0, 0] += bl[0]
    Dzz[Nz - 1, Nz - 1] += bh[-1]
    # cyclonic in lon-direction
    Dxx[0, Nx - 1] = 1
    Dxx[Nx - 1, 0] = 1

    Dxx = Dxx.tocsr()
    Dzz = Dzz.tocsr()

    # important: reshape shapes first k(z), then i(lat) and then j(lon)
    # -> in A closest to diag must be k, then i and then j and the kron
    # order must be: kron(Ix, kron(Iy, Iz))

    # A = 1 / cos(phi) ** 2 * Hxx + Hyy - tan(phi) * Hy + (sin(phi) + Sxx + Syy) * Hzz
    Axx = sparse.kron(sparse.kron(Dxx,\
                    Iy._divide_sparse(np.tile(ac.reshape(len(ac), 1) ** 2, (1, Ny)))), Iz)
    Ayy = sparse.kron(Ix, sparse.kron(Dyy, Iz))
    Ay  = sparse.kron(Ix, sparse.kron(
                         Dy.multiply(np.tile(at.reshape(len(at), 1) * np.radians(dy) / 2, (1, Ny))),
                         Iz))

    Azz = sparse.kron(Ix, sparse.kron(Iy, Dzz))
    vor = np.reshape(vor, (Nx * Ny * Nz), order='F')
    tmp = sparse.diags([vor[1:], vor, vor[:-1]], [-1, 0, 1])
    Azz = sparse.kron(Ix, sparse.kron(np.diag(fcor[:, 0]), Dzz)) + Azz.multiply(tmp)

    # negative inversion operator
    A = - (Axx + Ayy + Azz - Ay)
    return A


def A3D_S(Nz, Ny, Nx, ac, at, dy):
    # Einheitsmatrix und Einheitsvektor
    Ix = sparse.identity(Nx)  # UnitMatrix
    ex = np.full(Nx, 1)  # Unit vector
    Iy = sparse.identity(Ny)
    ey = np.full(Ny, 1)
    Iz = sparse.identity(Nz)
    ez = np.full(Nz, 0)

    Dxx = sparse.diags([ex[:-1], -2 * ex, ex[:-1]], [-1, 0, 1]).tolil()
    Dyy = sparse.diags([ey[:-1], -2 * ey, ey[:-1]], [-1, 0, 1])
    Dy = sparse.diags([-ey[:-1], ey[:-1]], [-1, 1])
    Dzz = sparse.diags([ez], [0])

    # cyclonic in lon-direction
    Dxx[0, Nx - 1] = 1
    Dxx[Nx - 1, 0] = 1

    Dxx = Dxx.tocsr()

    # important: reshape shapes first k(z), then i(lat) and then j(lon)
    # -> in A closest to diag must  be k, then i and then j and the kron
    # order must be: kron(Ix, kron(Iy, Iz))

    # A = 1 / cos(phi) ** 2 * Hxx + Hyy - tan(phi) * Hy + (sin(phi) + Sxx + Syy) * Hzz
    Axx = sparse.kron(sparse.kron(Dxx, Iy / np.tile(ac.reshape(len(ac), 1) ** 2, (1, Ny))), Iz)
    Ayy = sparse.kron(Ix, sparse.kron(Dyy, Iz))
    Ay = sparse.kron(Ix,
                     sparse.kron(Dy.multiply(np.tile(at.reshape(len(at), 1) * np.radians(dy) / 2,
                                                     (1, Ny))),
                                 Iz))

    Azz = sparse.kron(Ix, sparse.kron(Iy, Dzz))

    # negative inversion operator
    A = - (Axx + Ayy + Azz - Ay)

    return A

def laplacian(Nz, Ny, Nx, ac, at, dy):
    # Einheitsmatrix und Einheitsvektor
    Ix = sparse.identity(Nx)  # UnitMatrix
    ex = np.full(Nx, 1)  # Unit vector
    Iy = sparse.identity(Ny)
    ey = np.full(Ny, 1)
    Iz = sparse.identity(Nz)

    # inversion operator for each dimension
    # schreibt in die Hauptdiagonale das Array ex * (-2) (also überall den Wert -2) und
    # in die Nebendiagonalen das Array ex (also überall den Wert 1)
    Dxx = sparse.lil_matrix(sparse.diags([ex[:-1], -2 * ex, ex[:-1]], [-1, 0, 1]))
    # cyclic in lon-direction
    Dxx[0, Nx - 1] = 1
    Dxx[Nx - 1, 0] = 1

    Dyy = sparse.diags([ ey[:-1], -2 * ey, ey[:-1]], [-1, 0, 1])
    Dy  = sparse.diags([-ey[:-1],          ey[:-1]], [-1,    1])

    Dxx = Dxx.tocsr()

    Axx = sparse.kron(Dxx, sparse.kron(Iy / np.tile(ac[:,None] ** 2, (1, Ny)),Iz))
    Ayy = sparse.kron(Ix , sparse.kron(Dyy,Iz))
    Ay  = sparse.kron(Ix ,
                sparse.kron(Dy.multiply(np.tile(at[:,None] * np.radians(dy) / 2, (1, Ny))),Iz))

    A = Axx + Ayy - Ay

    return A


def linsolv(self,Lh, b, x0):
    '''
    used fgmres with preconditioning of SOR
    '''
    # if fgmres is used, A should be converted to csc format, since column slicing is used

    # convert to PETSc.Vec and PETSc.Mat Type
    A = PETSc.Mat().createAIJ(size = Lh.shape,
                 csr = (Lh.indptr, Lh.indices, Lh.data))
    x0 = PETSc.Vec().createWithArray(x0)
    b  = PETSc.Vec().createWithArray(b)

    # solve
    self.setOperators(A)
    self.setFromOptions()

    self.solve(b, x0)

    converged_reason = self.getConvergedReason()
    if converged_reason == 2:
        print('KSP_CONVERGED_RTOL: %s - number of iterations' % self.its)
    elif converged_reason == 3:
        print('KSP_CONVERGED_ATOL: %s - number of iterations' % self.its)
    elif converged_reason == -3:
        print('maximum number of iterations reached')
    else:
        print(converged_reason)

    y = x0.getArray()

    return y

@jit
def qltqmin(q, qmin):
    Nz,Ny,Nx = q.shape
    for i in range(1, Nz-1):
        qi = q[i]
        qdif = 0
        Nqmin = 0
        for j in range(len(qi)):
            for k in range(len(qi[0])):
                if qi[j][k] <= qmin[i]:
                    Nqmin += 1
                    qdif += qmin[i] - qi[j][k]

        qdif /= Nx * Ny - Nqmin

        for j in range(len(qi)):
            for k in range(len(qi[0])):
                if qi[j][k] <= qmin[i]:
                    qi[j][k] = qmin[i]
                else:
                    qi[j][k] -= qdif
        q[i] = qi
    return q
