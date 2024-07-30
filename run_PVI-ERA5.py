from prepare_PVI import prepare_PVI
from PVinversion import PVinversion
from basic_functions import gradm as gradient
from enstools.io.reader import read
import xarray as xr
#from xarray import open_dataset as read
import numpy as np


def plotting(uv,uvBG,uvNT,uvTD,lat,lon,title):

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from basics.plot.mapping import stereo

    llon,llat = np.meshgrid(lon,lat)

    fig=plt.figure(title,figsize=[20,12])
    ax1 = plt.subplot(221,
                      projection=ccrs.NorthPolarStereo(central_longitude=0,
                                                           true_scale_latitude=None,
                                                           globe=None))
    ax2 = plt.subplot(222,
                      projection=ccrs.NorthPolarStereo(central_longitude=0,
                                                           true_scale_latitude=None,
                                                           globe=None))
    ax3 = plt.subplot(223,
                      projection=ccrs.NorthPolarStereo(central_longitude=0,
                                                           true_scale_latitude=None,
                                                           globe=None))
    ax4 = plt.subplot(224,
                      projection=ccrs.NorthPolarStereo(central_longitude=0,
                                                               true_scale_latitude=None,
                                                               globe=None))
    stereo(ax1,(min(lat),max(lat)))
    stereo(ax2,(min(lat),max(lat)))
    stereo(ax3,(min(lat),max(lat)))
    stereo(ax4,(min(lat),max(lat)))

    # ------------------------------------------------------------------------------
    # ##############################################################################
    # ------------------------------------------------------------------------------

    CS1 = ax1.contourf(lon,lat,np.hypot(uv[0],uv[1])[p==250,...].squeeze(),
                      np.linspace(0,80,int((80/10)+1)),transform=ccrs.PlateCarree(),
                      cmap='pink_r')
    ax1.quiver(llon[::3,::3],llat[::3,::3],
               uv[1][p==250,::3,::3].squeeze(),uv[0][p==250,::3,::3].squeeze(),
               transform=ccrs.PlateCarree(),regrid_shape=50)
    fig.colorbar(CS1,ax=ax1, orientation='vertical',aspect=20,fraction=0.04)
    ax1.set_title('balanced flow (after inversion)')

    # ------------------------------------------------------------------------------
    CS2 = ax2.contourf(lon,lat,np.hypot(uvNT[0],uvNT[1])[p==250,...].squeeze(),
                      np.linspace(0,60,int((60/10)+1)),transform=ccrs.PlateCarree(),
                      cmap='pink_r')
    ax2.quiver(llon[::3,::3],llat[::3,::3],
               uvNT[1][p==250,::3,::3].squeeze(),uvNT[0][p==250,::3,::3].squeeze(),
               transform=ccrs.PlateCarree(),regrid_shape=50)
    fig.colorbar(CS2,ax=ax2, orientation='vertical',aspect=20,fraction=0.04)
    ax2.set_title('balanced flow of upper-level inversion')
    # ------------------------------------------------------------------------------
    CS3 = ax3.contourf(lon,lat,np.hypot(uvTD[0],uvTD[1])[p==250,...].squeeze(),
                      np.linspace(0,10,int((10/1)+1)),transform=ccrs.PlateCarree(),
                      cmap='pink_r')
    ax3.quiver(llon[::3,::3],llat[::3,::3],
               uvTD[1][p==250,::3,::3].squeeze(),uvTD[0][p==250,::3,::3].squeeze(),
               transform=ccrs.PlateCarree(),regrid_shape=50)
    fig.colorbar(CS3,ax=ax3, orientation='vertical',aspect=20,fraction=0.04)
    ax3.set_title('balanced flow of low-level inversion')
    # ------------------------------------------------------------------------------
    CS4 = ax4.contourf(lon,lat,np.hypot(uvBG[0],uvBG[1])[p==250,...].squeeze(),
                      np.linspace(0,60,int((60/10)+1)),transform=ccrs.PlateCarree(),
                      cmap='pink_r')
    ax4.quiver(llon[::3,::3],llat[::3,::3],
               uvBG[1][p==250,::3,::3].squeeze(),uvBG[0][p==250,::3,::3].squeeze(),
               transform=ccrs.PlateCarree(),regrid_shape=50)
    fig.colorbar(CS4,ax=ax4, orientation='vertical',aspect=20,fraction=0.04)
    ax4.set_title('balanced background flow (after inversion)')

def generateXarray(ubg,vbg,Phibg,qbg,thtbg,
                   u,v,Phi,q,tht,
                   uup,vup,Phiup,qup,thtup,
                   ulow,vlow,Philow,qlow,thtlow,
                   uTlow=None,vTlow=None,PhiTlow=None,qTlow=None,thtTlow=None,
                   uPVlow=None,vPVlow=None,PhiPVlow=None,qPVlow=None,thtPVlow=None,
                   p=None,lat=None,lon=None,day=None):
    
    result = xr.Dataset({"lon"              : ("lon", lon),
                         "lat"              : ("lat", lat),
                         "isobaricInhPa"    : ("isobaricInhPa", p),
                         "isobaricInhPa_NB" : ("isobaricInhPa_NB" ,[875,125]),
                         "U_velocity_of_BG" : (["isobaricInhPa","lat","lon"], ubg),
                         "V_velocity_of_BG" : (["isobaricInhPa","lat","lon"], vbg),
                          "Geopotential_BG" : (["isobaricInhPa","lat","lon"], Phibg),
                          "PV_BG"           : (["isobaricInhPa","lat","lon"], qbg),
                          "Temperature_BG"  : (["isobaricInhPa_NB","lat","lon"], thtbg),
                         "U_velocity_of_BAL": (["isobaricInhPa","lat","lon"], u),
                         "V_velocity_of_BAL": (["isobaricInhPa","lat","lon"], v),
                          "Geopotential_BAL": (["isobaricInhPa","lat","lon"], Phi),
                          "PV_BAL"          : (["isobaricInhPa","lat","lon"], q),
                          "Temperature_BAL" : (["isobaricInhPa_NB","lat","lon"], tht),
                         "U_velocity_of_UP" : (["isobaricInhPa","lat","lon"], uup),
                         "V_velocity_of_UP" : (["isobaricInhPa","lat","lon"], vup),
                          "Geopotential_UP" : (["isobaricInhPa","lat","lon"], Phiup),
                          "PV_UP"           : (["isobaricInhPa","lat","lon"], qup),
                          "Temperature_UP"  : (["isobaricInhPa_NB","lat","lon"], thtup),
                        "U_velocity_of_LOW" : (["isobaricInhPa","lat","lon"], ulow),
                        "V_velocity_of_LOW" : (["isobaricInhPa","lat","lon"], vlow),
                         "Geopotential_LOW" : (["isobaricInhPa","lat","lon"], Philow),
                         "PV_LOW"           : (["isobaricInhPa","lat","lon"], qlow),
                         "Temperature_LOW"  : (["isobaricInhPa_NB","lat","lon"], thtlow),
                        })
    
    if uTlow is not None:
        result = result.assign({ "U_velocity_of_TLOW" : (["isobaricInhPa","lat","lon"], uTlow),
                        "V_velocity_of_TLOW" : (["isobaricInhPa","lat","lon"], vTlow),
                         "Geopotential_TLOW" : (["isobaricInhPa","lat","lon"], PhiTlow),
                         "PV_TLOW" : (["isobaricInhPa","lat","lon"], qTlow),
                         "Temperature_TLOW"  : (["isobaricInhPa_NB","lat","lon"], thtTlow)})
    if uPVlow is not None:
        result = result.assign({ "U_velocity_of_PVLOW" : (["isobaricInhPa","lat","lon"], uPVlow),
                        "V_velocity_of_PVLOW" : (["isobaricInhPa","lat","lon"], vPVlow),
                         "Geopotential_PVLOW" : (["isobaricInhPa","lat","lon"], PhiPVlow),
                         "PV_PVLOW" : (["isobaricInhPa","lat","lon"], qPVlow),
                         "Temperature_PVLOW" : (["isobaricInhPa_NB","lat","lon"], thtPVlow)})

    # add description for the variables
    result["U_velocity_of_BG"].attrs["title"] = "Balanced U_velocity of background flow"
    result["U_velocity_of_BG"].attrs["units"] = "m s**-1"
    result["V_velocity_of_BG"].attrs["title"] = "Balanced V_velocity of background flow"
    result["V_velocity_of_BG"].attrs["units"] = "m s**-1"
    result["Geopotential_BG"].attrs["title"] = "Geopotential associated with background PV"
    result["Geopotential_BG"].attrs["units"] = "m**2 s**-2"
    result["Temperature_BG"].attrs["title"] = "Potential temperature for BGinversion"
    result["Temperature_BG"].attrs["units"] = "K"
    result["PV_BG"].attrs["title"] = "PV of background PV"
    result["PV_BG"].attrs["units"] = "PVU"

    result["U_velocity_of_BAL"].attrs["title"] = "Balanced U_velocity"
    result["U_velocity_of_BAL"].attrs["units"] = "m s**-1"
    result["V_velocity_of_BAL"].attrs["title"] = "Balanced V_velocity"
    result["V_velocity_of_BAL"].attrs["units"] = "m s**-1"
    result["Geopotential_BAL"].attrs["title"] = "Geopotential associated with full PV"
    result["Geopotential_BAL"].attrs["units"] = "m**2 s**-2"
    result["Temperature_BAL"].attrs["title"] = "Potential temperature for full inversion"
    result["Temperature_BAL"].attrs["units"] = "K"
    result["PV_BAL"].attrs["title"] = "full PV"
    result["PV_BAL"].attrs["units"] = "PVU"

    result["U_velocity_of_UP"].attrs["title"] = "Balanced U_velocity associated with \
                                                        upper-level PV anomalies"
    result["U_velocity_of_UP"].attrs["units"] = "m s**-1"
    result["V_velocity_of_UP"].attrs["title"] = "Balanced V_velocity associated \
                                                        with upper-level PV anomalies"
    result["V_velocity_of_UP"].attrs["units"] = "m s**-1"
    result["Geopotential_UP"].attrs["title"] = "Geopotential associated with upper-level PV"
    result["Geopotential_UP"].attrs["units"] = "m**2 s**-2"
    result["Temperature_UP"].attrs["title"] = "Potential temperature for UP inversion"
    result["Temperature_UP"].attrs["units"] = "K"
    result["PV_UP"].attrs["title"] = "upper-level PV"
    result["PV_UP"].attrs["units"] = "PVU"

    result["U_velocity_of_LOW"].attrs["title"] = "Balanced U_velocity associated with \
                                                    low-level PV anomalies"
    result["U_velocity_of_LOW"].attrs["units"] = "m s**-1"
    result["V_velocity_of_LOW"].attrs["title"] = "Balanced V_velocity associated with \
                                                    low-level PV anomalies"
    result["V_velocity_of_LOW"].attrs["units"] = "m s**-1"
    result["Geopotential_LOW"].attrs["title"] = "Geopotential associated with low-level PV"
    result["Geopotential_LOW"].attrs["units"] = "m**2 s**-2"
    result["Temperature_LOW"].attrs["title"] = "Potential temperature for LOWinversion"
    result["Temperature_LOW"].attrs["units"] = "K"
    result["PV_LOW"].attrs["title"] = "low-level PV"
    result["PV_LOW"].attrs["units"] = "PVU"

    # rotate the result back to the original shape and add a time coordinate
    result.coords["time"] = day
    result = result.transpose().expand_dims("time", 0)

    return result

def data_prepcrocessing(data,latlim,lonlim,dlatlon):
    import numpy as np

    data = data.sortby('isobaricInhPa',ascending=False)
    data = data.sortby('lat',ascending=False)
    data = data.sel(lat=np.linspace(latlim[1],latlim[0],
                                    int(np.diff(latlim)/dlatlon+1))).squeeze()
    # sometimes single nan values occur, which leads to no convergence of full inversion
    # -> interpolate them in lon direction
    if np.isnan(data.z.values).any():
        NNaN = np.isnan(data.z.values).sum()
        print(str(NNaN)+' NaN values replaced in Geopotential by interpolation')
        data.update({"z":data.z.interpolate_na(dim="lon")})

    return data


'''
    main file to execute piecewise PV-inversion as defined in Teubler and Riemer 2016
    1. first calls prepare_PVI, here input-variables for PV inversion are calculated based
    on wind field, temperature and geopotential
    2. PV-inversion is called
    3. wind fields calculated from streamfunction are saved in netcdf-file

    The input file for data is a standard grib-file from analysis IFS-data
    The input file for dataBG is a netcdf-file calculated from a 30-day time mean or any other background
'''

BGinversion     = True
FULLinversion   = True
UPinversion     = True
LOWinversion    = True
TLOWinversion   = True #(only low-level temperature inversion)
PVLOWinversion  = True #(only low-level PV inversion)

plot_figure   = False
save_data     = False


latlim  = [25, 80]
lonlim  = [0, 359]
dlatlon = 1

data   = read('data/E5day.grb').squeeze().isel(time=0).load()
dataBG = read('data/E5BG.nc').squeeze()
dataBG = dataBG.sel(time=data.time).load()

# ###################       no changes below        ##########################################

# reduce data to lat, -lon range of interest and sort pressure levels
data   = data_prepcrocessing(data,latlim,lonlim,dlatlon)
dataBG = data_prepcrocessing(dataBG,latlim,lonlim,dlatlon)

lon  = np.asarray(data['lon'])
lat  = np.asarray(data['lat'])
p0   = np.asarray(data['isobaricInhPa'])

if BGinversion:
    uBG = np.asarray(dataBG.u)
    vBG = np.asarray(dataBG.v)
    TBG = np.asarray(dataBG.t)
    zBG = np.asarray(dataBG.z)

    qbg, S, H, tht_bg, p = prepare_PVI(uBG,vBG,TBG,zBG,{'p':p0,'lon':lon,'lat':lat},'full')
    Psi_bg,Phi_bg   = PVinversion(qbg, S, H, tht_bg, lat, lon, p , underrelax=0.5)[:2]

    ubg, vbg = gradient(Psi_bg,lat,lon)
    ubg = -ubg

if FULLinversion:
    q, S, H, tht, p = prepare_PVI(data.u,data.v,data.t,data.z,data.coords,'full')
    Psi,Phi         = PVinversion(q, S, H, tht, lat, lon, p, underrelax=0.5)[:2]

    u, v = gradient(Psi,lat,lon)
    u = -u

if UPinversion:
    qup, S, H, tht_up, p = prepare_PVI(data.u,data.v,data.t,data.z,data.coords,'up',
                                  dataBG.u,dataBG.v,
                                  dataBG.t,dataBG.z)
    Psi_up,Phi_up     = PVinversion(qup, S, H, tht_up, lat, lon, p, underrelax=0.5)[:2]

    uup, vup = gradient(Psi_up,lat,lon)
    uup = -uup

    # calculate wind field of upper anomalies due to substraction method
    uUP = u - uup
    vUP = v - vup
    PhiUP = Phi-Phi_up
    qUP   = q-qup
    thtUP = tht-tht_up


if LOWinversion:
    qlow, S, H, tht_low, p = prepare_PVI(data.u,data.v,data.t,data.z,data.coords,'low',
                                  dataBG.u,dataBG.v,
                                  dataBG.t,dataBG.z)
    Psi_low,Phi_low    = PVinversion(qlow, S, H, tht_low, lat, lon , p, underrelax=0.5)[:2]

    ulow, vlow = gradient(Psi_low,lat,lon)
    ulow = -ulow

    # calculate wind field of lower anomalies due to substraction method
    uLOW = u - ulow
    vLOW = v - vlow
    PhiLOW = Phi-Phi_low
    qLOW   = q-qlow
    thtLOW = tht-tht_low

if TLOWinversion:
    qTlow, S, H, tht_Tlow, p = prepare_PVI(data.u,data.v,data.t,data.z,data.coords,'Tlow',
                                  dataBG.u,dataBG.v,
                                  dataBG.t,dataBG.z)
    Psi_Tlow,Phi_Tlow = PVinversion(qTlow, S, H, tht_Tlow, lat, lon , p, underrelax=0.5)[:2]

    uTlow, vTlow = gradient(Psi_Tlow,lat,lon)
    uTlow = -uTlow

    # calculate wind field of lower anomalies due to substraction method
    uTLOW = u - uTlow
    vTLOW = v - vTlow
    PhiTLOW = Phi-Phi_Tlow
    qTLOW   = q-qTlow
    thtTLOW = tht-tht_Tlow

if PVLOWinversion:
    qPVlow, S, H, tht_PVlow, p = prepare_PVI(data.u,data.v,data.t,data.z,data.coords,'PVlow',
                                  dataBG.u,dataBG.v,
                                  dataBG.t,dataBG.z)
    Psi_PVlow,Phi_PVlow = PVinversion(qPVlow, S, H, tht_PVlow, lat, lon , p, underrelax=0.5)[:2]

    uPVlow, vPVlow = gradient(Psi_PVlow,lat,lon)
    uPVlow = -uPVlow

    # calculate wind field of lower anomalies due to substraction method
    uPVLOW = u - uPVlow
    vPVLOW = v - vPVlow
    PhiPVLOW = Phi-Phi_PVlow
    qPVLOW   = q-qPVlow
    thtPVLOW = tht-tht_PVlow

data.close()
dataBG.close()

if plot_figure:
    plotting((v,u),(vbg,ubg),(vUP,uUP),(vLOW,uLOW),lat,lon,'flow of inversion')

if save_data:
    PVIXR = generateXarray(ubg,vbg,Phi_bg,qbg,tht_bg,
                           u,v,Phi,q,tht,
                           uUP,vUP,PhiUP,qUP,thtUP,
                           uLOW,vLOW,PhiLOW,qLOW,thtLOW,
                           uTlow=uTLOW,vTlow=vTLOW,PhiTlow=PhiTLOW,qTlow=qTLOW,thtTlow=thtTLOW,
                           uPVlow=uPVLOW,vPVlow=vPVLOW,PhiPVlow=PhiPVLOW,qPVlow=qPVLOW,thtPVlow=thtPVLOW,
                           p=p,lat=lat,lon=lon,day=data.time)

    PVIXR.to_netcdf('data/PVIout.nc')
