# import packages
import pandas                      as pd
import numpy                       as np
import cartopy.io.shapereader      as shpreader
import matplotlib                  as mpl
import matplotlib.pyplot           as plt
import xarray                      as xr
from   geographiclib.geodesic      import Geodesic    
from   shapely                     import geometry
from   shapely.ops                 import unary_union
from   descartes                   import PolygonPatch
from   scipy.optimize              import fsolve,root
from   shapely.geometry            import Point
import calendar
import cartopy.crs                 as ccrs
import cartopy.feature             as cfeature

class ocean_row:
    '''
    Class to simulate ocean rows
    '''
    def __init__(self,mcf):
        '''
        Initialize 
        
        self
        ----  
        mcf: dictionary
            start_lat: scalar
                starting location latitude [deg]
            start_lon: scalar
                starting location longitude [deg]
            start_name: scalar
                starting location name
            stop_lat: scalar
                final destination latitude [deg]
            stop_lon: scalar
                final destination longitude [deg]
            stop_name: scalar
                final destination name
            vpers: scalar
                velocity in perfect conditions [m/s]
            tod_0: datetime
                starting timestamp 
            tod_start: list of strings ('HH:MM:SS')
                list of timestamps to start rowing
            tod_stop:  list of strings ('HH:MM:SS')
                list of timestamps to stop rowing
            m: scalar
                mass
            Gw: scalar
                water friction constant
            GwA: scalar
                water friction constant sea anchor
            Ga: scalar
                air friction constant
        '''
        
        self.mcf = mcf
        
    
    def distance(self):
        '''
        Calculate distance between starting and final destination
        
        Parameters
        ----------
        self
        
        Returns
        -------
        distance: scalar
            distance between points [m]
        '''
        lat1 = self.mcf['start_lat']
        lon1 = self.mcf['start_lon']
        lat2 = self.mcf['stop_lat']
        lon2 = self.mcf['stop_lon']
        return Geodesic.WGS84.Inverse(lat1,lon1,lat2,lon2)['s12']
    
    def rowing_force(self):
        '''
        Calculate rowing force given the velocity in perfect conditions
        
        Parameters
        ----------
        self
        
        Returns
        -------
        Frow: scalar
            rowing force
        '''
        Ga    = self.mcf['Ga']
        Gw    = self.mcf['Gw']
        vperf = self.mcf['vperf']
        return (Ga+Gw)*vperf**2
    
    def rowing_activity(self):
        '''
        Construct intervals indicating rowing activity 

        Parameters
        ----------
        self

        Returns
        -------
        int0:   
            interval starting times [s of day]
        introw: 
            indicate per interval whether rowing (True) or not rowing (False) [True/False]
        intdur: 
            duration per interval [s]
        intN:   total number of intervals per day            
        k0:     interval index
        dt0:    timestep till next interval
        '''
        tod_0     = self.mcf['tod_0']
        tod_start = self.mcf['tod_start']
        tod_stop  = self.mcf['tod_stop']

        int0row   = np.array(pd.to_timedelta(tod_start).total_seconds()).astype(int) # start rowing [s]
        int0rest  = np.array(pd.to_timedelta(tod_stop).total_seconds()).astype(int) # start resting [s]
        int0      = np.append(int0row,int0rest) # interval starting times (both rowing and resting) [s]
        introw    = np.append(np.ones(int0row.shape).astype(bool),np.zeros(int0rest.shape).astype(bool)) # activity (True = rowing, False = resting)

        order     = np.argsort(int0) # sort intervals in time
        int0      = int0[order] # sort starting times [s]
        introw    = introw[order] # sort activity [bool]
        intdur    = np.diff(np.append(int0,24*3600+int0[0])) # calculate interval duration [s]
        intN      = len(int0) # total number of intervals per day

        sod_t0    = int(pd.to_timedelta(tod_0[11:19]).total_seconds()) # starting time [second of day]
        if sod_t0<int0[0]: # if in interval that started the day before and runs in the next day
            k0  = intN-1 # interval index
            dt0 = int0[0]-sod_t0 # timestep till next interval
        else:
            k0  = np.where(int0<=sod_t0)[0][-1] # interval index
            dt0 = intdur[k0]-(sod_t0-int0[k0]) # timestep till next interval
        return int0,introw,intdur,intN,k0,dt0
    
    def plt_rowing_activity(self):
        '''
        Plot rowing activity as a function of time
        
        Parameters
        ----------
        self:
            tod_0:     starting timestamp [datetime]
            tod_start: list of timestamps to start rowing
            tod_stop:  list of timestamps to stop rowing
        
        Returns
        -------
        fig: created figure
        ax: axes of created figure
        '''
        int0,introw,intdur,intN,k0,dt0 = self.rowing_activity() # row activity
        
        # plot rowing activity as function of time of day
        fig,ax = plt.subplots()
        plt.step(np.append(np.append(0,int0),3600*24)/(3600),np.append(np.append(introw[-1],introw),introw[-1]),where='post',color='black')
        ax.set_xlim([0,24])
        ax.set_ylabel('Rowing activity')
        ax.set_xlabel('$t$ (h)')
        plt.show()
        plt.close()
        return fig,ax
    
    @staticmethod
    def plt_modified_imshow(X,Y,Z,ax=None,cb=False,*args,**kwargs): # modified plt.imshow
        '''
        Modified plt.imshow that automatically includes colorbar 

        Parameters
        ----------
        X,Y:    x[Nx],y[Ny] OR meshgrid of x and y values [Nx*Ny]
        Z:      data of interest 2D matrx [Nx*Ny]
        ax:     ploting axis
        cb:     include colorbar when true
        args:   arguments for plt.imshow
        kwargs: kwargs for plt.imshow    

        Returns
        -------
        im:     modified plt.imshow     
        ''' 
        if X.ndim == 2:
            X = X[0,:]
            Y = Y[:,0]
        
        # draw modified imshow
        dx = (X[1]-X[0])/2 # spacing along x
        dy = (Y[1]-Y[0])/2 # spacing along y
        extent = [X[0]-dx, X[-1]+dx, Y[0]-dy, Y[-1]+dy] # extent imshow to correct size
        if ax is None: # if no ax is included
            ax = plt.gca() # create axis
        im = ax.imshow(np.flip(Z,axis=0), extent=extent,*args,**kwargs) # draw imshow  

        # plot colorbar
        if cb: 
            # data min and max
            zmin = np.nanmin(Z) # minimum of data 
            zmax = np.nanmax(Z) # maximum of data

            # get vmin value
            if 'vmin' in kwargs.keys(): # check if vmin is stated
                vmin = kwargs['vmin'] # get vmin value
            else: # if not stated
                vmin = zmin # set vmin from data

            # get vmax value
            if 'vmax' in kwargs.keys(): # check if vmax is stated
                vmax = kwargs['vmax'] # get vmax value
            else: # if not stated
                vmax = zmax # set vmax from data

            # determine extension of colorbar 
            if (zmin>=vmin) & (zmax<=vmax):
                extend = 'neither'
            elif (zmax>vmax) & (zmin<vmin):
                extend = 'both'
            elif zmax>vmax:
                extend = 'max'
            elif zmin<vmin:
                extend = 'min'

            # create colorbar on new axis
            plt.colorbar(im,extend=extend,fraction = 0.047*len(Y)/len(X),pad=0.04)
        return im
    
    def plt_overview(self):
        '''
        Create overview plot of rowing domain and shortest distance between starting point and destination
        
        Parameters
        ----------
        self

        Returns:
        fig: fig
            created figure
        ax: ax
            axes of created figure        
        '''
        start_lon  = self.mcf['start_lon']
        start_lat  = self.mcf['start_lat']
        start_name = self.mcf['start_name']
        stop_lon   = self.mcf['stop_lon']
        stop_lat   = self.mcf['stop_lat']
        stop_name  = self.mcf['stop_name']
        
        pdom = ocean_row.create_domain() # create allowed domain to row

        # create figure of domain, including start and ending destinations
        xlims = [30,130] # longitude limits [deg]
        ylims = [-40,0] # latitude limits [deg]
        fig,ax = ocean_row.plt_base_simple(xlims,ylims,10,10) # create figure

        [ax.add_patch(PolygonPatch(pdom.geoms[0], alpha=0.2)) for i in range(len(pdom.geoms))] # plot rowing domain

        # plot starting and destination locations 
        plt.scatter(start_lon,start_lat,color='blue') # add starting locations
        plt.scatter(stop_lon,stop_lat,color='red') # add ending locations
        plt.plot([start_lon,stop_lon],[start_lat,stop_lat],color='grey',zorder=0,transform=ccrs.Geodetic()) # plot line
        ax.set(title=f'From {start_name} to {stop_name}, {int(self.distance()/1000)} km')
        plt.show()
        return fig,ax
    
    @staticmethod
    def create_domain(): 
        '''
        Create allowed domain to row   

        Returns
        -------
        pdom: Shapely MultiPolygon 
            Allowed rowing domain

        '''    
        # define domain
        dom  = np.array([[40   ,-38  ],\
                         [35   ,-30  ],\
                         [35   ,-2   ],\
                         [95   ,-2   ],\
                         [102  ,-8   ],\
                         [113  ,-10  ],\
                         [122.5,-13.5],\
                         [122.5,-30  ],\
                         [115  ,-34  ],\
                         [110  ,-38  ]]) # coordinates domain (lon,lat) [deg]
        pdom = geometry.Polygon([[p[0], p[1]] for p in dom]) # create polygon

        # remove certain zones
        dom_excl  = np.array([[49.6 ,-15.2],\
                              [50.42,-16.1],\
                              [49.84,-17.2],\
                              [49.11,-17.19]]) # small cove Madagascar 
        pdom_excl = geometry.Polygon([[p[0], p[1]] for p in dom_excl]) # create polygon
        pdom      = pdom.difference(pdom_excl) # remove from domain

        # remove land from zones
        land_shp_fname = shpreader.natural_earth(resolution='50m',category='physical', name='land') # load in land
        land_geom      = unary_union(list(shpreader.Reader(land_shp_fname).geometries())) # get geometry 
        pdom           = pdom.difference(land_geom) # remove from domain
        return pdom
    
    @staticmethod
    def total_force(V,*arg): 
        '''
        Calculate sum of forces on rowing boat

        Parameters
        ----------
        V: vector [2]
            boat velocity
        arg: list of arguments:
            Vw:    velocity water [longitudinal,lateral] [m/s]
            Va:    velocity air [longitudinal,lateral] [m/s]
            ed:    boat orientation [longitudinal,lateral] [-]
            Frmag: rowing force [N]
            row:   True when rowing [True/False]
            anc:   Ture when anchor is dropped[True/False]
            Gw:    friction constant of boat with water
            GwA:   friction constant of anchor with water
            Ga:    friction constant of boat with air    

        Returns
        -------
        Ft: scalar
            total force [longitudinal,lateral] [N]    
        '''    
        Vw,Va,ed,Frmag,row,anc,Gw,GwA,Ga = arg # fill in arguments

        # rowing force 
        Fr = int(row)*Frmag*ed

        # water friction sea anchor    
        Fwterm = np.sqrt((Vw-V).dot(Vw-V))*(Vw-V)
        Fw     = Gw*Fwterm # water friction
        Fwa    = int(anc)*GwA*Fwterm # sea anchor water friction, anchor is in when resting (row = 0)

        # air friction 
        Fa  = Ga*np.sqrt((Va-V).dot(Va-V))*(Va-V)
        Fap = Fa.dot(ed)*ed # parallel component

        # total force
        Ft  = Fw + Fwa + Fap + Fr
        return Ft
    
    @staticmethod
    def tick_placer(ax,xlims,ylims,xtickspacing,ytickspacing): 
        '''
        Place ticks in cartopy plot

        Parameters
        ----------
        ax: ax
            axis of plot
        xlims: list [2]
            x-axis limits [deg]
        xlims: list [2]
            y-axis limits [deg]
        xtickspacing: scalar
            x tick spacing [deg]
        ytickspacing: scalar
            y tick spacing [deg]  

        Returns
        -------
        ax: ax
            axis including placed limits
        '''   
        # place x-ticks
        xticks,xticklabels = ocean_row.tick_single_ax(xlims,xtickspacing) # tick locations and labels
        ax.set_xticks(xticks, crs=ccrs.PlateCarree()) # set ticks
        ax.set_xticklabels(xticklabels) # set tick labels

        # place y-ticks
        yticks,yticklabels = ocean_row.tick_single_ax(ylims,ytickspacing) # tick locations and labels
        ax.set_yticks(yticks, crs=ccrs.PlateCarree()) # set ticks
        ax.set_yticklabels(yticklabels) # set tick labels
        return ax

    @staticmethod
    def tick_single_ax(lims,tickstep,dropticksmid = False): 
        '''
        Create ticks and ticklabels for tick_placer() for a single ax

        Parameters
        ----------
        lims: list [2]
            axis limits 
        tickstep: scalar     
            tick spacing [deg]
        dropticksmid: boolean
            drop middle ticks when True    

        Returns
        -------
        ticks: vector
            tick locations [deg]
        ticklables: vector
            ticklabels [deg]    
        '''   
        ticks       = np.arange(np.ceil(lims[0]/tickstep)\
                      *tickstep,np.floor((lims[1])/tickstep+1)\
                      *tickstep,tickstep) # range of ticks
        tickstepstr = str(tickstep) # ticks to dring
        ticksdec    = tickstepstr[::-1].find('.') # find decimal numbers
        if ticksdec < 0: # if no decimal numbers
            ticksdec = 0 # zero decimal numbers
        ticklabels  = [("%."+str(ticksdec)+"f"+"$^\circ$") % number for number in ticks] # make tick labels
        if dropticksmid: # when mid ticks are dropped
            ticklabels[1:-1] = [' '] * len(ticklabels[1:-1]) # drop mid ticks
        return ticks,ticklabels # place ticks along axis
    
    @staticmethod
    def plt_base_simple(xlims,ylims,ytickspacing,xtickspacing): 
        '''
        Create base of plot

        Parameters
        ----------
        xlims: list [2]
            x-axis limits [deg]
        xlims: list [2]
            y-axis limits [deg]
        xtickspacing: scalar
            x tick spacing [deg]
        ytickspacing: scalar
            y tick spacing [deg]

        Returns
        -------
        fig: fig
            created figure
        ax: ax
            axes of created figure
        '''

        fig = plt.figure() # create figure
        plateCr = ccrs.PlateCarree()
        plateCr._threshold = plateCr._threshold/10.  #set finer threshold
        ax = plt.axes(projection=plateCr)

        ax.set_extent([xlims[0],xlims[1],ylims[0],ylims[1]], ccrs.PlateCarree()) # set limits
        ax.add_feature(cfeature.LAND,facecolor=[0.95,0.95,0.95]) # add land
        ax.add_feature(cfeature.COASTLINE) # add coast

        ax  = ocean_row.tick_placer(ax,xlims,ylims,xtickspacing,ytickspacing) # place ticks
        return fig,ax
    
    @staticmethod
    def plt_hindcast(hc,pvar,ptype='month_mean_all'):
        '''
        Plot montly average of hindcast data
        
        Parameters
        ----------
        hc: dictonary 
            contains all hindcast datasets 
                dsa: air
                dsw: water 
                dswh: wave height
        pvar: string
            plot variable
                water velocity
                water temperature
                air velocity
                daily wave height max
                daily wave height mean
        ptype: string
            plot type
                month_mean_all: mean per month of all available data
                month_mean_annual: mean per month per year
        
        Returns
        -------
        fig: fig
            created figure
        ax: ax
            axes of created figure
        '''
        tokts = 1.94384449 # convert m/s to kts
        
        xlims = [30,130] # longitude limits [deg]
        ylims = [-40,0] # latitude limits [deg]

        if pvar == 'water velocity':
            ds     = hc['dsw']
            u      = ds.uw*tokts
            v      = ds.vw*tokts
            z      = np.sqrt(u**2+v**2)      

            pmin   = 0
            pmax   = 1.5
            qscale = .2
            unit   = 'kts'
        elif pvar == 'water temperature':
            ds     = hc['dsw']
            z      = ds.Tw     

            pmin   = 18
            pmax   = 32
            unit   = 'K'
        elif pvar == 'air velocity':
            ds     = hc['dsa']
            u      = ds.ua*tokts
            v      = ds.va*tokts
            z      = np.sqrt(u**2+v**2)  

            pmin   = 0
            pmax   = 20
            qscale = 2
            unit   = 'kts'
        elif pvar == 'daily wave height max':
            ds     = hc['dswh']
            z      = ds.H_day_max

            pmin   = 0
            pmax   = 4
            unit   = 'm'
        elif pvar == 'daily wave height mean':
            ds     = hc['dswh']
            z      = ds.H_day_mean

            pmin   = 0
            pmax   = 4
            unit   = 'm'   
            
        if ptype == 'month_mean_all':
            time      = pd.DatetimeIndex(ds.time.values).month
            time_unq  = np.sort(np.unique(time))
            timelabel = [calendar.month_name[timei] for timei in time_unq]
        elif ptype == 'month_mean_annual':
            time      = ds.time.values.astype('datetime64[M]')
            time_unq  = np.sort(np.unique(time))
            timelabel = [str(timei) for timei in time_unq]
        
        x   = ds.lon.values
        y   = ds.lat.values
        X,Y = np.meshgrid(x,y)      

        for i in range(len(time_unq)):
            zi     = z.isel(time=(time==time_unq[i])).mean('time').values
            fig,ax = ocean_row.plt_base_simple(xlims,ylims,10,10)
            im = ocean_row.plt_modified_imshow(x,y,zi,ax=ax,cb=True,vmin=pmin,vmax=pmax,cmap='turbo')
            if (pvar == 'water velocity') | (pvar == 'air velocity'):
                ui   = u.isel(time=(time==time_unq[i])).mean('time').values
                vi   = v.isel(time=(time==time_unq[i])).mean('time').values
                qidx = int(np.ceil(X.shape[1]/30))
                ax.quiver(X[::qidx,::qidx],Y[::qidx,::qidx],ui[::qidx,::qidx],vi[::qidx,::qidx],units='xy',scale_units='xy',scale=qscale)
            ax.set(xlim=xlims,ylim=ylims,title=f'{timelabel[i]}, {pvar} ({unit})')
            plt.show()  
        pass
    
    def plt_vrow_wind_only(self): 
        '''
        Plot velocity depending on head and tail wind, for a given speed in perfect conditions (no wind and no waves)

        Parameters
        ----------
        self   

        Returns
        -------
            fig: fig
                created figure
            ax: ax
                axes of created figure
        '''
        vperf = self.mcf['start_lon']
        Gw    = self.mcf['Gw']
        GwA   = self.mcf['GwA']
        Ga    = self.mcf['Ga']
        
        Frmag = self.rowing_force() # calculate total rowing force
        G     = Ga/Gw # ratio of air friction cosntant to water friction constant

        tokts    = 1.94384449 # convert m/s to kts
        Ngrid    = 1000 # number of air velocities
        vairr    = np.linspace(-30,30,Ngrid)/tokts # grid of air velocities

        # allocate
        vrowairr    = np.empty(Ngrid) # boat speed at vperf
        
        for i in range(Ngrid): # loop over all grid points 
            # calculate boat speeds
            vrowairr[i] = fsolve(ocean_row.total_force,\
                                 list((vperf+vairr[i]*0.1)*np.array([1,0])),\
                                 args=(np.array([0,0]),np.array([vairr[i],0]),np.array([1,0]),Frmag,True,False,Gw,GwA,Ga))[0] 

        fig,ax = plt.subplots(figsize=(4,6)) # create figure
        ax.axhline(y=0, color='black', linestyle='-',linewidth=.5) # add line at y=0        
        ax.plot(np.abs(vairr*tokts),vrowairr*tokts,color='blue',linewidth=2) # plot rowing speed as function of wind speed
        ax.set(xlim=[0,np.max(np.abs(vairr))*tokts],ylim=[-3,5],title='$G = {:.4f}$'.format(G),xlabel='Wind speed (kts)',ylabel='Rowing speed (kts)') # add labels
        return fig,ax
    
    @staticmethod
    def angle_between(p1,p2):
        '''
        Calculate angle between two points

        Parameters
        ----------
        p1: vector [2]
            point 1 [x,y]
        p2: vector [2] 
            point 2 [x,y]

        Returns
        -------
        angle: scalar
            anlge between points
        '''    
        ang1  = np.arctan2(*p1[::-1]) # take arctan of first point
        ang2  = np.arctan2(*p2[::-1]) # take arctan of second point
        angle = np.rad2deg((ang1 - ang2) % (2 * np.pi)) # calculate angle between points
        return angle

    @staticmethod
    def filter_circular(Xa,Xt,Np,keepsides=True,geo=False):
        '''
        Filter points close to target (bin by anlge with target)

        Parameters
        ----------
        Xa: list       
            boat coordinates [2]
        Xt: vector [2]
            target coordinate
        Np: scalar
            number of remaining locations after filtering
        keepsides: boolean
            most outer locations are kept in addition when True
        geo: boolean
            True: when applied to coordinates (in degree)
            False: when applied to locations (in m) 

        Returns
        -------
        Xaf: list
            filtered locations [2]
        idxkeep: boolean vector
            indexes of Xa that remain (not removed by the filtration)
        '''    
        NXa = Xa.shape[0] # number of points
        if NXa>Np: # apply this function if there are more points then requested segments
            Xa0 = Xa-Xt # set Xt as centre point (0,0)     
            if geo: # when applied to coordinates
                dis = np.array([Geodesic.WGS84.Inverse(x[1],x[0],Xt[1],Xt[0])['s12'] for x in Xa]) # calculate distance to target    
                phi = np.array([Geodesic.WGS84.Inverse(Xt[1],Xt[0],x[1],x[0])['azi1']/180*np.pi for x in Xa]) # calculate angle from target to boat
            else: # when applied to locations 
                dis = np.sqrt(Xa0[:,0]**2+Xa0[:,1]**2) # calculate distance to target 
                phi = np.arctan2(Xa0[:,1],Xa0[:,0])+np.pi # calculate angle from target to boat

            # find outer locations by finding the interval using the largest difference between angles 
            phis    = np.sort(phi) # sort angles
            phis2   = np.append(phis,phis+2*np.pi) # add angles 2pi displaced 
            idxmdif = np.argmax(np.diff(phis2)[:NXa]) # find idx of largest angle 

            # define interval
            if idxmdif == NXa-1: # when idx of largest angle is the last sorted location
                # interval boundaries
                phimin  = phis[0] # start of interval
                phimax  = phis[idxmdif] # end of interval

                # normalize interval by setting phimin to zero
                phinmin = 0 # normalized start of interval
                phinmax = phimax-phimin # normalized end of interval
            else: # when idx of largest angle is in between the sorted locations
                # interval boundaries
                phimin  = phis[idxmdif+1] # start of interval
                phimax  = phis[idxmdif] # end of interval

                # normalize interval by setting phimin to zero
                phinmin = 0 # normalized start of interval
                phinmax = 2*np.pi-(phimin-phimax) # normalized end of interval

            # normalize all angles
            phin                = phi-phimin # normalize all angles
            phin[phin<0]       += 2*np.pi # updated to boundaries
            phin[phin>2*np.pi] -= 2*np.pi # updated to boundaries

            Nseg  = Np # number of segments
            Nsegd = 0 # allocate number of filled segments (should equal Np)

            flag1 = True
            flag2 = True
            while (flag1 | flag2): # when there are less filled segments than requested number of locations
                binedge = np.linspace(0,phinmax,Nseg+1) # bin segments

                inds = np.digitize(phin,binedge)-1 # check if locoations in segments
                inds[inds>(Nseg-1)] = Nseg-1 # include angle at phimax in last segment

                indsunq = np.unique(inds) # find unique segments
                Nsegd   = len(indsunq) # total number of filled segments
                   
                # if there are less filled segments (which meanch less remainging locations) than requested locations
                if flag1:
                    if (Nsegd < Np): 
                        Nseg += 1 # increase number of segments
                    else:
                        flag1 = False
                        
                if flag1 == False:
                    # allocate array to indicate if location should be kept (= True) or removed (= False)
                    idxkeep  = np.zeros(NXa).astype(bool) 
                    idxside  = np.zeros(NXa).astype(bool) 
                    idxclose = np.zeros(NXa).astype(bool) 

                    for i in range(Nsegd): # loop over all segments
                        idx    = inds == indsunq[i] # get indexes of segment
                        Xa0i   = Xa0[idx,:] # coordinates of locations in segment
                        disi   = dis[idx] # distances of locations in segment
                        idxmin = np.argmin(disi) # find which location is closest to target
                        idxclose[np.where(idx)[0][idxmin]] = True # keep closest location

                    if keepsides: # when keepsides is requested
                        idxside[np.argmax(phin)] = True # keep minumum anlge
                        idxside[np.argmin(phin)] = True # keep maximum angle          

                        idxkeep = idxclose | idxside
                    else:
                        idxkeep = idxclose
                    
                  
                    if np.sum(idxkeep)>Np:
                        Nseg -= 1
                    else:
                        flag2 = False                

            Xa0b = Xa0[idxkeep] # filter locations
            Xaf  = Xa0b+Xt # transfer locations (with respect to target) back to original locations 
        else: # if there are less available points then requested segments, then no filtering is required
            Xaf = Xa # keep all locations
            idxkeep = np.ones(NXa).astype(bool) # note all indexes as kept
        return Xaf,idxkeep
    
    @staticmethod
    def linear_weight(x,xp):
        '''
        Parameters
        ----------
        x: array
            axis values
        xp: scalar
            point on axis
        
        Returns
        ib: list
            neighbour index
        wb: list
            weight neighbour
        -------
        '''
        ix  = np.argmin(np.abs(x-xp))
        xix = x[ix]

        if xix == xp:
            wb = [1]
            ib = [ix]
        else:
            if xix > xp:
                ix0 = ix-1
                ix1 = ix
            elif xix <= xp:
                ix0 = ix
                ix1 = ix+1
                
            if (ix0<0) | (ix1>=len(x)):
                ib = [np.nan]
                wb = [np.nan]
            else:
                x0 = x[ix0]
                x1 = x[ix1]
                dx  = x1-x0
                ib = [ix0,ix1]
                wb = [(x1-xp)/dx,(xp-x0)/dx]
        return ib,wb

    @staticmethod
    def get_hindcast_at_loc(X_a,datet,dsa_uv,dsw_uv,dsa_time,dsa_lon,dsa_lat,dsw_time,dsw_lon,dsw_lat):
        '''
        Get the hindcast velocities (air and water) at given boat locations using trilinear interpolation
        
        Parameters
        ----------
        X_a: matrix
            N boat locations [N,2]
        datet: datetime
            time
        dsa_uv: matrix [2,:,:,:]
            full dataset air velocities (stacked)
        dsw_uv: matrx [2,:,:,:]
            full dataset water velocities (stacked)
        dsa_time: vector
            coordinate
        dsa_lon: vector
            coordinate
        dsa_lat: vector
            coordinate
        dsw_time: vector
            coordinate
        dsw_lon: vector
            coordinate
        dsw_lat: vector
            coordinate

        Returns
        -------
        Va_a: matrix
            air velocity at N boat locations [N,2]
        Vw_a: matrix
            water velocity at N boat locations [N,2]
        '''
        N = X_a.shape[0]

        Va_a = np.empty([N,2])
        Vw_a = np.empty([N,2])

        iat,wat = ocean_row.linear_weight(dsa_time,datet)
        iwt,wwt = ocean_row.linear_weight(dsw_time,datet)

        for i in range(N): # loop over all locations
            iax,wax = ocean_row.linear_weight(dsa_lon,X_a[i,0])
            iay,way = ocean_row.linear_weight(dsa_lat,X_a[i,1])

            iaxf,iayf,iatf = [A.flatten() for A in np.meshgrid(iax,iay,iat,indexing='ij')]
            waxf,wayf,watf = [A.flatten() for A in np.meshgrid(wax,way,wat,indexing='ij')]

            if any(np.isnan(iaxf))|any(np.isnan(iayf))|any(np.isnan(iatf)):
                Va_a[i,0] = np.nan
                Va_a[i,1] = np.nan
            else:
                Va_a[i,0] = np.sum(dsa_uv[0,iatf,iayf,iaxf]*waxf*wayf*watf)
                Va_a[i,1] = np.sum(dsa_uv[1,iatf,iayf,iaxf]*waxf*wayf*watf)

            iwx,wwx = ocean_row.linear_weight(dsw_lon,X_a[i,0])
            iwy,wwy = ocean_row.linear_weight(dsw_lat,X_a[i,1])

            iwxf,iwyf,iwtf = [A.flatten() for A in np.meshgrid(iwx,iwy,iwt,indexing='ij')]
            wwxf,wwyf,wwtf = [A.flatten() for A in np.meshgrid(wwx,wwy,wwt,indexing='ij')]

            if any(np.isnan(iwxf))|any(np.isnan(iwyf))|any(np.isnan(iwtf)):
                Vw_a[i,0] = np.nan
                Vw_a[i,1] = np.nan
            else:
                Vw_a[i,0] = np.sum(dsw_uv[0,iwtf,iwyf,iwxf]*wwxf*wwyf*wwtf)
                Vw_a[i,1] = np.sum(dsw_uv[1,iwtf,iwyf,iwxf]*wwxf*wwyf*wwtf)
            

        return Va_a,Vw_a

    def run_ocean_row(self,hc,dmode,imax,rtmin):
        '''
        Run the ocean rowing model
        
        Parameters
        ----------
        self
        hc: dictonary 
            contains all hindcast datasets 
                dsa: air
                dsw: water 
                dswh: wave height
        dmode: dictionary
            mode: string
                rowing mode (destination, track, optimized)
            if mode == destination
                anchor_drop: boolean
                    True: always drop anchor when resting
                    False: never drop anchor when resting
                filters_on: boolean
                    True: remove location when on land
                    False: no filtering based on land
            if mode == track 
                Xa_pr: matrix [N_pr,2]
                    Locations of track
                tod_0: string *OPTIONAL*
                    string of datetime to start row
                    overwrites tod_0 of mcf in function
                anchor_drop: boolean
                    True: always drop anchor when resting
                    False: never drop anchor when resting   
                filters_on: boolean
                    True: remove location when on land
                    False: no filtering based on land
            if mode == optimze
                angle_max: scalar
                    maximum rowing angle w.r.t. destination
                angle_Nstep: scalar
                    number of rowing directions (should be odd)
                Np: scalar
                    number of filtered points
        imax: scalar
            maximum number of iterations
        rtmin: scalar
            threshold of distance till target [m]
                        
        Returns
        -------        
        Xs: matrix [Nend,Np,2]
            location [deg]
        Vas: matrix [Nend,Np,2]
            velocity air
        Vws: matrix [Nend,Np,2]
            velocity water
        Vs: matrix [Nend,Np,2]
            velocity boat (during previous segment)
        datets: vector [Nend]
            datetime 
        Nhists: matrix [Nend,Np]
            history of previous location
        Ns: vector [Nend]
            number of boat locations per iteration
        '''

        # values from self
        vperf      = self.mcf['vperf']
        Gw         = self.mcf['Gw']
        GwA        = self.mcf['GwA']
        Ga         = self.mcf['Ga']
        tod_start  = self.mcf['tod_start'] 
        tod_stop   = self.mcf['tod_stop']
        
        pdom = ocean_row.create_domain() # create allowed domain to row
               
        mode = dmode['mode'] # get mode
        
        if mode == 'destination':
            Np          = 1 # max number of points after filtering
            anchor_drop = dmode['anchor_drop'] # configure anchor
            filters_on  = dmode['filters_on']
        elif mode == 'track':
            Np          = 1 # max number of points after filtering
            anchor_drop = dmode['anchor_drop'] # configure anchor
            Xa_pr       = dmode['Xa_pr'] # all previous row locations [deg]
            N_pr        = Xa_pr.shape[0] # total number of locations
            filters_on  = dmode['filters_on']
        elif mode == 'optimize':
            tod_0       = self.mcf['tod_0'] # starting datetime
            phir        = np.linspace(-dmode['angle_max'],dmode['angle_max'],dmode['angle_Nstep']) # angle offsets to target
            Nphir       = len(phir) # number of angle offsets
            Np          = dmode['Np'] # max number of points after filtering
            filters_on = True
        
        if mode == 'track':
            if 'tod_0' in dmode: # if starting datetime is listed in dmode
                tod_0  = dmode['tod_0'] # starting datetime
            else: # take starting datetime from mcf
                tod_0  = self.mcf['tod_0'] # starting datetime
            start_lon  = Xa_pr[0,0] # starting longitude
            start_lat  = Xa_pr[0,1] # starting latitude
            stop_lon   = Xa_pr[N_pr-1,0] # destination longitude
            stop_lat   = Xa_pr[N_pr-1,1] # destination latitude
        else:
            tod_0      = self.mcf['tod_0'] # starting datetime
            start_lon  = self.mcf['start_lon'] # starting longitude
            start_lat  = self.mcf['start_lat'] # starting latitude
            stop_lon   = self.mcf['stop_lon'] # destination longitude
            stop_lat   = self.mcf['stop_lat'] # destination latitude
        
        # intialize model
        # calculate force from velocity in perfect conditions
        Frmag  = self.rowing_force() # calculate total rowing force
        int0,introw,intdur,intN,k,dt = self.rowing_activity() # row activity
        datet0 = np.array(np.datetime64(tod_0,'s'))

        # distances
        X0     = np.array([start_lon,start_lat]) # starting point
        Xt     = np.array([stop_lon,stop_lat]) # target point
        rt0    = Geodesic.WGS84.Inverse(X0[1],X0[0],Xt[1],Xt[0])['s12'] # distance to target      

        # intialize 
        N        = 1 # number of starting points
        No       = 1 # number of options
        X_a      = np.empty([N,2]) # allocate all locations
        X_a[0,:] = X0 # location 
        i        = 0 # iteration
        rto      = np.array([Geodesic.WGS84.Inverse(x[1],x[0],Xt[1],Xt[0])['s12'] for x in X_a]) # geodesic distance to target
        datet    = datet0 # datetime
        Nhist    = np.nan # previous boat location
        V_a      = np.array([0,0]) # boat velocity

        # datasets
        dsa_uv = np.stack([hc['dsa'].ua.values,hc['dsa'].va.values]) # hindcast air speeds
        dsw_uv = np.stack([hc['dsw'].uw.values,hc['dsw'].vw.values]) # hindcast water speeds
        
        # dataset coordinates
        dsa_time = hc['dsa'].time.values
        dsa_lon  = hc['dsa'].lon.values
        dsa_lat  = hc['dsa'].lat.values
        dsw_time = hc['dsw'].time.values
        dsw_lon  = hc['dsw'].lon.values
        dsw_lat  = hc['dsw'].lat.values

        # allocate
        Xs     = np.empty([imax,Np,2]); Xs.fill(np.nan) # locations
        Vas    = np.empty([imax,Np,2]); Vas.fill(np.nan) # air velocity
        Vws    = np.empty([imax,Np,2]); Vws.fill(np.nan) # water velocity
        Vs     = np.empty([imax,Np,2]); Vs.fill(np.nan) # boat velocity
        datets = np.empty([imax],dtype='datetime64[s]'); # time series     
        Nhists = np.zeros([imax,Np]) # history of previous boat location
        Ns     = np.zeros([imax]) # number of boat locations
                
        # create plot
        xlims = [30,130] # longitude limits [deg]
        ylims = [-40,0] # latitude limits [deg]
        fig,ax = ocean_row.plt_base_simple(xlims,ylims,10,10) # create figure
        plt.gcf().set_size_inches(13,6)
        ax.set_title(pd.to_timedelta(np.round((datet-datet0)/np.timedelta64(1, 's')),'s'))
        
        circt = np.array([[Geodesic.WGS84.Direct(Xt[1],Xt[0],angle,rtmin)[key] for key in ['lon2','lat2']] for angle in np.arange(0,360,1)])
        
        ax.plot(circt[:,0],circt[:,1],'r-')        
        ax.plot(X0[0],X0[1],'o',color='blue',markersize=7)
        ax.plot(Xt[0],Xt[1],'o',color='red' ,markersize=7)
        
        if mode == 'optimize':
            pltraj = ax.plot(X_a[:,0],X_a[:,1],'o',color='r',markersize=2)
        else:
            pltraj = ax.plot(Xs[:i+1,0,0],Xs[:i+1,0,1],'r-',linewidth=1)
        
        if dmode['mode']=='track':
            plt.plot(Xa_pr[:,0],Xa_pr[:,1],'k-',linewidth=1)

        while (i < (imax-1)) & np.all(rto > rtmin): # start loop       
            
            # get hindcast data
            Va_a,Vw_a = self.get_hindcast_at_loc(X_a,datet,dsa_uv,dsw_uv,dsa_time,dsa_lon,dsa_lat,dsw_time,dsw_lon,dsw_lat)

            if filters_on == False:
                # set hindcast velocities to zero when no data is available
                Va_a[np.isnan(Va_a)] = 0
                Vw_a[np.isnan(Vw_a)] = 0

            # store values
            Xs    [i,0:N,:] = X_a 
            Vas   [i,0:N,:] = Va_a
            Vws   [i,0:N,:] = Vw_a
            Vs    [i,0:N,:] = V_a
            Nhists[i,0:N]   = Nhist
            Ns    [i]       = N
            datets[i]       = datet
            
            i += 1  # next iteration

            # rowing activity
            rowt = introw[k] # check rowing activity options
            
            if (mode == 'destination')|(mode == 'track'):
                No   = 1 # number of options
                if rowt == True: # if rowing
                    phio = np.array([0]) # rowing direction options
                    rowo = np.array([True]) # rowing activity
                    anco = np.array([False]) # achor
                if rowt == False: # if resting    
                    phio = np.array([np.nan]) # rowing direction options
                    rowo = np.array([False]) # rowing activity
                    anco = np.array([anchor_drop]) # anchor 
            elif mode == 'optimize':
                if rowt == True: # if rowing
                    No   = Nphir+2 # can row (Nphir options) or rest(2 options: with and without anchor)
                    phio = np.append(phir,[np.nan,np.nan]) # rowing direction options
                    rowo = np.append(np.tile(True,Nphir),[False,False]) # rowing options
                    anco = np.append(np.tile(False,Nphir),[False,True]) # anchor options
                if rowt == False: # if resting
                    No   = 2 # 2 options: with and without anchor
                    phio = np.array([np.nan,np.nan]) # rowing direction options
                    rowo = np.array([False,False]) # rowing options
                    anco = np.array([False,True]) # anchor options

            # allocate all new locations and velocities per time step
            X_aold = X_a 
            X_a    = np.empty([N*No,2])
            Va_a   = np.empty([N*No,2])
            Vw_a   = np.empty([N*No,2])
            V_a    = np.empty([N*No,2])
            Nhist  = np.zeros([N*No]).astype(int)

            for ii in range(N): # loop over every starting location  
                X = np.copy(X_aold[ii,:]) # get starting location

                # air and water velocity
                Va = Vas[i-1,ii,:]
                Vw = Vws[i-1,ii,:]
                
                for jj in range(No): # loop over every activity option
                    Nhist[ii*No+jj] = ii # set history index

                    # set activity specifics
                    phi = phio[jj] 
                    row = rowo[jj]
                    anc = anco[jj]

                    # direction
                    if row: # when rowing
                        if (mode == 'destination')|(mode == 'optimize'):
                            # row in direction of destination (bearing geodesic)
                            theta  = Geodesic.WGS84.Inverse(X[1],X[0],Xt[1],Xt[0])['azi1'] 
                            theta += phi # deviate from direction
                            ed     = np.array([np.sin(theta/180*np.pi),np.cos(theta/180*np.pi)])
                        elif mode == 'track':
                            # find rowing direction based on previous row
                            # find distance between boat and previous row locations
                            rt_pr = np.array([Geodesic.WGS84.Inverse(x[1],x[0],X[1],X[0])['s12'] for x in Xa_pr]) 
                            idxpr = np.argmin(rt_pr) # find which location is closest to boat

                            if idxpr == N_pr-1: # if last location is closest
                                idxpr_aim = N_pr-1 # row towards destination
                            else: # otherwise
                                idxpr_aim = idxpr+1 # row towards next stop

                            # rowing direction
                            # find angle towards next stop
                            theta = Geodesic.WGS84.Inverse(X[1],X[0],Xa_pr[idxpr_aim,1],Xa_pr[idxpr_aim,0])['azi1'] 
                            # convert angle into rowing directions in 2D
                            ed    = (np.array([np.sin(theta/180*np.pi),np.cos(theta/180*np.pi)])).flatten() 

                    else: # when resting 
                        if Va.dot(Va)!=0: # if there is wind
                            ed    = Va/np.sqrt(Va.dot(Va)) # boat turns in direction of wind
                        else: # if there is no wind
                            if (mode == 'destination')|(mode == 'optimize'):
                                # in direction of destination (bearing geodesic)
                                theta = Geodesic.WGS84.Inverse(X[1],X[0],Xt[1],Xt[0])['azi1'] 
                                ed    = np.array([np.sin(theta/180*np.pi),np.cos(theta/180*np.pi)])
                            elif mode == 'track':
                                # find rowing direction based on previous row
                                # find distance between boat and previous row locations
                                rt_pr = np.array([Geodesic.WGS84.Inverse(x[1],x[0],X[1],X[0])['s12'] for x in Xa_pr]) 
                                idxpr = np.argmin(rt_pr) # find which location is closest to boat

                                if idxpr == N_pr-1: # if last location is closest
                                    idxpr_aim = idxpr # row towards destination
                                else: # otherwise
                                    idxpr_aim = idxpr+1 # row towards next stop

                                # rowing direction
                                # find angle towards next stop
                                theta = Geodesic.WGS84.Inverse(X[1],X[0],Xa_pr[idxpr_aim,1],Xa_pr[idxpr_aim,0])['azi1'] 
                                # convert angle into rowing directions in 2D
                                ed    = (np.array([np.sin(theta/180*np.pi),np.cos(theta/180*np.pi)])).flatten() 

                    # calculate boat velocity                   
                    V = root(ocean_row.total_force,\
                           list(ed*vperf*int(row)+Vw*0.5+Va*0.3),\
                           args=(Vw,Va,ed,Frmag,row,anc,Gw,GwA,Ga),\
                           tol=.001).x

                    # update velocity and position
                    vmag = np.sqrt(V[0]**2+V[1]**2)
                    if vmag != 0: # only update X if there is a V
                        azi1 = ocean_row.angle_between([0,1],V)
                        s12  = vmag*dt
                        Xnew = np.array(list(map(Geodesic.WGS84.Direct(X[1],X[0],azi1,s12).get, ['lon2','lat2']))) # new location
                    else:
                        Xnew = np.copy(X)

                    # get distances from start and to destiantion
                    r0 = Geodesic.WGS84.Inverse(Xnew[1],Xnew[0],X0[1],X0[0])['s12'] # distance from starting point
                    rt = Geodesic.WGS84.Inverse(Xnew[1],Xnew[0],Xt[1],Xt[0])['s12'] # distance to target point

                    if filters_on & np.logical_not(Point([Xnew[0],Xnew[1]]).intersects(pdom)) & (r0 > rtmin) & (rt > rtmin): 
                        #if on land & far from starting ro ending location
                        Xnew  = np.array([np.nan,np.nan])

                    # store values
                    X_a[ii*No+jj,:] = Xnew 
                    V_a[ii*No+jj,:] = V

            # filter new locations
            idx = np.ones(N*No).astype(bool) # allocate idx to keep
            
            # remove NaN values 
            keepidx = np.logical_not(np.any(np.isnan(X_a[idx]),axis=1))
            idx[np.where(idx)[0][~keepidx]] = False

            if mode == 'optimize':
                # drop duplicates 
                _,idxuniq = np.unique(X_a[idx],axis=0,return_index=True)
                keepidx   = np.zeros(len(X_a[idx])).astype(bool)
                keepidx[idxuniq] = True
                idx[np.where(idx)[0][~keepidx]] = False

                # select points close to target (bin by anlge with target)
                _,keepidx = ocean_row.filter_circular(X_a[idx],Xt,Np,keepsides=True,geo=True)
                idx[np.where(idx)[0][~keepidx]] = False
            
            # keep filtered values
            X_a   = X_a[idx,:]
            Nhist = Nhist[idx]
            N     = X_a.shape[0]
            V_a   = V_a[idx,:]

            if N == 0:
                pltrajpop = pltraj.pop(0)
                pltrajpop.remove()
                if mode == 'optimize':
                    pltraj = ax.plot(X_a[:,0],X_a[:,1],'o',color='r',markersize=2)
                else:
                    pltraj = ax.plot(Xs[:i+1,0,0],Xs[:i+1,0,1],'r-',linewidth=1)
                fig.canvas.draw()
                
                raise ValueError('No valid boat location(s)')
            
            # distance to target
            rto = np.array([Geodesic.WGS84.Inverse(x[1],x[0],Xt[1],Xt[0])['s12'] for x in X_a])# geodesic distance to target
                
            # update datet
            datet = datet+np.timedelta64(dt,'s')  

            # for next step
            k  = (k+1)%intN
            dt = intdur[k]  

            if (i% 20 == 0) | (i == (imax-1)) | np.any(rto <= rtmin): # plot locations
                
                print('Iteration: '+str(round(i/imax*1000)/10)+'%, Distance: '+str(round((1-np.min(rto)/rt0)*1000)/10)+'%', end='\r')
                
                pltrajpop = pltraj.pop(0)
                pltrajpop.remove()
                if mode == 'optimize':
                    pltraj = ax.plot(X_a[:,0],X_a[:,1],'o',color='r',markersize=2)
                else:
                    pltraj = ax.plot(Xs[:i+1,0,0],Xs[:i+1,0,1],'r-',linewidth=1)
                fig.canvas.draw()
                ax.set_title(pd.to_timedelta(np.round((datet-datet0)/np.timedelta64(1,'s')),'s'))
                
        if i==imax-1: 
            raise ValueError('Maximum number of iterations reached')
            
        iend = i # last iteration
        
        Va_a,Vw_a = self.get_hindcast_at_loc(X_a,datet,dsa_uv,dsw_uv,dsa_time,dsa_lon,dsa_lat,dsw_time,dsw_lon,dsw_lat)
        
        # store last iteration 
        Xs    [iend,0:N,:] = X_a
        Vas   [iend,0:N,:] = Va_a
        Vws   [iend,0:N,:] = Vw_a
        Vs    [iend,0:N,:] = V_a
        Nhists[iend,0:N]   = Nhist
        Ns    [iend]       = N
        datets[iend]       = datet
        
        # crop stored values
        Xs     = Xs    [:iend+1,:,:] 
        Vas    = Vas   [:iend+1,:,:] 
        Vws    = Vws   [:iend+1,:,:] 
        Vs     = Vs    [:iend+1,:,:] 
        Nhists = Nhists[:iend+1,:]   
        Ns     = Ns    [:iend+1]      
        datets = datets[:iend+1]           
        
        print('Iteration: '+str(round(i/imax*1000)/10)+'%, Distance: '+str(round((1-np.min(rto)/rt0)*1000)/10)+'%', end='\r')

        pltrajpop = pltraj.pop(0)
        pltrajpop.remove()
        if mode == 'optimize':
            pltraj = ax.plot(X_a[:,0],X_a[:,1],'o',color='r',markersize=2)
        else:
            pltraj = ax.plot(Xs[:i+1,0,0],Xs[:i+1,0,1],'r-',linewidth=1)
        fig.canvas.draw()
        ax.set_title(pd.to_timedelta(np.round((datet-datet0)/np.timedelta64(1,'s')),'s'))
        plt.show()        
        
        return Xs,Vas,Vws,Vs,datets,Nhists,Ns    
    
    def get_fastest_route(self,Nhists,Xs):
        '''
        Get fastest route from series of rows (from isochrone map)

        Parameters
        ----------
        self
        Nhists: matrix [Nend,Np]
            history of previous location
        Xs: matrix [Nend,Np,2]
            location [deg]    

        Returns
        -------
        Xf: matrix [Nend,2]
            locations of fastest route
        '''        
        stop_lon = self.mcf['stop_lon']
        stop_lat = self.mcf['stop_lat']

        iend = Nhists.shape[0]-1 # get last timestep
        Xt   = np.array([stop_lon,stop_lat]) # target location
        rto  = np.array([Geodesic.WGS84.Inverse(x[1],x[0],Xt[1],Xt[0])['s12'] for x in Xs[iend,:,:]])# geodesic distance to target
        Xf   = np.empty([iend+1,2]) # allocate track of fastest route
        
        iwin       = int(np.argmin(rto)) # get nearest location = end of winning track
        Xf[iend,:] = Xs[iend,iwin,:] # set last point of fastest track
        
        for i in np.flip(np.arange(0,iend)): # loop over all points in fastes track (from end till strat)
            iwin    = int(Nhists[i+1,iwin]) # get previous index
            Xf[i,:] = Xs[i,iwin,:] # get previous location
        return Xf
    
    @staticmethod
    def coords_to_boat_velocity(X1,X2,datet1,datet2):
        '''
        Calculate velocity given two coordinates and the time between them
        
        Parameters
        ----------
        X1: vector [2]
            coordinates of point 1
        X2: vector [2]
            coordinates of point 2
        datat1: datetime
            timestamp of point 1
        datat2: datetime
            timestamp of point 2
        '''
        s12  = Geodesic.WGS84.Inverse(X1[1],X1[0],X2[1],X2[0])['s12'] # distance from 1 to 2 (meters)
        azi1 = Geodesic.WGS84.Inverse(X1[1],X1[0],X2[1],X2[0])['azi1'] # azimuth of line at point 1 (degrees)
        dt   = (datet2-datet1)/np.timedelta64(1,'s') # datetime 

        V = s12/dt*np.array([np.sin(azi1/180*np.pi),np.cos(azi1/180*np.pi)]) # velocity vector
        return V
    
    @staticmethod
    def coords_to_all_velocities(Xa,datets,hc):
        '''
        Compute boat, air and water velocities from coordinates and time series
        
        Parameters
        ----------
        Xa: matrix [Nend,2]
            locations (in deg)
        datets: vector [Nend]
            datetimes 
        hc: dictonary 
            contains all hindcast datasets 
                dsa: air
                dsw: water 
                dswh: wave height
                
        Returns
        -------
        Vb: [N,2]
            boat velocities
        Va: [N,2]
            air velocities
        Vw: [N,2]
            water velocities
        '''
        N = len(datets) # number of steps

        # datasets
        dsa_uv = np.stack([hc['dsa'].ua.values,hc['dsa'].va.values])
        dsw_uv = np.stack([hc['dsw'].uw.values,hc['dsw'].vw.values])

        # dataset coordinates
        dsa_time = hc['dsa'].time.values
        dsa_lon  = hc['dsa'].lon.values
        dsa_lat  = hc['dsa'].lat.values
        dsw_time = hc['dsw'].time.values
        dsw_lon  = hc['dsw'].lon.values
        dsw_lat  = hc['dsw'].lat.values

        # allocate
        Vb = np.empty([N,2])
        Va = np.empty([N,2])
        Vw = np.empty([N,2])

        for i in range(N):
            # boat speed
            if i == N-1:
                Vb[i] = np.nan
            else:
                Vb[i] = ocean_row.coords_to_boat_velocity(Xa[i],Xa[i+1],datets[i],datets[i+1])

            # hindcast
            Va[i],Vw[i] = [x[0] for x in ocean_row.get_hindcast_at_loc(np.array([Xa[i]]),datets[i],dsa_uv,dsw_uv,dsa_time,dsa_lon,dsa_lat,dsw_time,dsw_lon,dsw_lat)]
        
        return Vb,Va,Vw

    @staticmethod
    def coords_to_polar(Xa,datets,hc,xlims=[],ylims=[],zlims=[]):
        '''
        Convert track into polar plot
        
        Parameters
        ----------
        Xa: matrix [Nend,2]
            locations (in deg)
        datets: vector [Nend]
            datetimes 
        hc: dictonary 
            contains all hindcast datasets 
                dsa: air
                dsw: water 
                dswh: wave height
        *xlims: list [2]
            x plot limits
        *ylims: list [2]
            y plot limits
        *zlims: list [2]
            z plot limits
        '''
        # get velocities of boat, air and water
        Vb,Va,Vw = ocean_row.coords_to_all_velocities(Xa,datets,hc)

        # calculate relative velocities (remove currents)
        Vb_rel = Vb-Vw
        Va_rel = Va-Vw

        # calculate anlge w.r.t. rowing direction
        theta_b_rel = np.arctan(Vb_rel[:,0]/Vb_rel[:,1])
        theta_a     = np.arctan(Va_rel[:,0]/Va_rel[:,1])
        theta_a_rel = theta_a-theta_b_rel

        # speed of boat and air
        speed_b_rel   = np.sqrt(Vb_rel[:,0]**2+Vb_rel[:,1]**2)
        speed_a_rel   = np.sqrt(Va_rel[:,0]**2+Va_rel[:,1]**2)
        speed_a_along = np.cos(theta_a_rel)*speed_a_rel # along
        speed_a_perp  = np.sin(theta_a_rel)*speed_a_rel # perpendicular 

        tokts = 1.94384449 # convert m/s to kts

        # start building elements of polar plot
        x = np.concatenate([-np.abs(speed_a_perp),np.abs(speed_a_perp)])*tokts # x-axis: wind perpendicular to boat
        y = np.tile(speed_a_along,2)*tokts # y-axis: wind along boat
        z = np.tile(speed_b_rel,2)*tokts # z-

        # set x and y limits
        if len(xlims) == 0:
            xlims = [np.nanmin(x),np.nanmax(x)]
        if len(ylims) == 0:
            ylims = [np.nanmin(y),np.nanmax(y)] 

        # create grid
        X,Y   = np.meshgrid(np.linspace(xlims[0],xlims[1],200),np.linspace(ylims[0],ylims[1],200))
        Ny,Nx = X.shape # grid size
        Z     = np.empty([Ny,Nx]) # allocate z
        Z.fill(np.nan) 
        dismin = 1 # set size of marker (allowed to overlap)

        # loop over all grid elements
        for ii in range(Nx): 
            for jj in range(Ny):
                xp = X[jj,ii] # get x value
                yp = Y[jj,ii] # get y value

                dis = np.sqrt((xp-x)**2+(yp-y)**2) # get distance to all grid points
                if np.any(dis<=dismin): # if within minimum distance
                    Z[jj,ii] = np.nanmean(z[dis<=dismin]) # average values at grid point
        
        # create plot
        fig,ax = plt.subplots(figsize=(8,8)) 

        # plot base lines
        plt.axvline(x=0,c='black',linewidth=1)
        plt.axhline(y=0,c='black',linewidth=1)

        # set z limits
        if len(zlims) == 0:
            zlims = [-np.nanmin(Z),np.nanmax(Z)] 

        # add polar plot
        ocean_row.plt_modified_imshow(X,Y,Z,cmap='jet',interpolation='bilinear',vmin=zlims[0],vmax=zlims[1])

        # add colorbar
        cax  = fig.add_axes([ax.get_position().x1+0.015,ax.get_position().y0,0.02,ax.get_position().height])
        norm = mpl.colors.Normalize(vmin=zlims[0],vmax=zlims[1])
        cb   = mpl.colorbar.ColorbarBase(cax,cmap='jet', norm=norm)

        ax.set(xlabel='relative wind perpendicular to boat (kts)',ylabel='relative wind parallel to boat (kts)',title='boat speed (kts)',xlim=xlims,ylim=ylims)
        pass