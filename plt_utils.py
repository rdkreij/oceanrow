


class plt_utils:
    
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
    def plt_hindcast(hc,ptype):
        '''
        Plot montly average of hindcast data
        
        Parameters
        ----------
        hc: dictonary 
            contains all hindcast datasets 
                dsa: air
                dsw: water 
                dswh: wave height
        ptype: string
            plot type
                water velocity
                water temperature
                air velocity
                daily wave height max
                daily wave height mean
        
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

        if ptype == 'water velocity':
            ds     = hc['dsw']
            months = pd.DatetimeIndex(ds.time.values).month
            u      = ds.uw*tokts
            v      = ds.vw*tokts
            z      = np.sqrt(u**2+v**2)      

            pmin   = 0
            pmax   = 1.5
            qscale = .2
            unit   = 'kts'
        elif ptype == 'water temperature':
            ds     = hc['dsw']
            months = pd.DatetimeIndex(ds.time.values).month
            z      = ds.Tw     

            pmin   = 18
            pmax   = 32
            unit   = 'K'
        elif ptype == 'air velocity':
            ds     = hc['dsa']
            months = pd.DatetimeIndex(ds.time.values).month
            u      = ds.ua*tokts
            v      = ds.va*tokts
            z      = np.sqrt(u**2+v**2)  

            pmin   = 0
            pmax   = 15
            qscale = 2
            unit   = 'kts'
        elif ptype == 'daily wave height max':
            ds     = hc['dswh']
            months = pd.DatetimeIndex(ds.time.values).month
            z      = ds.H_day_max

            pmin   = 0
            pmax   = 4
            unit   = 'm'
        elif ptype == 'daily wave height mean':
            ds     = hc['dswh']
            months = pd.DatetimeIndex(ds.time.values).month
            z      = ds.H_day_mean

            pmin   = 0
            pmax   = 4
            unit   = 'm'   
        x = ds.lon.values
        y = ds.lat.values
        X,Y = np.meshgrid(x,y)

        for i in range(1,13):
            zi     = z.isel(time=(months==i)).mean('time').values
            fig,ax = ocean_row.plt_base_simple(xlims,ylims,10,10)
            im = ocean_row.plt_modified_imshow(x,y,zi,ax=ax,cb=True,vmin=pmin,vmax=pmax,cmap='turbo')
            if (ptype == 'water velocity') | (ptype == 'air velocity'):
                ui   = u.isel(time=(months==i)).mean('time').values
                vi   = v.isel(time=(months==i)).mean('time').values
                qidx = int(np.ceil(X.shape[1]/30))
                ax.quiver(X[::qidx,::qidx],Y[::qidx,::qidx],ui[::qidx,::qidx],vi[::qidx,::qidx],units='xy',scale_units='xy',scale=qscale)
            ax.set(xlim=xlims,ylim=ylims,title=f'{calendar.month_name[i]}, {ptype} ({unit})')
            plt.show()  
        pass
    