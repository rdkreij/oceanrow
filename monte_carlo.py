import numpy as np
from   scipy.optimize import root
from   geographiclib.geodesic      import Geodesic    
from mod_ocean_row import ocean_row 
from dataclasses import dataclass

@dataclass
class State:
    t: np.datetime64
    lat: float
    lon: float

@dataclass
class Action:    
    is_rowing: bool
    is_anchored: bool
    phi: float

    def __init__(self, is_rowing: bool, anchor = None, phi = None):
        self.is_rowing = is_rowing
        
        if (is_rowing):
            if (phi == None):
                raise ValueError("Anchor must be specified for rowing action")
            else:
                self.phi = phi

            self.is_anchored = False
        else:
            if (anchor == None):
                raise ValueError("Anchor must be specified for rowing action")
            else:
                self.is_anchored = anchor

            self.phi = 0


@dataclass
class boat:
    vperf: float 
    m: float
    Gw: float
    GwA: float
    Ga: float
    start_lat: float
    start_lon: float

    @property
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
        return (self.Ga + self.Gw)* self.vperf**2
    
    

class ocean:
    def __init__(self, hc, start_time, target, boat):
        self.hc = hc
        self.boat = boat
        self.start_time = start_time
        self.target = target


        self.TIMESTEP_SIZE = 10_000
        
    def get_velocities(self, time, positions):
        # datasets
        dsa = self.hc['dsa'] # hindcast air speeds
        dsw = self.hc['dsw'] # hindcast water speed
        
        # dataset coordinates
        dsa_time = dsa.time.values
        dsa_lon  = dsa.lon.values
        dsa_lat  = dsa.lat.values
        dsw_time = dsw.time.values
        dsw_lon  = dsw.lon.values
        dsw_lat  = dsw.lat.values
        
        Va_a, Vw_a = ocean_row.get_hindcast_at_loc(np.reshape(positions, (1, 2)), time, dsa, dsw, dsa_time, dsa_lon, dsa_lat, dsw_time, dsw_lon, dsw_lat)
        
        # set hindcast velocities to zero when no data is available
        Va_a[np.isnan(Va_a)] = 0
        Vw_a[np.isnan(Vw_a)] = 0
        
        return Va_a, Vw_a
    
        
    def solve_v(self, state: State, angle, row, anc):
        vw_all, va_all = self.get_velocities(state.t, [[state.lat], [state.lon]])

        Vw = vw_all[0]
        Va = va_all[0]

        guess = self.boat.vperf + Va * 0.1

        V = root(self.total_force, guess, args=(Vw, Va, self.get_ed(state.lat, state.lon, angle), row, anc), tol=.001).x
        return V
    
    
    def total_force(self, V, *args): 
        '''
        Calculate sum of forces on rowing boat

        Parameters
        ----------
        V: vector [2]
            boat velocity

        args:
            vw: water vector [2]
            va: air vector [2]
            ed:    Row direction [longitudinal,lateral] [-]
            row:   True when rowing [True/False]
            anc:   Ture when anchor is dropped[True/False]

        Returns
        -------
        Ft: scalar
            total force [longitudinal,lateral] [N]    
        '''    

        # unpack args
        Vw = args[0]
        Va = args[1]
        ed = args[2]
        row = args[3]
        anc = args[4]
        
        Frmag = self.boat.rowing_force

        # rowing force 
        Fr = int(row)*Frmag*ed

        # water friction sea anchor    


        Fwterm = np.sqrt((Vw-V).dot(Vw-V))*(Vw-V)
        Fw     = self.boat.Gw * Fwterm # water friction

        Fwa    = int(anc) * self.boat.GwA * Fwterm # sea anchor water friction, anchor is in when resting (row = 0)

        # air friction 
        Faterm = np.sqrt((Va-V).dot(Va-V))*(Va-V)
        Fa  = self.boat.Ga * Faterm
        Fap = Fa.dot(ed) * ed # parallel component

        # total force
        Ft  = Fw + Fwa + Fap + Fr
        return Ft
    
    def start(self):
        # Returns a representation of the starting state of the ocean
        lat, lon = self.boat.start_lat, self.boat.start_lon
        return (self.start_time, lat, lon)
    
    def get_ed(self, lat, lon, phi):
        # Returns the rowing direction
        # row in direction of destination (bearing geodesic)
        theta  = Geodesic.WGS84.Inverse(lat, lon, self.target[0], self.target[1])['azi1'] 
        theta += phi # deviate from direction
        ed     = np.array([np.sin(theta/180*np.pi),np.cos(theta/180*np.pi)])

        return ed

    def update_coordate_with_velocity(self, lat, lon, v, dt):
        # Returns the updated coordinates
        # update position
        vmag = np.linalg.norm(v)
       
        azi1 = ocean_row.angle_between([0,1], v)
        s12  = vmag * dt
        
        print("geodesic:",Geodesic.WGS84.Direct(lat, lon, azi1, s12))

        Xnew = np.array(list(  map(Geodesic.WGS84.Direct(lat, lon, azi1, s12).get, ['lon2','lat2'])  )) # new location
        
        return Xnew[0], Xnew[1]
    
    def next_state(self, s: State, action: Action):
        # Takes the state, and the move to be applied.
        # Returns the new state.

        phi = action.phi

        # Calculate the velocity applied to the boat
        V = self.solve_v(s, phi, action.is_rowing, action.is_anchored)    

        updated_lat, updated_lon = self.update_coordate_with_velocity(s.lat, s.lon, V, self.TIMESTEP_SIZE)

        return State(s.t + self.TIMESTEP_SIZE, updated_lat, updated_lon)


    def legal_moves(self, state_history):
        # Takes a sequence of game states representing the full
        # game history, and returns the full list of moves that
        # are legal plays for the current player.
        
        # get_rowing_activity
        pass
        
    def winner(self, state_history):
        # Takes a sequence of game states representing the full
        # game history.  If the target is reached, return 1.  
        # If the row did not reach the target, return zero.
        pass
        
class MonteCarlo:
    def __init__(self, sea, **kwargs):
        # Takes an instance of an ocean and optionally some keyword
        # arguments.  Initializes the list of game states and the
        # statistics tables.
        self.sea = sea
        self.states = []
        seconds = kwargs.get('time', 30)
        self.calculation_time = datetime.timedelta(seconds=seconds)
        self.max_moves = kwargs.get('max_moves', 100)


    def update(self, state):
        # Takes an ocean state, and appends it to the history.
        self.states.append(state)

    def get_play(self):
        # Causes the AI to calculate the best move from the
        # current state and return it.
        begin = datetime.datetime.utcnow()
        while datetime.datetime.utcnow() - begin < self.calculation_time:
            self.run_simulation()

    def run_simulation(self):
        # Plays out a "random" game from the current position,
        # then updates the statistics tables with the result.
        states_copy = self.states[:]
        state = states_copy[-1]

        for t in xrange(self.max_moves):
            legal = self.sea.legal_moves(states_copy)

            play = choice(legal)
            state = self.sea.next_state(state, play)
            states_copy.append(state)

            winner = self.board.winner(states_copy)
            if winner:
                break
        
        