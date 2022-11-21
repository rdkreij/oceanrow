import numpy as np

class boat:
    def __init__(self, vperf, m, Gw, GwA, Ga, start_lat, start_lon):
        '''
        vperf: scalar
            velocity in perfect conditions [m/s]
        m: scalar
            mass
        Gw: scalar
            water friction constant
        GwA: scalar
            water friction constant sea anchor
        Ga: scalar
            air friction constant
        start_lat: scalar
            Starting position of the boat (latitude)
        start_lon: scalar
            Starting position of the boat (longitude)
        '''
        self.vperf = vperf
        self.m = m
        self.Gw = Gw
        self.GwA = GwA
        self.Ga = Ga
        self.lat = start_lat
        self.lon = start_lon
        
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
        
        Va_a, Vw_a = ocean_row.get_hindcast_at_loc(positions, time, dsa, dsw, dsa_time, dsa_lon, dsa_lat, dsw_time, dsw_lon, dsw_lat)
        
        # set hindcast velocities to zero when no data is available
        Va_a[np.isnan(Va_a)] = 0
        Vw_a[np.isnan(Vw_a)] = 0
        
        return Va_a, Vw_a
    
    def friction_coeff(self, time, V):
        '''
        Calculates the friction coefficient Fwterm at a particular time
        
        Parameters
        ----------
        V: vector [2]
            velocity vector (e.g. of a boat)
        '''
        Va_a, Vw_a = get_velocities(time, [[self.boat.lat], [self.boat.lon]])
        
        
        
    
    def total_force(self, time, Frmag, ed, row, anc): 
        '''
        Calculate sum of forces on rowing boat

        Parameters
        ----------
        V: vector [2]
            boat velocity
        
        ed:    Row direction [longitudinal,lateral] [-]
        Frmag: rowing force [N]
        row:   True when rowing [True/False]
        anc:   Ture when anchor is dropped[True/False]

        Returns
        -------
        Ft: scalar
            total force [longitudinal,lateral] [N]    
        '''    

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
    
    def start(self):
        # Returns a representation of the starting state of the ocean
        lat, long = self.startps
        return (self.start_time, lat, long)
    
    def next_state(self, state, action):
        # Takes the state, and the move to be applied.
        # Returns the new state.
        pass

    def legal_moves(self, state_history):
        # Takes a sequence of game states representing the full
        # game history, and returns the full list of moves that
        # are legal plays for the current player.
        
        # get_rowing_activity
        
        
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
        
        