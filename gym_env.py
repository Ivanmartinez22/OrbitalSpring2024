# #############################################################################
# ## Restricted Rights
# ## WARNING: This is a restricted distribution L3HARRIS REPOSITORY file.
# ##          Do Not Use Under A Government Charge Number Without Permission.
# #############################################################################

# *****************************************************************************
# FILE:             gym_env.py
#
#    Copyright (C) 2024 L3Harris Technologies, Inc.  All rights reserved.
#
# CLASSIFICATION:   Unclassified
#
# DESCRIPTION:
#  Constructs OpenAI Gym environment from Orekit and allows for RL training to 
#  occur
#
# LIMITATIONS:
#  Requires a static goal state and supports single agent training
#
# SOFTWARE HISTORY:
#  01FEB24 90PA PTR#MISSANGR-01026  P. Sridhar
#               Initial coding.
# *****************************************************************************

# An adaptation of the Satmind environment from A Reinforcement Learning Approach to Spacecraft Trajectory
# Converts original implementation into a valid OpenAI gym environment

import gym
from gym import spaces

import orekit
from math import radians, degrees, pi
import datetime
import numpy as np
import os, random
import time

orekit.initVM()

from org.orekit.frames import FramesFactory
from org.orekit.bodies import OneAxisEllipsoid
from org.orekit.orbits import KeplerianOrbit
from org.orekit.propagation import SpacecraftState
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.orbits import OrbitType, PositionAngleType
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.utils import IERSConventions, Constants
# from org.orekit.forces.maneuvers import ConstantThrustManeuver
from org.orekit.forces.maneuvers import ImpulseManeuver
from org.orekit.propagation.events import DateDetector
from org.orekit.frames import LOFType
from org.orekit.attitudes import LofOffset
from orekit.pyhelpers import setup_orekit_curdir  
from org.orekit.forces.gravity import NewtonianAttraction
from org.orekit.utils import Constants

from org.hipparchus.geometry.euclidean.threed import Vector3D
from orekit import JArray_double


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3d plotting
from matplotlib.widgets import Slider, Button


import pyorb
import time
import csv





def add_vel(ax):
    r = orb._cart[:3, 0]
    v = orb._cart[3:, 0]
    vel = ax.quiver(
        r[0], r[1], r[2],
        v[0], v[1], v[2],
        length=ax_lims*0.05,
    )
    return vel


#plot variables


fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.25, bottom=0.25)
# plt.ion()
num = 1000
ax_lims = 7000000
global asymtote_limit
asymtote_limit = 0.99




orb = pyorb.Orbit(
    M0 = 1.0,
    G = pyorb.get_G(length='AU', mass='Msol', time='y'),
    num = num,
    a = 1.0,
    e = 0,
    i = 0,
    omega = 0,
    Omega = 0,
    anom = np.linspace(0, 360, num=num),
    degrees = True,
    type = 'true',
)
# for some optimization
orb.direct_update = False


target = pyorb.Orbit(
    M0 = 1.0,
    G = pyorb.get_G(length='AU', mass='Msol', time='y'),
    num = num,
    a = 6300000,
    e = 0.23*4,
    i = 5.3*4,
    omega = 24.0*4,
    Omega = 24.0*4,
    anom = np.linspace(0, 360, num=num),
    degrees = True,
    type = 'true',
)




r = orb.r






l, = ax.plot(r[0, :], r[1, :], r[2, :], '-g', label='Current State') #training orbits


fin, = ax.plot(r[0, :], r[1, :], r[2, :], '-r', label='Target State')


dot, = ax.plot([r[0, 0]], [r[1, 0]], [r[2, 0]], 'ob') # Current orbit
global vel
vel = add_vel(ax)
ax.plot([0], [0], [0], 'og', label='Earth')
plt.legend(loc="upper left")


# Display Sliders
axr_b = fig.add_axes([0.05, 0.10, 0.05, 0.02])
r_b = Button(axr_b, 'Reset')






ax.set_title('Orbit', fontsize=22)
ax.set_xlabel('X-position [m]', fontsize=15, labelpad=20)
ax.set_ylabel('Y-position [m]', fontsize=15, labelpad=20)
ax.set_zlabel('Z-position [m]', fontsize=15, labelpad=20)


ax.set_xlim([-ax_lims, ax_lims])
ax.set_ylim([-ax_lims, ax_lims])
ax.set_zlim([-ax_lims, ax_lims])


axcolor = 'lightgoldenrodyellow'
ax_a = plt.axes([0.25, 0.05, 0.2, 0.03], facecolor=axcolor)
ax_e = plt.axes([0.25, 0.1, 0.2, 0.03], facecolor=axcolor)
ax_i = plt.axes([0.25, 0.15, 0.2, 0.03], facecolor=axcolor)


ax_omega = plt.axes([0.6, 0.05, 0.2, 0.03], facecolor=axcolor)
ax_Omega = plt.axes([0.6, 0.1, 0.2, 0.03], facecolor=axcolor)
ax_nu = plt.axes([0.6, 0.15, 0.2, 0.03], facecolor=axcolor)


s_a = Slider(ax_a, 'a [m]', 5400*1e3, 6400*1e3, valinit=1)
s_e = Slider(ax_e, 'e [1]', 0, 1, valinit=0)
# s_e.is_hyp = False
s_i = Slider(ax_i, 'i [deg]', 4, 6, valinit=0)


s_omega = Slider(ax_omega, 'omega [deg]', 0, 40, valinit=0)
s_Omega = Slider(ax_Omega, 'Omega [deg]', 0, 40, valinit=0)
s_nu = Slider(ax_nu, 'nu [deg]', -180, 180, valinit=0)








#functions






def draw():
    r = orb.r
    t = target.r


    l.set_xdata(r[0, 1:])
    l.set_ydata(r[1, 1:])
    l.set_3d_properties(r[2, 1:])


    dot.set_xdata([r[0, 0]])
    dot.set_ydata([r[1, 0]])
    dot.set_3d_properties([r[2, 0]])


    fin.set_xdata(t[0, 1:])
    fin.set_ydata(t[1, 1:])
    fin.set_3d_properties(t[2, 1:])


    global vel
    vel.remove()
    vel = add_vel(ax)


    fig.canvas.draw_idle()




def update_orb(val):
    a, e, i, omega, Omega, nu = val
    orb.a = a
    orb.e = e * 4 # x4 scaling to better observe slight element changes
    orb.i = i * 4
    orb.omega = omega * 4
    orb.Omega = Omega * 4
    orb._kep[5, 0] = nu * 4
    draw()




def update_sat(a, e, i, omega, Omega, nu):
   
   


   


    def set_state(event, source):
        if source == 'Reset':
            current_state()


    r_b.on_clicked(lambda x: set_state(x, 'Reset'))
   


    def current_state():
       
        # time.sleep(1)
        # Iterate through each state


        update_orb([a, e, i, omega, Omega, nu])
        s_a.set_val(a)
        s_e.set_val(e)
        s_i.set_val(i)
        s_omega.set_val(omega)
        s_Omega.set_val(Omega)
        s_nu.set_val(nu)
        plt.pause(0.1)


    current_state()

# Loads orekit-data.zip from current directory
setup_orekit_curdir()

FUEL_MASS = "Fuel Mass"

# Set constants
UTC = TimeScalesFactory.getUTC()
inertial_frame = FramesFactory.getEME2000()
attitude = LofOffset(inertial_frame, LOFType.LVLH)
EARTH_RADIUS = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
MU = Constants.WGS84_EARTH_MU

dir = "results"
# Check whether the specified path exists or not
isExist = os.path.exists(dir)
if not isExist:
   os.makedirs("results/reward")
   os.makedirs("results/state")
   os.makedirs("results/action")
   os.makedirs("models")

# Inherit from OpenAI gym
class OrekitEnv(gym.Env):
    """
    This class uses Orekit to create an environment to propagate a satellite
    """

    def __init__(self, state, state_targ, date, duration, mass, stepT, live_viz):
        """
        Params:
        state: Kepler coordinates of current state
        state_targ: Kepler coordinates of target state
        date: Start date
        duration: Simulation duration
        mass: [dry_mass, fuel_mass]
        stepT: Time between time steps (seconds)
        live_viz: show live visualizer while training (boolean)
        """

        """
        (this was preexisting but did not match any of the actual parameters)
        initializes the orekit VM and included libraries
        Params:
        _prop: The propagation object
        _initial_date: The initial start date of the propagation
        _orbit: The orbit type (Keplerian, Circular, etc...)
        _currentDate: The current date during propagation
        _currentOrbit: The current orbit paramenters
        _px: spacecraft position in the x-direction
        _py: spacecraft position in the y-direction
        _sc_fuel: The spacecraft with fuel accounted for
        _extrap_Date: changing date during propagation state
        _sc_state: The spacecraft without fuel
        """

        super(OrekitEnv, self).__init__()
        # ID for reward/state output files (Can create better system)
        self.id = random.randint(1,100000)
        self.alg = ""

        # satellite position at each time step
        self.px = [] # satellite x position
        self.py = [] # satellite y position
        self.pz = [] # satellite z position

        # satellite target position at each time step
        self.target_px = [] # target x
        self.target_py = [] # target y
        self.target_pz = [] # target z

        # model parameters (equinoctial elements) at each time step
        self.a_orbit = [] # semimajor axis
        self.ex_orbit = [] # eccentricity x
        self.ey_orbit = [] # eccentricity y
        self.hx_orbit = [] # angular momentum x
        self.hy_orbit = [] # angular momentum y
        self.lv_orbit = [] # true longitude and latitude

        # rate of change of state params at each time step
        self.adot_orbit = []
        self.exdot_orbit = []
        self.eydot_orbit = []
        self.hxdot_orbit = []
        self.hydot_orbit = []

        # Kepler coordinates at each time step
        # https://en.wikipedia.org/wiki/Orbital_elements
        self.e_orbit = [] # eccentricity
        self.i_orbit = [] # inclination
        self.w_orbit = [] # argument of periapsis
        self.omega_orbit = [] # longitude of ascending node
        self.v_orbit = [] # true anomaly at epoch

        self._sc_state = None
        self._extrap_Date = None
        self._targetOrbit = None

        self.total_reward = 0
        self.episode_num = 0
        # List of rewards/episode over entire lifetime of env instance
        self.episode_reward = []
        #List of actions and thrust magnitudes
        self.actions = []
        self.thrust_mags = []
        self.n_actions = 0
        self.consecutive_actions = 0
        self.curr_dist = 0
        self.n_hits = 0
        self.initial_dist = 0

        # Fuel params
        self.dry_mass = mass[0]
        self.fuel_mass = mass[1]
        self.curr_fuel_mass = self.fuel_mass
        self.initial_mass = self.dry_mass + self.fuel_mass

        # Accpetance tolerance
        self._orbit_tolerance = {'a': 10000, 'ex': 0.01, 'ey': 0.01, 'hx': 0.001, 'hy': 0.001, 'lv': 0.01}

        # Allows for randomized reset
        self.randomize = False
        self._orbit_randomizer = {'a': 4000.0e3, 'e': 0.2, 'i': 2.0, 'w': 10.0, 'omega': 10.0, 'lv': 5.0}
        self.seed_state = state
        self.seed_target = state_targ
        self.target_hit = False

        # Environment initialization -----------------------------------
        # Set Dates
        self.set_date(date)
        self._extrap_Date = self._initial_date
        self.final_date = self._initial_date.shiftedBy(duration)

        # create orbits (in Keplerian coordinates)
        self.initial_orbit = self.create_orbit(state, self._initial_date)
        self._currentOrbit = self.create_orbit(state, self._initial_date)
        self._targetOrbit= self.create_orbit(state_targ, self.final_date)

        self.set_spacecraft(self.initial_mass, self.curr_fuel_mass) # sets self._sc_state
        self.create_Propagator() # set self._prop with NumericalPropagator
        self.setForceModel() # update self._prop to include HolmesFeatherstoneAttractionModel ForceModel

        self.stepT = stepT
        self.live_viz = live_viz

        # OpenAI API to define 3D continous action space vector [a,b,c]
        # Velocity in the following directions:
            # radial: line formed from center of earth to satellite, 
            # tangential: facing in the direction of movement perpendicular to radial
            # normal: perpendicular to orbit plane
        # self.action_space = spaces.Box(
        #     low=-1,
        #     high=1,
        #     shape=(3,),
        #     dtype=np.float32
        # )

        # new discrete action space
        self.thrust_values = [-100, -75, -50, -25, 0, 25, 50, 75, 100]
        self.action_space = spaces.MultiDiscrete([len(self.thrust_values)] * 3)


        # self.observation_space = 10  # states | Equinoctial components + derivatives
        #   changing to 18: 6 for current equinoctial components, 6 for current derivatives, 5 for target equinoctial components (minus lv), 1 for n_actions
        # OpenAI API
        # state params + derivatives (could include target in future)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,shape=(18,),
                                        dtype=np.float32)
        self.action_bound = 0.6  # Max thrust limit
        self._isp = 3100.0 

        # set self.r_target_state and self.r_initial_state with data from _targetOrbit and _orbit (convert from KeplerianOrbit to np.array of equinoctial components)
        # (originally from state and state_targ parameters)
        self.r_target_state = self.get_state(self._targetOrbit)
        self.r_initial_state = self.get_state(self.initial_orbit)
        
        #visulizer plot variables 
        # self.fig = plt.figure(figsize=(15, 15))
        # self.ax = self.fig.add_subplot(111, projection='3d')
        if self.live_viz is True:
            plt.ion()



    def set_date(self, date=None, absolute_date=None, step=0):
        """
        Set up the date for an orekit secnario
        :param date: list [year, month, day, hour, minute, second] (optional)
        :param absolute_date: a orekit Absolute Date object (optional)
        :param step: seconds to shift the date by (int)
        :return:
        """
        if date != None:
            year, month, day, hour, minute, sec = date
            self._initial_date = AbsoluteDate(year, month, day, hour, minute, sec, UTC)
        elif absolute_date != None and step != 0:
            self._extrap_Date = AbsoluteDate(absolute_date, step, UTC)
        else:
            # no argument given, use current date and time
            now = datetime.datetime.now()
            year, month, day, hour, minute, sec = now.year, now.month, now.day, now.hour, now.minute, float(now.second)
            self._initial_date = AbsoluteDate(year, month, day, hour, minute, sec, UTC)


    def create_orbit(self, state, date):
        """
         Crate the initial orbit using Keplarian elements
        :param state: a state list [a, e, i, omega, raan, lM]
        :param date: A date given as an orekit absolute date object
        :return:
        """
        a, e, i, omega, raan, lM = state # get keplerian coordinates

        # Add Earth size offset
        a += EARTH_RADIUS
        # Convert to radians
        i = radians(i)
        omega = radians(omega)
        raan = radians(raan)
        lM = radians(lM)

        # Initialize Derivatives 
        aDot, eDot, iDot, paDot, rannDot, anomalyDot = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # Set inertial frame
        set_orbit = KeplerianOrbit(a, e, i, omega, raan, lM,
                                   aDot, eDot, iDot, paDot, rannDot, anomalyDot,
                                   PositionAngleType.TRUE, inertial_frame, date, MU)
          
        return set_orbit


    def convert_to_keplerian(self, orbit):
        ko = KeplerianOrbit(orbit)
        return ko


    def set_spacecraft(self, mass, fuel_mass):
        """
        Add the fuel mass to the spacecraft
        :param mass: dry mass of spacecraft (kg, flaot)
        :param fuel_mass:
        :return:
        """
        sc_state = SpacecraftState(self.initial_orbit, mass)
        self._sc_state = sc_state.addAdditionalState (FUEL_MASS, fuel_mass)


    def create_Propagator(self):
        """
        Creates and initializes the propagator
        :return:
        """
        minStep = 0.001
        maxStep = 500.0

        # define tolerances of integrator
        # lower tolerances are more accurate but usually require shorter step times (less efficient, more accurate)
        # higher tolerances allow larger step times but sacrifice accuracy (more efficient, less accurate)
        position_tolerance = 60.0
        tolerances = NumericalPropagator.tolerances(position_tolerance, self.initial_orbit, self.initial_orbit.getType())
        abs_tolerance = JArray_double.cast_(tolerances[0])
        rel_telerance = JArray_double.cast_(tolerances[1])

        # define integrator
        # Dormand Prince algorithm is able to adaptively adjust step size
        # step size used to estimate integration by breaking it into discrete intervals
        # larger step sizes = more efficient, less accurate
        # smaller step sizes = less efficient, more accurate
        integrator = DormandPrince853Integrator(minStep, maxStep, abs_tolerance, rel_telerance)
        integrator.setInitialStepSize(10.0)

        # create propagator
        numProp = NumericalPropagator(integrator)
        numProp.setInitialState(self._sc_state) # self._sc_state also contains data about orbit
        numProp.setMu(MU)
        numProp.setOrbitType(OrbitType.KEPLERIAN)
        numProp.setAttitudeProvider(attitude)

        # numProp.setSlaveMode() # was not commented out before

        self._prop = numProp


    def setForceModel(self):
        """
        Set up environment force models
        """
        # force model gravity field
        newattr = NewtonianAttraction(MU)
        itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)  # International Terrestrial Reference Frame, earth fixed

        earth = OneAxisEllipsoid(EARTH_RADIUS,
                                Constants.WGS84_EARTH_FLATTENING,itrf)
        gravityProvider = GravityFieldFactory.getNormalizedProvider(8, 8)
        self._prop.addForceModel(HolmesFeatherstoneAttractionModel(earth.getBodyFrame(), gravityProvider))

    # required for Gym Env, called when new episode starts
    def reset(self):
        """
        Resets the orekit enviornment
        :return:
        """

        print("RESET")
        print('Total Reward:', self.total_reward)
        print(f'Fuel Remaining: {self.curr_fuel_mass}/{self.fuel_mass}')
        print('Actions:', len(self.actions))

        self.write_episode_stats()

        self._prop = None
        self._currentDate = None
        self._currentOrbit = None

        self.n_actions = 0
        self.consecutive_actions = 0

        # Randomizes the initial orbit (initial state +- random variable)
        if self.randomize:
            self.initial_orbit = None
            a_rand = self.seed_state[0]
            e_rand = self.seed_state[1]
            w_rand = self.seed_state[3]
            omega_rand = self.seed_state[4]
            lv_rand = self.seed_state[5]

            a_rand = random.uniform(self.seed_state[0]-self._orbit_randomizer['a'], self._orbit_randomizer['a']+ self.seed_state[0])
            e_rand = random.uniform(self.seed_state[1]-self._orbit_randomizer['e'], self._orbit_randomizer['e']+ self.seed_state[1])
            i_rand = random.uniform(self.seed_state[2]-self._orbit_randomizer['i'], self._orbit_randomizer['i']+ self.seed_state[2])
            w_rand = random.uniform(self.seed_state[3]-self._orbit_randomizer['w'], self._orbit_randomizer['w']+ self.seed_state[3])
            omega_rand = random.uniform(self.seed_state[4]-self._orbit_randomizer['omega'], self._orbit_randomizer['omega']+ self.seed_state[4])
            lv_rand = random.uniform(self.seed_state[5]-self._orbit_randomizer['lv'], self._orbit_randomizer['lv']+ self.seed_state[5])
            
            state = [a_rand, e_rand, i_rand, w_rand, omega_rand, lv_rand]

            self.initial_orbit = self.create_orbit(state, self._initial_date)
            self._currentOrbit = self.create_orbit(state, self._initial_date)
        else:
            self._currentOrbit = self.initial_orbit

        # reset dates
        self._currentDate = self._initial_date
        self._extrap_Date = self._initial_date

        # reset spacecraft state
        self.set_spacecraft(self.dry_mass + self.fuel_mass, self.fuel_mass)
        self.curr_fuel_mass = self.fuel_mass

        # create propagator and force model
        self.create_Propagator()
        self.setForceModel()

        # recreate arrays for post analysis
        self.px = []
        self.py = []
        self.pz = []

        self.a_orbit = []
        self.ex_orbit = []
        self.ey_orbit = []
        self.hx_orbit = []
        self.hy_orbit = []
        self.lv_orbit = []

        self.adot_orbit = []
        self.exdot_orbit = []
        self.eydot_orbit = []
        self.hxdot_orbit = []
        self.hydot_orbit = []

        # Kepler
        self.e_orbit = []
        self.i_orbit = []
        self.w_orbit = []
        self.omega_orbit = []
        self.v_orbit = []

        self.actions = []
        self.thrust_mags = []


        state = self.get_state(self._currentOrbit)

        if self.total_reward != 0:
            self.episode_reward.append(self.total_reward) 
        self.total_reward = 0
        self.episode_num += 1
        
        return state

    @property
    def getTotalMass(self):
        """
        Get the total mass of the spacecraft
        :return: dry mass + fuel mass (kg)
        """
        return self._sc_state.getAdditionalState(FUEL_MASS)[0] + self._sc_state.getMass()


    # return state as list from orbit object
    # used by the model as input for NN (must match observation space dimensions)
    def get_state(self, orbit, with_derivatives=True):
        # basic equinoctial components
        state = [orbit.getA(), orbit.getEquinoctialEx(), orbit.getEquinoctialEy(),
                       orbit.getHx(), orbit.getHy(), orbit.getLv()] 
        
        # add derivatives
        if with_derivatives: 
            state += [orbit.getADot(), orbit.getEquinoctialExDot(),
                     orbit.getEquinoctialEyDot(),
                     orbit.getHxDot(), orbit.getHyDot(), orbit.getLvDot()]

        # add target and current fuel
        state += [self._targetOrbit.getA(), self._targetOrbit.getE(), self._targetOrbit.getI(), self._targetOrbit.getPerigeeArgument(), self._targetOrbit.getRightAscensionOfAscendingNode(),
                 self.curr_fuel_mass]
        
        return state
    

    # gets the state with the current and target position and velocities in each direction + fuel mass
    # 18 dimensions
    def get_state_keplerian(self, orbit):
        orbit = self.convert_to_keplerian(orbit)
        target = self.convert_to_keplerian(self._targetOrbit)
        return [ orbit.getA(), orbit.getE(), orbit.getI(), orbit.getPerigeeArgument(), orbit.getRightAscensionOfAscendingNode(), orbit.getTrueAnomaly(),
                 orbit.getADot(), orbit.getEDot(), orbit.getIDot(), orbit.getPerigeeArgumentDot(), orbit.getRightAscensionOfAscendingNodeDot(), orbit.getTrueAnomalyDot(),
                 target.getA(), target.getE(), target.getI(), target.getPerigeeArgument(), target.getRightAscensionOfAscendingNode(),
                 self.curr_fuel_mass]


    # takes in action and computes state after action is performed
    # returns observation, reward, done, info
    def step(self, input):
        """
        Take a propagation step
        :param input: 3D velocity vector (m/s, float)
        :return: spacecraft state (np.array), reward value (float), done (bool)
        """
        self._prevOrbit = self._currentOrbit
        self.prev_fuel = self.curr_fuel_mass
        self.did_action = False

        compute_thrust_val = lambda x: float(self.thrust_values[x])
        input = list(map(compute_thrust_val, input))
        vel = Vector3D(*input)

        if(any(input)):
            self.consecutive_actions += 1
            self.did_action = True
            self.n_actions += 1
        else:
            self.consecutive_actions = 0

        if(any(input)):
            self.n_actions += 1

        # Remove previous event detectors
        self._prop.clearEventsDetectors()

        # Add force model
        event_detector = DateDetector(self._extrap_Date.shiftedBy(0.01)) # detects when date is reached during propagation
        impulse = ImpulseManeuver(event_detector, attitude, vel, self._isp) # applies velocity vector when event triggered
        self._prop.addEventDetector(impulse) # add detector to propagator

        # Propagate
        try:
            currentState = self._prop.propagate(self._extrap_Date, self._extrap_Date.shiftedBy(float(5000)))
        except: 
            print('orekit error')
            state = self.get_state(self._currentOrbit)
            return state, -1, True, {}
        

        # set current state equal to newly propagated state
        self.curr_fuel_mass = currentState.getMass() - self.dry_mass
        self._currentDate = currentState.getDate()
        self._extrap_Date = self._currentDate
        self._currentOrbit = currentState.getOrbit()

        # Saving for post analysis
        coord = currentState.getPVCoordinates().getPosition()
        self.px.append(coord.getX())
        self.py.append(coord.getY())
        self.pz.append(coord.getZ())
        self.a_orbit.append(currentState.getA())
        self.ex_orbit.append(currentState.getEquinoctialEx())
        self.ey_orbit.append(currentState.getEquinoctialEy())
        self.hx_orbit.append(currentState.getHx())
        self.hy_orbit.append(currentState.getHy())
        self.lv_orbit.append(currentState.getLv())

        k_orbit = self.convert_to_keplerian(self._currentOrbit)
        self.e_orbit.append(k_orbit.getE())
        self.i_orbit.append(k_orbit.getI())
        self.w_orbit.append(k_orbit.getPerigeeArgument())
        self.omega_orbit.append(k_orbit.getRightAscensionOfAscendingNode())
        self.v_orbit.append(k_orbit.getTrueAnomaly())

        self.actions.append(vel)
        self.thrust_mags.append(vel)

        self.adot_orbit.append(self._currentOrbit.getADot())
        self.exdot_orbit.append(self._currentOrbit.getEquinoctialExDot())
        self.eydot_orbit.append(self._currentOrbit.getEquinoctialEyDot())
        self.hxdot_orbit.append(self._currentOrbit.getHxDot())
        self.hydot_orbit.append(self._currentOrbit.getHyDot())

        # Calc reward / termination state for this step
        reward, done = self.dist_reward(vel)
        state = self.get_state(self._currentOrbit)
        info = {} # OpenAI debug option
        
        if self.live_viz is True:
            update_sat((self.a_orbit[-1]-EARTH_RADIUS),self.e_orbit[-1],degrees(self.i_orbit[-1]),degrees(self.w_orbit[-1]),degrees(self.omega_orbit[-1]),degrees(self.v_orbit[-1]))

        return np.array(state), reward, done, info
    

    # compute the difference between 2 angles
    def angle_diff(self, angle1, angle2):
        diff = angle2 - angle1
        diff = (diff + pi) % (2 * pi) - pi
        return diff


    def dist_reward(self, action):
        """
        Computes the reward based on the state of the agent
        :return: reward value (float), done state (bool)
        """
        done = False

        curr_state = self.get_state(self._currentOrbit)
        curr_velocity = curr_state[6:12]
        prev_state = self.get_state(self._prevOrbit, with_derivatives=False)
        initial_state = self.get_state(self.initial_orbit)

        # prev_k = self.convert_to_keplerian(self._prevOrbit)
        # curr_k = self.convert_to_keplerian(self._currentOrbit)
        # start_k = self.convert_to_keplerian(self.initial_orbit)
        # target_k = self.convert_to_keplerian(self._targetOrbit)

        curr_dist = np.zeros(5)
        prev_dist = np.zeros(5)
        initial_dist = np.zeros(5)

        curr_dist[0] = abs(self.r_target_state[0] - curr_state[0]) / self.r_target_state[0]
        curr_dist[1] = abs(self.r_target_state[1] - curr_state[1]) / self.r_target_state[1]
        curr_dist[2] = abs(self.r_target_state[2] - curr_state[2]) / self.r_target_state[2]
        curr_dist[3] = abs(self.r_target_state[3] - curr_state[3]) / self.r_target_state[3]
        curr_dist[4] = abs(self.r_target_state[4] - curr_state[4]) / self.r_target_state[4]
        curr_dist_value = np.sum(curr_dist)
        self.curr_dist = curr_dist_value

        prev_dist[0] = abs(self.r_target_state[0] - prev_state[0]) / self.r_target_state[0]
        prev_dist[1] = abs(self.r_target_state[1] - prev_state[1]) / self.r_target_state[1]
        prev_dist[2] = abs(self.r_target_state[2] - prev_state[2]) / self.r_target_state[2]
        prev_dist[3] = abs(self.r_target_state[3] - prev_state[3]) / self.r_target_state[3]
        prev_dist[4] = abs(self.r_target_state[4] - prev_state[4]) / self.r_target_state[4]
        prev_dist_value = np.sum(prev_dist)

        initial_dist[0] = abs(self.r_target_state[0] - initial_state[0]) / self.r_target_state[0]
        initial_dist[1] = abs(self.r_target_state[1] - initial_state[1]) / self.r_target_state[1]
        initial_dist[2] = abs(self.r_target_state[2] - initial_state[2]) / self.r_target_state[2]
        initial_dist[3] = abs(self.r_target_state[3] - initial_state[3]) / self.r_target_state[3]
        initial_dist[4] = abs(self.r_target_state[4] - initial_state[4]) / self.r_target_state[4]
        initial_dist_value = np.sum(initial_dist)

        self.initial_dist = initial_dist_value


        curr_velocity_mag = np.linalg.norm(curr_velocity)
        curr_velocity_mag = 1 if curr_velocity_mag < 1 else curr_velocity_mag # only scale if velocity mag >= 1

        # positive = current state closer to target than previous
        # scaled by current velocity magnitude so it doesnt get big rewards for going fast
        # might replace with a constant value if its closer
        distance_change = (prev_dist_value - curr_dist_value) / curr_velocity_mag 
        distance_change_constant = 0.1 if distance_change > 0 else -0.1

        # penalize having lower altitude than both the target and initial state (hopefully will discourage crashing into)

        a_penalty = curr_state[0] - 1000000 + EARTH_RADIUS # positive = A is greater than min altitude (1000 km)
        a_penalty = 0 if a_penalty > 0 else a_penalty * 100

        fuel_consumed = self.prev_fuel - self.curr_fuel_mass
        total_fuel_consumed = self.fuel_mass - self.curr_fuel_mass
        consecutive_action_penalty = self.consecutive_actions * -10 if self.consecutive_actions > 5 else 0
        action_penalty = 0.1 if self.did_action else 0
        initial_difference = initial_dist_value - curr_dist_value # positive value = closer than initial
        initial_difference = 0 if initial_difference < 0 else initial_difference # dont penalize being further


        # reward = -1 * 10*curr_dist_value + 5*distance_change - 0.1*fuel_consumed - a_penalty**2 + consecutive_action_penalty
        # reward = -curr_dist_value - fuel_consumed
        # reward = -curr_dist_value + distance_change_constant - a_penalty - action_penalty
        reward = -curr_dist_value + initial_difference - a_penalty - action_penalty
        # print('distance reward:', curr_dist_value)
        # print('distance change:', distance_change)
        # print('a penalty:', a_penalty)
        # print('action penalty:', action_penalty)

        # print(distance_change_reward)
        # print(curr_dist_value)

        # reward = -(reward_a + reward_hx*10 + reward_hy*10 + reward_ex + reward_ey) * self.n_actions
        # print(reward)

        # TERMINAL STATES
        # Target state (with tolerance)
        if abs(self.r_target_state[0] - curr_state[0]) <= self._orbit_tolerance['a'] and \
           abs(self.r_target_state[1] - curr_state[1]) <= self._orbit_tolerance['ex'] and \
           abs(self.r_target_state[2] - curr_state[2]) <= self._orbit_tolerance['ey'] and \
           abs(self.r_target_state[3] - curr_state[3]) <= self._orbit_tolerance['hx'] and \
           abs(self.r_target_state[4] - curr_state[4]) <= self._orbit_tolerance['hy']:
            print('\nhit')
            reward = 100000 # multiply by % fuel left
            # if not self.randomize: # if the initial states are not randomized at each episode (initial states always the same)
            #     self.fuel_mass = total_fuel_consumed * 1.5 # set max fuel usage to current fuel
            done = True
            self.target_hit = True
            self.n_hits += 1
            # Create state file for successful mission
            self.write_state(curr_dist_value)

        # Out of fuel
        elif self.curr_fuel_mass <= 0:
            print('\nRan out of fuel')
            print('Distance:', curr_dist_value)
            reward = -total_fuel_consumed
            done = True

        # Crash into Earth
        elif self._currentOrbit.getA() < EARTH_RADIUS:
            print('\nIn earth')
            print('Distance:', curr_dist_value)
            reward = -100000
            done = True

        # Mission duration exceeded
        elif self._extrap_Date.compareTo(self.final_date) >= 0:
            print("\nOut of time")
            print('Distance:', curr_dist_value)
            reward = -total_fuel_consumed
            if self.n_actions == 0: # penalize doing nothing
                reward = -10000000
            done = True

        elif curr_dist_value > initial_dist_value * 20:
            print('\nToo far from target')
            reward = -1000
            done = True

        self.total_reward += reward

        return reward, done
    

    # State/Action Output files

    def write_state(self, distance):
        # State file (Equinoctial)
        with open("results/state/"+str(self.id)+"_"+self.alg+"_state_equinoctial_"+str(self.episode_num)+".txt", "w") as f:
            #Add episode number line 1
            f.write("Episode: " + str(self.episode_num) + '\n')
            f.write('Fuel: ' + str(self.curr_fuel_mass)  + '/' + str(self.fuel_mass) + '\n')
            f.write('fuel consumed: ' + str(self.fuel_mass - self.curr_fuel_mass))
            f.write('Distance: ' + str(distance)+ '\n')
            f.write('Total Reward: ' + str(self.total_reward)+ '\n')
            f.write('Number of Steps: ' + str(len(self.actions)) + '\n')
            for i in range(len(self.a_orbit)):
                try:
                    f.write(str(self.a_orbit[i]/1e3)+","+str(self.ex_orbit[i])+","+str(self.ey_orbit[i])+","+str(self.hx_orbit[i])+","+ \
                        str(self.hy_orbit[i])+","+str(self.lv_orbit[i])+","+str(self.px[i]/1e3)+","+str(self.py[i]/1e3)+","+str(self.pz[i]/1e3)+'\n')
                except Exception as err:
                    print("Unexpected error")
                    print("Writing '-' in place")
                    f.write('-\n')
        # State file (Kepler)
        with open("results/state/"+str(self.id)+"_"+self.alg+"_state_kepler_"+str(self.episode_num)+".txt", "w") as f:
            #Add episode number line 1
            f.write("Episode: " + str(self.episode_num) + '\n')
            for i in range(len(self.a_orbit)):
                try:
                    f.write(str(self.a_orbit[i]-EARTH_RADIUS)+","+str(self.e_orbit[i])+","+str(degrees(self.i_orbit[i]))+","\
                            +str(degrees(self.w_orbit[i]))+","+str(degrees(self.omega_orbit[i]))+","+str(degrees(self.v_orbit[i]))+'\n')
                except Exception as err:
                    print("Unexpected error", err)
                    # f.write('-\n')

        # Action file
        with open("results/action/"+str(self.id)+"_"+self.alg+"_action_"+str(self.episode_num)+".txt", 'w') as f:
            f.write("Fuel Mass: " + str(self.curr_fuel_mass) + "/" + str(self.fuel_mass) + '\n')
            for i in range(len(self.actions)):
                for j in range(3):
                    try:
                        f.write(str(self.actions[i][j])+",")
                    except Exception as err:
                        print("Unexpected error")
                        print("Writing '-' in place")
                        f.write('-\n')
                f.write(str(self.thrust_mags[i])+'\n')
      

    def write_reward(self):
        with open("results/reward/"+str(self.id)+"_"+self.alg+"_reward"+".txt", "w") as f:
            for reward in self.episode_reward:
                f.write(str(reward)+'\n')


    def write_episode_stats(self):
        with open('results/episode_stats/' + str(self.id) + "_" + self.alg + ".csv", "a") as f:
            f.write(str(self.episode_num) + ',' + str(self.total_reward) + ',' + str(self.curr_fuel_mass) + ',' + str(self.curr_dist) + ', ' + str(self.n_hits) + ', ' + str(self.initial_dist) + '\n')