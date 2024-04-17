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
from math import radians, degrees
import datetime
import numpy as np
import os, random
import time
import traceback
from math import radians, degrees, pi

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

    def __init__(self, state, state_targ, date, duration, mass, stepT, live_viz, action_space_type, thrust_type):
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

        # state params (used in model) at each time step
        self.a_orbit = [] # semimajor axis
        
        self.last_a = state_targ[0]
        self.targ_a = state_targ[0]
        self.targ_e = state_targ[1]
        self.ex_orbit = [] # eccentricity x
        self.ey_orbit = [] # eccentricity y
        self.hx_orbit = [] # inclination x
        self.hy_orbit = [] # inclination y
        self.lv_orbit = [] # ???

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
        self.curr_dist = 0
        #List of actions and thrust magnitudes
        self.actions = []
        self.thrust_mags = []
        self.n_actions = 0

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
        
        self.hit_multiple = False
        self.hit_2 = False
        self.hit_3 = False
        self.hit_4 = False
        self.number_of_moves = 0

        # Environment initialization -----------------------------------
        # Set Dates
        self.set_date(date)
        self._extrap_Date = self._initial_date
        self.final_date = self._initial_date.shiftedBy(duration)

        # create orbits (in Keplerian coordinates)
        self.initial_orbit = self.create_orbit(state, self._initial_date)
        self._currentOrbit = self.create_orbit(state, self._initial_date) # might have to reference same object as _orbit
        self._targetOrbit= self.create_orbit(state_targ, self.final_date)

        self.set_spacecraft(self.initial_mass, self.curr_fuel_mass) # sets self._sc_state
        self.create_Propagator() # set self._prop with NumericalPropagator
        self.setForceModel() # update self._prop to include HolmesFeatherstoneAttractionModel ForceModel

        self.stepT = stepT
        self.live_viz = live_viz
        self.thrust_type = thrust_type
        self.action_space_type = action_space_type

        # OpenAI API to define 3D continous action space vector [a,b,c]
        # Velocity in the following directions:
            # radial: line formed from center of earth to satellite, 
            # tangential: facing in the direction of movement perpendicular to radial
            # normal: perpendicular to orbit plane
        self.thrust_values = []
        self.action_space = spaces.Box(
                low=-1,
                high=1,
                shape=(4,),
                dtype=np.float32
            )
        self.consecutive_actions = 0
        if action_space_type == "discrete":
            self.thrust_values = [-100.0, -99.5, -99.0, -98.5, -98.0, -97.5, -97.0, -96.5, -96.0, -95.5, -95.0, -94.5, -94.0, -93.5, -93.0, -92.5, -92.0, -91.5, -91.0, -90.5, -90.0, -89.5, -89.0, -88.5, -88.0, -87.5, -87.0, -86.5, -86.0, -85.5, -85.0, -84.5, -84.0, -83.5, -83.0, -82.5, -82.0, -81.5, -81.0, -80.5, -80.0, -79.5, -79.0, -78.5, -78.0, -77.5, -77.0, -76.5, -76.0, -75.5, -75.0, -74.5, -74.0, -73.5, -73.0, -72.5, -72.0, -71.5, -71.0, -70.5, -70.0, -69.5, -69.0, -68.5, -68.0, -67.5, -67.0, -66.5, -66.0, -65.5, -65.0, -64.5, -64.0, -63.5, -63.0, -62.5, -62.0, -61.5, -61.0, -60.5, -60.0, -59.5, -59.0, -58.5, -58.0, -57.5, -57.0, -56.5, -56.0, -55.5, -55.0, -54.5, -54.0, -53.5, -53.0, -52.5, -52.0, -51.5, -51.0, -50.5, -50.0, -49.5, -49.0, -48.5, -48.0, -47.5, -47.0, -46.5, -46.0, -45.5, -45.0, -44.5, -44.0, -43.5, -43.0, -42.5, -42.0, -41.5, -41.0, -40.5, -40.0, -39.5, -39.0, -38.5, -38.0, -37.5, -37.0, -36.5, -36.0, -35.5, -35.0, -34.5, -34.0, -33.5, -33.0, -32.5, -32.0, -31.5, -31.0, -30.5, -30.0, -29.5, -29.0, -28.5, -28.0, -27.5, -27.0, -26.5, -26.0, -25.5, -25.0, -24.5, -24.0, -23.5, -23.0, -22.5, -22.0, -21.5, -21.0, -20.5, -20.0, -19.5, -19.0, -18.5, -18.0, -17.5, -17.0, -16.5, -16.0, -15.5, -15.0, -14.5, -14.0, -13.5, -13.0, -12.5, -12.0, -11.5, -11.0, -10.5, -10.0, -9.5, -9.0, -8.5, -8.0, -7.5, -7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5, 26.0, 26.5, 27.0, 27.5, 28.0, 28.5, 29.0, 29.5, 30.0, 30.5, 31.0, 31.5, 32.0, 32.5, 33.0, 33.5, 34.0, 34.5, 35.0, 35.5, 36.0, 36.5, 37.0, 37.5, 38.0, 38.5, 39.0, 39.5, 40.0, 40.5, 41.0, 41.5, 42.0, 42.5, 43.0, 43.5, 44.0, 44.5, 45.0, 45.5, 46.0, 46.5, 47.0, 47.5, 48.0, 48.5, 49.0, 49.5, 50.0, 50.5, 51.0, 51.5, 52.0, 52.5, 53.0, 53.5, 54.0, 54.5, 55.0, 55.5, 56.0, 56.5, 57.0, 57.5, 58.0, 58.5, 59.0, 59.5, 60.0, 60.5, 61.0, 61.5, 62.0, 62.5, 63.0, 63.5, 64.0, 64.5, 65.0, 65.5, 66.0, 66.5, 67.0, 67.5, 68.0, 68.5, 69.0, 69.5, 70.0, 70.5, 71.0, 71.5, 72.0, 72.5, 73.0, 73.5, 74.0, 74.5, 75.0, 75.5, 76.0, 76.5, 77.0, 77.5, 78.0, 78.5, 79.0, 79.5, 80.0, 80.5, 81.0, 81.5, 82.0, 82.5, 83.0, 83.5, 84.0, 84.5, 85.0, 85.5, 86.0, 86.5, 87.0, 87.5, 88.0, 88.5, 89.0, 89.5, 90.0, 90.5, 91.0, 91.5, 92.0, 92.5, 93.0, 93.5, 94.0, 94.5, 95.0, 95.5, 96.0, 96.5, 97.0, 97.5, 98.0, 98.5, 99.0, 99.5, 100.0]

            self.action_space = spaces.MultiDiscrete([len(self.thrust_values)] * 3)
            
        # self.observation_space = 10  # states | Equinoctial components + derivatives
        # OpenAI API
        # state params + derivatives (could include target in future)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,shape=(10,),
                                        dtype=np.float32)
        self.action_bound = 0.6  # Max thrust limit
        self._isp = 3100.0 

        # set self.r_target_state and self.r_initial_state with data from _targetOrbit and _orbit (convert from KeplerianOrbit to np.array)
        # (originally from state and state_targ parameters)
        self.r_target_state = self.get_state(self._targetOrbit)
        self.r_initial_state = self.get_state(self.initial_orbit)
        
        
        self.no_prop_counter = 0
        self.last_e = self.targ_e

        self.one_hit_per_episode = 0
        
        #visulizer plot variables 
        # self.fig = plt.figure(figsize=(15, 15))
        # self.ax = self.fig.add_subplot(111, projection='3d')
        if self.live_viz is True:
            plt.ion()
            
        # new discrete action space
        



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
        self.write_episode_stats()
        print(self.id)

        self._prop = None
        self._currentDate = None
        self._currentOrbit = None
        self.one_hit_per_episode = 0
        self.number_of_moves = 0
        
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
            self._currentOrbit = self.create_orbit(state, self._initial_date) # might have to reference same object as _orbit
        else:
            self._currentOrbit = self.initial_orbit

        # reset dates
        self._currentDate = self._initial_date
        self._extrap_Date = self._initial_date

        # reset spacecraft state
        self.set_spacecraft(self.initial_mass, self.fuel_mass)
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
        self.n_actions = 0
        
        self.last_a = self.targ_a
        self.last_e = self.targ_e

        state = np.array([self.initial_orbit.getA() / self.r_target_state[0],
                          self.initial_orbit.getEquinoctialEx(),
                          self.initial_orbit.getEquinoctialEy(),
                          self.initial_orbit.getHx(),
                          self.initial_orbit.getHy(), 0, 0, 0, 0, 0])

        print("RESET: ", self.total_reward)
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
    def get_state(self, orbit, with_derivatives=True):

        if with_derivatives:
            state = [orbit.getA(), orbit.getEquinoctialEx(), orbit.getEquinoctialEy(),
                     orbit.getHx(), orbit.getHy(), orbit.getLv(),
                     orbit.getADot(), orbit.getEquinoctialExDot(),
                     orbit.getEquinoctialEyDot(),
                     orbit.getHxDot(), orbit.getHyDot(), orbit.getLvDot()]
        else:
            state = [orbit.getA(), orbit.getEquinoctialEx(), orbit.getEquinoctialEy(),
                       orbit.getHx(), orbit.getHy(), orbit.getLv()]

        return state
    

    #Ivan Step Function 
    # takes in action and computes state after action is performed
    # returns observation, reward, done, info
    def step(self, input):
        """
        Take a propagation step
        :param input: 3D velocity vector (m/s, float)
        :return: spacecraft state (np.array), reward value (float), done (bool)
        """

        self._prevOrbit = self._currentOrbit
        # thrust_mag = np.linalg.norm(thrust)
        # thrust_dir = thrust / thrust_mag
        # DIRECTION = Vector3D(float(thrust_dir[0]), float(thrust_dir[1]), float(thrust_dir[2]))
        # vel = Vector3D(float(vel[0]), float(vel[1]), float(vel[2]))

        # radial, tangential, normal (not sure what order)
        # vel = Vector3D(float(input[0])*50, float(input[1])*50, float(input[2])*50)
        # thrust_bool = input[3] > 0 # model decides if it actually does performs a maneuver
        if self.thrust_type == "discrete":
            compute_thrust_val = lambda x: float(self.thrust_values[x])
            input = list(map(compute_thrust_val, input))
            vel = Vector3D(*input)
            self.number_of_moves += 1
        else:
            vel = Vector3D(float(input[0])*50, float(input[1])*50, float(input[2])*50)
            thrust_bool = input[3] > 0 # model decides if it actually does performs a maneuver
            if thrust_bool: 
                event_detector = DateDetector(self._extrap_Date.shiftedBy(0.01)) # detects when date is reached during propagation
                impulse = ImpulseManeuver(event_detector, attitude, vel, self._isp) # applies velocity vector when event triggered
                self._prop.addEventDetector(impulse) # add detector to propagator
                self.n_actions += 1
                self.number_of_moves += 1

        
        # Remove previous event detectors
        self._prop.clearEventsDetectors()

        # Add force model
        if self.thrust_type == "discrete":
            event_detector = DateDetector(self._extrap_Date.shiftedBy(0.01)) # detects when date is reached during propagation
            impulse = ImpulseManeuver(event_detector, attitude, vel, self._isp) # applies velocity vector when event triggered
            self._prop.addEventDetector(impulse) # add detector to propagator
            self.n_actions += 1
            

        # Propagate
        # if self.last_a > 0:
        #     try:
        #         currentState = self._prop.propagate(self._extrap_Date, self._extrap_Date.shiftedBy(float(self.stepT)))
        #         self.curr_fuel_mass = currentState.getMass() - self.dry_mass
        #         self._currentDate = currentState.getDate()
        #         self._extrap_Date = self._currentDate
        #         self._currentOrbit = currentState.getOrbit()
        #         coord = currentState.getPVCoordinates().getPosition()

                

        #         # Saving for post analysis
        #         self.px.append(coord.getX())
        #         self.py.append(coord.getY())
        #         self.pz.append(coord.getZ())
        #         self.a_orbit.append(currentState.getA())
        #         self.ex_orbit.append(currentState.getEquinoctialEx())
        #         self.ey_orbit.append(currentState.getEquinoctialEy())
        #         self.hx_orbit.append(currentState.getHx())
        #         self.hy_orbit.append(currentState.getHy())
        #         self.lv_orbit.append(currentState.getLv())
        #     except Exception as err:
        #         reward = -100000000
        #         print(err)
        #         print("Orbit error a < 0")
        if self.last_a > 0 or self.last_e < 1:
            try:
                currentState = self._prop.propagate(self._extrap_Date, self._extrap_Date.shiftedBy(float(self.stepT)))
                self.curr_fuel_mass = currentState.getMass() - self.dry_mass
                self._currentDate = currentState.getDate()
                self._extrap_Date = self._currentDate
                self._currentOrbit = currentState.getOrbit()
                coord = currentState.getPVCoordinates().getPosition()
            except:
                print("Orekit error")
                # state = self.get_state(self._currentOrbit, with_derivatives=True)
                # state = np.append(state, self.get_state(self._targetOrbit, with_derivatives=False))[:-1]
                # state = np.append(state, self.n_actions)
                state_1 = [(self._currentOrbit.getA()) / self.r_target_state[0],
                   self._currentOrbit.getEquinoctialEx(), self._currentOrbit.getEquinoctialEy(),
                   self._currentOrbit.getHx(), self._currentOrbit.getHy(),
                   self._currentOrbit.getADot(), self._currentOrbit.getEquinoctialExDot(),
                   self._currentOrbit.getEquinoctialEyDot(),
                   self._currentOrbit.getHxDot(), self._currentOrbit.getHyDot()
                   ]
                
                return state_1, -1, True, {}
            

            # Saving for post analysis
            self.px.append(coord.getX())
            self.py.append(coord.getY())
            self.pz.append(coord.getZ())
            self.a_orbit.append(currentState.getA())
            self.ex_orbit.append(currentState.getEquinoctialEx())
            self.ey_orbit.append(currentState.getEquinoctialEy())
            self.hx_orbit.append(currentState.getHx())
            self.hy_orbit.append(currentState.getHy())
            self.lv_orbit.append(currentState.getLv())
           
        if self.no_prop_counter >= 1:
            reward = -100000000
            print("No prop count: " + str(self.no_prop_counter))

       

        k_orbit = self.convert_to_keplerian(self._currentOrbit)

        # print(f"Orbit after: {currentState.getA()-EARTH_RADIUS} {k_orbit.getE()} {k_orbit.getI()} {k_orbit.getPerigeeArgument()} {k_orbit.getRightAscensionOfAscendingNode()} {k_orbit.getTrueAnomaly()}")


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
        reward, done = self.dist_reward() # was self.dist_reward(thrust)

        # penalize doing nothing
        if done and self.n_actions == 0:
            reward = -1000
            print("doing nothing")

        state_1 = [(self._currentOrbit.getA()) / self.r_target_state[0],
                   self._currentOrbit.getEquinoctialEx(), self._currentOrbit.getEquinoctialEy(),
                   self._currentOrbit.getHx(), self._currentOrbit.getHy(),
                   self._currentOrbit.getADot(), self._currentOrbit.getEquinoctialExDot(),
                   self._currentOrbit.getEquinoctialEyDot(),
                   self._currentOrbit.getHxDot(), self._currentOrbit.getHyDot()
                   ]
        # OpenAI debug option
        info = {}
        
        if self.live_viz is True:
        
            print(len(self.a_orbit))
            if len(self.a_orbit) >= 1:
                self.last_a = self.a_orbit[-1]-EARTH_RADIUS
                self.last_e = self.e_orbit[-1]
                update_sat((self.a_orbit[-1]-EARTH_RADIUS),self.e_orbit[-1],degrees(self.i_orbit[-1]),degrees(self.w_orbit[-1]),degrees(self.omega_orbit[-1]),degrees(self.v_orbit[-1]))
            

        return np.array(state_1), reward, done, info
    
    
    
    
        # compute the difference between 2 angles
    def angle_diff(self, angle1, angle2):
        diff = angle2 - angle1
        diff = (diff + pi) % (2 * pi) - pi
        return diff


    def dist_reward(self):
        """
        Computes the reward based on the state of the agent
        :return: reward value (float), done state (bool)
        """
        # a, ecc, i, w, omega, E, adot, edot, idot, wdot, omegadot, Edot = state

        done = False

        state = self.get_state(self._currentOrbit, with_derivatives=False)

        prev_k = self.convert_to_keplerian(self._prevOrbit)
        curr_k = self.convert_to_keplerian(self._currentOrbit)
        target_k = self.convert_to_keplerian(self._targetOrbit)

        prev_dist = np.zeros(5)
        curr_dist = np.zeros(5)

        prev_dist[0] = (target_k.getA() - prev_k.getA()) / target_k.getA()
        prev_dist[1] = target_k.getE() - prev_k.getE()
        prev_dist[2] = self.angle_diff(target_k.getI(), prev_k.getI())
        prev_dist[3] = self.angle_diff(target_k.getPerigeeArgument(), prev_k.getPerigeeArgument())
        prev_dist[4] = self.angle_diff(target_k.getRightAscensionOfAscendingNode(), prev_k.getRightAscensionOfAscendingNode())
        prev_dist_value = np.linalg.norm(prev_dist)
        # prev_dist_value = np.sum(prev_dist)

        curr_dist[0] = (target_k.getA() - curr_k.getA()) / target_k.getA()
        curr_dist[1] = target_k.getE() - curr_k.getE()
        curr_dist[2] = self.angle_diff(target_k.getI(), curr_k.getI())
        curr_dist[3] = self.angle_diff(target_k.getPerigeeArgument(), curr_k.getPerigeeArgument())
        curr_dist[4] = self.angle_diff(target_k.getRightAscensionOfAscendingNode(), curr_k.getRightAscensionOfAscendingNode())
        curr_dist_value = np.linalg.norm(curr_dist)
       
        self.curr_dist = curr_dist_value
        # distance_change_reward = (prev_dist_value - curr_dist_value)  # reward being closer than the previous

        # reward = distance_change_reward - curr_dist_value
        
        reward = 0

        # TERMINAL STATES
        # Target state (with tolerance)
        
        
        if abs(self.r_target_state[0] - state[0]) <= self._orbit_tolerance['a'] and \
           abs(self.r_target_state[1] - state[1]) <= self._orbit_tolerance['ex'] and \
           abs(self.r_target_state[2] - state[2]) <= self._orbit_tolerance['ey'] and \
           abs(self.r_target_state[3] - state[3]) <= self._orbit_tolerance['hx'] and \
           abs(self.r_target_state[4] - state[4]) <= self._orbit_tolerance['hy']:
            reward += 1000000
            self.total_reward += reward
            done = True
            print('hit')
            self.target_hit = True
            # Create state file for successful mission
            self.write_state()
            return reward, done
        
        # if self.number_of_moves >= 11:
        #     done = True
        #     print('too many moves')
        #     return reward, done
              
        # Give more reward for when individual elements get close to target value
        hit_Number = []
        if abs(self.r_target_state[0] - state[0]) <= self._orbit_tolerance['a']:
            #reward += 1
            #print('hit a')
            # self.target_hit = True
            # Create state file for successful mission
            # self.write_state()
            hit_Number.append("a")
            #self.one_hit_per_episode += 1
            #self.total_reward += reward
        
        if abs(self.r_target_state[1] - state[1]) <= self._orbit_tolerance['ex']:
            # reward += 1
            # print('hit ex')
            # self.target_hit = True
            # Create state file for successful mission
            # self.write_state()
            hit_Number.append("ex")
            # self.one_hit_per_episode += 1
            # self.total_reward += reward
            
        if abs(self.r_target_state[2] - state[2]) <= self._orbit_tolerance['ey']:
            # reward += 1
            # print('hit ey')
            # self.target_hit = True
            # Create state file for successful mission
            # self.write_state()
            hit_Number.append("ey")
            # self.one_hit_per_episode += 1
            # self.total_reward += reward
        
        if abs(self.r_target_state[3] - state[3]) <= self._orbit_tolerance['hx']:
            # reward += 1
            # print('hit hx')
            # self.target_hit = True
            # Create state file for successful mission
            # self.write_state()
            hit_Number.append("hx")
            # self.one_hit_per_episode += 1
            # self.total_reward += reward

        if abs(self.r_target_state[4] - state[4]) <= self._orbit_tolerance['hy']:
            #reward += 1
            #print('hit hy')
            # self.target_hit = True
            # Create state file for successful mission
            # self.write_state()
            hit_Number.append("hy")
            #self.one_hit_per_episode += 1
            #self.total_reward += reward

        # if(len(hit_Number) == 2):
        #     self.hit_2 = True
        # if(len(hit_Number) == 3):
        #     self.hit_3 = True
        # if(len(hit_Number) == 2):
        #     self.hit_2 = True
        
        if (len(hit_Number) > 4):
            reward += len(hit_Number) * 100000
            
        
            print(hit_Number)
            self.total_reward += reward
        elif(len(hit_Number) == 1):
            print("hit one", hit_Number)
            if self.hit_multiple is True:
                reward -= 1
            # if(hit_Number[0] != "a"):
            #     self.one_hit_per_episode += 1
            self.total_reward += reward
        else: 
            self.total_reward -= 1
            
            


        
                
        # Major axis critique 
        if state[0] < self.r_target_state[0]:
            reward -= 100
            self.total_reward += reward
            # print("Applying penalty for smaller a")

        if abs(self.r_target_state[0] - state[0]) <= self._orbit_tolerance['a']:
            reward += 1000
            self.total_reward += reward
            print("Applying reward for better a")

        if (self.r_target_state[0]) < state[0]:
            reward -= 100
            self.total_reward += reward
            # print("Applying penalty for bigger a")
        

        if (2 * self.r_target_state[0]) < state[0]:
            reward -= 10000000
            self.total_reward += reward
            done = True
            print("Shot out into space")
            return reward, done
        
        # if self.one_hit_per_episode >= 10:
        #     penalty = self.one_hit_per_episode / 10
        #     penalty *= -10
        #     self.total_reward += penalty
            # print(penalty)
            # print("Applying penalty for only hitting 1")
            # print(self.total_reward)



        # Out of fuel
        if self.curr_fuel_mass <= 0:
            print('Ran out of fuel')
            done = True
            reward += -10
            self.total_reward += reward
            return reward, done




        # Crash into Earth
        if self._currentOrbit.getA() < EARTH_RADIUS:
            reward += -10000000
            done = True
            print('In earth')
            self.total_reward += reward
            return reward, done


        # Mission duration exceeded
        if self._extrap_Date.compareTo(self.final_date) >= 0:
            reward += -1000
            print("Out of time")
            # self.write_state() DEBUG
            done = True
            return reward, done


        return reward, done

    

    # State/Action Output files

    def write_state(self):
        # State file (Equinoctial)
        with open("results/state/"+str(self.id)+"_"+self.alg+"_state_equinoctial_"+str(self.episode_num)+".txt", "w") as f:
            #Add episode number line 1
            f.write("Episode: " + str(self.episode_num) + '\n')
            for i in range(len(self.a_orbit)):
                try:
                    f.write(str(self.a_orbit[i]/1e3)+","+str(self.ex_orbit[i])+","+str(self.ey_orbit[i])+","+str(self.hx_orbit[i])+","+ \
                        str(self.hy_orbit[i])+","+str(self.lv_orbit[i])+","+str(self.px[i]/1e3)+","+str(self.py[i]/1e3)+","+str(self.pz[i]/1e3)+'\n')
                except Exception as err:
                    print(err)
                    print("Unexpected error")
                    print("Writing '-' in place 1")
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
                    print("Unexpected error 2", err)
                    # f.write('-\n')

        # Action file
        with open("results/action/"+str(self.id)+"_"+self.alg+"_action_"+str(self.episode_num)+".txt", 'w') as f:
            f.write("Fuel Mass: " + str(self.curr_fuel_mass) + "/" + str(self.fuel_mass) + '\n')
            for i in range(len(self.actions)):
                for j in range(3):
                    try:
                        f.write(str(self.actions[i][j])+",")
                    except Exception as err:
                        print(err)
                        print("Unexpected error 3")
                        print("Writing '-' in place")
                        f.write('-\n')
                f.write(str(self.thrust_mags[i])+'\n')
      

    def write_reward(self):
        with open("results/reward/"+str(self.id)+"_"+self.alg+"_reward"+".txt", "w") as f:
            for reward in self.episode_reward:
                f.write(str(reward)+'\n')
                
    def write_episode_stats(self):
        with open('results/episode_stats/' + str(self.id) + "_" + self.alg + ".csv", "a") as f:
            f.write(str(self.episode_num) + ',' + str(self.total_reward) + ',' + str(self.curr_fuel_mass) + ',' + str(self.curr_dist) + '\n')
