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
from org.orekit.forces.maneuvers import ConstantThrustManeuver
from org.orekit.frames import LOFType
from org.orekit.attitudes import LofOffset
from orekit.pyhelpers import setup_orekit_curdir  
from org.orekit.forces.gravity import NewtonianAttraction
from org.orekit.utils import Constants

from org.hipparchus.geometry.euclidean.threed import Vector3D
from orekit import JArray_double

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

    def __init__(self, state, state_targ, date, duration, mass, stepT):
        """
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
        self.n_hits = 0
        self.initial_dist = 0
        # state params
        self.px = []
        self.py = []
        self.pz = []
        self.a_orbit = []
        self.ex_orbit = []
        self.ey_orbit = []
        self.hx_orbit = []
        self.hy_orbit = []
        self.lv_orbit = []

        # Kepler
        self.e_orbit = []
        self.i_orbit = []
        self.w_orbit = []
        self.omega_orbit = []
        self.v_orbit = []

        self.adot_orbit = []
        self.exdot_orbit = []
        self.eydot_orbit = []
        self.hxdot_orbit = []
        self.hydot_orbit = []

        self._sc_fuel = None
        self._extrap_Date = None
        self._targetOrbit = None
        self.target_px = []
        self.target_py = []
        self.target_pz = []

        self.total_reward = 0
        self.episode_num = 0
        # List of rewards/episode over entire lifetime of env instance
        self.episode_reward = []
        #List of actions and thrust magnitudes
        self.actions = []
        self.thrust_mags = []

        # Fuel params
        self.dry_mass = mass[0]
        self.fuel_mass = mass[1]
        self.cuf_fuel_mass = self.fuel_mass
        self.initial_mass = self.dry_mass + self.fuel_mass

        # Accpetance tolerance
        self._orbit_tolerance = {'a': 10000, 'ex': 0.01, 'ey': 0.01, 'hx': 0.001, 'hy': 0.001, 'lv': 0.01}

        # Allows for randomized reset
        self.randomize = False
        self._orbit_randomizer = {'a': 4000.0e3, 'e': 0.2, 'i': 2.0, 'w': 10.0, 'omega': 10.0, 'lv': 5.0}
        self.seed_state = state
        self.seed_target = state_targ
        self.target_hit = False
        


        # Environment initialization
        
        self.curr_fuel_mass = self.fuel_mass
        self.set_date(date)
        self._extrap_Date = self._initial_date
        self.create_orbit(state, self._initial_date)

        self.final_date = self._initial_date.shiftedBy(duration)
        self.create_orbit(state_targ, self.final_date)

        self.stepT = stepT
        
        self.curr_dist = 0
        
        self._targetOrbit= self.create_orbit(state_targ, self.final_date)
        
        
                
        self.initial_orbit = self.create_orbit(state, self._initial_date)
        self.r_target_state = self.get_state(self._targetOrbit)
        self.r_initial_state = self.get_state(self.initial_orbit)
        
        self.set_spacecraft(self.initial_mass, self.curr_fuel_mass)
        self.create_Propagator()
        self.setForceModel()

        # self.action_space = 3  # output thrust directions
        # OpenAI API to define 3D continous action space vector [a,b,c]
        self.action_space = spaces.Box(
            low=-0.6,
            high=0.6,
            shape=(3,),
            dtype=np.float32
        )
        # self.observation_space = 10  # states | Equinoctial components + derivatives
        # OpenAI API
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,shape=(10,),
                                        dtype=np.float32)
        self.action_bound = 0.6  # Max thrust limit
        self._isp = 3100.0 


        self.r_target_state = self.get_state(self._targetOrbit)
        self.r_initial_state = self.get_state(self.initial_orbit)

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
        self._sc_fuel = sc_state.addAdditionalState (FUEL_MASS, fuel_mass)

    def create_Propagator(self):
        """
        Creates and initializes the propagator
        :return:
        """
        minStep = 0.001
        maxStep = 500.0

        position_tolerance = 60.0
        tolerances = NumericalPropagator.tolerances(position_tolerance, self.initial_orbit, self.initial_orbit.getType())
        abs_tolerance = JArray_double.cast_(tolerances[0])
        rel_telerance = JArray_double.cast_(tolerances[1])

        integrator = DormandPrince853Integrator(minStep, maxStep, abs_tolerance, rel_telerance)

        integrator.setInitialStepSize(10.0)

        numProp = NumericalPropagator(integrator)
        numProp.setInitialState(self._sc_fuel)
        numProp.setMu(MU)
        numProp.setOrbitType(OrbitType.KEPLERIAN)

        # numProp.setSlaveMode() # was not commented out before

        self._prop = numProp
        self._prop.setAttitudeProvider(attitude)


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
        
        
    def get_state_keplerian(self, orbit):
        orbit = self.convert_to_keplerian(orbit)
        target = self.convert_to_keplerian(self._targetOrbit)
        return [ orbit.getA(), orbit.getE(), orbit.getI(), orbit.getPerigeeArgument(), orbit.getRightAscensionOfAscendingNode(), orbit.getTrueAnomaly(),
                 orbit.getADot(), orbit.getEDot(), orbit.getIDot(), orbit.getPerigeeArgumentDot(), orbit.getRightAscensionOfAscendingNodeDot(), orbit.getTrueAnomalyDot(),
                 target.getA(), target.getE(), target.getI(), target.getPerigeeArgument(), target.getRightAscensionOfAscendingNode(),
                 self.curr_fuel_mass]

    def reset(self):
        """
        Resets the orekit enviornment
        :return:
        """

        self._prop = None
        self._currentDate = None
        self._currentOrbit = None
        
        
        print("RESET")
        # print('Total Reward:', self.total_reward)
        # print(f'Fuel Remaining: {self.curr_fuel_mass}/{self.fuel_mass}')
        # print('Actions:', len(self.actions))

        self.write_episode_stats()
        
        

        # Randomizes the initial orbit
        if self.randomize:
            self._orbit = None
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
            self.create_orbit(state, self._initial_date)
        else:
            self._currentOrbit = self.initial_orbit

        self._currentDate = self._initial_date
        self._extrap_Date = self._initial_date

        self.set_spacecraft(self.initial_mass, self.fuel_mass)
        self.cuf_fuel_mass = self.fuel_mass
        self.curr_fuel_mass = self.fuel_mass
        self.create_Propagator()
        self.setForceModel()

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
        
        print(self.id)
        
        return state

    @property
    def getTotalMass(self):
        """
        Get the total mass of the spacecraft
        :return: dry mass + fuel mass (kg)
        """
        return self._sc_fuel.getAdditionalState(FUEL_MASS)[0] + self._sc_fuel.getMass()

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

    def step(self, thrust):
        """
        Take a propagation step
        :param thrust: 3D Thrust vector (Newtons, float)
        :return: spacecraft state (np.array), reward value (float), done (bool)
        """
        
        self._prevOrbit = self._currentOrbit
        thrust_mag = np.linalg.norm(thrust)
        thrust_dir = thrust / thrust_mag
        DIRECTION = Vector3D(float(thrust_dir[0]), float(thrust_dir[1]), float(thrust_dir[2]))
        # print(thrust)
        if thrust_mag <= 0:
            # DIRECTION = Vector3D.MINUS_J
            thrust_mag = abs(float(thrust_mag))
        else:
            # DIRECTION = Vector3D.PLUS_J
            thrust_mag = float(thrust_mag)

        # Add force model
        thrust_force = ConstantThrustManeuver(self._extrap_Date, self.stepT, thrust_mag, self._isp, attitude, DIRECTION)
        self._prop.addForceModel(thrust_force)
        # Propagate
        currentState = self._prop.propagate(self._extrap_Date.shiftedBy(self.stepT))

        self.cuf_fuel_mass = currentState.getMass() - self.dry_mass
        self._currentDate = currentState.getDate()
        self._extrap_Date = self._currentDate
        self._currentOrbit = currentState.getOrbit()
        coord = currentState.getPVCoordinates().getPosition()

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

        k_orbit = self.convert_to_keplerian(self._currentOrbit)

        self.e_orbit.append(k_orbit.getE())
        self.i_orbit.append(k_orbit.getI())
        self.w_orbit.append(k_orbit.getPerigeeArgument())
        self.omega_orbit.append(k_orbit.getRightAscensionOfAscendingNode())
        self.v_orbit.append(k_orbit.getTrueAnomaly())

        self.actions.append(thrust)
        self.thrust_mags.append(thrust_mag)

        # Calc reward / termination state for this step
        reward, done = self.dist_reward() # was self.dist_reward(thrust)

        state_1 = [(self._currentOrbit.getA()) / self.r_target_state[0],
                   self._currentOrbit.getEquinoctialEx(), self._currentOrbit.getEquinoctialEy(),
                   self._currentOrbit.getHx(), self._currentOrbit.getHy(),
                   self._currentOrbit.getADot(), self._currentOrbit.getEquinoctialExDot(),
                   self._currentOrbit.getEquinoctialEyDot(),
                   self._currentOrbit.getHxDot(), self._currentOrbit.getHyDot()
                   ]
        # OpenAI debug option
        info = {}

        self.adot_orbit.append(self._currentOrbit.getADot())
        self.exdot_orbit.append(self._currentOrbit.getEquinoctialExDot())
        self.eydot_orbit.append(self._currentOrbit.getEquinoctialEyDot())
        self.hxdot_orbit.append(self._currentOrbit.getHxDot())
        self.hydot_orbit.append(self._currentOrbit.getHyDot())

        return np.array(state_1), reward, done, info

    def dist_reward(self):
        """
        Computes the reward based on the state of the agent
        :return: reward value (float), done state (bool)
        """
        # a, ecc, i, w, omega, E, adot, edot, idot, wdot, omegadot, Edot = state

        done = False
        
        
        
        curr_state = self.get_state(self._currentOrbit)
        curr_velocity = curr_state[6:12]
        prev_state = self.get_state(self._prevOrbit, with_derivatives=False)
        initial_state = self.get_state(self.initial_orbit)
        
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

        

        state = np.array([self._currentOrbit.getA(), self._currentOrbit.getEquinoctialEx(), self._currentOrbit.getEquinoctialEy(),
                          self._currentOrbit.getHx(), self._currentOrbit.getHy(), self._currentOrbit.getLv()])

        reward_a = np.sqrt((self.r_target_state[0] - state[0])**2) / self.r_target_state[0]
        reward_ex = np.sqrt((self.r_target_state[1] - state[1])**2)
        reward_ey = np.sqrt((self.r_target_state[2] - state[2])**2)
        reward_hx = np.sqrt((self.r_target_state[3] - state[3])**2)
        reward_hy = np.sqrt((self.r_target_state[4] - state[4])**2)

        reward = -(reward_a + reward_hx*10 + reward_hy*10 + reward_ex + reward_ey)

        # TERMINAL STATES
        # Target state (with tolerance)
        if abs(self.r_target_state[0] - state[0]) <= self._orbit_tolerance['a'] and \
           abs(self.r_target_state[1] - state[1]) <= self._orbit_tolerance['ex'] and \
           abs(self.r_target_state[2] - state[2]) <= self._orbit_tolerance['ey'] and \
           abs(self.r_target_state[3] - state[3]) <= self._orbit_tolerance['hx'] and \
           abs(self.r_target_state[4] - state[4]) <= self._orbit_tolerance['hy']:
            reward += 1
            done = True
            print('hit')
            self.target_hit = True
            # Create state file for successful mission
            self.n_hits += 1
            self.write_state()
            return reward, done

        # Out of fuel
        if self.cuf_fuel_mass <= 0:
            print('Ran out of fuel')
            done = True
            reward = -1
            return reward, done

        # Crash into Earth
        if self._currentOrbit.getA() < EARTH_RADIUS:
            reward = -1
            done = True
            print('In earth')
            return reward, done

        # Mission duration exceeded
        if self._extrap_Date.compareTo(self.final_date) >= 0:
            reward = -1
            print("Out of time")
            # self.write_state() DEBUG
            done = True

        self.total_reward += reward
        
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
            f.write("Fuel Mass: " + str(self.cuf_fuel_mass) + "/" + str(self.fuel_mass) + '\n')
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
