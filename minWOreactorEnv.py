import gym
from gym import spaces
import numpy as np
from casadi import *
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise
from scipy.optimize import fsolve
from stable_baselines3 import DDPG
import random

noise = 0.00008 #measurement noise
  

def normAction(u):
    Fb = (u[0]-5.5)/1.5
    Tr = (u[1]-85)/15
    return np.array([Fb,Tr], dtype=np.float32)

def denormAction(u):
    Fb = (u[0])*1.5+5.5
    Tr = (u[1])*15+85
    return np.array([Fb,Tr], dtype=np.float32)

def normObs(x):
    normX = []
    for i in x:
        normX.append(i*2-1)
    return np.array(normX, dtype=np.float32)

def denormObs(x):
    normX = []
    for i in x:
        normX.append((i+1)/2)
    return np.array(normX, dtype=np.float32)

def normXmin(x):
    normX = []
    for i in range(len(x)):
        if i==0:
            normX.append((x[i]+75)/175)
        elif i==1:
            normX.append((x[i]-0.38)/0.5)
        elif i==2:
            normX.append((x[i]-0.42)/0.5)
    return np.array(normX, dtype=np.float32)

def denormXmin(x):
    normX = []
    for i in range(len(x)):
        if i==0:
            normX.append((x[i]*175-75))
        elif i==1:
            normX.append((x[i]*0.5+0.38))
        elif i==2:
            normX.append((x[i]*0.5+0.42))
    return np.array(normX, dtype=np.float32)


def Js(u,X):
  return (1043.38 * X[4] * (1.8275 + u[0]) +
              20.92 * X[3] * (1.8275 + u[0]) - 
              79.23 * 1.8275 -
              118.34 * u[0]) #scale reward to around 1

def root(x,u):
  # Define constants:
  Fa = 1.8275
  Mt = 2105.2

  # Define vectors with names of states system:
  Fb = u[0]
  Tr = u[1]

  # Kinetic rate constants for each reaction
  k1s = 1.6599e6 * exp(-6666.7 / (Tr + 273.15))
  k2s = 7.2117e8 * exp(-8333.3 / (Tr + 273.15))
  k3s = 2.6745e12 * exp(-11111. / (Tr + 273.15))

  # Total mass flow
  Fr = Fa + Fb

  # Reaction rates
  r1s = k1s * x[0] * x[1] * Mt
  r2s = k2s * x[1] * x[2] * Mt
  r3s = k3s * x[2] * x[4] * Mt

  # Model equations
  F = np.zeros(6)
  F[0] = (Fa - r1s - Fr * x[0]) / Mt
  F[1] = (Fb - r1s - r2s - Fr * x[1])/Mt
  F[2] = (+ 2 * r1s - 2 * r2s - r3s - Fr * x[2]) / Mt
  F[3] = (+ 2 * r2s - Fr * x[3]) / Mt 
  F[4] =  (+   r2s - 0.5 * r3s - Fr * x[4]) / Mt
  F[5] = (+ 1.5 * r3s - Fr * x[5]) / Mt

  return F

def Xf(uk):
    x0 = [0,0,0,0,0,0]
    args = (uk)
    states = fsolve(lambda x: root(x,uk), x0)
    return states.reshape(6,1)


def G1(xa):
    return (xa-0.12)

def G2(xg):
    return (xg-0.08)


class minWOreactorEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, maxSteps=100):
    super(minWOreactorEnv, self).__init__()
    #Initialize parameters
    self.maxSteps = maxSteps
    # Define action and observation space
    # They must be gym.spaces objects
    
    #Actions possible: manipulate the inputs FB and TR
    #FB can range from 4 to 7 kg/s
    #TR can range from 70 to 100 kg/s

    lowAction = np.array([-1, -1], dtype=np.float32)
    highAction = np.array([1, 1], dtype=np.float32)
    
    self.action_space = spaces.Box(low=lowAction, high=highAction, dtype=np.float32)
    
    #Observations possible: all measured parameters, which means all mass fractions
    # ['Xa', 'Xb', 'Xc', 'Xe', 'Xp', 'Xg']
    # NB: we have constraints on xa and xg, but we enforce these in the reward function
    highObs = np.array([1.0, 1.0, 1.0], dtype=np.float32) 
    self.observation_space = spaces.Box(low=-highObs, high=highObs, dtype=np.float32)

  def step(self, action):
    # Execute one time step within the environment
    # Here it equals evaluation at one iteration:
    # Given 1 action we calculate the next state
    # Return takes the form: state, reward, done, info = step(action)
    self.t += 1
    self.lastState = self._get_obs()
    dnAction = denormAction(action)
    stateVec = Xf(dnAction)
    

    
    ####ADD NOISE TO MEASUREMENTS############################
    for i in range(len(stateVec)):
      stateVec[i] += np.random.normal(0,noise, size=(1,1))[0][0]
      if stateVec[i]<0:
        stateVec[i]=0
      elif stateVec[i]>1:
        stateVec[i]=1
    
    
    #Reshape and append
    J = float((Js(dnAction,stateVec)))
    costs = J
    g1 = G1(stateVec[0][0])
    g2 = G2(stateVec[-1][0])

    minX = [J,g1,g2]
    nminX = normXmin(minX)
    self.state = nminX.reshape(3,) #get vector of all Xs
    self.FbList.append(dnAction[0])
    self.TList.append(dnAction[1])


    #CONSTRAINTS, LP terms
    if g1>0:
      costs -= 410*abs(stateVec[0][0]-0.12)

    if g2>0:
      costs -= 2000*abs(stateVec[-1][0]-0.08)

    
    percentChange = []
    for i in range(len(self.lastState)):
      pCi = (self.lastState[i]-self.state[i])/self.state[i]
      percentChange.append(bool(pCi<10e-5))
    done = bool(False not in percentChange)
    if self.t>=self.maxSteps:
        done = bool(self.t>=self.maxSteps)
    
    return self._get_obs(), costs, done, {}
    
  def reset(self):
    #Visualiztion: plots Fb(TR) for the simulation of the WO reactor
    u0F = random.uniform(5.5,7)
    u0T = random.uniform(75,86)
    u0 = np.array([u0F,u0T])
    a = Xf(u0)
    J = 0
    g1 = G1(a[0][0])
    g2 = G2(a[-1][0])
    minX = [J,g1,g2]
    a = normXmin(minX)
    self.t = 0
    self.state = a.reshape(3,)
    self.lastState = None
    self.FbList = [7]
    self.TList = [70]
    self.lastRew = 0
    return self._get_obs()
  
  def _get_obs(self):
    Xs = self.state
    return np.array(Xs, dtype=np.float32)

  def _get_time(self):
    return self.t

  def render(self, mode='human', close=False):

    #Visualiztion: plots Fb(TR) for the simulation of the WO reactor
    iter = np.linspace(0,20,21)
    G1_Fb = [4,4.125,4.3,4.39,4.45,4.5,4.75,5,5.25,5.36,5.5,5.75,6,6.1]
    G1_Tr = [84,82.65,81.2,80.5,80,79.5,77.6,76,74.5,73.9,73.2,71.85,70.5,70]
    a_BSpline = make_interp_spline(G1_Fb, G1_Tr)
    # 300 represents number of points to make between T.min and T.max
    G1_Fb_new = np.linspace(4, 6.1, 300) 

    spl = make_interp_spline(G1_Fb, G1_Tr, k=3)  # type: BSpline
    G1_Tr_smooth = spl(G1_Fb_new)

    G2_Fb = [4,4.3,4.39,4.5,4.7,5.05,5.25,5.5,7]
    G2_Tr = [77.7,80,80.5,81.5,82.5,85,86,88,98.5]
    G2_Fb = [4,7]
    G2_Tr = [77.605,98.5]

    #%%

    plt.plot(G1_Fb_new,G1_Tr_smooth,'r',label='$g_1$')
    plt.plot(G2_Fb,G2_Tr,'firebrick',label='$g_2$')
    plt.plot(self.FbList,self.TList,'--',color='tab:blue')
    plt.plot(self.FbList,self.TList,'*',label='$u_k$',color='tab:blue')
    print(self.FbList,self.TList)

    plt.grid(color='0.95')
    plt.legend()
    plt.title('Plot of input iterations')
    plt.xlabel("$F_b$ [kg/s]")
    plt.ylabel("$T_R$ [°C]")
    plt.axis([4, 7, 70, 100])

    
    plt.show()