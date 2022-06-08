from WOreactorEnv import *
from extWOreactorEnv import *
from minWOreactorEnv import *

#Define name for files
filename = "RLRTO_DDPG"

#Set parameters
T = 5000 #total number of training time steps
runs = 10
actionDev = 0.09537452347834259 #std dev to action noise


#Define the environment
env = WOreactorEnv()     #Full state
#env = minWOreactorEnv() #Minimal state 
#env = extWOreactorEnv() #Extended state

# The action noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma= actionDev  * np.ones(n_actions))



for l in range(runs): #for averageing
    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps = T,log_interval=10, n_eval_episodes=2)
    

    # To store reward history of each episode
    ep_reward_list = []
    obs = env.reset()
    prevListLength = 0

    for j in range(10):
        #action, _states = model.predict(obs)
        action, _states = model.predict(obs,deterministic= True) #for SAC
        obs, rewards, dones, info = env.step(action)
        
        if len(env.FbList)>prevListLength:
            ep_reward_list.append(rewards)
            FbList = env.FbList
            TList = env.TList
        prevListLength = len(FbList)

    #deviation from optimum
    Fbdev = abs(FbList[-1]-4.3894)
    Tdev = abs(TList[-1]-80.4948)
    #FEIL vil bare ha siste deviation ikke alle variablene
    
    with open("Fb"+filename+".txt","a") as f:
        f.write(str(FbList)+"\n")
    with open("T"+filename+".txt","a") as f:
        f.write(str(TList)+"\n")