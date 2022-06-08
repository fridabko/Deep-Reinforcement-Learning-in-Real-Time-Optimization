from WOreactorEnv import *
from stable_baselines3 import SAC

#Set parameters
T = 5000 #total number of training time steps
actionDev = 0.09537452347834259 #std dev to action noise

#Define the environment
env = WOreactorEnv()

#Check that the environment is sufficient
check_env(env)



obs = env.reset()

# The action noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma= actionDev  * np.ones(n_actions))



#TRAINING AGENT:
#MLP is basic feedforward network
model = SAC("MlpPolicy", env, action_noise=action_noise, verbose=1)

model.learn(total_timesteps = T,log_interval=10, n_eval_episodes=2)


# To store reward history of each episode
ep_reward_list = []


obs = env.reset()

for i in range(10):
  action, _states = model.predict(obs)
  obs, rewards, dones, info = env.step(action)
  ep_reward_list.append(rewards)
  env.render()
#True optimum:(4.3894),(80.4948)