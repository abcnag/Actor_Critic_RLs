import gym
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical

device = 'cpu'
sb.set()

class Actor_Critic_Nets(nn.Module):
  def __init__(self, observation_space, action_space):
    super().__init__()
    #---#
    self.shared_layers = nn.Sequential(
        nn.Linear(observation_space, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU()
    )
    #---#
    self.policy_layers = nn.Sequential(
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, action_space)
    )
    #---#
    self.value_layers = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

  def value(self, obs):
    z = self.shared_layers(obs)
    values = self.value_layers(z)
    return values
  
  def policy(self, obs):
    z = self.shared_layers(obs)
    policy_logits = self.policy_layers(z)
    return policy_logits

  def forward(self, obs):
    z = self.shared_layers(obs)
    values = self.value_layers(z)
    policy_logits = self.policy_layers(z)
    return policy_logits, values

class PPO:
  def __init__(self,
               actor_critic,
               ppo_clip=0.2,
               target_kl=0.01,
               max_actor_iter=80,
               max_critic_iter=80,
               actor_lr=3e-4,
               critic_lr=1e-2):
    self.ac = actor_critic
    self.ppo_clip = ppo_clip
    self.target_kl = target_kl
    self.max_actor = max_actor_iter
    self.max_critic = max_critic_iter
    #---#
    policy_params = list(self.ac.shared_layers.parameters()) + list(self.ac.policy_layers.parameters())
    self.policy_optim = optim.Adam(policy_params, lr=actor_lr)
    value_params = list(self.ac.shared_layers.parameters()) + list(self.ac.value_layers.parameters())
    self.value_optim = optim.Adam(value_params, lr=critic_lr)
    #---#
  def train_policy(self, obs, act, old_log_probs, gaes):
   for _ in range(self.max_actor):
    self.policy_optim.zero_grad()
    new_logits = Categorical(logits=self.ac.policy(obs))
    new_log_probs = new_logits.log_prob(act)
    policy_ratio = torch.exp(new_log_probs - old_log_probs) 
    clipped_ratio = policy_ratio.clamp(1 - self.ppo_clip, 1 + self.ppo_clip)
    #---#
    clipped_loss = clipped_ratio * gaes
    full_loss = policy_ratio * gaes
    policy_loss = -torch.min(full_loss, clipped_loss).mean()
    #---#
    policy_loss.backward()
    self.policy_optim.step()
    #---#
    # kl_dive = (old_log_probs - new_log_probs).mean()
    # if kl_dive >= self.target_kl:
    #   break
    
  def train_value(self, obs, returns):
    for _ in range(self.max_critic):
      self.value_optim.zero_grad()
      values = self.ac.value(obs)
      value_loss = ((returns - values)**2).mean()
      #---#
      value_loss.backward()
      self.value_optim.step()

def discount_rewards(reward, gamma=0.99):
  new_rewards = [float(reward[-1])]
  for i in reversed(range(len(reward)-1)):
    new_rewards.append(float(reward[i] + gamma * new_rewards[-1]))
  return np.array(new_rewards[::-1])
##################
def caculate_gaes(rewards, values, gamma=0.99, decay=0.97):
  next_values = np.concatenate([values[1:],[0]])
  #---#
  delta = [rew +  gamma * next_val - val for rew,val,next_val in zip(rewards, values, next_values)]
  #---#
  new_gaes = [float(delta[-1])]
  for i in reversed(range(len(delta)-1)):
    new_gaes.append(float(delta[i] + gamma * decay * new_gaes[-1]))
  return np.array(new_gaes[::-1])

def rollout(model, env, max_steps=1000):
  train_data = [[], [], [], [], []] # obs, act, rew, val, log_prob
  obs = env.reset()
  episode_reward = 0
  #---#
  for ep in range(max_steps):
    logits, vals = model(torch.tensor([obs], dtype=torch.float32, device=device))
    actions = Categorical(logits=logits)
    action = actions.sample()
    action_logprob = actions.log_prob(action).item()
    action,vals = action.item(), vals.item()
    #---#
    next_obs, reward, done, _ = env.step(action)
    #---#
    for i,item in enumerate((obs, action, reward, vals, action_logprob)):
      train_data[i].append(item)
    #---#
    obs = next_obs
    episode_reward += reward
    #---#
    if done:
      break
  train_data = [np.asarray(x) for x in train_data]
  train_data[3] = caculate_gaes(train_data[2], train_data[3])
  return train_data, episode_reward

env = gym.make('CartPole-v0')
model = Actor_Critic_Nets(env.observation_space.shape[0], env.action_space.n).to(device)
train_data, ep_reward = rollout(model, env)
#---#
n_episodes = 20000
print_freq = 100
#---#
ppo = PPO(model,
          ppo_clip=0.2,
          target_kl=0.02,
          max_actor_iter=100,
          max_critic_iter=100,
          actor_lr=0.001,
          critic_lr=0.01)

ep_rewards = []
for ep in range(n_episodes):
  train_data, reward = rollout(model, env)
  ep_rewards.append(reward)
  #---#
  perm = np.random.permutation(len(train_data[0]))
  #---#
  obs = torch.tensor(train_data[0][perm], dtype=torch.float32, device=device)
  act = torch.tensor(train_data[1][perm], dtype=torch.int32, device=device)
  gaes = torch.tensor(train_data[3][perm], dtype=torch.float32, device=device) 
  act_logprob = torch.tensor(train_data[4][perm], dtype=torch.float32, device=device)
  #---#
  returns = discount_rewards(train_data[2])[perm]
  returns = torch.tensor(returns, dtype=torch.float32, device=device)
  #---#
  ppo.train_policy(obs, act, act_logprob, gaes)
  ppo.train_value(obs, returns)
  #---#
  if (ep+1) % print_freq == 0:
    print(f'reward avg 200 /ep{ep+1} -----> {np.mean(ep_rewards[-print_freq:])}')
    print(f'reward avg all /ep{ep+1} -----> {np.mean(ep_rewards)}\n-------')
