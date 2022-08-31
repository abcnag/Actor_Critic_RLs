import numpy as np
import os
import tensorflow as tf
import keras
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

class Replay_Buffer:
  def __init__(self, max_size, input_shape, n_actions):
    self.mem_size = max_size
    self.mem_counter = 0
    #---#
    self.state_memory = np.zeros((self.mem_size, *input_shape))
    self.new_state_memory = np.zeros((self.mem_size, *input_shape))
    self.action_memory = np.zeros((self.mem_size, n_actions))
    self.reward_memory = np.zeros(self.mem_size)
    self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
  
  def store_transition(self, state, action, reward, state_, done):
    index = self.mem_counter % self.mem_size
    #---#
    self.state_memory[index] = state
    self.new_state_memory[index] = state_
    self.action_memory[index] = action
    self.reward_memory[index] = reward
    self.terminal_memory[index] = done
    self.mem_counter += 1
  
  def sample_buffer(self, batch_size):
    max_memory = min(self.mem_counter, self.mem_size)
    batch = np.random.choice(max_memory, batch_size)
    #---#
    states = self.state_memory[batch]
    states_ = self.new_state_memory[batch]
    actions = self.action_memory[batch]
    rewards = self.reward_memory[batch]
    dones = self.terminal_memory[batch]
    #---#
    return states, actions, rewards, states_, dones

class CriticNetwork(keras.Model):
  def __init__(self, n_actions, fc1_dims=256, fc2_dims=256, chkp='tmp/sac', name='critic'):
    super(CriticNetwork, self).__init__()
    #---#
    self.fc1d = fc1_dims
    self.fc2d = fc2_dims
    self.n_actions = n_actions
    self.model_name = name
    self.chkp_dir = chkp
    self.chkp_file = os.path.join(self.chkp_dir, name+'_sac')
    #---#
    self.fc1 = Dense(self.fc1d, activation='relu')
    self.fc2 = Dense(self.fc2d, activation='relu')
    self.q = Dense(1, activation=None)
  
  def call(self, state, action):
    act_val = self.fc1(tf.concat([state, action], axis=1))
    act_val = self.fc2(act_val)
    q = self.q(act_val)
    #---#
    return q

class ValueNetwork(keras.Model):
  def __init__(self, fc1_dims=256, fc2_dims=256, chkp='tmp/sac', name='value'):
    super(ValueNetwork, self).__init__()
    #---#
    self.fc1d = fc1_dims
    self.fc2d = fc2_dims
    self.model_name = name
    self.chkp_dir = chkp
    self.chkp_file = os.path.join(self.chkp_dir, name+'_sac')
    #---#
    self.fc1 = Dense(self.fc1d, activation='relu')
    self.fc2 = Dense(self.fc2d, activation='relu')
    self.v = Dense(1, activation=None)
  
  def call(self, state):
    state_val = self.fc1(state)
    state_val = self.fc2(state_val)
    v = self.v(state_val)
    #---#
    return v

class ActorNetwork(keras.Model):
  def __init__(self, max_action, n_actions=2, fc1_dims=256, fc2_dims=256, chkp='tmp/sac', name='value'):
    super(ActorNetwork, self).__init__()
    #---#
    self.fc1d = fc1_dims
    self.fc2d = fc2_dims
    self.model_name = name
    self.chkp_dir = chkp
    self.chkp_file = os.path.join(self.chkp_dir, name+'_sac')
    self.max_action = max_action
    self.n_actions = n_actions
    self.noise = 1e-6
    #---#
    self.fc1 = Dense(self.fc1d, activation='relu')
    self.fc2 = Dense(self.fc2d, activation='relu')
    self.mu = Dense(self.n_actions, activation=None)
    self.sigma = Dense(self.n_actions, activation=None)
  
  def call(self, state):
    prob = self.fc1(state)
    prob = self.fc2(prob)
    mu = self.mu(prob)
    sigma = self.sigma(prob)
    sigma = tf.clip_by_value(sigma, self.noise, 1)
    #---#
    return mu, sigma
  
  def sample_normal(self, state):
    mu, sigma = self.call(state)
    probs = tfp.distributions.Normal(mu, sigma)
    action = probs.sample()
    action = tf.math.tanh(action) * self.max_action
    log_prob = probs.log_prob(action)
    log_prob -= tf.math.log(1-tf.math.pow(action, 2) + self.noise)
    log_prob = tf.math.reduce_sum(log_prob, axis=1, keepdims=True)
    #---#
    return action, log_prob


class Agent:
  def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8], env=None, gamma=0.99, n_actions=2,
               tau=0.005, max_size=100000, layer1=256, layer2=256, batch_size=256, reward_scale=2):
    self.gamma = gamma
    self.tau = tau
    self.n = n_actions
    self.batch_size = batch_size
    self.memory = Replay_Buffer(max_size, input_dims, self.n)
    #---#
    self.actor = ActorNetwork(env.action_space.high, self.n)
    self.critic1 = CriticNetwork(self.n)
    self.critic2 = CriticNetwork(self.n)
    self.value = ValueNetwork()
    self.target_value = ValueNetwork()
    #---#
    self.actor.compile(optimizer=Adam(learning_rate=alpha))
    self.critic1.compile(optimizer=Adam(learning_rate=beta))
    self.critic2.compile(optimizer=Adam(learning_rate=beta))
    self.value.compile(optimizer=Adam(learning_rate=beta))
    self.target_value.compile(optimizer=Adam(learning_rate=beta))
    #---#
    self.scale = reward_scale
    self.update_net_params(tau=1)
  
  def choice_action(self, obs):
    state = tf.convert_to_tensor([obs])
    act, _ = self.actor.sample_normal(state)
    #---#
    return act[0]
  
  def remember(self, state, act, rew, news, done):
    self.memory.store_transition(state, act, rew, news, done)
  
  def update_net_params(self, tau=None):
    if tau is None:
      tau = self.tau
    #---#
    weights = []
    targets = self.target_value.weights
    for i, weight in enumerate(self.value.weights):
      weights.append(weight*tau + targets[i]*(1-tau))
    self.target_value.set_weights(weights)
  
  def save_model(self):
    print('...save model...')
    self.actor.save_weights(self.actor.chkp_file)
    self.critic1.save_weights(self.critic1.chkp_file)
    self.critic2.save_weights(self.critic2.chkp_file)
    self.value.save_weights(self.value.chkp_file)
    self.target_value.save_weights(self.target_value.chkp_file)

  def load_model(self):
    print('...save model...')
    self.actor.load_weights(self.actor.chkp_file)
    self.critic1.load_weights(self.critic1.chkp_file)
    self.critic2.load_weights(self.critic2.chkp_file)
    self.value.load_weights(self.value.chkp_file)
    self.target_value.load_weights(self.target_value.chkp_file)

  def learn(self):
    if self.memory.mem_size < self.batch_size:
      return
    #---#
    state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
    states = tf.convert_to_tensor(state, dtype=tf.float32)
    new_states = tf.convert_to_tensor(new_state, dtype=tf.float32)
    rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
    actions = tf.convert_to_tensor(action, dtype=tf.float32)
    #---#
    with tf.GradientTape() as g:
      value = tf.squeeze(self.value(states), 1)
      target_value = tf.squeeze(self.target_value(new_states), 1)
      current_action, log_probs = self.actor.sample_normal(states)
      log_probs = tf.squeeze(log_probs, 1)
      q1 = self.critic1(states, current_action)
      q2 = self.critic2(states, current_action)
      critic_val = tf.squeeze(tf.math.minimum(q1,q2), 1) # Q min val
      #---#
      value_target = critic_val - log_probs  
      value_loss = 0.5 * keras.losses.MSE(value, value_target)
    value_gradient = g.gradient(value_loss, self.value.trainable_variables)
    self.value.optimizer.apply_gradients(zip(value_gradient, self.value.trainable_variables))
    #---#
    with tf.GradientTape() as g:
      new_act, log_probs = self.actor.sample_normal(states)
      log_probs = tf.squeeze(log_probs, 1)
      q1 = self.critic1(states, current_action)
      q2 = self.critic2(states, current_action)
      critic_val = tf.squeeze(tf.math.minimum(q1,q2), 1) # Q min val
      #---#
      actor_loss = tf.reduce_mean(log_probs - critic_val)
    actor_gradient = g.gradient(actor_loss, self.actor.trainable_variables)
    self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))
    #---#
    with tf.GradientTape(persistent=True) as g:
      q_hat = self.scale*reward + self.gamma*target_value*(1-done)
      q1old = self.critic1(states, current_action)
      q2old = self.critic2(states, current_action)
      #---#
      c1_loss = 0.5 * keras.losses.MSE(q1old, q_hat)
      c2_loss = 0.5 * keras.losses.MSE(q2old, q_hat)
    c1_gradient = g.gradient(c1_loss, self.critic1.trainable_variables)
    self.critic1.optimizer.apply_gradients(zip(c1_gradient, self.critic1.trainable_variables))
    c2_gradient = g.gradient(c2_loss, self.critic2.trainable_variables)
    self.critic2.optimizer.apply_gradients(zip(c2_gradient, self.critic2.trainable_variables))
    #---#
    self.update_net_params()


!pip install pybullet
import pybullet_envs
import gym

env = gym.make('InvertedPendulumBulletEnv-v0')
agent = Agent(input_dims=env.observation_space.shape, n_actions=env.action_space.shape[0], env=env)
n_games = 250
filename = 'env.png'
best_score = env.reward_range[0]
score_history = []
load_chkp = False
#---#
if load_chkp:
  agent.load_model()
  env.render('human')
#---#
for i in range(n_games):
  obs = env.reset()
  done = False
  score = 0
  while not done:
    act = agent.choice_action(obs)
    obs_, reward, done, _ = env.step(act)
    score += reward
    agent.remember(obs, act, reward, obs_, done)
    #---#
    if not load_chkp:
      agent.learn()
    obs = obs_
  score_history.append(score)
  avg_score = np.mean(score_history[-100:])
  #---#
  if avg_score > best_score:
    best_score = avg_score
    if not load_chkp:
      agent.save_model()
  #---#
  print(f'episode {i+1} --> avg = {avg_score} \\ score = {score}')
