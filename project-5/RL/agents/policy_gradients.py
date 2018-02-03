"""Policy search agent."""
import os
import signal
import sys
import pandas as pd
import numpy as np

from quad_controller_rl import util
from quad_controller_rl.agents.base_agent import BaseAgent

from quad_controller_rl.agents.actor import Actor
from quad_controller_rl.agents.critic import Critic
from quad_controller_rl.agents.memory import Memory
from quad_controller_rl.agents.ou_noise import OUNoise

class DDPG(BaseAgent):    
    def __init__(self, task):
        # debugger
        # import pdb; pdb.set_trace()

        # Task (environment) information
        self.task = task
        
        # constrain size
        self.state_size = 3 # np.prod(self.task.observation_space.shape) # position only (x,y,z),  self.task.observation_space.__dict__
        self.action_size = 3 # np.prod(self.task.action_space.shape) # force only (x,y,z), self.task.action_space.shape.__dict__

        # Actor (Policy) Model
        self.action_low =  self.task.action_space.low[0:3] # force 
        self.action_high = self.task.action_space.high[0:3] # force
        
        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.6 

        # Save episode stats
        self.stats_filename = os.path.join(util.get_param('out'),"stats_{}.csv".format(util.get_timestamp()))
        self.stats_columns = ['episode', 'total_reward']  # specify columns to save
        self.episode_num = 1        

        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.noise = OUNoise(self.action_size)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = Memory(self.buffer_size)

        # Algorithm parameters
        self.gamma = 0.95  # discount factor
        self.tau = 0.001  # for soft update of target parameters

        # Episode variables
        self.reset_episode_vars()
        signal.signal(signal.SIGINT, self.save_models_on_exit)

    def reset_episode_vars(self):
        # debugger
        # import pdb; pdb.set_trace()

        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0

    def step(self, state, reward, done):
        # debugger
        # import pdb; pdb.set_trace()

        state = self.preprocess_state(state)        

        # Choose an action        
        action = self.act(state)
        
        if self.last_state is not None and self.last_action is not None:
            self.memory.add(self.last_state, self.last_action, reward, state, done)

        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(self.memory.sample(self.batch_size))

        self.total_reward += reward
        self.count += 1
        self.last_state = state
        self.last_action = action

        if done:
            self.episode_num += 1
            self.write_stats([self.episode_num, self.total_reward])
            self.reset_episode_vars()            
        
        return self.postprocess_action(action)

    def act(self, states):
        # debugger
        # import pdb; pdb.set_trace()

        states = np.reshape(states, [-1, self.state_size])
        actions = self.actor_local.model.predict(states)
        return actions + self.noise.sample()  # add some noise for exploration

    def learn(self, experiences):
        # debugger
        # import pdb; pdb.set_trace()
        
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        # Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)
    
    def soft_update(self, local_model, target_model):
        # debugger
        # import pdb; pdb.set_trace()

        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

    def preprocess_state(self, state):
        # debugger
        # import pdb; pdb.set_trace()
        
        return state[0:3]  # position only

    def postprocess_action(self, action):
        # debugger
        # import pdb; pdb.set_trace()

        complete_action = np.zeros(self.task.action_space.shape)
        complete_action[0:3] = action  # linear force only
        return complete_action

    def write_stats(self, stats):
        # debugger
        # import pdb; pdb.set_trace()

        df_stats = pd.DataFrame([stats], columns=self.stats_columns)
        df_stats.to_csv(self.stats_filename, mode='a', index=False,
            header=not os.path.isfile(self.stats_filename))

    def save_models_on_exit(self, signal, frame):
        print("Save Models on Exit: ")
        self.critic_target.model.save('/home/robond/catkin_ws/critic_takeoff_model.h5')
        self.actor_target.model.save('/home/robond/catkin_ws/actor_takeoff_model.h5')
        sys.exit(0)
