import jax.numpy as jnp
import numpy as np
import haiku as hk
import time
import jax
import sys




"""
Manages the memory for the rl agent
"""
class Memory:
    def __init__(self, env, input_size):
        self.rng = jax.random.PRNGKey(42)
        self.size = 0
        self.max_size = 10000
        self.gamma = 0.97

        self.r_keys = jnp.zeros((1,6))
        self.r_states = jnp.zeros((1, 100))

        self.t_state_mem = []
        self.t_action_mem = []
        self.t_log_prob_mem = []
        self.t_reward_mem = []
        self.t_done_mem = []
        self.t_ml_cell_states_mem = []
        self.t_ml_rnn_states_mem = []
        self.t_Rts_mem = []


    #Add the state, action, reward, state_next and done to the current episode memory
    def add(self, state, action, log_prob, reward, ml_cell_state, ml_rnn_state):
        self.t_state_mem.append(state)
        self.t_action_mem.append(action)
        self.t_log_prob_mem.append(log_prob)
        self.t_reward_mem.append(reward) 
        self.t_ml_cell_states_mem.append(ml_cell_state)
        self.t_ml_rnn_states_mem.append(ml_rnn_state)


    #Computes the Rts
    def compute_rewards_to_go(self):
        Rt = 0
        for t in reversed(range(len(self.t_reward_mem))):
            Rt = self.t_reward_mem[t] + self.gamma * Rt
            self.t_Rts_mem.insert(0, Rt)


    #Gets all lstm states and state_mem
    def get_reinstatement_states(self):
        return self.r_keys, self.r_states


    #Gets the last episode sequence
    def get_last_episode(self):
        self.compute_rewards_to_go()
        """
        rand_index = 0
        if max_size < len(self.t_state_mem):
            self.rng, rng_key = jax.random.split(self.rng)
            rand_index = jax.random.randint(rng_key, (1,), 0, len(self.t_state_mem) - max_size + 2)[0]
        return (
            jnp.array(self.t_state_mem[rand_index:rand_index + max_size]), 
            jnp.array(self.t_action_mem[rand_index:rand_index + max_size]), 
            jnp.array(self.t_log_prob_mem[rand_index:rand_index + max_size]), 
            jnp.array(self.t_reward_mem[rand_index:rand_index + max_size]),
            jnp.array(self.t_Rts_mem[rand_index:rand_index + max_size]))
        """
        self.t_Rts_mem = jnp.array(self.t_Rts_mem)
        #self.t_Rts_mem = (self.t_Rts_mem - jnp.mean(self.t_Rts_mem)) / (jnp.std(self.t_Rts_mem) + 1e-10)
        self.t_Rts_mem = self.t_Rts_mem / 10

        self.t_state_mem = jnp.array(self.t_state_mem)
        self.t_action_mem = jnp.array(self.t_action_mem)
        return (
            self.t_state_mem,
            self.t_action_mem,
            jnp.array(self.t_log_prob_mem), 
            jnp.array(self.t_reward_mem),
            self.t_Rts_mem,
            self.t_ml_rnn_states_mem)


    #Clears the memory
    def clear_episode_memory(self):
        """
        keys = jnp.concatenate([self.t_state_mem, jnp.expand_dims(self.t_action_mem, axis=1), jnp.expand_dims(self.t_Rts_mem, axis=1)], axis=1)
        self.r_keys = jnp.concatenate([self.r_keys, keys])[-1000:]
        self.r_states = jnp.concatenate([self.r_states, jnp.array(self.t_ml_cell_states_mem)])[-1000:]
        """

        self.t_state_mem = []
        self.t_action_mem = []
        self.t_log_prob_mem = []
        self.t_reward_mem = []
        self.t_ml_cell_states_mem = []
        self.t_ml_rnn_states_mem = []
        self.t_Rts_mem = []
