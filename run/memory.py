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
        self.last_combined = self.size
        self.max_size = 10000

        self.state_mem = jnp.zeros((self.max_size,input_size))
        self.action_mem = jnp.zeros((self.max_size,2))
        self.reward_mem = jnp.zeros((self.max_size, 1))
        self.state_next_mem = jnp.zeros((self.max_size, input_size))
        self.done_mem = jnp.zeros((self.max_size, 1))
        self.ml_state_mem = jnp.zeros((self.max_size, 300))
        self.ml_state_next_mem = jnp.zeros((self.max_size, 300))

        """
        self.state_mem = jnp.zeros((0,input_size))
        self.action_mem = jnp.zeros((0,2))
        self.reward_mem = jnp.zeros((0, 1))
        self.state_next_mem = jnp.zeros((0, input_size))
        self.done_mem = jnp.zeros((0, 1))
        self.ml_state_mem = jnp.zeros((0, 300))
        self.ml_state_next_mem = jnp.zeros((0, 300))
        self.t_size = 0
        self.t_state_mem = np.zeros((50,input_size))
        self.t_action_mem = np.zeros((50,2))
        self.t_reward_mem = np.zeros((50, 1))
        self.t_state_next_mem = np.zeros((50, input_size))
        self.t_done_mem = np.zeros((50, 1))
        self.t_ml_state_mem = np.zeros((50, 300))
        self.t_ml_state_next_mem = np.zeros((50, 300))
        """


    #Add the state, action, reward, state_next and done to the current episode memory
    def add(self, state, action, reward, state_next, done, ml_state, ml_state_next):
        self.state_mem = self.state_mem.at[self.size].set(state)
        self.action_mem = self.action_mem.at[self.size].set(action)
        self.reward_mem = self.reward_mem.at[self.size].set(reward)
        self.state_next_mem = self.state_next_mem.at[self.size].set(state_next)
        self.done_mem = self.done_mem.at[self.size].set(done)
        self.ml_state_mem = self.ml_state_mem.at[self.size].set(ml_state)
        self.ml_state_next_mem = self.ml_state_next_mem.at[self.size].set(ml_state_next)
        self.size += 1
        """
        self.t_state_mem[self.t_size] = state
        self.t_action_mem[self.t_size] = action
        self.t_reward_mem[self.t_size] = reward
        self.t_state_next_mem[self.t_size] = state_next
        self.t_done_mem[self.t_size] = done 
        self.t_ml_state_mem[self.t_size] = ml_state
        self.t_ml_state_next_mem[self.t_size] = ml_state_next 
        self.t_size += 1
        if self.t_size == 50:
            self.state_mem = jnp.concatenate([self.state_mem, jnp.array(self.t_state_mem)], axis=0) 
            self.action_mem = jnp.concatenate([self.action_mem, jnp.array(self.t_action_mem)], axis=0) 
            self.reward_mem = jnp.concatenate([self.reward_mem, jnp.array(self.t_reward_mem)], axis=0) 
            self.state_next_mem = jnp.concatenate([self.state_next_mem, jnp.array(self.t_state_next_mem)], axis=0) 
            self.done_mem = jnp.concatenate([self.done_mem, jnp.array(self.t_done_mem)], axis=0) 
            self.ml_state_mem = jnp.concatenate([self.ml_state_mem, jnp.array(self.t_ml_state_mem)], axis=0) 
            self.ml_state_next_mem = jnp.concatenate([self.ml_state_next_mem, jnp.array(self.t_ml_state_next_mem)], axis=0) 
            self.size += self.t_size
            self.t_size = 0
        """


    #Gets the lstm for the associated state
    def get_reinstatement_states(self):
        return self.state_mem, self.ml_state_mem


    def sample_batch(self, batch_size, sequence_size):
        def sample_batch_sequence(size, rand_index, state_mem, action_mem, reward_mem, state_next_mem, done_mem, ml_state_mem, ml_state_next_mem):
            return (jax.lax.dynamic_slice(state_mem, (rand_index,0), (size,4)),
            jnp.reshape(jax.lax.dynamic_slice(action_mem, (rand_index,0), (size,1)), (size,)),
            jnp.reshape(jax.lax.dynamic_slice(reward_mem, (rand_index,0), (size,1)), (size,)),
            jax.lax.dynamic_slice(state_next_mem, (rand_index,0), (size,4)),
            jnp.reshape(jax.lax.dynamic_slice(done_mem, (rand_index,0), (size,1)), (size,)),
            jnp.zeros((batch_size,)),
            jax.lax.dynamic_slice(ml_state_mem, (rand_index,0), (size,300)),
            jax.lax.dynamic_slice(ml_state_next_mem, (rand_index,0), (size,300)),
            jax.lax.dynamic_slice(ml_state_mem, (rand_index,0), (size,300)),
            jax.lax.dynamic_slice(ml_state_next_mem, (rand_index,0), (size,300)))
        if sequence_size > self.size:
            return False
        self.rng, rng_key =  jax.random.split(self.rng)
        return jax.vmap(sample_batch_sequence, in_axes=(None,0,None,None,None,None,None,None,None))(
                sequence_size, 
                jax.random.randint(rng_key, (batch_size,), minval=0, maxval=self.size - sequence_size - 1), 
                self.state_mem, 
                self.action_mem, 
                self.reward_mem, 
                self.state_next_mem, 
                self.done_mem, 
                self.ml_state_mem, 
                self.ml_state_next_mem)
