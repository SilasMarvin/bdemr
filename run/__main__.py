import gym
import jax.numpy as jnp
import numpy as np
import time
import sys

import model
import memory


train_batch_size = 16 
updates_between_episodes = 5 


"""
The main loop for the program 
"""
if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    mem = memory.Memory(env, input_size=5)
    ml = model.Model(input_size=5, output_size=2)
    start_time = time.time()

    for i in range(1000):
        state = np.concatenate(([0], env.reset()), axis=0)
        done = False
        step = 0
        while not done:
            ml_cell_states = ml.get_current_cell_states()
            ml_rnn_state = ml.get_current_state()
            action, log_prob = ml.act(state, mem.get_reinstatement_states())
            next_state, reward, done, _ = env.step(action)
            next_state = np.concatenate(([reward], next_state), axis=0)
            if done and not step == 199:
                reward = -5
            mem.add(state, action, log_prob, reward, ml_cell_states, ml_rnn_state)
            state = jnp.array(next_state)
            step += 1
        print(f"EPISODE: {i}  MADE TO STEP: {step}  EPSILON: {ml.epsilon}  RUNNING TIME: {time.time() - start_time}")

        batch = mem.get_last_episode()
        mem.clear_episode_memory()
        mem_states = mem.get_reinstatement_states()
        ml.update(batch, mem_states)
        ml.reset_state()
