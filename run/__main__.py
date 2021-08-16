import gym
import jax.numpy as jnp
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
    mem = memory.Memory(env, input_size=4)
    ml = model.Model(input_size=4, output_size=2)
    start_time = time.time()

    for i in range(2000): 
        state = jnp.array(env.reset())
        done = False
        step = 0
        while not done:
            ml_state = ml.get_current_state()
            action = int(ml.act(state, mem.get_reinstatement_states()))
            next_state, reward, done, _ = env.step(action)
            next_state = jnp.array(next_state)
            mem.add(state, action, reward, next_state, done, ml_state, ml.get_current_state())
            state = next_state
            step += 1
        print(f"EPISODE: {i}  MADE TO STEP: {step}  EPSILON: {ml.epsilon}  RUNNING TIME: {time.time() - start_time}")

        #ml.episode_end()

        batch = mem.sample_batch(64, 16)
        if type(batch) == type(False) and batch == False:
            continue
        mem_states = mem.get_reinstatement_states()
        ml.update(batch, mem_states)
