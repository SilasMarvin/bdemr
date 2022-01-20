import gym
import jax.numpy as jnp
import numpy as np
import time
import sys

import model
import bandits
import memory

import matplotlib.pyplot as plt


train_batch_size = 16 
arm_count = 10 


"""
The main loop for the program 
"""
if __name__ == "__main__":
    #env = gym.make('CartPole-v0')
    env = bandits.BanditsManager(arm_count, 100)
    mem = memory.Memory(env, input_size=arm_count + 1)
    ml = model.Model(input_size=arm_count + 1, output_size=arm_count)
    start_time = time.time()

    average_epoch_rewards = []

    epoch_rewards = []
    epoch = 0

    try:
        env.reset_epoch()
        for i in range(2000000):
            if not i == 0 and i % (arm_count * 10) == 0:
                average_epoch_rewards.append(np.average(epoch_rewards))
                print(f"reseting epoch: {epoch} average rewards: {np.average(epoch_rewards)} epsilon: {ml.epsilon}")
                env.reset_epoch()
                epoch_rewards = []
                epoch += 1
            state = np.concatenate(([0], env.reset()), axis=0)
            done = False
            rewards = []
            step = 0
            while not done:
                ml_cell_states = ml.get_current_cell_states()
                ml_rnn_state = ml.get_current_state()
                action, log_prob = ml.act(state, mem.get_reinstatement_states())
                next_state, reward, done, _ = env.step(action)
                next_state = np.concatenate(([reward], next_state), axis=0)
                rewards.append(reward)
                mem.add(state, action, log_prob, reward, ml_cell_states, ml_rnn_state)
                state = jnp.array(next_state)
                step += 1
            epoch_rewards.append(np.sum(rewards))

            batch = mem.get_last_episode()
            mem.clear_episode_memory()
            mem_states = mem.get_reinstatement_states()
            ml.update(batch, mem_states)
            ml.reset_state()
    except:
        print("Exited early")
    plt.plot(range(len(average_epoch_rewards)), average_epoch_rewards)
    plt.ylabel("Average reward per epoch")
    plt.show()
