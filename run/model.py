from typing import Any, NamedTuple, Optional, Sequence, Tuple, Union
from haiku._src import base
from haiku._src import stateful

import jax.numpy as jnp
import numpy as np
import haiku as hk
import pickle
import optax
import math
import jax
import sys



"""
A copy and pase rewrite with added reinstatment state of hk dyanmic_unroll
"""
def dynamic_unroll(core, input_sequence, initial_state, reinstatement_state_keys, reinstatement_states, time_major=True, reverse=False):
    scan = stateful.scan if base.inside_transform() else jax.lax.scan
    def scan_f(prev_state, inputs):
        policy_dist, value, next_state = core(inputs[0], prev_state, reinstatement_state_keys, reinstatement_states)
        return next_state, (policy_dist, value)
    final_state, outputs = scan(
        scan_f,
        initial_state,
        [input_sequence],
        reverse=reverse)
    return outputs, final_state




"""
Subclassed LSTM that follows the architecture described in "Been There, Done That"
"""
class CLSTM(hk.LSTM):
    def __init__(self, hidden_size: int, name: Optional[str] = None):
        super().__init__(hidden_size, name)


    def __call__(self, inputs: jnp.ndarray, prev_state: hk.LSTMState, reinstatement_cell_state: jnp.ndarray) -> Tuple[jnp.ndarray, hk.LSTMState]:
        if len(inputs.shape) > 2 or not inputs.shape:
          raise ValueError("LSTM input must be rank-1 or rank-2.")
        x_and_h = jnp.concatenate([inputs, prev_state.hidden], axis=-1)
        gated = hk.Linear(5 * self.hidden_size)(x_and_h)
        # i = input, g = cell_gate, f = forget_gate, o = output_gate, r = reinstatement-gate
        i, g, f, o, r = jnp.split(gated, indices_or_sections=5, axis=-1)
        f = jax.nn.sigmoid(f + 1)  # Forget bias, as in sonnet.
        c = (f * prev_state.cell) + (jax.nn.sigmoid(i) * jnp.tanh(g)) + (jax.nn.sigmoid(r) * jax.nn.tanh(reinstatement_cell_state))
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        return h, hk.LSTMState(h, c)




"""
Subclasses the DeepRNN and augments it to follow the architecture described in "Been There, Done That"
"""
class CDeepRNN():
    def __init__(self):
        #self.layer1 = CLSTM(100)
        #self.layer1 = hk.LSTM(100)
        self.layer1 = hk.Linear(100)
        #self.layer2 = hk.Linear(50)
        self.layer2 = hk.LSTM(50)
        self.layer3_1 = hk.Linear(20)
        self.layer3_2 = hk.Linear(10)
        self.layer4_1 = hk.Linear(2)
        self.layer4_2 = hk.Linear(1)

        """
        self.mem_layer_1 = hk.Linear(50, name="mem_1")
        self.mem_layer_2 = hk.Linear(25, name="mem_2")
        self.mem_layer_3 = hk.Linear(1, name="mem_3")
        """


    def __call__(self, inputs, state, reinstatement_state_keys, reinstatement_states):
        next_states = []

        """
        tiled_inputs = jnp.tile(inputs, (jnp.shape(reinstatement_state_keys)[0], 1)) 
        reinstatement_state_key_inputs = jnp.concatenate([reinstatement_state_keys, tiled_inputs], axis=1)
        mem_out1 = jax.nn.tanh(self.mem_layer_3(jax.nn.relu(self.mem_layer_2(jax.nn.relu(self.mem_layer_1(reinstatement_state_key_inputs))))))
        r_state0 = jnp.sum(mem_out1 * reinstatement_states, axis=0)
        out1, next_state = self.layer1(inputs, state[0], r_state0)
        """
        
        #Use if layer one is an LSTM and not using mem layers
        #out1, next_state = self.layer1(inputs, state[0], reinstatement_states[0])
        #out1, next_state = self.layer1(inputs, state[0])
        #next_states.append(next_state)

        #Use if layer one is linear
        out1 = self.layer1(inputs)

        out1 = jax.nn.relu(out1)

        #out2 = jax.nn.relu(self.layer2(out1))
        out2, next_state = self.layer2(out1, state[0])
        next_states.append(next_state)

        out2 = jax.nn.relu(out2) 
        policy_dist = jax.nn.softmax(self.layer4_1(jax.nn.relu(self.layer3_1(out2))))
        value = self.layer4_2(jax.nn.relu(self.layer3_2(out2)))
        return policy_dist, value, tuple(next_states)


    def initial_state(self, batch_size: Optional[int]):
        #return tuple() #Only use if layer 1 is  linear
        return tuple([self.layer2.initial_state(batch_size)])




"""
A standard RL agent with act and update methods
"""
class Model:
    def __init__(self, input_size, output_size):
        self.rng = jax.random.PRNGKey(42)
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = 0.01
        self.epsilon = 1
        self.epsilon_decay = 0.9993
        self.ppo_epsilon = 0.2
        self.min_epsilon = 0.1
        self.update_step = 0 

        def make_network():
            mlp = CDeepRNN()
            return mlp

        def __get_initial_state():
            core = make_network()
            return core.initial_state(None)
        self._get_initial_state = hk.without_apply_rng(hk.transform(__get_initial_state))
        self._get_initial_state_params = self._get_initial_state.init(None)

        def forward(x, state, reinstatement_keys, reinstatement_states):
            core = make_network()
            return core(x, state, reinstatement_keys, reinstatement_states)

        self.ml_state = self.get_initial_state()
        self.ml = hk.transform(forward)
        self.ac_params = self.ml.init(self.rng, jnp.ones((self.input_size,)), self.ml_state, jnp.zeros((1,6)), jnp.zeros((1,100)))
        self.ml_apply = jax.jit(self.ml.apply)
        self.opt = optax.chain(optax.clip_by_global_norm(2.0), optax.adam(self.learning_rate))
        self.opt_state = self.opt.init(self.ac_params)

        #Unrolls the network
        def unroll(state_mem, rnn_start_state, reinstatement_state_keys, reinstatement_states):
            core = make_network()
            (probs, V), _ = dynamic_unroll(core, state_mem, rnn_start_state, reinstatement_state_keys, reinstatement_states)
            return probs, V
        _, self._unroll = hk.without_apply_rng(hk.transform(unroll))
        self._unroll = jax.jit(self._unroll)

        def loss(state_mem, rts_mem, action_mem, log_probs_mem, advantages, rnn_start_state, reinstatement_state_keys, reinstatement_states):
            core = make_network()
            (probs, V), _ = dynamic_unroll(core, state_mem, rnn_start_state, reinstatement_state_keys, reinstatement_states)
            V = jnp.squeeze(V, 1)
            log_probs = jnp.log(probs)
            log_probs_a = jax.vmap(lambda b, a: b[a])(log_probs, action_mem) #Log probs for a specific action
            ratios = jnp.exp(log_probs_a - log_probs_mem)
            s1 = ratios * advantages
            s2 = jnp.clip(ratios, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon) * advantages 
            actor_loss = jnp.mean(-jnp.fmin(s1, s2))
            critic_loss = jnp.mean(jnp.square(rts_mem - V))
            #entropy = - jnp.sum(probs * log_probs)
            return actor_loss + critic_loss
        _, self._loss = hk.without_apply_rng(hk.transform(loss))
        self._loss = jax.jit(self._loss)


    #Updates the agent's online network and potentially copies the online networks weights to the target network
    def update(self, batch, reinstatement_states):
        self.rng, rng_key = jax.random.split(self.rng)
        state_mem, action_mem, log_probs_mem, reward_mem, rts_mem, rnn_states = batch
        reinstatement_state_keys, reinstatement_states = reinstatement_states

        batch_probs, values = self._unroll(self.ac_params, state_mem, self.get_initial_state(), reinstatement_state_keys, reinstatement_states)
        values = jnp.squeeze(values, 1)
        #rts_mem = (rts_mem - jnp.mean(rts_mem)) / (jnp.std(rts_mem) + 1e-10) Already normalizing in the memory
        advantages = rts_mem - values 

        for _ in range(3):
            grads = jax.grad(self._loss)(self.ac_params, state_mem, rts_mem, action_mem, log_probs_mem, advantages, self.get_initial_state(), reinstatement_state_keys, reinstatement_states)
            updates, self.opt_state = self.opt.update(grads, self.opt_state)
            self.ac_params = optax.apply_updates(self.ac_params, updates)

        if self.update_step % 20 == 0:
            self.save()
        self.update_step += 1
            

    #Gets an action from the agent
    def act(self, x, reinstatement_states):
        self.rng, rng_key = jax.random.split(self.rng)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
        reinstatement_keys, reinstatement_states = reinstatement_states
        prediction, values, self.ml_state = self.ml_apply(self.ac_params, rng_key, x, self.ml_state, reinstatement_keys, reinstatement_states)
        if jax.random.uniform(rng_key, (1,), minval=0.0, maxval=1.0) > self.epsilon:
            action = jax.random.choice(rng_key, jnp.arange(2), (), p=prediction)
        else:
            action = jax.random.randint(rng_key, (1,), minval=0, maxval=self.output_size)[0]
        log_prob = jnp.log(prediction[action])
        return int(action), log_prob


    #Gets the initial ml state
    def get_initial_state(self):
        return self._get_initial_state.apply(self._get_initial_state_params)


    #Gets the current cell ml state
    def get_current_cell_states(self):
        state = []
        for x in self.ml_state:
            state += list(x.cell)
        return jnp.array(state)


    #Gets the current state
    def get_current_state(self):
        return self.ml_state


    #Resets the state to the initial state
    def reset_state(self):
        self.ml_state = self.get_initial_state()


    #Saves the model
    def save(self):
        params = hk.data_structures.to_mutable_dict(self.ac_params)
        pickle.dump(params, open("checkpoints/{}.pickle".format(self.update_step), "wb"))





























"""
        #Gets lstm cell state from flattened array
        def array_to_cell_states(state):
            splits = jnp.split(state, [100])
            return [splits[1],splits[2]]
        self._array_to_cell_states = jax.jit(array_to_cell_states)

        #Flat array to lstm state
        def array_to_lstm_state(ml_state_mem):
            splits = jnp.split(ml_state_mem, [100])
            return (hk.LSTMState(splits[0], splits[1]),)
        self._array_to_lstm_state = array_to_lstm_state
"""
