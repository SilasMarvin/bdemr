from typing import Any, NamedTuple, Optional, Sequence, Tuple, Union
from haiku._src import base
from haiku._src import stateful

import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import math
import jax
import sys



"""
A copy and pase rewrite with added reinstatment state of hk dyanmic_unroll
"""
def dynamic_unroll(core, input_sequence, initial_state, reinstatement_states, time_major=True, reverse=False):
    scan = stateful.scan if base.inside_transform() else jax.lax.scan
    # Swap the input and output of core.
    def scan_f(prev_state, inputs):
        outputs, next_state = core(inputs[0], prev_state, inputs[1])
        return next_state, outputs
    # TODO(hamzamerzic): Remove axis swapping once scan supports time axis arg.
    if not time_major:
        input_sequence = _swap_batch_time(input_sequence)
    final_state, output_sequence = scan(
        scan_f,
        initial_state,
        [input_sequence, reinstatement_states],
        reverse=reverse)
    if not time_major:
        output_sequence = _swap_batch_time(output_sequence)
    return output_sequence, final_state




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
        # TODO(slebedev): Consider aligning the order of gates with Sonnet.
        # i = input, g = cell_gate, f = forget_gate, o = output_gate, r = reinstatement-gate
        i, g, f, o, r = jnp.split(gated, indices_or_sections=5, axis=-1)
        f = jax.nn.sigmoid(f + 1)  # Forget bias, as in sonnet.
        c = f * prev_state.cell + jax.nn.sigmoid(i) * jnp.tanh(g) + jax.nn.sigmoid(r) * jax.nn.tanh(reinstatement_cell_state)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        return h, hk.LSTMState(h, c)




"""
Subclasses the DeepRNN and augments it to follow the architecture described in "Been There, Done That"
"""
class CDeepRNN(hk.DeepRNN):
    def __init__(self, layers: Sequence[Any], name: Optional[str] = None):
        super().__init__(layers, name=name)


    def __call__(self, inputs, state, reinstatement_state):
        current_inputs = inputs
        next_states = []
        outputs = []
        state_idx = 0
        concat = lambda *args: jnp.concatenate(args, axis=-1)
        for idx, layer in enumerate(self.layers):
          if self.skip_connections and idx > 0:
            current_inputs = jax.tree_multimap(concat, inputs, current_inputs)
          if isinstance(layer, hk.RNNCore):
            current_inputs, next_state = layer(current_inputs, state[state_idx], reinstatement_state[state_idx])
            outputs.append(current_inputs)
            next_states.append(next_state)
            state_idx += 1
          else:
            current_inputs = layer(current_inputs)
        if self.skip_connections:
          out = jax.tree_multimap(concat, *outputs)
        else:
          out = current_inputs
        return out, tuple(next_states)




"""
A standard RL agent with act and update methods
It utilizes a online and target network as did the reference paper
"""
class Model:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = 0.01
        self.epsilon = 1
        self.epsilon_decay = 0.9994
        self.min_epsilon = 0.1
        self.gamma = 0.99
        self.update_step = 0 
        self.copy_weights_every = 20 
        self.rng = jax.random.PRNGKey(42)

        def make_network():
            mlp = CDeepRNN([
              CLSTM(100), jax.nn.relu,
              CLSTM(50), jax.nn.relu,
              hk.Linear(output_size),
            ])
            return mlp

        def __get_initial_state():
            core = make_network()
            return core.initial_state(None)
        self._get_initial_state = hk.without_apply_rng(hk.transform(__get_initial_state))
        self._get_initial_state_params = self._get_initial_state.init(None)

        def forward(x, state, reinstatement_state):
            core = make_network()
            return core(x, state, reinstatement_state)

        self.ml_state = self.get_initial_state()
        self.ml = hk.transform(forward)
        self.online_params = self.ml.init(self.rng, jnp.ones((self.input_size,)), self.ml_state, jnp.zeros((100,)))
        self.target_params = self.online_params
        self.ml_apply = jax.jit(self.ml.apply)
        self.opt = optax.chain(optax.clip_by_global_norm(0.25), optax.adam(self.learning_rate))
        self.opt_state = self.opt.init(self.online_params)

        #Gets lstm cell state from flattened array
        def array_to_cell_states(state):
            splits = jnp.split(state, [100,200,250])
            return [splits[1],splits[2]]
        self._array_to_cell_states = jax.jit(array_to_cell_states)

        #Flat array to lstm state
        def array_to_lstm_state(ml_state_mem):
            splits = jnp.split(ml_state_mem, [100,200,250])
            return (hk.LSTMState(splits[0], splits[1]), hk.LSTMState(splits[2], splits[3]))
        self._array_to_lstm_state = array_to_lstm_state

        #Computes the updated q values
        def get_updated_q_values(next_state_mem, reward_mem, done_mem, ml_state_mem, reinstatement_state_keys, reinstatement_states):
            core = make_network()
            def map_fn(next_state_mem, reward_mem, done_mem, ml_state_mem, reinstatement_state_keys, reinstatement_states):
                ml_state_mem = self._array_to_lstm_state(ml_state_mem)
                future_rewards, _ = dynamic_unroll(core, next_state_mem, ml_state_mem, jnp.zeros((16,100)))
                updated_q_values = reward_mem + self.gamma * jnp.amax(future_rewards, axis=1)
                updated_q_values = updated_q_values * (1 - done_mem)
                return updated_q_values
            return jax.vmap(map_fn, in_axes=(0,0,0,None,None,None))(next_state_mem, reward_mem, done_mem, ml_state_mem, reinstatement_state_keys, reinstatement_states)
        _, self._get_updated_q_values = hk.without_apply_rng(hk.transform(get_updated_q_values))
        self._get_updated_q_values = jax.jit(self._get_updated_q_values)

        #Computes the loss in refernce to paper as defined in Equation 4
        def loss(updated_q_values, state_mem, action_masks, graph_value_mem, ml_state_mem, reinstatement_state_keys, reinstatement_states):
            core = make_network()
            def map_fn(updated_q_values, state_mem, action_masks, graph_value_mem, ml_state_mem, reinstatement_state_keys, reinstatement_states):
                ml_state_mem = self._array_to_lstm_state(ml_state_mem)
                q_values, _ = dynamic_unroll(core, state_mem, ml_state_mem, jnp.zeros((16,100)))
                q_values = jnp.sum(jnp.multiply(q_values, action_masks), axis=1)
                #loss = jnp.sum(jnp.square(jnp.subtract(updated_q_values[-1], q_values[-1])) + 0.1 * jnp.square(jnp.subtract(graph_value_mem[-1], q_values[-1]))) #Only compare the last value of the individual batch sequence
                loss = jnp.sum(jnp.square(jnp.subtract(updated_q_values[-1], q_values[-1]))) #Only compare the last value of the individual batch sequence
                return loss
            return jnp.sum(jax.vmap(map_fn, in_axes=(0,0,0,0,None,None,None))(updated_q_values, state_mem, action_masks, graph_value_mem, ml_state_mem, reinstatement_state_keys, reinstatement_states)) / jnp.shape(updated_q_values)[0]
        _, self._loss = hk.without_apply_rng(hk.transform(loss))
        self._loss = jax.jit(self._loss)


    #Updates the agent's online network and potentially copies the online networks weights to the target network
    def update(self, batch, reinstatement_states):
        state_mem, action_mem, reward_mem, next_state_mem, done_mem, graph_value_mem, ml_state_mem, ml_state_next_mem, ml_reinstatement_state_mem, ml_reinstatement_state_next_mem = batch
        reinstatement_state_keys, reinstatement_states = reinstatement_states

        ml_state_mem = self.get_current_state()
        ml_state_next_mem = self.get_current_state()

        updated_q_values = self._get_updated_q_values(self.target_params, next_state_mem, reward_mem, done_mem, ml_state_next_mem, reinstatement_state_keys, reinstatement_states)
        action_masks = jax.nn.one_hot(action_mem, self.output_size)
        grads = jax.grad(self._loss)(self.online_params, updated_q_values, state_mem, action_masks, graph_value_mem, ml_state_mem, reinstatement_state_keys, reinstatement_states)
        updates, self.opt_state = self.opt.update(grads, self.opt_state)
        self.online_params = optax.apply_updates(self.online_params, updates)
        if self.update_step % self.copy_weights_every == 0:
            self.target_params = self.online_params
        self.update_step += 1


    #Gets an action from the agent
    def act(self, x, reinstatement_states):
        self.rng, rng_key = jax.random.split(self.rng)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
        if jax.random.uniform(rng_key, (1,), minval=0.0, maxval=1.0) > self.epsilon:
            prediction, self.ml_state = self.ml_apply(self.online_params, rng_key, x, self.ml_state, jnp.zeros((100,)))
            return jnp.argmax(prediction)
        else:
            return jax.random.randint(rng_key, (1,), minval=0, maxval=self.output_size)[0]


    #Gets the initial ml state
    def get_initial_state(self):
        return self._get_initial_state.apply(self._get_initial_state_params)


    #Gets the current ml state
    def get_current_state(self):
        state = []
        for x in self.ml_state:
            state += list(x.hidden)
            state += list(x.cell)
        return jnp.array(state)


    #Sets the state back to the initial state
    def episode_end(self):
        self.ml_state = self.get_initial_state()
