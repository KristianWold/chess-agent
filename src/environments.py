import numpy as np
import torch
import math
import cmath
import qiskit.quantum_info as qi
import random
import config


class HaarMeasure():

    def __call__(self, other, seed=None, train=True):
        if not seed is None:
            U_target = np.array(qi.random_unitary(other.dim, seed).data)
        else:
            U_target = np.array(qi.random_unitary(other.dim).data)

        gate_sequence = None

        return U_target, gate_sequence


class FixedUnitary():

    def __init__(self,
                 U=None):
        self.U = U

    def __call__(self, other, seed=None, train=True):
        gate_sequence = None

        return self.U, gate_sequence


class BasisCurriculum():

    def __init__(self,
                 min_length=None,
                 schedule_length=None,
                 max_length=None,
                 interval=1000,
                 weighted = False):

        self.min_length = min_length
        self.max_length = max_length
        self.schedule_length = schedule_length
        self.interval = interval
        self.weighted = weighted


    def __call__(self, other, seed=None, train=True):
        rng = random.Random(seed)

        if other.episode_count % self.interval == 0:
            self.schedule_length += 1

        if train:
            current_length = min(self.max_length, self.schedule_length)
        else:
            current_length = self.max_length

        if self.weighted:
            elements = list(range(self.min_length, current_length + 1))
            length = rng.choices(population = elements,
                                 weights = elements,
                                 k = 1)[0]
        else:
            length = rng.randint(self.min_length, current_length)


        phase = rng.uniform(0, 2 * math.pi)
        U_target = cmath.exp(1j * phase) * np.eye(other.dim)
        #U_target = np.eye(other.dim)

        index_sequence = []
        for i in range(length):
            indicies = list(range(other.basis_gates.num_gates))
            probs = other.basis_gates.probs
            index = rng.choices(population = indicies,
                                weights = probs,
                                k=1)[0]

            index_sequence.append(index)
            U_target = other.basis_gates[index]@U_target

        return U_target, index_sequence


class DenseReward():

    def __init__(self, fidelity, tol=None, max_num_actions = None,):
        self.fidelity = fidelity
        self.tol = tol
        self.max_num_actions = max_num_actions

    def __call__(self, other):
        U_target = other.U_target
        U_current = other.U_current
        action_count = other.action_count

        fid = self.fidelity(U_target, U_current)
        
        if fid > 1 - self.tol:
            reward = self.max_num_actions - action_count + 1
            done = True
        else:
            reward = (fid - 1)
            done = False

        return reward, done


class SparseReward():

    def __init__(self, fidelity, tol=None, max_num_actions = None,):
        self.fidelity = fidelity
        self.tol = tol
        self.max_num_actions = max_num_actions

    def __call__(self, other):
        U_target = other.U_target
        U_current = other.U_current

        fid = self.fidelity(U_target, U_current)
        
        if fid > 1 - self.tol:
            reward = 0
            done = True
        else:
            reward = -1
            done = False

        return reward, done


def agf(U1, U2):
    K = U1.conj().T@U2
    dim = U1.shape[0]
    fid = (np.real(np.trace(K) * np.trace(K.conj().T)) +
           dim) / (dim + dim**2)
    return fid


def state_fidelity(U1, U2):
    fid = np.abs((U1.conj().T @ U2)[0,0])**2
    return fid


def fix_phase(U):
    trace = np.trace(U)
    phase = cmath.phase(trace)
    U = cmath.exp(-1j * phase) * U

    return U


class Environment:

    def __init__(self,
                 dim=None,
                 reward_func=None,
                 basis_gates=None,
                 data_generator=None,
                 max_num_actions=None,
                 fix_phase=True,
                 fisc = False):

        self.dim = dim
        self.reward_func = reward_func
        self.fidelity = self.reward_func.fidelity
        self.basis_gates = basis_gates
        self.data_generator = data_generator
        self.max_num_actions = max_num_actions
        self.fix_phase = fix_phase
        self.fisc = fisc

        self.episode_count = 1
        self.action_count = 0
        self.action_history = []

    def get_state(self):
        """get state from current enviroment"""

        if self.fisc:
            U = self.U_o[0,:]
        else:
            U = self.U_o

        state = np.concatenate(
            [np.real(U).flatten(), np.imag(U).flatten()])
        state = torch.tensor(state, dtype=torch.float32, device=config.device).view(1, -1)

        return state

    def get_reward(self):
        """get reward from current enviroment"""

        reward, _ = self.reward_func(self)
        return torch.tensor(reward, dtype=torch.float, device=config.device).view(1, 1)

    def check_done(self):
        _, done = self.reward_func(self)

        return bool(done)

    def step(self, action):
        self.action_count += 1
        self.action_history.append(action)
        self.U_current = self.U_current@self.basis_gates[action]
        self.U_o = self.U_o@self.basis_gates[action]

        # fix phase of environment
        if self.fix_phase:
            self.U_o = fix_phase(self.U_o)

        return self.get_state(), self.get_reward(), self.check_done()

    def reset(self, input_state=None, seed=None):
        if input_state is None:
            self.episode_count += 1
            self.U_target, _ = self.data_generator(self, seed=seed)
        else:
            self.U_target = input_state

        self.U_current = np.eye(self.dim)
        self.U_o = self.U_target.conj().T

        # fix phase of environment
        if self.fix_phase:
            self.U_o = fix_phase(self.U_o)

        self.action_count = 0
        self.action_history = []
        return self.get_state(), self.check_done()

    def generate_episode(self, seed=None):
        U_target, action_sequence = self.data_generator(self, seed=seed, train=False)
        
        state, done = self.reset(input_state=U_target)

        episode = []
        for action in action_sequence:
            next_state, reward, done = self.step(action)
            done_tensor = torch.tensor(bool(done), dtype=torch.bool, device=config.device)
            transition = [state, action, next_state, reward, done_tensor]
            episode.append(transition)
            state = next_state

        return episode
