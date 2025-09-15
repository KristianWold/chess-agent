import numpy as np
import matplotlib.pyplot as plt
import pickle
import config
import qiskit.quantum_info as qi
import multiprocessing as mp
import random
import math
import warnings


from itertools import count
from tqdm import tqdm
from scipy.optimize import minimize

from basis_gates import *
from copy import copy


class Metropolis:
    def __init__(self, T_start=0, gamma=1):
        self.T_start = T_start  # temperature, higher means more exploration
        self.T = T_start
        self.gamma = gamma  # decay factor

    def __call__(self, fidelity_new, fidelity_current):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress overflow in exp warning

            if self.T > 0:
                amplitude = np.exp((fidelity_new - fidelity_current) / self.T)
                u = random.uniform(0, 1)
                accept = (amplitude > u)
            else:
                # greedy approach, T=0
                accept = (fidelity_new > fidelity_current)

        self.T *= self.gamma

        return accept

    def reset(self):
        self.T = self.T_start


class MonteCarloCompiler:

    def __init__(self,
                 basis_gates,
                 length,
                 variable_length,
                 tol,
                 max_steps,
                 acceptance_criterion,
                 verbose = False):

        self.basis_gates = basis_gates
        self.length = length
        self.variable_length = variable_length
        self.tol = tol
        self.max_steps = max_steps
        self.acceptance_criterion = acceptance_criterion
        self.verbose = verbose

        self.num_gates = self.basis_gates.num_gates

    #def compile_list(self, U_list, verbose=False):
    #    U_compiled_list = []
    #    seq_list = []
    #    fidelity_list = []
    #    steps_list = []

    #    if verbose:
    #        decorator = tqdm
    #    else:
    #        def decorator(x): return x

    #    with mp.Pool(self.num_threads) as p:
    #        result = p.map(self.compile, U_list)

    #    U_compiled_list, seq_list, fidelity_list, steps_list, reject_ratio_list = zip(*result)

    #    return U_compiled_list, seq_list, fidelity_list, steps_list, reject_ratio_list

    def compile(self, U_target):
        self.acceptance_criterion.reset()
        self.U_target = U_target

        self.sequence_current = [random.randrange(
            self.num_gates) for i in range(self.length)]
        self.U_current = self.seq_to_unitary(self.sequence_current)

        self.fidelity_current = agf(self.U_target, self.U_current)

        step_list = []
        fidelity_list = []
        num_rejected = 0

        for i in range(self.max_steps):
            if self.verbose and i%(self.max_steps//100) == 0:
                print(f"{i}/{self.max_steps}")

            if self.step():
                pass
            else:
                num_rejected += 1

            step_list.append(i + 1)
            fidelity_list.append(self.fidelity_current)

            if self.fidelity_current > self.tol:
                break

        reject_ratio = num_rejected/i

        returns = []
        returns.append(self.seq_to_unitary(self.sequence_current))
        returns.append(self.sequence_current)
        returns.append(fidelity_list)
        returns.append(step_list)
        returns.append(reject_ratio)

        return tuple(returns)

    def seq_to_unitary(self, sequence):
        dim = self.basis_gates[0].shape[0]
        U = np.eye(dim)
        for index in sequence:
            U = U@self.basis_gates[index]

        return U

    def step(self):

        sequence_new = copy(self.sequence_current)
        if len(sequence_new) > 0:
            index = random.randrange(len(sequence_new))
        else:
            index = 0

        if self.variable_length:
            action = random.randrange(3)
        else:
            action = 0

        # change random gate
        if action == 0 and len(sequence_new) > 0:
            gate = sequence_new[index]
            sequence_new[index] = random.choice(list(range(1, gate)) +
                                                list(range(gate + 1, self.num_gates)))
        # remove random gate
        if action == 1 and len(sequence_new) > 0:
            sequence_new.pop(index)
        # add random gate
        if action == 2:
            gate_new = random.randrange(self.num_gates)
            sequence_new.insert(index, gate_new)

        U_new = self.seq_to_unitary(sequence_new)
        fidelity_new = agf(self.U_target, U_new)

        if self.acceptance_criterion(fidelity_new, self.fidelity_current):
            self.fidelity_current = fidelity_new
            self.sequence_current = sequence_new
            return True

        return False


def f(params, X, y):
    c = params[0]
    k = params[1]

    loss = np.sum((np.log(1 / (1 - X))**c + k - y)**2)
    return loss


def pad_list(x_list, max_length=1000):
    x_list_new = []
    for x in x_list:
        x_new = x + (max_length - len(x)) * [x[-1]]
        x_list_new.append(x_new)

    return x_list_new
