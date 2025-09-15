import numpy as np
import math
import cmath
from utils import *


class SmallRotations:

    def __init__(self, frac):
        I = np.array([[1, 0], [0, 1]])
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        R1 = math.cos(frac * math.pi / 2) * I - 1j * \
            math.sin(frac * math.pi / 2) * X
        R2 = math.cos(frac * math.pi / 2) * I - 1j * \
            math.sin(frac * math.pi / 2) * Y
        R3 = math.cos(frac * math.pi / 2) * I - 1j * \
            math.sin(frac * math.pi / 2) * Z
        R4 = math.cos(frac * math.pi / 2) * I + 1j * \
            math.sin(frac * math.pi / 2) * X
        R5 = math.cos(frac * math.pi / 2) * I + 1j * \
            math.sin(frac * math.pi / 2) * Y
        R6 = math.cos(frac * math.pi / 2) * I + 1j * \
            math.sin(frac * math.pi / 2) * Z

        self.basis_gates = [R1, R2, R3, R4, R5, R6]
        self.num_gates = 6

    def __getitem__(self, index):
        return self.basis_gates[index]


class SmallRotations2:

    def __init__(self, frac):
        I = np.array([[1, 0], [0, 1]])
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        Rx_pos = math.cos(frac * math.pi / 2) * I - 1j * \
            math.sin(frac * math.pi / 2) * X

        Rx_neg = math.cos(frac * math.pi / 2) * I + 1j * \
            math.sin(frac * math.pi / 2) * X

        Ry_pos = math.cos(frac * math.pi / 2) * I - 1j * \
            math.sin(frac * math.pi / 2) * Y

        Ry_neg = math.cos(frac * math.pi / 2) * I + 1j * \
            math.sin(frac * math.pi / 2) * Y

        Rz_pos = math.cos(frac * math.pi / 2) * I - 1j * \
            math.sin(frac * math.pi / 2) * Z

        Rz_neg = math.cos(frac * math.pi / 2) * I + 1j * \
            math.sin(frac * math.pi / 2) * Z

        B1 = math.cos(frac * math.pi / 2) * np.kron(I, I) - 1j * \
            math.sin(frac * math.pi / 2) * np.kron(X, X)

        B2 = math.cos(frac * math.pi / 2) * np.kron(I, I) + 1j * \
            math.sin(frac * math.pi / 2) * np.kron(X, X)

        B3 = math.cos(frac * math.pi / 2) * np.kron(I, I) - 1j * \
            math.sin(frac * math.pi / 2) * np.kron(Y, Y)

        B4 = math.cos(frac * math.pi / 2) * np.kron(I, I) + 1j * \
            math.sin(frac * math.pi / 2) * np.kron(Y, Y)

        B5 = math.cos(frac * math.pi / 2) * np.kron(I, I) - 1j * \
            math.sin(frac * math.pi / 2) * np.kron(Z, Z)

        B6 = math.cos(frac * math.pi / 2) * np.kron(I, I) + 1j * \
            math.sin(frac * math.pi / 2) * np.kron(Z, Z)

        B7 = np.kron(Rx_pos, I)
        B8 = np.kron(Rx_neg, I)
        B9 = np.kron(I, Rx_pos)
        B10 = np.kron(I, Rx_neg)

        B11 = np.kron(Ry_pos, I)
        B12 = np.kron(Ry_neg, I)
        B13 = np.kron(I, Ry_pos)
        B14 = np.kron(I, Ry_neg)

        B15 = np.kron(Rz_pos, I)
        B16 = np.kron(Rz_neg, I)
        B17 = np.kron(I, Rz_pos)
        B18 = np.kron(I, Rz_neg)

        self.basis_gates = [B1, B2, B3, B4, B5, B6, B7, B8,
                            B9, B10, B11, B12, B13, B14, B15, B16, B17, B18]
        self.num_gates = 18

    def __getitem__(self, index):
        return self.basis_gates[index]
    


class SmallRotations3:

    def __init__(self, frac):
        I = np.array([[1, 0], [0, 1]])
        X = np.array([[0, 1], [1, 0]])
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]])

        X_pos = math.cos(frac * math.pi / 2) * I - 1j * \
            math.sin(frac * math.pi / 2) * X

        X_neg = math.cos(frac * math.pi / 2) * I + 1j * \
            math.sin(frac * math.pi / 2) * X

        Y_pos = math.cos(frac * math.pi / 2) * I - 1j * \
            math.sin(frac * math.pi / 2) * Y

        Y_neg = math.cos(frac * math.pi / 2) * I + 1j * \
            math.sin(frac * math.pi / 2) * Y

        Z_pos = math.cos(frac * math.pi / 2) * I - 1j * \
            math.sin(frac * math.pi / 2) * Z

        Z_neg = math.cos(frac * math.pi / 2) * I + 1j * \
            math.sin(frac * math.pi / 2) * Z
        
        
        R_list = [X_pos, X_neg, Y_pos, Y_neg, Z_pos, Z_neg]

        XX_pos = math.cos(frac * math.pi / 2) * kron(I, I) - 1j * \
            math.sin(frac * math.pi / 2) * kron(X, X)

        XX_neg = math.cos(frac * math.pi / 2) * kron(I, I) + 1j * \
            math.sin(frac * math.pi / 2) * kron(X, X)

        YY_pos = math.cos(frac * math.pi / 2) * kron(I, I) - 1j * \
            math.sin(frac * math.pi / 2) * kron(Y, Y)

        YY_neg = math.cos(frac * math.pi / 2) * kron(I, I) + 1j * \
            math.sin(frac * math.pi / 2) * kron(Y, Y)

        ZZ_pos = math.cos(frac * math.pi / 2) * kron(I, I) - 1j * \
            math.sin(frac * math.pi / 2) * kron(Z, Z)

        ZZ_neg = math.cos(frac * math.pi / 2) * kron(I, I) + 1j * \
            math.sin(frac * math.pi / 2) * kron(Z, Z)
        
        RR_list = [XX_pos, XX_neg, YY_pos, YY_neg, ZZ_pos, ZZ_neg]

        self.basis_gates = []

        for R in R_list:
            G1 = kron(R, I, I)
            G2 = kron(I, R, I)
            G3 = kron(I, I, R)
            self.basis_gates.extend([G1, G2, G3])

        swap = np.array([[1,0,0,0],[0,0,1,0], [0,1,0,0], [0,0,0,1]])
        swap23 = kron(I,swap)
                         

        for RR in RR_list:
            G1 = kron(RR, I)
            G2 = kron(I, RR)
            G3 = swap23@kron(RR, I)@swap23
            self.basis_gates.extend([G1, G2, G3])

        self.num_gates = len(self.basis_gates)

    def __getitem__(self, index):
        return self.basis_gates[index]


class HRC:

    def __init__(self):
        B1 = 1 / np.sqrt(5) * np.array([[1, 2 * 1j], [2 * 1j, 1]])
        B2 = 1 / np.sqrt(5) * np.array([[1, 2], [-2, 1]])
        B3 = 1 / np.sqrt(5) * np.array([[1 + 2 * 1j, 0], [0, 1 - 2 * 1j]])

        self.basis_gates = [B1, B2, B3]
        self.num_gates = len(self.basis_gates)
        self.probs = [1, 1, 1]

    def __getitem__(self, index):
        return self.basis_gates[index]


class HRC2:

    def __init__(self):
        I = np.eye(2)

        G1 = 1 / np.sqrt(5) * np.array([[1, 2 * 1j], [2 * 1j, 1]])
        G2 = 1 / np.sqrt(5) * np.array([[1, 2], [-2, 1]])
        G3 = 1 / np.sqrt(5) * np.array([[1 + 2 * 1j, 0], [0, 1 - 2 * 1j]])

        B1 = np.kron(G1, I)
        B2 = np.kron(G2, I)
        B3 = np.kron(G3, I)
        B4 = np.kron(I, G1)
        B5 = np.kron(I, G2)
        B6 = np.kron(I, G3)
        B7 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        B8 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

        self.basis_gates = [B1, B2, B3, B4, B5, B6, B7, B8]
        self.num_gates = len(self.basis_gates)
        self.probs = [1, 1, 1, 1, 1, 1, 1, 1]

    def __getitem__(self, index):
        return self.basis_gates[index]


class HRC2Swap:

    def __init__(self):
        I = np.eye(2)

        G1 = 1 / np.sqrt(5) * np.array([[1, 2 * 1j], [2 * 1j, 1]])
        G2 = 1 / np.sqrt(5) * np.array([[1, 2], [-2, 1]])
        G3 = 1 / np.sqrt(5) * np.array([[1 + 2 * 1j, 0], [0, 1 - 2 * 1j]])

        B1 = np.kron(G1, I)
        B2 = np.kron(G2, I)
        B3 = np.kron(G3, I)
        B4 = np.kron(I, G1)
        B5 = np.kron(I, G2)
        B6 = np.kron(I, G3)
        B7 = np.array([[1, 0, 0, 0], [0, (1 + 1j)/2, (1 - 1j)/2, 0], [0, (1 - 1j)/2, (1 + 1j)/2, 0], [0, 0, 0, 1]])

        self.basis_gates = [B1, B2, B3, B4, B5, B6, B7]
        self.num_gates = len(self.basis_gates)
        self.probs = [1, 1, 1, 1, 1, 1, 2]

    def __getitem__(self, index):
        return self.basis_gates[index]


class HRC3Swap:

    def __init__(self):
        self.basis_gates = []
        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])

        G1 = 1 / np.sqrt(5) * np.array([[1, 2 * 1j], [2 * 1j, 1]])
        G2 = 1 / np.sqrt(5) * np.array([[1, 2], [-2, 1]])
        G3 = 1 / np.sqrt(5) * np.array([[1 + 2 * 1j, 0], [0, 1 - 2 * 1j]])

        G = [G1, G2, G3]

        for g in G:
            self.basis_gates.append(kron(g, I, I))
            self.basis_gates.append(kron(I, g, I))
            self.basis_gates.append(kron(I, I, g))

        sqrt_swap = np.array([[1, 0, 0, 0], [0, (1 + 1j)/2, (1 - 1j)/2, 0], [0, (1 - 1j)/2, (1 + 1j)/2, 0], [0, 0, 0, 1]])
        
        swap = sqrt_swap@sqrt_swap
        swap23 = kron(I, swap)

        B1 = kron(sqrt_swap, I)
        B2 = kron(I, sqrt_swap)
        B3 = swap23@kron(sqrt_swap, I)@swap23

        self.basis_gates.extend([B1, B2, B3])
        self.num_gates = len(self.basis_gates)

        self.probs = self.num_gates*[1]

    def __getitem__(self, index):
        return self.basis_gates[index]
    

class HRC3:

    def __init__(self):
        self.basis_gates = []
        I = np.eye(2)
        X = np.array([[0, 1], [1, 0]])

        G1 = 1 / np.sqrt(5) * np.array([[1, 2 * 1j], [2 * 1j, 1]])
        G2 = 1 / np.sqrt(5) * np.array([[1, 2], [-2, 1]])
        G3 = 1 / np.sqrt(5) * np.array([[1 + 2 * 1j, 0], [0, 1 - 2 * 1j]])
        zerozero = np.array([[1, 0], [0, 0]])
        oneone = np.array([[0, 0], [0, 1]])

        G = [G1, G2, G3]

        for g in G:
            self.basis_gates.append(kron(g, I, I))
            self.basis_gates.append(kron(I, g, I))
            self.basis_gates.append(kron(I, I, g))

        self.basis_gates.append(kron(zerozero, I, I) + kron(oneone, X, I))
        self.basis_gates.append(kron(I, zerozero, I) + kron(X, oneone, I))

        self.basis_gates.append(kron(zerozero, I, I) + kron(oneone, I, X))
        self.basis_gates.append(kron(I, I, zerozero) + kron(X, I, oneone))

        self.basis_gates.append(kron(I, zerozero, I) + kron(I, oneone, X))
        self.basis_gates.append(kron(I, I, zerozero) + kron(I, X, oneone))

        self.num_gates = len(self.basis_gates)

        self.probs = self.num_gates*[1]

    def __getitem__(self, index):
        return self.basis_gates[index]
