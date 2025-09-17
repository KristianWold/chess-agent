import numpy as np
import random
import os
import pickle
from itertools import count



def saver(object, filename):
    pickle.dump(object, open(filename, "wb"))


def loader(filename):
    object = pickle.load(open(filename, "rb"))

    return object
