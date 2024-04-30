"""
ya boys trying to create a Mask R-CNN model

copyright (c) 2024 Ben's bed any violatioin should pay in marlboro red cartons

mwehehehehehe
 
"""

import os
import random as rd
import datetime
import re
import math 
import logging
import multiprocessing
import numpy as np 
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM

from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

