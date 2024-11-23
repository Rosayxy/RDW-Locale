from utils.constants import *
from utils.space import *
import math
from utils.misc import calc_move_with_gain

def calc_gain(user, physical_space, delta):
    # implement your own logic
    return MIN_TRANS_GAIN, MIN_ROT_GAIN, MIN_CUR_GAIN_R, 1

def update_reset(user, physical_space, delta):
    # implenent reset logic
    user.angle = ( user.angle + math.pi ) % (2 * math.pi)
    return user