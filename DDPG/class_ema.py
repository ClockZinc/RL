import tensorflow as tf
import tensorflow.keras as tl
import numpy as np
class ema_V1(object):
    # tf.train.ExponentialMovingAverage 
    # try to replace it
    # shadow_variable -= (1 - decay) * (shadow_variable - variable)
    # or shadow_variable = decay * shadow_variable + (1 - decay) * variable(not recommended)
    # Reasonable values for decay are close to 1.0, typically in the multiple-nines range: 0.999, 0.9999, etc.
    # if has the num_updates
    # min(decay, (1 + num_updates) / (10 + num_updates))
    def __init__(self,decay):
        self.decay=decay

    def apply(self,shadow):
        self.shadow=shadow
    
    def average(self,variable):
        self.shadow.assign_sub((1 - self.decay) * (self.shadow - variable))
        return self.shadow
    
    def average_num(self,variable,num_updates):
        decay=min(self.decay, (1 + num_updates) / (10 + num_updates))
        self.shadow.assign_sub((1 - decay) * (self.shadow - variable))
        return self.shadow

class ema_V2(object):
    # tf.train.ExponentialMovingAverage 
    # try to replace it
    # shadow_variable -= (1 - decay) * (shadow_variable - variable)
    # or shadow_variable = decay * shadow_variable + (1 - decay) * variable(not recommended)
    # Reasonable values for decay are close to 1.0, typically in the multiple-nines range: 0.999, 0.9999, etc.
    # if has the num_updates
    # min(decay, (1 + num_updates) / (10 + num_updates))
    def __init__(self,decay):
        self.decay=decay
        self.num_updates=0

    def apply(self,shadow):
        self.shadow=shadow

    def average(self,variable):
        self.shadow.assign_sub((1 - self.decay) * (self.shadow - variable))
        return self.shadow
    
    def average_num(self,variable):
        decay=min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        self.shadow.assign_sub((1 - decay) * (self.shadow - variable))
        self.num_updates+=1
        return self.shadow