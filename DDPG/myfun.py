import tensorflow as tf


class ema_V0(object):
    def __init__(self,TAU):
        self.TAU=TAU
    # shadow_variable -= (1 - decay) * (shadow_variable - variable)
    # shadow_variable = decay * shadow_variable + (1 - decay) * variable(not recommended)
    def update_target_network_V0(self,network, target_network):
        TAU=self.TAU
        for shadow_variable, variable in zip(target_network.weights, network.weights):
            shadow_variable.assign(TAU * variable + (1.0 - TAU) * shadow_variable)

    def update_target_network_V1(self,network, target_network):
        TAU=self.TAU
        for shadow_variable, variable in zip(target_network.weights, network.weights):
            shadow_variable.assign_sub(TAU *(shadow_variable-variable))

class ema_V1(object):
    # tf.train.ExponentialMovingAverage 
    # try to replace it
    # shadow_variable -= (1 - decay) * (shadow_variable - variable)
    # or shadow_variable = decay * shadow_variable + (1 - decay) * variable(not recommended)
    # Reasonable values for decay are close to 1.0, typically in the multiple-nines range: 0.999, 0.9999, etc.
    # if has the num_updates
    # min(decay, (1 + num_updates) / (10 + num_updates))

    def __init__(self,TAU):
        # TAU is the soft update rate
        self.decay=1-TAU

    def update_target_network_V1(self,network, target_network):
        decay=self.decay
        for shadow_variable, variable in zip(target_network.weights, network.weights):
            shadow_variable.assign_sub((1-decay)*(shadow_variable-variable))

class ema_V2(object):
    # tf.train.ExponentialMovingAverage 
    # try to replace it
    # shadow_variable -= (1 - decay) * (shadow_variable - variable)
    # or shadow_variable = decay * shadow_variable + (1 - decay) * variable(not recommended)
    # Reasonable values for decay are close to 1.0, typically in the multiple-nines range: 0.999, 0.9999, etc.
    # if has the num_updates
    # min(decay, (1 + num_updates) / (10 + num_updates))
    def __init__(self,TAU,update_rate=1):
        # TAU is the soft update rate
        self.decay=1-TAU
        self.num_updates=0
        self.update_rate=update_rate

    def update_target_network(self,network, target_network):
        decay=self.decay
        for shadow_variable, variable in zip(target_network.weights, network.weights):
            shadow_variable.assign_sub((1-decay)*(shadow_variable-variable))
    
    def update_target_network_num(self,network, target_network):
        num_updates=self.num_updates
        decay=min(self.decay, (1 + num_updates) / (10 + num_updates))
        for shadow_variable, variable in zip(target_network.weights, network.weights):
            shadow_variable.assign_sub((1-decay)*(shadow_variable-variable))
        self.num_updates+=self.update_rate
