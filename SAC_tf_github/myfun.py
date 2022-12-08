# import tensorflow as tf
import numpy as np
import random
import time,multiprocessing,os

PI = np.pi
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


class ReplayBuffer_V0:
    """
    a ring buffer for storing transitions and sampling for training
    :state: (state_dim,)
    :action: (action_dim,)
    :reward: (,), scalar
    :next_state: (state_dim,)
    :done: (,), scalar (0 and 1) or bool (True and False)
    """
    '''
    需要注意的是在小batch size<=10000 的时候 random sample 更快
    超出后还是numpy.rando.choice更快
    '''
    '''
    主要实现存储、采样功能
    '''

    def __init__(self, capacity,s_dim,a_dim):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.s_dim=s_dim
        self.a_dim=a_dim

    def store(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        transition=np.hstack((state, action, reward, next_state, done)).astype(np.float32)
        self.buffer[self.position] = (transition)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        transition = np.stack(batch)  # stack for each element
        bs  = transition[:, :self.s_dim]                         #从bt获得数据s
        ba  = transition[:, self.s_dim:self.s_dim + self.a_dim]  #从bt获得数据a
        br  = transition[:, -self.s_dim - 2:-self.s_dim-1]         #从bt获得数据r
        bs_ = transition[:, -self.s_dim -1:-1]
        bd  = transition[:,-1:]
        """ 
        
        the * serves as unpack: sum(a,b) <=> batch=(a,b), sum(*batch) ;
        zip: a=[1,2], b=[2,3], zip(a,b) => [(1, 2), (2, 3)] ;
        the map serves as mapping the function on each list element: map(square, [2,3]) => [4,9] ;
        np.stack((1,2)) => array([1, 2])
        """
        return bs, ba, br, bs_, bd
    # def store_data(self,transition):
    #     if len(self.buffer) < self.capacity:
    #         self.buffer.append(None) 
    #     self.buffer[self.position] = (transition)
    #     self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def __len__(self):
        return len(self.buffer)



def PrintProgressBar(t1,i,MAX_EPISODES,ep_reward):
    # progress = i / MAX_EPSODES * 100 %
    cost_time=time.time() - t1
    rest_time=(MAX_EPISODES-i)*cost_time
    hour=rest_time / 3600
    minute=rest_time % 3600 /60
    print(
        '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}|est: {}h {:2.1f}min'.format(
            i, MAX_EPISODES, ep_reward,cost_time,np.int32(hour),minute
        ), end=''
    )


def PrintProgressBar_init(t1,k,INIT_SIZE):
    print(
    '\rinitializing : | Episode: {:3.0f}%  | Running Time: {:.4f}'.format(
        k/INIT_SIZE*100,
        time.time() - t1
    ), end=''
)
def PrintProgressBAR_V0(t1,i,MAX_EPISODES,ep_reward):
    print(
        '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
            i, MAX_EPISODES, ep_reward,
            time.time() - t1
        )
    )

