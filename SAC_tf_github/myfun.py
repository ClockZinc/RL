# import tensorflow as tf
import numpy as np
import random
import time,multiprocessing,os,imageio

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
        '\rEpisode: {:4d}/{:4d}  | Episode Reward: {:9.4f}  | Running Time: {:2.4f}|est: {}h {:2.1f}min'.format(
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
        '\rEpisode: {:4d}/{:4d}  | Episode Reward: {:9.4f}  | Running Time: {:2.4f}'.format(
            i, MAX_EPISODES, ep_reward,
            time.time() - t1
        )
    )



def create_video(filename,video_name,env, actor, fps=30,max_step=10000):
    if not os.path.exists(filename):
        os.makedirs(filename)
    filename+="/"+video_name
    with imageio.get_writer(filename, fps=fps) as video:
        done = False
        state = env.reset()
        frame = env.render(mode="rgb_array")
        video.append_data(frame)
        while not done:    
            state = np.expand_dims(state, axis=0)
            action = actor(state)[0]
            state, _, done, _ = env.step(action)
            frame = env.render(mode="rgb_array")
            video.append_data(frame)
            if not max_step:
                break
            else:
                max_step-=1




class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity
        self.Full_flag=False

    def add(self, p, data):
        self.tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(self.tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0
            self.Full_flag=True

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    def total_p(self):
        return self.tree[0]  # the root
    
class Memory_V1(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """


    def __init__(self, capacity,s_dim,a_dim,epsilon = 0.01,alpha = 0.6,beta = 0.4,beta_increment_per_sampling = 0.0001,abs_err_upper = 1.):
        self.tree = SumTree(capacity)
        self.s_dim=s_dim
        self.a_dim=a_dim
        self.epsilon = epsilon  # small amount to avoid zero priority
        self.alpha = alpha  # [0~1] convert the importance of TD error to priority
        self.beta = beta  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.abs_err_upper = abs_err_upper # clipped abs error
        self.tree_size=2 * capacity - 1

    def store(self,s,a,r,s_,done):
        transition=np.hstack((s,a,r,s_,done)).astype(np.float32)# 横向堆叠这些变量
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size),np.float32), np.empty((n, 1),dtype=np.float32)
        pri_seg = self.tree.total_p() / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        if (not self.tree.Full_flag): # 如果没满的话
            last_idx=self.tree.tree_idx# 最后一个的index就是确定的最后新加进去的
        else:
            last_idx=self.tree_size
        min_prob = np.min(self.tree.tree[-self.tree.capacity:last_idx]) / self.tree.total_p() # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p()
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data

        bs = b_memory[:, :self.s_dim]                         #从bt获得数据s
        ba = b_memory[:, self.s_dim:self.s_dim + self.a_dim]  #从bt获得数据a
        br = b_memory[:, -self.s_dim - 2:-self.s_dim-1]         #从bt获得数据r
        bs_ = b_memory[:, -self.s_dim-1:-1]
        bd = b_memory[:, -1:]
        return bs,ba,br,bs_,bd,b_idx,ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)