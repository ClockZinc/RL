# -*- coding: utf-8 _*_

'''
improved from 4.3
保存文件中增加了当前的版本名字
POLICY_DELAY=2
'''

# 系统级的函数
import os,time,datetime,argparse
# 尝试通过网上的方法屏蔽tensorflow的警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 基本配置
import numpy as np
import gym
import matplotlib.pyplot as plt

## tensorflow
import tensorflow as tf
# 一些层和模型的搭建
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.layers import Lambda as layer_Lambda
from tensorflow.keras.models import Model

# 自己编写的软更新函数
from myfun import ema_V2

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)    # 未键入--train时，args.train=Flase
parser.add_argument('--test', dest='test', action='store_true') # 未键入--test时，args.test=Flase
parser.add_argument('--seed',dest='seed',default=3407,type=int) # 未键入--seed 123 时，args.seed=3047
args = parser.parse_args()  # 整合到args内

##########################################################################################

ENV_NAME='Pendulum-v1'  # 环境
RANDOMSEED=args.seed   # 随机种子

LR_A=0.001              # actor learning rate
LR_C=0.002              # critic learning rate
GAMMA=0.9               # reward discount rate
REPLAYBUFFER_SIZE=10000 # size of replay buffer
BATCH_SIZE=32           # learn per batch size
VAR=3                   # variance of the action for exploration
TAU=0.001               # soft update parameter
MAX_EPISODES=200        # maximum exploration times(from reset() to done/max_EP_STEP)
MAX_EP_STEPS=500        # maximum step per episode
TEST_PER_EPISODES=10    # step that test the model

POLICY_DELAY=2          # train the actor per POLICY_DELAY of update of critic

##########################################################################################

class DDPG(object):
    
    '''
    replay_buffer

    '''

    def __init__(self,s_dim,a_dim,a_bound):
        # 先初始化transition (s_t,a_t,r_t,s_{t+1})
        # numpy.zeros((row,col),dtype)
        self.replay_buffer=np.zeros((REPLAYBUFFER_SIZE,s_dim*2+a_dim*1+1),dtype=np.float32)
        self.pointer=0  # 指针
        self.s_dim,self.a_dim,self.a_bound=s_dim,a_dim,a_bound
        self.critic_update_times=0

        W_init = tf.random_normal_initializer(mean=0,stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        # 建立四个网络，先定义两个网络架构
        def get_actor(input_state_shape,output_shape,name=''):
            lb=np.array(output_shape[0])
            ub=np.array(output_shape[1])
            gap=ub-lb
            '''
            input:s[n,s_dim]
            output:a[n,a_dim] in a_bound

            with a struct of
            input 64+dense 64+dense 1+tanh 1+(layer_Lambda)

            where n is the number of states input
            '''
            inputs=Input(input_state_shape,name=name+'_input')
            x=Dense(units=64,activation='relu',kernel_initializer=W_init, bias_initializer=b_init,name=name+'_invisiable_layer1')(inputs)
            x=Dense(units=64,activation='relu',kernel_initializer=W_init, bias_initializer=b_init,name=name+'_invisiable_layer2')(x)
            x=Dense(units=1,activation='sigmoid',kernel_initializer=W_init, bias_initializer=b_init,name=name+'_sigmoid_layer')(x)
            x=layer_Lambda(lambda x:lb+gap*x,name=name+'_Lambda_layer')(x)
            return Model(inputs=inputs,outputs=x,name=name+'_network')
        
        def get_critic(input_s_shape,input_a_shape,name=''):
            '''
            input :s,a
            output:q_value
            with a struct of 
            [input_s,input_a]
            '''
            input_s=Input(input_s_shape,name=name+'_state_input')
            input_a=Input(input_a_shape,name=name+'_action_input')
            x=tf.concat([input_s,input_a],1)
            x=Dense(units=64,activation='relu',kernel_initializer=W_init, bias_initializer=b_init,name=name+'_invisiable_layer1')(x)
            x=Dense(units=64,activation='relu',kernel_initializer=W_init, bias_initializer=b_init,name=name+'_invisiable_layer2')(x)
            x=Dense(units=1,kernel_initializer=W_init, bias_initializer=b_init,name=name+'_linear_output')(x)
            return Model(inputs=[input_s,input_a],outputs=x,name=name+'_network')
        
        # 帮助target网络硬更新
        def copy_para(from_model, to_model):
            '''
            copy the parameter from from_model to to_model
            '''
            for i,j in zip(from_model.trainable_weights,to_model.trainable_weights):
                j.assign(i)
        
        # 建立四个网络
        self.actor          =get_actor(s_dim,output_shape=a_bound,name='actor')
        self.actor_target   =get_actor(s_dim,output_shape=a_bound,name='actor_target')
        self.critic         =get_critic(s_dim,a_dim,name='critic')
        self.critic_target  =get_critic(s_dim,a_dim,name='critic_target')

        # 赋值
        copy_para(self.actor,self.actor_target)
        copy_para(self.critic,self.critic_target)

        # 初始化 ExponentialMovingAverage
        self.ema=ema_V2(TAU=TAU)
        # self.ema=ema_V2(decay=1-TAU)

        # 设置优化器
        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)

        ############*end* __init__ *end*############

    # 动作选择函数
    def choose_action(self,s):
        '''
        input: state s
        output: the action chosen by actor
        '''
        return self.actor(np.array([s], dtype=np.float32))[0]# 限定输入为float32,输出为一维向量
    
    # 存储transition
    def store_transition(self,s,a,r,s_):
        '''
        store transition(s,a,r,s') into the replay buffer
        '''
        s =s.astype(np.float32)
        s_ =s_.astype(np.float32)
        
        transition=np.hstack((s,a,r,s_))# 横向堆叠这些变量

        index = self.pointer % REPLAYBUFFER_SIZE
        self.replay_buffer[index,:]=transition
        self.pointer+=1
    
    # 选取transition
    def choose_transiton(self):
        indices = np.random.choice(REPLAYBUFFER_SIZE,size=BATCH_SIZE) # 先确定一批transition的位置
        bt = self.replay_buffer[indices, :]                    #根据indices，选取数据bt，相当于随机
        bs = bt[:, :self.s_dim]                         #从bt获得数据s
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]  #从bt获得数据a
        br = bt[:, -self.s_dim - 1:-self.s_dim]         #从bt获得数据r
        bs_ = bt[:, -self.s_dim:]
        return bs,ba,br,bs_

    # 通过tf.function 提升求导时间
    @tf.function #提升performance，减少训练时间
    def update_actor(self,bs):
        with tf.GradientTape() as tape:
            a = self.actor(bs)
            q = self.critic([bs, a])
            a_loss = -tf.reduce_mean(q)  # 【敲黑板】：注意这里用负号，是梯度上升！也就是离目标会越来越远的，就是越来越大。
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights) # 得到*负值*偏Q偏theta
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))# 进行一次梯度下降（但是实际是梯度上升）
    
    # 通过tf.function 提升求导时间
    @tf.function #提升performance，减少训练时间
    def update_critic(self,bs,ba,br,bs_):
        with tf.GradientTape() as tape:         # 通过这种形式获得梯度
            a_ = self.actor_target(bs_)         
            q_ = self.critic_target([bs_, a_])
            y = br + GAMMA * q_
            q = self.critic([bs, ba])
            td_error = tf.losses.mean_squared_error(y, q) # TD_error MSE(y,q)
        c_grads = tape.gradient(td_error, self.critic.trainable_weights) # 得到偏LOSS偏omega
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))# 进行一次梯度下降（目的是降低LOSS）

    # 学习一轮
    # @tf.function #提升performance，减少训练时间
    def learn(self):
        '''
        upgrade the parameter using TD_error and Policy Gradient Descent
        '''
        bs,ba,br,bs_=self.choose_transiton()

        # 更新critic先
        self.update_critic(bs,ba,br,bs_)
        self.critic_update_times+=1
        # 更新actor 其频率应该低于critic
        if (self.critic_update_times % POLICY_DELAY ==0):
            self.update_actor(bs)

        # 更新一次参数
        self.ema.update_target_network(network=self.critic,target_network=self.critic_target)
        self.ema.update_target_network(network=self.actor,target_network=self.actor_target)
            

    # 保存变量和load变量
    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')
        else:
            now=datetime.datetime.now()
            now=now.strftime('%Y-%m-%d-%H%M%S')
            new_name='model_'+now
            old_name='model'
            os.rename(old_name,new_name)
            os.makedirs(old_name)
            
        
        self.actor.save_weights('model/ddpg_actor.hdf5')
        self.actor_target.save_weights('model/ddpg_actor_target.hdf5')
        self.critic.save_weights('model/ddpg_critic.hdf5')
        self.critic_target.save_weights('model/ddpg_critic_target.hdf5')

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        self.actor.load_weights('model/ddpg_actor.hdf5')
        self.actor_target.load_weights('model/ddpg_actor_target.hdf5')
        self.critic.load_weights('model/ddpg_critic.hdf5')
        self.critic_target.load_weights('model/ddpg_critic_target.hdf5')
    
    def save_log(self,t):
        '''
        save some message
        '''
        fileID=open('model/result.txt',mode='w+')
        print(
            f'python name={os.path.basename(__file__)}',
            f'ENV_NAME={ENV_NAME}',
            f'RANDOMSEED={RANDOMSEED}',
            f'LR_A={LR_A}',
            f'LR_C={LR_C}',
            f'GAMMA={GAMMA}',
            f'REPLAYBUFFER_SIZE={REPLAYBUFFER_SIZE}',
            f'BATCH_SIZE={BATCH_SIZE}',
            f'VAR={VAR}',
            f'MAX_EPISODES={MAX_EPISODES}',
            f'MAX_EP_STEPS={MAX_EP_STEPS}',
            f'TEST_PER_EPISODES={TEST_PER_EPISODES}',
            f'TAU={TAU}',
            f'POLICY_DELAY={POLICY_DELAY}'
            f'\nRunning time: {t}',
            sep='\n',file=fileID)
        fileID.close()
    
    def save_pic(self,plt):
        plt.savefig('model/result.png')

# init
env=gym.make(ENV_NAME)
env.unwrapped

# set seed 方便复现结果
env.reset(seed=RANDOMSEED)
np.random.seed(RANDOMSEED)
tf.random.set_seed(RANDOMSEED)

# 得到动作空间参数
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound_lb = env.action_space.low
a_bound_ub = env.action_space.high
a_bound = [a_bound_lb,a_bound_ub]

# 初始化DDPG
ddpg=DDPG(s_dim,a_dim,a_bound)

# 开始训练 --train
if args.train:
    # 部分信息输出
    print('s_dim',s_dim)
    print('a_dim',a_dim)
    print('a_bound',a_bound)
    print('\n**********',
    f'\n Random Seed is :\t {RANDOMSEED}',
    '\n**********')

    reward_buffer = []      #用于记录每个EP的reward，统计变化
    t0 = time.time()        #统计时间
    for i in range(MAX_EPISODES):
        t1 = time.time()
        s = env.reset()
        ep_reward = 0       #记录当前EP的reward
        for j in range(MAX_EP_STEPS):
            # Add exploration noise
            a = ddpg.choose_action(s)       #这里很简单，直接用actor估算出a动作

            # 为了能保持开发，这里用了另外一种方式增加探索。
            # 因此需要需要以a为均值，VAR为标准差，建立正态分布，再从正态分布采样出a
            # 因为a是均值，所以a的概率是最大的。但a相对其他概率由多大，是靠VAR调整。这里我们其实可以增加更新VAR，动态调整a的确定性
            # 然后进行裁剪
            a = np.clip(np.random.normal(a, VAR), -2, 2)  
            # 与环境进行互动
            s_, r, done, info = env.step(a)

            # 保存s，a，r，s_
            ddpg.store_transition(s, a, r / 10, s_)

            # 第一次数据满了，就可以开始学习
            if ddpg.pointer > REPLAYBUFFER_SIZE:
                ddpg.learn()

            #输出数据记录
            s = s_  
            ep_reward += r  #记录当前EP的总reward
            if j == MAX_EP_STEPS - 1:
                print(
                    '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                        i, MAX_EPISODES, ep_reward,
                        time.time() - t1
                    ), end=''
                )
            plt.show()
        # test
        if i and not i % TEST_PER_EPISODES:
            t1 = time.time()
            s = env.reset()
            ep_reward = 0
            for j in range(MAX_EP_STEPS):

                a = ddpg.choose_action(s)  # 注意，在测试的时候，我们就不需要用正态分布了，直接一个a就可以了。
                s_, r, done, info = env.step(a)

                s = s_
                ep_reward += r
                if j == MAX_EP_STEPS - 1:
                    print(
                        '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                            i, MAX_EPISODES, ep_reward,
                            time.time() - t1
                        )
                    )

                    reward_buffer.append(ep_reward)

        if reward_buffer:
            plt.ion()
            plt.cla()
            plt.title('DDPG')
            plt.plot(np.array(range(len(reward_buffer))) * TEST_PER_EPISODES, reward_buffer)  # plot the episode vt
            plt.xlabel('episode steps')
            plt.ylabel('normalized state-action value')
            plt.ylim(-MAX_EP_STEPS*10, 100)# MAX_EP_STEPS=5000 -50000 # MAX_EP_STEPS=200 -2000
            plt.show()
            plt.pause(0.1)
    print('\n***end***')
    plt.ioff()    
    print('\nRunning time: ', time.time() - t0)

    ddpg.save_ckpt()
    ddpg.save_pic(plt)
    ddpg.save_log(t=(time.time() - t0))
    

    # print('store time cost is:{:.4f}'.format(time.perf_counter()-time_start_store_weights))
    

# test
if args.test:
    ddpg.load_ckpt()
    while True:
        s = env.reset()
        for i in range(MAX_EP_STEPS):
            env.render()
            s, r, done, info = env.step(ddpg.choose_action(s))
            if done:
                break




