'''
尝试下低batch_size能不能收敛
'''
import time,argparse,os
import gym
import numpy as np
import tensorflow as tf
from myfun import PrintProgressBar_init,PrintProgressBar
from class_sac_bw3_1 import SAC


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('-t','--train', dest='train', action='store_true', default=False)    # 未键入--train时，args.train=Flase
parser.add_argument('-e','--test', dest='test', action='store_true',default=False) # 未键入--test时，args.test=Flase
parser.add_argument('--seed',dest='seed',default=3407,type=int) # 未键入--seed 123 时，args.seed=3047
parser.add_argument('-l','--load',dest='load',action='store_true',default=False) # 未键入--seed 123 时，args.seed=3047
args = parser.parse_args()  # 整合到args内


# ENV_NAME='Pendulum-v1'  # 环境
ENV_NAME='BipedalWalker-v3'  # 环境
RANDOMSEED=args.seed   # 随机种子

LR_A=1e-3               # actor learning rate
LR_C=1e-3               # critic learning rate
GAMMA=0.995              # reward discount rate
REPLAYBUFFER_SIZE=100000# size of replay buffer
BATCH_SIZE=128           # learn per batch size
VAR=1                   # variance of the action for exploration
TAU=0.005                # soft update parameter 不能太小
MAX_EPISODES=500        # maximum exploration times(from reset() to done/max_EP_STEP)
MAX_LE_STEPS=200        # maximum step per episode
MAX_EP_STEPS=5000        # maximum step per episode
TEST_PER_EPISODES=10    # step that test the model

POLICY_DELAY=1          # train the actor per POLICY_DELAY of update of critic
INIT_SIZE=10000


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action

# init
env=NormalizedActions(gym.make(ENV_NAME)) 
env.unwrapped

# set seed 方便复现结果
env.reset(seed=RANDOMSEED)
tf.random.set_seed(RANDOMSEED)

# 得到动作空间参数
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound_lb = env.action_space.low
a_bound_ub = env.action_space.high
a_bound = [a_bound_lb,a_bound_ub]

# 初始化sac
sac=SAC(s_dim,a_dim,a_bound, 
        ENV_NAME=ENV_NAME ,                  # 环境
        RANDOMSEED=RANDOMSEED       ,        # 随机种子
        LR_A=LR_A                    ,       # actor learning rate
        LR_C=LR_C                    ,      # critic learning rate
        GAMMA=GAMMA                     ,    # reward discount rate
        REPLAYBUFFER_SIZE=REPLAYBUFFER_SIZE, # size of replay buffer
        BATCH_SIZE=BATCH_SIZE         ,      # learn per batch size
        VAR=VAR                         ,    # variance of the action for exploration
        TAU=TAU                        ,     # soft update parameter
        POLICY_DELAY=POLICY_DELAY)
if args.load:
    sac.load_ckpt(mode=True)
# 开始训练 --train
if args.train:
    # 部分信息输出
    print('\n------------',
    f'\n Random Seed is :\t {RANDOMSEED}',
    '\n------------')
    writer = sac.writer
    #####################      INIT        #####################
    t1 = time.time()
    ep_reward = 0       #记录当前EP的reward

    while sac.pointer_store_times < INIT_SIZE:
        s = env.reset()
        for init_j in range(MAX_EP_STEPS):
            # Add exploration noise
            # a = sac.choose_action(s)       #这里很简单，直接用actor估算出a动作
            # a = np.clip(np.random.normal(a, VAR), -MAX_YAW_CONTROL, MAX_YAW_CONTROL)  
            a = sac.choose_action_random(s)       #这里很简单，直接用actor估算出a动作
            # a = np.random.uniform(low=a_bound_lb,high=a_bound_ub,size=(4,))
            
            s_, r, done, info = env.step(a)
            sac.store_transition(s, a, r, s_,done)


            #输出数据记录
            s = s_  
            ep_reward += r  #记录当前EP的总reward
            if done:
                break
            PrintProgressBar_init(t1,sac.pointer_store_times,INIT_SIZE)
    #####################    __END__       #####################
    reward_buffer = []      #用于记录每个EP的reward，统计变化
    t0 = time.time()        #统计时间
    for i in range(MAX_EPISODES):
        t1 = time.time()
        s = env.reset()
        ep_reward = 0       #记录当前EP的reward
        ####################  TRAIN  ####################
        for j in range(MAX_EP_STEPS):
            # Add exploration noise
            a = sac.choose_action(s)       #这里很简单，直接用actor估算出a动作
            s_, r, done, info = env.step(a)

            # 保存s，a，r，s_
            sac.store_transition(s, a, r, s_,done)

            #输出数据记录
            s = s_  
            ep_reward += r  #记录当前EP的总reward
            # plt.show()
            if done: break
        writer.add_scalar('Reward',ep_reward,global_step=i)
        ####################   END   ####################
        # 第一次数据满了，就可以开始学习
        if sac.pointer_store_times > INIT_SIZE:
            # if VAR>0.1:
            #     VAR*=0.9999
            # else:
            #     VAR=0.1
            for LE_STEPS in range(MAX_LE_STEPS):    
                sac.learn()

        PrintProgressBar(t1,i,MAX_EPISODES,ep_reward)


        ####################  TEST  ####################
        if i and not i % TEST_PER_EPISODES:
            t1 = time.time()
            s = env.reset()
            ep_reward = 0
            for j in range(MAX_EP_STEPS):

                a = sac.choose_action_test(s)  # 注意，在测试的时候，我们就不需要用正态分布了，直接一个a就可以了。
                s_, r, done, info = env.step(a)

                s = s_
                ep_reward += r
                if j == MAX_EP_STEPS - 1 or done:
                    print(
                        '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                            i, MAX_EPISODES, ep_reward,
                            time.time() - t1
                        )
                    )
                    reward_buffer.append(ep_reward)
                if done:break
        ####################   END   ####################

        ####################   SAVE   ######################
        if i and (not i % 100) : 
            sac.save_ckpt()
        ####################   END   ######################


    print('\n***end***')
    print('\nRunning time: ', time.time() - t0)
    sac.save_ckpt()
    

    # print('store time cost is:{:.4f}'.format(time.perf_counter()-time_start_store_weights))
    

# test
if args.test:
    sac.load_ckpt()
    while True:
        s = env.reset()
        for i in range(MAX_EP_STEPS):
            env.render()
            s, r, done, info = env.step(sac.choose_action_test(s))
            if done:
                break
