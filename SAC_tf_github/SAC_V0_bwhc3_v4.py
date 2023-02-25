'''
This Version uses SAC 2 temp auto adjust
'''    
import time,argparse,os,logging
import gym
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
os.environ['CUDA_VISIBLE_DEVICES']='2'
import myfun
from class_PER_sac_2_temp_auto_adjust import SAC
# from class_sac_2_temp_auto_adjust import SAC
'''
测试后发现，二者训练速度相差不大,2.0在Pendulum-v1的环境下效果好一点
增大网络层数将会增加训练时间，但是网络规模的增加并不会，即神经元个数
'''


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('-t','--train', dest='train', action='store_true', default=False)    # 未键入--train时，args.train=Flase
parser.add_argument('-e','--test', dest='test', action='store_true',default=False) # 未键入--test时，args.test=Flase
parser.add_argument('--seed',dest='seed',default=3407,type=int) # 未键入--seed 123 时，args.seed=3047
parser.add_argument('-l','--load',dest='load',action='store_true',default=False) # 未键入--seed 123 时，args.seed=3047
parser.add_argument('-v','--video',dest='video',action='store_true',default=False) # 未键入--seed 123 时，args.seed=3047
args = parser.parse_args()  # 整合到args内


ENV_NAME='Pendulum-v1'  # 环境
# ENV_NAME='BipedalWalker-v3'  # 环境
# ENV_NAME='BipedalWalkerHardcore-v3'  # 环境
RANDOMSEED=args.seed   # 随机种子

LR_A=3e-4               # actor learning rate
LR_C=3e-4              # critic learning rate
GAMMA=0.995              # reward discount rate
REPLAYBUFFER_SIZE=100000# size of replay buffer
BATCH_SIZE=32           # learn per batch size
FULL_COE=1                # when Replaybuffer is full then the MAX_LE_STEPS *=FULL_COE
VAR=1                   # variance of the action for exploration
TAU=0.005                # soft update parameter 不能太小
MAX_EPISODES=200        # maximum exploration times(from reset() to done/max_EP_STEP)
MAX_LE_STEPS=200        # maximum step per episode
MAX_EP_STEPS=200        # maximum step per episode
TEST_PER_EPISODES=10    # step that test the model

POLICY_DELAY=1          # train the actor per POLICY_DELAY of update of critic
INIT_SIZE=min(BATCH_SIZE*MAX_LE_STEPS*2,REPLAYBUFFER_SIZE)
# INIT_SIZE=10000


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
        POLICY_DELAY=POLICY_DELAY,
        writer_mode=args.train)
if args.load:
    sac.load_ckpt(mode=True)
# 开始训练 --train
if args.train:
    writer = sac.writer
    #####################      INIT        #####################
    t1 = time.time()
    ep_reward = 0       #记录当前EP的reward

    while sac.pointer_store_times < INIT_SIZE:
        s = env.reset()
        for init_j in range(MAX_EP_STEPS):
            # Add exploration noise
            a = sac.choose_action_random(s)       #这里很简单，直接用actor估算出a动作
            
            s_, r, done, info = env.step(a)
            sac.store_transition(s, a, r/10, s_,done)


            #输出数据记录
            s = s_  
            ep_reward += r  #记录当前EP的总reward
            if done:
                break
            myfun.PrintProgressBar_init(t1,sac.pointer_store_times,INIT_SIZE)
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
            sac.store_transition(s, a, r/10, s_,done)
            if sac.pointer_store_times == REPLAYBUFFER_SIZE:
                MAX_LE_STEPS *= FULL_COE
                print("\n FULL_COE \n")

            #输出数据记录
            s = s_  
            ep_reward += r  #记录当前EP的总reward
            # plt.show()
            if done: break
        writer.add_scalar('Reward',ep_reward,global_step=i)
        ####################   END   ####################
        # 第一次数据满了，就可以开始学习
        if sac.pointer_store_times > INIT_SIZE:

            for LE_STEPS in range(MAX_LE_STEPS):    
                sac.learn()

        myfun.PrintProgressBar(t1,i,MAX_EPISODES,ep_reward)


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
                    myfun.PrintProgressBAR_V0(t1,i,MAX_EPISODES,ep_reward)
                    reward_buffer.append(ep_reward)
                if done:break
        ####################   END   ####################

        ####################   SAVE   ######################
        if i and (not i % 100) : 
            sac.save_ckpt()
            s = env.reset()
            for i in range(MAX_EP_STEPS):
                env.render()
                s, r, done, info = env.step(sac.choose_action_test(s))
                if done:
                    break
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
            # s, r, done, info = env.step(sac.choose_action(s))
            if done:
                break

# gene video
if args.video:
    logging.getLogger().setLevel(logging.ERROR)
    sac.load_ckpt()
    filename = "./video"
    video_name = ENV_NAME+sac.now_time+".mp4"
    myfun.create_video(filename, video_name, env, actor=sac.choose_action_test, fps=env.metadata["render_fps"])
