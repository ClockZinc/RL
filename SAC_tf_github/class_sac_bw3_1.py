# env DDPG_V3
# tf version of SAC
# dual Q network Version
# reward scaled
import os,sys,time,datetime
import tensorflow as tf
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.layers import Lambda as layer_Lambda
from tensorflow.keras.models import Model

from myfun import ema_V2,ReplayBuffer_V0,PI
from tensorboardX import SummaryWriter

class SAC_Actor_Model(tf.keras.Model):
    def __init__(self,s_dim,a_dim, W_init_coe=0.1, b_init_coe=0.001,log_std_min=-10, log_std_max=2,name=''):
        super(SAC_Actor_Model, self).__init__()

        self.log_std_min        = log_std_min
        self.log_std_max        = log_std_max

        W_init                  = tf.random_normal_initializer(mean=0,stddev=W_init_coe)
        b_init                  = tf.constant_initializer(b_init_coe)

        self.feature1           = Dense(256, activation=tf.nn.relu,kernel_initializer=W_init, bias_initializer=b_init)
        self.feature2           = Dense(256, activation=tf.nn.relu,kernel_initializer=W_init, bias_initializer=b_init)

        self.mu1                = Dense(32, activation=tf.nn.relu,kernel_initializer=W_init, bias_initializer=b_init)
        self.mu_out             = Dense(a_dim, activation='linear',kernel_initializer=W_init, bias_initializer=b_init)

        self.log_std1           = Dense(32, activation='relu',kernel_initializer=W_init, bias_initializer=b_init)
        self.log_std_out        = Dense(a_dim, activation='linear',kernel_initializer=W_init, bias_initializer=b_init)

    def call(self, state):
        feature = self.feature1(state)
        feature = self.feature2(feature)

        mu      = self.mu1(feature)
        mu      = self.mu_out(mu)

        log_std = self.log_std1(feature)
        log_std = self.log_std_out(log_std)

        log_std = tf.clip_by_value(log_std,clip_value_max=2,clip_value_min=-20)

        return mu, log_std
    

def get_critic(s_dim,a_dim,name=''):
    '''
    input :s,a
    output:q_value
    with a struct of 
    [input_s,input_a]
    '''
    W_init = tf.random_normal_initializer(mean=0,stddev=0.1)
    b_init = tf.constant_initializer(0.001)
    input_s=Input(s_dim,name=name+'_state_input')
    input_a=Input(a_dim,name=name+'_action_input')
    x=tf.concat([input_s,input_a],1)
    x=Dense(units=256,activation='relu',kernel_initializer=W_init, bias_initializer=b_init,name=name+'_invisiable_layer1')(x)
    x=Dense(units=128,activation='relu',kernel_initializer=W_init, bias_initializer=b_init,name=name+'_invisiable_layer2')(x)
    x=Dense(units=32,activation='relu',kernel_initializer=W_init, bias_initializer=b_init,name=name+'_invisiable_layer3')(x)
    x=Dense(units=1,kernel_initializer=W_init, bias_initializer=b_init,name=name+'_linear_output')(x)
    return Model(inputs=[input_s,input_a],outputs=x,name=name+'_network')

def get_value(s_dim,name=''):
    '''
    input :s
    output:v_value
    '''
    W_init = tf.random_normal_initializer(mean=0,stddev=0.1)
    b_init = tf.constant_initializer(0.001)
    inputs=Input(s_dim,name=name+'_state_input')
    x=Dense(units=256,activation='relu',kernel_initializer=W_init, bias_initializer=b_init,name=name+'_invisiable_layer1')(inputs)
    x=Dense(units=128,activation='relu',kernel_initializer=W_init, bias_initializer=b_init,name=name+'_invisiable_layer2')(x)
    x=Dense(units=32,activation='relu',kernel_initializer=W_init, bias_initializer=b_init,name=name+'_invisiable_layer3')(x)
    x=Dense(units=1,kernel_initializer=W_init, bias_initializer=b_init,name=name+'_linear_output')(x)
    return Model(inputs=inputs,outputs=x,name=name+'_network')

def copy_para(from_model, to_model):
    '''
    copy the parameter from from_model to to_model
    '''
    for i,j in zip(from_model.trainable_weights,to_model.trainable_weights):
        j.assign(i)

log_sqrt_2PI = tf.math.log(tf.math.sqrt(2 * PI))
def get_log_prob(x,mu,sigma):
    '''
    得到x在Normal(loc = mu, scale = sigma)的概率的log值
    log(f(x))=-(x-mu)**2/(2*sigma**2)-tf.math.log(sigma/tf.math.sqrt(2*PI))
    '''
    log_prob=- tf.pow((x - mu),2)/ (2 * tf.pow(sigma,2) ) - tf.math.log(sigma) - log_sqrt_2PI
    return log_prob

class SAC(object):
    def __init__(self,s_dim,a_dim,a_bound,
            ENV_NAME='Pendulum-v1',     # 环境
            RANDOMSEED=3407,            # 随机种子
            LR_A=0.004,                 # actor learning rate
            LR_C=0.004,                 # critic learning rate
            GAMMA=0.9,                  # reward discount rate
            REPLAYBUFFER_SIZE=10000,    # size of replay buffer
            BATCH_SIZE=64,              # learn per batch size
            VAR=3,                      # variance of the action for exploration
            TAU=0.01,                   # soft update parameter
            POLICY_DELAY=1,
            alpha=0.2,
            test_mode=False
    ):
        ##########################################################
        self.s_dim,self.a_dim,self.a_bound=s_dim,a_dim,a_bound 
        self.alpha              = alpha
        self.ENV_NAME=ENV_NAME                   # 环境
        self.RANDOMSEED=RANDOMSEED               # 随机种子
        self.LR_A=LR_A                           # actor learning rate
        self.LR_C=LR_C                           # critic learning rate
        self.GAMMA=GAMMA                         # reward discount rate
        self.REPLAYBUFFER_SIZE=REPLAYBUFFER_SIZE # size of replay buffer
        self.BATCH_SIZE=BATCH_SIZE               # learn per batch size
        self.VAR=VAR                             # variance of the action for exploration
        self.TAU=TAU                             # soft update parameter
        self.POLICY_DELAY=POLICY_DELAY
        ##########################################################
        # self info
        self.model_name     ='model_'+os.path.basename(sys.argv[0]).split(".")[0]
        if not test_mode:
            now_time            = str(datetime.datetime.now().day)+str(datetime.datetime.now().hour)+str(datetime.datetime.now().minute)
            self.algorithm_name ='SAC'

        # 先初始化ReplayBuffer,tensorboradX.summaryWriter
        self.replay_buffer  = ReplayBuffer_V0(REPLAYBUFFER_SIZE,s_dim=s_dim,a_dim=a_dim)
        self.writer         = SummaryWriter(self.model_name+'/tensorboard'+now_time)

        # some pointers
        self.pointer_update_times   =0
        self.pointer_store_times    =0  # 指针
        
        # 建立五个网络
        self.actor          =SAC_Actor_Model(s_dim = s_dim, a_dim=a_dim, name='actor')
        self.value          =get_value(s_dim = s_dim, name='value')
        self.value_target   =get_value(s_dim = s_dim, name='value_target')
        self.q_value_1      =get_critic(s_dim = s_dim,a_dim=a_dim,name='critic_1')
        self.q_value_2      =get_critic(s_dim,a_dim,name='critic_2')

        # 赋值,初始化actor的输入端
        copy_para(self.value,self.value_target)
        _=self.actor(tf.zeros((1,s_dim)))

        # 初始化 ExponentialMovingAverage
        self.ema=ema_V2(TAU=TAU)

        # 设置优化器
        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.value_opt = tf.optimizers.Adam(LR_C)
        self.q_value_1_opt = tf.optimizers.Adam(LR_C)
        self.q_value_2_opt = tf.optimizers.Adam(LR_C)

        ############*end* __init__ *end*############
    
    def choose_action(self,state):
        '''
        input: state s
        output: the action chosen by actor
                action.shape = state.shape
        '''
        state = tf.constant([state])
        mu, log_std = self.actor(state)
        std     = tf.math.exp(log_std)
        z       = tf.random.normal(shape=mu.shape ,mean= mu, stddev= std)
        action  = tf.tanh(z)

        return action[0]
    
    def choose_action_random(self,state):
        action  = tf.random.uniform((self.a_dim,),minval=-1,maxval=1)
        return action

    def choose_action_test(self,state):
        '''
        input: state s
        output: the action chosen by actor
                action.shape = state.shape
        '''
        state = tf.constant([state])
        mu, log_std = self.actor(state)
        # std     = tf.math.exp(log_std)
        # z       = tf.random.normal(shape=mu.shape ,mean= mu, stddev= std)
        action  = tf.tanh(mu)

        return action[0]
    
    def evaluation(self,state,epsilon=1e-6):
        '''            
        get_log_prob
        get the probability of the new_action
            input : states and epsilon(defalut=1e-6,in case of the log(0))
            output: actions , log_prob , z , mu , log_std
        where
            z  : Normal.sample()
            mu : mean
            log_std : std
        '''
        mu, log_std = self.actor(state)
        sigma   = tf.math.exp(log_std)
        z       = tf.random.normal(mu.shape)
        niu     = mu + sigma * z
        action  = tf.tanh(niu)
        log_prob= get_log_prob(niu, mu= mu, sigma= sigma) - tf.math.log(1 - action**2 + epsilon) 
        return action, log_prob, z, mu, log_std

    def store_transition(self,s,a,r,s_,d):
        '''
        store transition(s,a,r,s') into the replay buffer\
        这边参考了https://blog.csdn.net/weixin_48370148/article/details/114549032
        对reward进行调整
        '''
        if r == -100:
            r = -1
        r = r * 10
        self.replay_buffer.store(s,a,r,s_,d)
        self.pointer_store_times+=1
    
    def choose_transition(self):
        bs,ba,br,bs_,bd=self.replay_buffer.sample(self.BATCH_SIZE)
        return bs,ba,br,bs_,bd

    @tf.function
    def update_value(self,bs):
        with tf.GradientTape() as tape:
            action, log_prob, z, mu, log_std = self.evaluation(bs)
            Q_value_1       = self.q_value_1([bs,action])
            Q_value_2       = self.q_value_2([bs,action])
            Q_min           = tf.math.minimum(Q_value_1,Q_value_2)
            V_value         = self.value(bs)
            target_V_value  = Q_min - log_prob*self.alpha
            V_loss          = tf.losses.mean_squared_error(V_value, target_V_value)
        V_grads = tape.gradient(V_loss, self.value.trainable_weights) # 得到偏LOSS偏omega
        self.value_opt.apply_gradients(zip(V_grads, self.value.trainable_weights))# 进行一次梯度下降（目的是降低LOSS）

    @tf.function
    def update_actor(self,bs):
        with tf.GradientTape() as tape:
            action, log_prob, z, mu, log_std = self.evaluation(bs)
            Q_value_1       = self.q_value_1([bs,action])
            Q_value_2       = self.q_value_2([bs,action])
            Q_min           = tf.math.minimum(Q_value_1,Q_value_2)
            Policy_loss     = tf.reduce_mean(self.alpha*log_prob-Q_min)
        Policy_grads = tape.gradient(Policy_loss, self.actor.trainable_weights) # 得到偏LOSS偏omega
        self.actor_opt.apply_gradients(zip(Policy_grads, self.actor.trainable_weights))# 进行一次梯度下降（目的是降低LOSS）

    @tf.function
    def update_q_value_1(self,bs,ba,br,bs_,bd):
        with tf.GradientTape() as tape:         # 通过这种形式获得梯度
            Q_value       = self.q_value_1([bs,ba])
            target_V_value= self.value_target(bs_)
            target_Q_value= br + (1 - bd) * self.GAMMA * target_V_value
            td_error = tf.losses.mean_squared_error(Q_value, target_Q_value) # TD_error MSE(y,q)
        Q_grads = tape.gradient(td_error, self.q_value_1.trainable_weights) # 得到偏LOSS偏omega
        self.q_value_1_opt.apply_gradients(zip(Q_grads, self.q_value_1.trainable_weights))# 进行一次梯度下降（目的是降低LOSS）

    @tf.function
    def update_q_value_2(self,bs,ba,br,bs_,bd):
        with tf.GradientTape() as tape:         # 通过这种形式获得梯度
            Q_value       = self.q_value_2([bs,ba])
            target_V_value= self.value_target(bs_)
            target_Q_value= br + (1 - bd) * self.GAMMA * target_V_value
            td_error = tf.losses.mean_squared_error(Q_value, target_Q_value) # TD_error MSE(y,q)
        Q_grads = tape.gradient(td_error, self.q_value_2.trainable_weights) # 得到偏LOSS偏omega
        self.q_value_2_opt.apply_gradients(zip(Q_grads, self.q_value_2.trainable_weights))# 进行一次梯度下降（目的是降低LOSS）

    def learn(self):
        # get transit
        bs,ba,br,bs_,bd=self.choose_transition()

        self.update_value(bs)
        self.update_q_value_1(bs,ba,br,bs_,bd)
        self.update_q_value_2(bs,ba,br,bs_,bd)

        if (self.pointer_update_times % self.POLICY_DELAY ==0):
            self.update_actor(bs)
        # ema
        self.ema.update_target_network(network=self.value,target_network=self.value_target)

        self.pointer_update_times+=1
        
    # 保存变量和load变量
    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists(self.model_name):
            os.makedirs(self.model_name)
            

        self.actor.save_weights(self.model_name+'/'+self.algorithm_name+'_actor.hdf5')
        self.value.save_weights(self.model_name+'/'+self.algorithm_name+'_value.hdf5')
        self.value_target.save_weights(self.model_name+'/'+self.algorithm_name+'_value_target.hdf5')
        self.q_value_1.save_weights(self.model_name+'/'+self.algorithm_name+'_q_1_target.hdf5')
        self.q_value_2.save_weights(self.model_name+'/'+self.algorithm_name+'_q_2_target.hdf5')
        print("\n------------\n model saved \n------------\n")

    def load_ckpt(self,mode=False):
        """
        load trained weights
        :return: None
        """
        self.actor.load_weights(self.model_name+'/'+self.algorithm_name+'_actor.hdf5')
        if mode:
            self.value.load_weights(self.model_name+'/'+self.algorithm_name+'_value.hdf5')
            self.value_target.load_weights(self.model_name+'/'+self.algorithm_name+'_value_target.hdf5')
            self.q_value_1.load_weights(self.model_name+'/'+self.algorithm_name+'_q_1_target.hdf5')
            self.q_value_2.load_weights(self.model_name+'/'+self.algorithm_name+'_q_2_target.hdf5')
        print("\n------------\n model loaded \n------------\n")