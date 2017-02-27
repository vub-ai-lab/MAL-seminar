import tensorflow as tf
import numpy as np
from experience_replay import ReplayDB


def copy_vars(to_vars, from_vars):
    op_list = []
    for v1, v2 in zip(to_vars, from_vars):
        op_list.append(tf.assign(v1, v2))
    return op_list


class Agent(object):
    ''' Learning agent base class
    '''

    def __init__(self, num_actions):
        self._n_actions = num_actions
        self._n_steps = 0

    def num_actions(self):
        '''Size of agent's action set '''
        return self._n_actions

    def num_steps(self):
        '''Number of learning steps performed'''
        return self._n_steps

    def select_action(self, obs, **kwargs):
        '''select action for given state observation

        Args:
            obs: state observation
        '''
        return np.random.choice(self.num_actions())

    def update(self, s, a, r, t):
        ''' Performs one learning step

        Args:
            s (ndarray, float): state observation
            a (int): action taken
            r (float): reward recieved
            t (bool): final transition of episode?
        '''
        self._n_steps += 1


class DQNAgent(Agent):
    ''' Deep Q-learning agent


    Args:
        network_fn (callable): constructor for the deep Q-network. Should take
            inputs (tf.placeholder) and n_actions (int) as inputs and return
            a tf.tensor representing the qvalues corresponding to the inputs
    '''

    def __init__(self, network_fn, num_actions, obs_shape, alpha=0.001,
                 gamma=.99, epsilon=0.05, min_replay_size=10000,
                 batch_size=32, replay_size=100000,
                 train_freq=1, target_freq=1000):
        super(DQNAgent, self).__init__(num_actions)
        # store parameters
        self.learning_rate = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.train_freq = train_freq
        self.target_freq = target_freq
        self.min_replay_size = min_replay_size
        self.batch_size = batch_size

        # create replaydb
        self.db = ReplayDB(obs_shape, replay_size)

        # construct placeholders for input batch
        self.s = tf.placeholder("float", (None,)+obs_shape)
        self.a = tf.placeholder("int32", (None,))
        self.r = tf.placeholder("float", (None,))
        self.t = tf.placeholder("bool", (None,))
        self.sp = tf.placeholder("float", (None,)+obs_shape)

        # create q_network and target network
        with tf.variable_scope("train_scope"):
            self.q_net = network_fn(self.s, self._n_actions)
        with tf.variable_scope("target_scope"):
            self.target_net = network_fn(self.sp, self._n_actions)

        # extract weights from both networks
        self.q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                        scope="train_scope")
        self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                             scope="target_scope")

        # create target net sync operation
        self.sync_op = copy_vars(self.target_vars, self.q_vars)

        # calculate targets for batch
        vals = tf.stop_gradient(self.target_net)  # don't use target net grads
        self.target_op = self.r + self.gamma * (tf.reduce_max(vals, 1) *
                                                (1.-tf.to_float(self.t)))

        # get qvals for actions taken
        act_one_hot = tf.one_hot(self.a, self.num_actions())
        # we need to multiply network output with 1-hot action encoding
        # since tf currently doesn't allow indexing with vector
        self.q_vals = tf.reduce_sum(self.q_net * act_one_hot, 1)

        # squared errors for batch
        squared_error = (self.target_op - self.q_vals)**2

        # take mean over batch to get loss
        self.loss_op = tf.reduce_mean(squared_error, 0)

        # create train op
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss_op, self.q_vars)
        clipped_gvs = [(tf.clip_by_value(grad, -5., 5.), var)
                       for grad, var in grads_and_vars]
        self.train_op = optimizer.apply_gradients(clipped_gvs)

        # init op
        init_op = tf.global_variables_initializer()

        # create a tensorflow session to run ops in
        self.session = tf.Session()

        # initialize all variables
        self.session.run(init_op)

        # sync target network
        self.update_target_network()

    def add_transition(self, s, a, r, t):
        ''' add transition to replay database'''
        self.db.insert(s, a, r, t)

    def update(self, s, a, r, t):
        super(DQNAgent, self).update(s, a, r, t)
        self.add_transition(s, a, r, t)

        # train deep q network
        if (self.num_steps() % self.train_freq == 0 and
             self.db.num_samples() >= self.min_replay_size):

            # sample batch from replay database
            (S, A, R, T, Sp) = self.db.sample(self.batch_size)
            feed_dict = {self.s: S,
                         self.a: A,
                         self.r: R,
                         self.t: T,
                         self.sp: Sp}
            cost, _ = self.session.run([self.loss_op, self.train_op],
                                       feed_dict=feed_dict)
        # update target network
        if self.num_steps() % self.target_freq:
            self.update_target_network()

    def update_target_network(self):
        ''' Sync the target network with current q_network'''
        self.session.run(self.sync_op)

    def e_greedy(self, obs, eps=0.05):
        ''' epsilon- greedy action selection

        Args:
            obs (ndarray,float): state observation
            eps (float): probability of random action (in [0,1])
        '''
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions())
        else:
            vals = self.get_values(obs)
            max_idx = np.arange(self.num_actions())[vals == np.max(vals)]
            return np.random.choice(max_idx)

    def select_action(self, obs):
        return self.e_greedy(obs, self.epsilon)

    def get_values(self, s):
        '''Get Q-values for input state'''
        inp = s.reshape(1, -1)  # inputs must be batch_size x dim
        return self.session.run(self.q_net, feed_dict={self.s: inp}).flatten()


class DPGAgent(Agent):
    '''
    Policy Gradient agent
    '''
    def __init__(self, network_fn, num_actions, obs_shape, alpha=0.001,
             gamma=.99):
        super(DPGAgent, self).__init__(num_actions)
        # store parameters
        self.learning_rate = alpha
        self.gamma = gamma

        # construct placeholders for input batch
        self.s = tf.placeholder("float", (None,)+obs_shape)
        self.a = tf.placeholder("int32", (None,))
        self.R = tf.placeholder("float", (None,))

        # create policy network
        self.policy_net = network_fn(self.s, self._n_actions)

        #define pg loss

        # note: need to reduce to single dimension because of tf
        # indexing limitations
        batch_size = tf.shape(self.policy_net)[0]
        idx = tf.range(batch_size)
        # probabilities of selected actions
        probs = tf.gather(tf.reshape(self.policy_net, [-1]),
                          idx * self.num_actions() + self.a)
        # pg loss
        self.loss_op = -tf.reduce_mean(tf.log(probs) * self.R)

        # create train op
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss_op)
        #no gradient processing for now
        self.train_op = optimizer.apply_gradients(grads_and_vars)

        # init op
        init_op = tf.global_variables_initializer()

        #buffers to store episode samples
        self.ep_rew, self.ep_obs, self.ep_act= [],[],[]

        #create tf sesion to run ops
        self.session = tf.Session()

        # initialize all variables
        self.session.run(init_op)

    def select_action(self, obs):
        inp = obs.reshape((1, -1))  # inputs must be batch_size x dim
        probs = self.session.run(self.policy_net, feed_dict={self.s:inp})
        return np.random.choice(self.num_actions(), p=probs.flatten())

    def get_returns(self, rewards, bootstrap=0.):
        ''' Discounted episode returns

        Calculate the discounted return (sum of discounted rewards) for every
        step, given a stream of rewards.

        Args:
            rewards (iterable,float): rewards stream from t=0,...T
            bootstrap (float, optional): bootstrap value (defaults to 0.). Can
                be used to bootstrap with value of final state in case the
                reward stream is cut off at non-terminal state.
        Returns:
            1d-array (float) cummulative discounted return for every step.
            Same length as input rewards.
        '''
        result = np.zeros(np.size(rewards))
        cumm_R = bootstrap
        for idx in reversed(xrange(0,np.size(rewards))):
            cumm_R = rewards[idx] + self.gamma * cumm_R
            result[idx] = cumm_R
        return result

    def update(self, s, a, r, t):
        super(DPGAgent, self).update(s, a, r, t)
        if t: #terminal
            # calculate episode discounted returns
            ep_returns = self.get_returns(self.ep_rew)
            # gather graph inputs
            feed_dict = {self.s: np.array(self.ep_obs),
                         self.a: np.array(self.ep_act),
                         self.R: ep_returns}
            # update policy network
            self.session.run(self.train_op, feed_dict=feed_dict)
            # clear buffers
            self.ep_rew, self.ep_obs, self.ep_act= [],[],[]
        else:
            # simply store transition, don't update
            self.ep_obs.append(s)
            self.ep_act.append(a)
            self.ep_rew.append(r)
