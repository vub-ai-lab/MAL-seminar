import numpy as np

class ReplayDB(object):
    ''' Database for storing transition samples

    Basic experience replay database implementation based on numpy arrays.
    Stores a fixed number of transition samples in memory. Assumes samples are
    added in sequence, i.e. a single agent adds all transitions in the order
    they are experienced.

    Args:
        shape (tuple): shape of observations to be stored. Database only
                    accepts fixed size float array observations.
        capacity (int): maximum number of samples that database can store

    '''

    def __init__(self, shape, capacity=10000):
        self._capacity = capacity
        self._obs_array = np.zeros((capacity,)+shape, dtype=np.float)
        self._rew_array = np.zeros(self._capacity, dtype=np.float)
        self._term_array = np.zeros(self._capacity, dtype=np.bool)
        self._act_array = np.zeros(self._capacity, dtype=np.int)
        self._insert_ptr = 0
        self._n_samples = 0

    def capacity(self):
        '''int: maximum database capacity (in number of samples)'''
        return self._capacity

    def num_samples(self):
        '''int: number of transition samples stored'''
        return self._n_samples

    def insert(self, s, a, r, t):
        ''' inserts a single transition sample into the db.

        Inserts a new transition into the database. Will overwrite  previously
        stored samples if database is filled to  max capacity. The database
        operates as a circular buffer so oldest samples are overwritten first.

        args:
            s (ndarray float): state observation
            a (int): action
            r (float): reward
            t (bool): termination indicator
        '''
        idx = self._insert_ptr
        self._obs_array[idx, ] = s
        self._act_array[idx] = a
        self._rew_array[idx] = r
        self._term_array[idx] = t
        self._insert_ptr = (self._insert_ptr+1) % self.capacity()
        self._n_samples = min(self._n_samples+1, self.capacity())

    def sample(self, batch_size=32):
        ''' Randomly sample transition from database.

        Returns:
            (s,a,r,t,s') tuple consisting of state observation s, action a,
                reward r, termination t and next state observation s'
        '''
        assert self.num_samples() >= batch_size, 'insufficient samples'
        idx = np.random.choice(self.num_samples(),
                               size=batch_size,
                               replace=False)
        # insertion ptr -1  is not a valid index since we don't have next state
        while np.any(idx == self._insert_ptr - 1):
            idx[idx == self._insert_ptr - 1] = np.random.choice(self.num_samples())
        s = self._obs_array[idx, ]
        a = self._act_array[idx]
        r = self._rew_array[idx]
        t = self._term_array[idx]
        # next states, only valid for non terminal states
        sp = self._obs_array[idx+1, ]
        return (s, a, r, t, sp)
