from typing import Literal


class Hyper():
    '''
    The hyperparams used in a training - testing session.
    '''

    def __init__(self, hyper: dict={}):
        self.dataset: Literal['SST5', '20news'] = '20news'

        self.trigram = False # generate trigrams
        self.stopword = True # remove stopwords
        self.lemmatize = True # do lemmatize
        self.rand_lr = False # random learning rate
        self.feature_pick: Literal['top', 'freq'] = 'top'
        self.feature_drop = 0
        
        self.feature_size = 20000
        self.penalty = 0 # penalty rate in regularization

        self.lr_fixed = 0.01
        self.lr_rand_coef = 0.01
        
        self.iteration = 3
        self.batch = 500
        self.epoch = 10
        
        for key in hyper:
            if key not in self.__dict__ or key.startswith('__'):
                raise KeyError(key)
          
            self.__dict__[key] = hyper[key]
