import math
from typing import Dict, List

import numpy as np
import pandas as pd

import calc
import preprocess
from hyper import Hyper
from terminal import Progress, console, hook_advance, print, progress_layout


class Model():
    def __init__(self, preset:dict=None, **hyper):
        if preset == None:
            self.hyper = Hyper(hyper)
            self.df_train = pd.read_csv(f'data/{self.hyper.dataset}/train.csv')

            preprocess.segmentate(self, self.df_train, 'training')
            preprocess.extract_ngram(self, self.df_train, 'training')
            self.count_label, self.features = preprocess.generate_features(self, self.df_train)
            self.weights = np.zeros((self.count_label, len(self.features)))
            self.df_train['prob'] = [np.zeros(self.count_label)] * len(self.df_train)
        else:
            self.hyper = Hyper(preset['hyper'])
            self.count_label = preset['count_label']
            self.features = np.array(preset['features'], dtype='<U45')
            self.weights = np.array(preset['weights'])

        self.hash_features = dict(zip(self.features, range(len(self.features))))


    def train(self):
        count_batch = math.ceil(len(self.df_train) / self.hyper.batch)

        print(f'[yellow]Get {count_batch} batch(s) of {self.hyper.batch} each.')

        with Progress(*progress_layout, console=console) as progress:
            taskid_training = progress.add_task('[magenta]Training...', total=count_batch)
            for index_batch in range(count_batch):
                progress.update(taskid_training, description=f'[magenta]Training on batch [yellow]#{index_batch+1}[/yellow] of {count_batch}...')
                sample = range(index_batch*self.hyper.batch,min(len(self.df_train), (index_batch+1) * self.hyper.batch))

                self.__batch(self.df_train, progress, sample, 'train')
                progress.advance(taskid_training)

        calc.calc_predict(self, self.df_train, range(len(self.df_train)))
        calc.calc_accuracy(self, self.df_train, True)
        print('[green bold]Training completed!')

    def __batch(self, df, progress, sample, set_type):
        if len(sample) == 0: return

        taskid_cache_feature_vector = progress.add_task(f'Generating feature vector...', total=len(sample))
        calc.cache_feature_vector(self, df, sample, hook_advance(progress, taskid_cache_feature_vector))
        calc.cache_target_matrix(self, df, sample)
        
        if set_type == 'train':
            self.__train_iterate(progress, sample)
        else:
            self.__test_iterate(progress, sample)
        
        calc.dump_feature_vector(self, df, sample)
        progress.update(taskid_cache_feature_vector, visible=False)

    def __train_iterate(self, progress, sample):
        taskid_iteration = progress.add_task(f'Iterating...', total=self.hyper.iteration)
        list_taskid_iteration = []
        for iteration in range(self.hyper.iteration):
            taskid_this_iteration = progress.add_task(f'Iteration #{iteration+1} of {self.hyper.iteration}...', total=len(sample))
            list_taskid_iteration.append(taskid_this_iteration)

            calc.calc_prob(self, self.df_train, sample, True, hook_advance(progress, taskid_this_iteration))
            progress.update(taskid_this_iteration, description=f'[green]Iteration #{iteration+1} done.')      
            progress.advance(taskid_iteration)

        for taskid_this_iteration in list_taskid_iteration: progress.update(taskid_this_iteration, visible=False)
        progress.update(taskid_iteration, visible=False)

    def prepare_test(self):
        self.df_test = pd.read_csv(f'data/{self.hyper.dataset}/test.csv')
        preprocess.segmentate(self, self.df_test, 'testing')
        preprocess.extract_ngram(self, self.df_test, 'testing')

    def test(self):
        count_batch = math.ceil(len(self.df_test) / self.hyper.batch)

        print(f'[yellow]Get {count_batch} batch(s) of {self.hyper.batch} each.')

        with Progress(*progress_layout, console=console) as progress:
            taskid_testing = progress.add_task('[magenta]Testing...', total=count_batch)
            for index_batch in range(count_batch):
                progress.update(taskid_testing, description=f'[magenta]Testing on batch [yellow]#{index_batch+1}[/yellow] of {count_batch}...')
                sample = range(index_batch*self.hyper.batch,min(len(self.df_test), (index_batch+1) * self.hyper.batch))

                self.__batch(self.df_test, progress, sample, 'test')
                progress.advance(taskid_testing)

        print('[green bold]Testing completed!')
        calc.calc_accuracy(self, self.df_test)

    def __test_iterate(self, progress, sample):
        taskid_this_iteration = progress.add_task(f'Iterating...', total=len(sample))
        calc.calc_prob(self, self.df_test, sample, False, hook_advance(progress, taskid_this_iteration))
        calc.calc_predict(self, self.df_test, sample)
        progress.update(taskid_this_iteration, visible=False)
