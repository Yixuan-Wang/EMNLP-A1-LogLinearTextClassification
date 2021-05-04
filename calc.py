import random
import time

import numpy as np
import numpy.ma as npma

from terminal import console, use_hook


def cache_feature_vector(model, df, sample, hook=None):
    @use_hook(hook)
    def row_calc_feature_vector(ngrams):
        feature_vector = np.zeros(len(model.features))
        for ngram in ngrams:
            index = model.hash_features.get(ngram, -1)
            if index > -1:
                feature_vector[index] = 1
        return feature_vector

    df.loc[sample, 'feature_vector'] = df.loc[sample, 'ngram'].apply(row_calc_feature_vector)

def dump_feature_vector(model, df, sample, hook=None):
    df.loc[sample] = df.loc[sample].drop('feature_vector', axis=1)

def cache_target_matrix(model, df, sample, hook=None):
    model.target_matrix = np.zeros((len(sample), model.count_label))
    for index, target in enumerate(df.loc[sample, 'target']): model.target_matrix[index][target] = 1

def calc_prob(model, df, sample, is_train=False, hook=None):
    def p(feature_vector, weights):
        exp = np.exp(weights @ feature_vector)
        return exp / exp.sum()

    @use_hook(hook)
    def row_calc_prob(row):
        prob = p(row['feature_vector'], model.weights)
        if is_train: descend(model, df, sample, row, prob)
        return prob
    
    df.loc[sample, 'prob'] = df.loc[sample].apply(row_calc_prob, axis=1)

def descend(model, df, sample, row, prob, hook=None):
    shape = (model.count_label, len(model.features))

    @use_hook(hook)
    def row_calc_grad(row, prob) -> np.ndarray:
        target = np.zeros(model.count_label); target[row['target']] = 1
        empirical = np.zeros(shape); empirical[row['target']] = row['feature_vector']
        expected = row['feature_vector'] * prob[:,np.newaxis]
        derta = empirical - expected
        if model.hyper.penalty != 0: derta -= model.hyper.penalty * model.weights
        return derta
        
    grad = row_calc_grad(row, prob)    
    if model.hyper.rand_lr:
        model.weights = model.weights + random.random() * model.hyper.lr_rand_coef * grad

    else: model.weights = model.weights + model.hyper.lr_fixed * grad

def calc_predict(model, df, sample, hook=None):
    @use_hook(hook)
    def row_calc_predict(prob):
        return np.argmax(prob)

    df.loc[sample, 'predict'] = df.loc[sample, 'prob'].apply(row_calc_predict)

def calc_accuracy(model, df, is_train=False):
    df_correct = df[df.target == df.predict]
    if is_train:
        model.train_accuracy = len(df_correct) / len(df)
    else:
        model.accuracy = len(df_correct) / len(df)
    
