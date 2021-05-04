import itertools
import random
import re
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from terminal import (Progress, console, format_bool, hook_advance, print,
                      progress_layout, use_hook)

dict_lemmatize: dict = None
set_stopwords: set = None

def segmentate(model, df, name_df):
    global dict_lemmatize
    with Progress(*progress_layout, console=console) as progress:
        taskid = progress.add_task(f'Segmentation on {name_df} set...', total=len(df))

        if model.hyper.lemmatize and dict_lemmatize == None:
            df_lemmatize = pd.read_csv('corpora/lemmatize.csv', sep='\t')
            dict_lemmatize = dict(zip(df_lemmatize['word'], df_lemmatize['lemma']))

        # def split_old(string):
        #     '''perform splitting on raw strings while removing unnecessary punctuation
        #     '''
        #     return list(filter(lambda x: bool(re.match(r'\'?\w[\w/\-\'.]+$', x)),
        #         map(
        #             lambda s: s.lstrip('_*').strip('_*,.\n').lower(),
        #             string.split(' ')
        #         )
        #     ))

        def split_new(string):
            '''perform splitting on raw strings while removing unnecessary punctuation, based upon [this source](https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py)
            '''
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
            string = re.sub(r"\'s", " \'s", string) 
            string = re.sub(r"\'ve", " \'ve", string) 
            string = re.sub(r"n\'t", " n\'t", string) 
            string = re.sub(r"\'re", " \'re", string) 
            string = re.sub(r"\'d", " \'d", string) 
            string = re.sub(r"\'ll", " \'ll", string) 
            string = re.sub(r",", " , ", string) 
            string = re.sub(r"!", " ! ", string) 
            string = re.sub(r"\(", " \( ", string) 
            string = re.sub(r"\)", " \) ", string) 
            string = re.sub(r"\?", " \? ", string)

            string = re.sub(r'_{1,2}(\w+?)_{1,2}', '\g<1>', string) # MD Punctuation
            string = re.sub(r'\*{1,2}(\w+?)\*{1,2}', '\g<1>', string)
            string = re.sub(r'[\w\-_.]+@[\w\-_.]+', '', string) # Email

            string = re.sub(r"\s{2,}", " ", string)
            return string.strip().lower().split(' ')

        def lemmatization(lst):
            '''perform lemmatization on word lists, lemmatization list is from [this source](https://github.com/michmech/lemmatization-lists/blob/master/lemmatization-en.txt)
            '''
            return list(map(
                lambda s: dict_lemmatize.get(s, s),
                lst
            ))

        @use_hook(hook_advance(progress, taskid))
        def row_segmentate(string):
            lst = split_new(string)
            if model.hyper.lemmatize: lst = lemmatization(lst)
            return lst

        df['data'] = df['data'].apply(row_segmentate)

def extract_ngram(model, df, name_df):
    global set_stopwords
    def generate_trigram(arr: List[str]) -> Set[Tuple[str, str, str]]:
        return set(map(tuple, np.array((arr[:-3], arr[1:-2], arr[2:-1])).T.tolist()))

    def generate_bigram(arr: List[str]) -> Set[Tuple[str, str]]:
        return set(map(tuple, np.array((arr[:-2], arr[1:-1])).T.tolist()))

    def generate_unigram(arr: List[str]):
        return set(tuple(arr))

    if model.hyper.stopword and set_stopwords == None:
        '''remove stopwords in word lists, stopwords list is a joint from [this source](https://code.google.com/archive/p/stop-words/) and [this source](https://web.archive.org/web/20111226085859/http://oxforddictionaries.com/words/the-oec-facts-about-the-language)
        '''

        df_stopwords = pd.read_csv('corpora/stopwords.csv', sep='\t')
        set_stopwords = set(df_stopwords['stopword'])
        set_stopwords.update(set(itertools.product(df_stopwords['stopword'], df_stopwords['stopword'])))


    with Progress(*progress_layout, console=console) as progress:
        taskid = progress.add_task(f'Extracting ngrams from {name_df} set...', total=len(df))

        @use_hook(hook_advance(progress, taskid))
        def row_extract_ngram(arr):
            ngrams = generate_unigram(arr) | generate_bigram(arr)
            if model.hyper.trigram: ngrams = ngrams | generate_trigram(arr)
            if model.hyper.stopword: ngrams = ngrams.difference(set_stopwords)
            return np.array(list(map(str, ngrams)), dtype='<U45')

        df['ngram'] = df['data'].apply(row_extract_ngram)

def generate_features(model, df):
    with Progress(*progress_layout, console=console) as progress:
        taskid = progress.add_task(f'Generating features...', total=len(df))

        dict_set_features: Dict[int, set] = {}

        @use_hook(hook_advance(progress, taskid))
        def row_generate_features(row):
            set_feature = set(row['ngram'])
            if row['target'] in dict_set_features:
                dict_set_features[row['target']].update(set_feature)
            else: dict_set_features[row['target']] = set_feature
        df.apply(row_generate_features, axis=1)

        count_label = len(dict_set_features)
        if not model.hyper.feature_drop:
            for label in dict_set_features:
                dict_set_features[label] = set(random.sample(list(dict_set_features[label]), int((1 - model.hyper.feature_drop) * len(dict_set_features[label]))))

        features = tuple(set().union(*dict_set_features.values()))

    with Progress(*progress_layout, console=console) as progress:
        taskid = progress.add_task(f'Sampling features...', total=len(df))

        
        dict_frequency_feature = dict.fromkeys(features, 0)

        @use_hook(hook_advance(progress, taskid))
        def row_count_frequency(ngrams):
            for ngram in ngrams:
                if ngram in dict_frequency_feature: dict_frequency_feature[ngram] += 1

        df['ngram'].apply(row_count_frequency)

        if model.hyper.feature_pick == 'freq':
            features = random.choices(features, k=model.hyper.feature_size)
        elif model.hyper.feature_pick == 'top':
            features = tuple(sorted(dict_frequency_feature.keys(), key=lambda x:dict_frequency_feature[x], reverse=True))
            features = features[:model.hyper.feature_size]
        
    return count_label, np.array(features, dtype='<U45')
