#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
utils.py:
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
nltk.download('punkt')

def pad_sents_char(sents, char_pad_token):
    """ Pad list of sentences according to the longest sentence in the batch and longest words in all sentences.
    @param sents (list[list[list[int]]]): list of sentences, result of `words2charindices()`
        from `vocab.py`
    @param char_pad_token (int): index of the character-padding token
    @returns sents_padded (list[list[list[int]]]): list of sentences where sentences/words shorter
        than the max length sentence/word are padded out with the appropriate pad token, such that
        each sentence in the batch now has same number of words and each word has an equal
        number of characters
        Output shape: (batch_size, max_sentence_length, max_word_length)
    """

    sents_padded = []
    max_word_length = max(len(w) for s in sents for w in s )
    max_sent_len = max(len(s) for s in sents)
    batch_size = len(sents)            #the length of sentences list

    for k in range(batch_size):
        sentence = sents[k]
        sent_padded = []

        for w in sentence:   #char_pad_token is the index of padding token
            data = [c for c in w] + [char_pad_token for _ in range(max_word_length-len(w))]     #w在vocab.py中会添加{ 和 }
            if len(data) > max_word_length:
                data = data[:max_word_length]          #若长度大于最长长度，则截取，长度大于最长长度的可能就是max_word_length=len(w)，在上面加入了一个[""]
            sent_padded.append(data)
#将每个word都对齐了，拥有一样的长度，接下来将每个句子对齐，使其长度一样
        sent_padded = sent_padded[:max_sent_len] + [[char_pad_token]*max_word_length] * max(0, max_sent_len - len(sent_padded))
        sents_padded.append(sent_padded)

    return sents_padded


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[int]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (int): padding token
    @returns sents_padded (list[list[int]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
        Output shape: (batch_size, max_sentence_length)
    """
    sents_padded = []

    ### COPY OVER YOUR CODE FROM ASSIGNMENT 4
    max_len = max([len(i) for i in sents])
    for sentence in sents:
        sents_padded.append(sentence + (max_len - len(sentence)) * [pad_token])
    ### END YOUR CODE FROM ASSIGNMENT 4

    return sents_padded



def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = nltk.word_tokenize(line)
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data

def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

# import math
# from typing import List
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# def pad_sents_char(sents, char_pad_token):
#     """ Pad list of sentences according to the longest sentence in the batch and max_word_length.
#     @param sents (list[list[list[int]]]): list of sentences, result of `words2charindices()`
#         from `vocab.py`
#     @param char_pad_token (int): index of the character-padding token
#     @returns sents_padded (list[list[list[int]]]): list of sentences where sentences/words shorter
#         than the max length sentence/word are padded out with the appropriate pad token, such that
#         each sentence in the batch now has same number of words and each word has an equal
#         number of characters
#         Output shape: (batch_size, max_sentence_length, max_word_length)
#     """
#     # Words longer than 21 characters should be truncated
#     max_word_length = 21
#
#     ### YOUR CODE HERE for part 1f
#     ### TODO:
#     ###     Perform necessary padding to the sentences in the batch similar to the pad_sents()
#     ###     method below using the padding character from the arguments. You should ensure all
#     ###     sentences have the same number of words and each word has the same number of
#     ###     characters.
#     ###     Set padding words to a `max_word_length` sized vector of padding characters.
#     ###
#     ###     You should NOT use the method `pad_sents()` below because of the way it handles
#     ###     padding and unknown words.
#
#     sents_padded = []
#
#     max_len = max(len(s) for s in sents)
#     batch_size = len(sents)
#
#     sent_padded1 = [[ w[:max_word_length] + [char_pad_token]*(max_word_length-len(w)) for w in s] for s in sents]
#     sents_padded = [sent + [[char_pad_token]*max_word_length]*(max_len-len(sent)) for sent in sent_padded1]
#     #print(sents_padded)
#     ### END YOUR CODE
#
#     return sents_padded
#
#
# def pad_sents(sents, pad_token):
#     """ Pad list of sentences according to the longest sentence in the batch.
#     @param sents (list[list[int]]): list of sentences, where each sentence
#                                     is represented as a list of words
#     @param pad_token (int): padding token
#     @returns sents_padded (list[list[int]]): list of sentences where sentences shorter
#         than the max length sentence are padded out with the pad_token, such that
#         each sentences in the batch now has equal length.
#         Output shape: (batch_size, max_sentence_length)
#     """
#     sents_padded = []
#
#     max_len = max(len(s) for s in sents)
#     batch_size = len(sents)
#
#     for s in sents:
#         padded = [pad_token] * max_len
#         padded[:len(s)] = s
#         sents_padded.append(padded)
#
#     return sents_padded
#
#
#
# def read_corpus(file_path, source):
#     """ Read file, where each sentence is dilineated by a `\n`.
#     @param file_path (str): path to file containing corpus
#     @param source (str): "tgt" or "src" indicating whether text
#         is of the source language or target language
#     """
#     data = []
#     for line in open(file_path):
#         sent = line.strip().split(' ')
#         # only append <s> and </s> to the target sentence
#         if source == 'tgt':
#             sent = ['<s>'] + sent + ['</s>']
#         data.append(sent)
#
#     return data
#
#
# def batch_iter(data, batch_size, shuffle=False):
#     """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
#     @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
#     @param batch_size (int): batch size
#     @param shuffle (boolean): whether to randomly shuffle the dataset
#     """
#     batch_num = math.ceil(len(data) / batch_size)
#     index_array = list(range(len(data)))
#
#     if shuffle:
#         np.random.shuffle(index_array)
#
#     for i in range(batch_num):
#         indices = index_array[i * batch_size: (i + 1) * batch_size]
#         examples = [data[idx] for idx in indices]
#
#         examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
#         src_sents = [e[0] for e in examples]
#         tgt_sents = [e[1] for e in examples]
#
#         yield src_sents, tgt_sents