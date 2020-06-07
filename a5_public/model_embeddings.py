#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h
        self.word_embed_size = word_embed_size
        self.vocab = vocab
        self.source = None
        self.target = None
        self.e_char = 50
        self.char_embeding = nn.Embedding(len(self.vocab.char2id),self.e_char)
        self.cnn = CNN(self.e_char,self.word_embed_size)
        self.highway = Highway(self.word_embed_size)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h
        # input_size = list(input.size())
        # input_reshape = input.reshape(-1, input_size[-2], input_size[-1])
        char_vector = self.char_embeding(input)  # shape (sentence_length,batch_size, max_word_length,e_char)
        char_vector = char_vector.transpose(-1,-2)    #reshape (sentence_length,batch_size,e_char, max_word_length)
        x_conv = self.cnn(char_vector)
        x_word_embed = self.highway(x_conv)        #dropout 我放在Highway模块里面实现了
        # x_word_embed_reshape = x_word_embed.reshape(input_size[0], input_size[1], -1)
        return x_word_embed      #shape (sentence_length, batch_size, wrod_embed_size)

        ### END YOUR CODE
#
# #
# import torch.nn as nn
#
# # Do not change these imports; your module names should be
# #   `CNN` in the file `cnn.py`
# #   `Highway` in the file `highway.py`
# # Uncomment the following two imports once you're ready to run part 1(j)
#
# from cnn import CNN
# from highway import Highway
#
#
# # End "do not change"
#
# class ModelEmbeddings(nn.Module):
#     """
#     Class that converts input words to their CNN-based embeddings.
#     """
#
#     def __init__(self, embed_size, vocab):
#         """
#         Init the Embedding layer for one language
#         @param embed_size (int): Embedding size (dimensionality) for the output
#         @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
#         """
#         super(ModelEmbeddings, self).__init__()
#
#         ## A4 code
#         # pad_token_idx = vocab.src['<pad>']
#         # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
#         ## End A4 code
#
#         ### YOUR CODE HERE for part 1j
#         pad_token_idx = vocab.char2id['<pad>']
#         e_char = 50
#         self.embeddings = nn.Embedding(len(vocab.char2id), e_char, padding_idx=pad_token_idx)
#         self.embed_size = embed_size
#         self.cnn = CNN(in_ch=e_char, out_ch=embed_size, k=5)
#         self.highway = Highway(D_in=embed_size, H=embed_size, D_out=embed_size, prob=0.3)
#         self.dropout = nn.Dropout(0.3)
#         ### END YOUR CODE
#
#     def forward(self, input):
#         """
#         Looks up character-based CNN embeddings for the words in a batch of sentences.
#         @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
#             each integer is an index into the character vocabulary
#         @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the
#             CNN-based embeddings for each word of the sentences in the batch
#         """
#         ## A4 code
#         # output = self.embeddings(input)
#         # return output
#         ## End A4 code
#
#         ### YOUR CODE HERE for part 1j
#         # Map input (x_padded) to output (x_word_embed)
#         # print('shape of input is',input.size())
#         # print('*** input is',input)
#
#         x_embed = self.embeddings(input)
#
#         # x_embed = F.embedding(input,self.embeddings)
#         # print('shape of x_embed is,',x_embed.size())
#         # x_reshaped = x_embed.permute(1,0,3,2) #batch_size,sentence_length,max_word_len,chat_embed_size
#         # print('shape of x_reshape is,',x_reshaped.size())
#         x_reshaped_list = list(x_embed.size())
#         # print(x_reshaped_list)
#         x_reshaped_red = x_embed.reshape(-1, x_reshaped_list[3], x_reshaped_list[2])
#         # print('shape of x_reshape_red is,',x_reshaped_red.size())
#         x_cov_out = self.cnn(x_reshaped_red)
#         # print('shape of x_cov_out is,',x_cov_out.size())
#         # x_cov_out = x_cov_out.squeeze(2)
#
#         # print('shape of x_cov_out is,',x_cov_out.size())
#
#         x_cov_out = x_cov_out.reshape(x_reshaped_list[0], x_reshaped_list[1], -1)
#
#         # print('to highway: shape of x_cov_out is,',x_cov_out.size())
#         x_word_embed = self.highway(x_cov_out)
#         # print('in:shape of x_word_embed is,',x_word_embed.size())
#         x_word_embed = self.dropout(x_word_embed)
#         # x_word_embed = x_word_embed.reshape(x_reshaped_list[1],x_reshaped_list[0],-1)
#
#         # print('out:size of x_word_embed',x_word_embed.size())
#         return x_word_embed
#         ### END YOUR CODE