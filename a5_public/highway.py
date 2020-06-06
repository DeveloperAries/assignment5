#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class Highway(nn.Module):
    #pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self,word_embed_size, dropout_rate=0.3):
        super(Highway, self).__init__()
        self.word_embed_size = word_embed_size
        self.dropout_rate = dropout_rate
        self.project = None
        self.gate = None
        self.project = nn.Linear(word_embed_size, word_embed_size, bias=True)
        self.gate = nn.Linear(word_embed_size, word_embed_size, bias=True)
        self.Dropout = nn.Dropout(p=self.dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x_conv):
        x_project = self.relu(self.project(x_conv)).clamp(min=0)
        x_gate = self.sigmoid(self.gate(x_conv))
        x_highway = x_project*x_gate + (1 - x_gate)*x_conv
        x_word_emb = self.Dropout(x_highway)
        return x_word_emb


    ### END YOUR CODE

# ### YOUR CODE HERE for part 1h
# import torch
# import numpy as np
# class Highway(torch.nn.Module):
#     def __init__(self, D_in, H, D_out,prob):
#         """
#         Apply the output of the convolution later (x_conv) through a highway network
#                 @param D_in (int): Size of input layer
#                 @param H (int): Size of Hidden layer
#                 @param D_out (int): Size of output layer
#                 @param prob (float): Probability of dropout
#         """
#         super(Highway, self).__init__()
#         self.linear1 = torch.nn.Linear(D_in, H)
#         self.linear2 = torch.nn.Linear(H, D_out)
#         self.sigmoid = torch.nn.Sigmoid()
#         self.dopout = torch.nn.Dropout(prob)
#         #Initializing weights
#         #torch.nn.init.xavier_normal_(self.linear1.weight, gain=np.sqrt(2.0))
#         #torch.nn.init.xavier_normal_(self.linear2.weight, gain=np.sqrt(2.0))
#
#
#     def forward(self, x):
#         """
#         Apply the output of the convolution later (x_conv) through a highway network
#                 @param x (Tensor): Input x_cov gets applied to Highway network - shape of input tensor [batch_size,1,e_word]
#                 @returns x_pred (Tensor): Size of Hidden layer -- NOTE: check the shapes
#         """
#         #print('** shape of x_conv is',x.size())
#         x_proj = self.linear1(x).clamp(min=0)
#
#         #print('** shape of x_proj is',x_proj.size())
#
#         x_gate_i = self.linear2(x)
#
#         #print('** shape of x_gate_i is',x_gate_i.size())
#
#         x_gate = self.sigmoid(x_gate_i)
#
#
#         #print('** shape of x_gate is',x_gate.size())
#
#         x_pred = x_gate * x_proj + (1-x_gate) * x
#
#
#         #print('** shape of x_pred is',x_pred.size())
#         x_emb = self.dopout(x_pred)
#         return x_emb
#
#
#
#
# ### END YOUR CODE