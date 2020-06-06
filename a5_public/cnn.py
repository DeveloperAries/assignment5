# # #!/usr/bin/env python3
# # # -*- coding: utf-8 -*-
# #
# # """
# # CS224N 2019-20: Homework 5
# # """
# #
import torch
import torch.nn as nn

class CNN(nn.Module):
    #pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, e_char,word_embed_size,k=5):
        super(CNN, self).__init__()
        self.word_embed_size = word_embed_size
        self.k = k
        self.char = e_char      #字符向量的维度
        self.conv = nn.Conv1d(in_channels=self.char,out_channels=self.word_embed_size,kernel_size=self.k,padding=1,bias=True)
        self.relu = nn.ReLU()
        self.MaxPooling = nn.AdaptiveMaxPool1d(1)

    def forward(self,input):
        input_size = list(input.size())
        input_reshape = input.reshape(-1,input_size[-2],input_size[-1])
        x_conv = self.conv(input_reshape)
        x_conv_out = self.MaxPooling(self.relu(x_conv)).squeeze()
        x_conv_out_reshape = x_conv_out.reshape(input_size[0],input_size[1],-1)
        return x_conv_out_reshape



    ### END YOUR CODE

# ### YOUR CODE HERE for part 1i
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# class CNN(nn.Module):
#     def __init__(self, in_ch, out_ch,k=5):
#         """
#         Apply the output of the convolution later (x_conv) through a highway network
#                 @param D_in (int): Size of input layer
#                 @param H (int): Size of Hidden layer
#                 @param D_out (int): Size of output layer
#                 @param prob (float): Probability of dropout
#         """
#         super(CNN, self).__init__()
#         self.conv1d = nn.Conv1d(in_ch, out_ch,k)
#         #self.maxpool = nn.MaxPool1d(max_word_len-k+1)
#         self.admaxpool = nn.MaxPool1d(21-k+1)#nn.AdaptiveMaxPool1d(1)#out_ch)
#         #Initializing weights
#         #nn.init.xavier_normal_(self.conv1d.weight, gain=np.sqrt(2.0))
#
#
#     def forward(self, x):
#         """
#         Apply the output of the convolution later (x_conv) through a highway network
#                 @param x (Tensor): Input x_cov gets applied to Highway network - shape of input tensor [batch_size,1,e_word]
#                 @returns x_pred (Tensor): Size of Hidden layer -- NOTE: check the shapes
#         """
#         # print('** shape of x is',x.size())
#         x_conv = self.conv1d(x)
#         # print('** shape of x_conv is',x_conv.size())
#         x_conv_act = F.relu(x_conv)
#         #print('** shape of x_connv_act is',x_conv_act.size())
#         #x_maxpool = self.maxpool(x_conv_act)
#         #print('** shape of x_maxpool is',x_maxpool.size())
#         x_admaxpool = self.admaxpool(x_conv_act)
#         # print('** shape of x_admaxpool is',x_admaxpool.size())
#         return x_admaxpool
# ### END YOUR CODE