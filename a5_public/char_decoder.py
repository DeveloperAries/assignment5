#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size,batch_first=True)    #看清楚了  这不是LSTMCell
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size,
                                           padding_idx=self.target_vocab.char2id['<pad>'])

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.
        input = input.permute(1,0).contiguous()      # change shape to (batch_size,length)
        input_embed = self.decoderCharEmb(input)
        o_t,(dec_hidden_h,dec_hidden_c) = self.charDecoder(input_embed,dec_hidden)
        scores = self.char_output_projection(o_t)
        scores = scores.permute(1,0,2).contiguous()     #shape (length, batch_size, self.vocab_size)   length = max_word_len
        return scores,(dec_hidden_h,dec_hidden_c)


        ### END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.
        注意  这里的length其实是max-word-len    batch_size=max_sentence_len*sentence_batch
        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss
        # char_sequence (Tensor)还是index序列，index是每个char的index
        input_char = char_sequence[:-1]
        target_char = char_sequence[1:].reshape(-1)
        scores,(dec_hidden_h,dec_hidden_c) = self.forward(input_char,dec_hidden)
        scores = scores.reshape(-1,scores.shape[2])
        # target_masks = (target_char != self.target_vocab.tgt['<pad>']).float()
        loss = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'],reduction='sum')
        return loss(scores,target_char)




        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        decodedWords = []
        batch_size = initialStates[0].shape[1]
        start = torch.tensor([self.target_vocab.start_of_word],device=device)
        batch_start = start.repeat(batch_size,1).transpose(1,0)
        batch_current_char = batch_start
        (dec_hidden_h, dec_hidden_c) = initialStates
        decode_all = torch.zeros((batch_size,1),dtype=torch.long,device=device)    #decode_all是包含开头以及}之后的所有index，统一decode到max_length时停止解码，为了可以一批同时解码
        for i in range(max_length):
            scores, (dec_hidden_h, dec_hidden_c) = self.forward(batch_current_char, (dec_hidden_h, dec_hidden_c))
            batch_current_char = torch.argmax(nn.functional.softmax(scores,2),dim=2)          # shape is (length,batch_size)
            decode_all = torch.cat((decode_all,batch_current_char.transpose(1,0)),1)          # shape is (batch_size,length)  length will become the word_length
        decode_all_list = decode_all[:,1:].tolist()    #将tensor转化为list
        decode_all_char = [[self.target_vocab.id2char[char] for char in row_char] for row_char in decode_all_list]
        for word_char in decode_all_char:
            word = []
            for char in word_char:
                if char == '}':
                    break
                word.append(char)
            decodedWords.append(''.join(word))
        return decodedWords

        ### END YOUR CODE

