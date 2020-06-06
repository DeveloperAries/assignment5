# from cnn import CNN
# import torch.nn as nn
# import torch
#
# a = torch.tensor([[[1.0,2.0,3.0,4.0],
#                   [1.0,2.0,3.0,4.0],
#                   [2.0,3.0,4.0,5.0],
#                   [2.0,3.0,4.0,5.0],
#                   [2.0,3.0,4.0,6.0]],
#                   [[1.0, 2.0, 3.0, 4.0],
#                    [1.0, 2.0, 3.0, 4.0],
#                    [2.0, 3.0, 4.0, 5.0],
#                    [2.0, 3.0, 4.0, 5.0],
#                    [2.0, 3.0, 4.0, 6.0]]
#                  ])
# a_size = list(a.size())
# print(a.transpose(1,0).size(),a.size())
# b = torch.argmax(a,dim=2)
# decode_all = torch.zeros((2,1),dtype=torch.long)
# print(b.size(),decode_all.size())
# print(torch.cat((decode_all,b),1)[:,1:])
#
# # model = CNN(5,3)
# # out = model(a)
# #
# # print(out)
# # print(out.size())
# # conv1 = nn.Conv1d(in_channels=256,out_channels=100,kernel_size=2)
# # input = torch.randn(32,35,256)
# # # batch_size x text_len x embedding_size -> batch_size x embedding_size x text_len
# # input = input.permute(0,2,1)
# # out = conv1(input)
# # print(out.size())
# #
# # b = torch.tensor([[1],[2]])
# # print(b.squeeze())

a = [[[1,2],[2,3]]]
print(len(a),len(a[0]),len(a[0][0]))

# #python run.py train --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en --dev-src=./en_es_data/dev_tiny.es --dev-tgt=./en_es_data/dev_tiny.en --vocab=vocab_tiny_q1.json --batch-size=2 --valid-niter=100 --max-epoch=101 --no-char-decoder