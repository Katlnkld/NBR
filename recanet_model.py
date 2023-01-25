import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ReCaNet(nn.Module):
    def __init__(self, num_items, item_embed_size, num_users, user_embed_size, history_len, h1, h2, h3, h4, h5):
        super(ReCaNet, self).__init__()
        self.num_items = num_items
        self.item_embed_size = item_embed_size
        self.num_users = num_users
        self.user_embed_size = user_embed_size
        self.history_len = history_len
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3
        self.h4 = h4
        self.h5 = h5

        self.item_embedding = nn.Embedding(self.num_items, self.item_embed_size)
        self.user_embedding = nn.Embedding(self.num_users, self.user_embed_size)
        
        self.fc1 = nn.Linear(self.item_embed_size + self.user_embed_size, self.h1)
        self.fc2 = nn.Linear(self.h1+1, self.h2)
        self.lstm1 = nn.LSTM(self.h2, self.h3, batch_first=True, num_layers=2)
        #self.lstm2 = nn.LSTM(self.h3, self.h4, batch_first=True)
        self.fc5 = nn.Linear(self.h4, self.h5)
        self.fc6 = nn.Linear(self.h4, self.h5)
        self.fc7 = nn.Linear(self.h5, 1)

    def forward(self, input1, input2, input3, input4):
        x1 = self.item_embedding(input1.to(torch.int64))
        x1 = x1.view(-1, self.item_embed_size) #1x32
       
        x2 = self.user_embedding(input2.to(torch.int64))
        x2 = x2.view(-1, self.user_embed_size) #1x128
        x3 = input3 # 1x5
        x4 = input4 # 1x5
        x4 = x4.unsqueeze(2)
        
        conc1 = torch.cat((x1, x2), axis=1)
        x11 = F.relu(self.fc1(conc1)) # 1*128

        x12 = x11.unsqueeze(1).repeat(1, self.history_len, 1)
        
        conc2 = torch.cat((x12, x4), 2)#.transpose(-2,-1)
        x14 = F.relu(self.fc2(conc2))
        
        #packing
        #mask = input4.bool()
        #seq_lengths = torch.sum(mask, axis=1)
        #pack = pack_padded_sequence(x14, seq_lengths.cpu().numpy() ,batch_first = True, enforce_sorted=False)
        #x21, (hx, cx) = self.lstm1(pack)

        ##unpack, _ = pad_packed_sequence(x21, batch_first=True)
        ##x22 = unpack[:, -1, :]
        #x22 = x21[:, -1, :]
        
        x21, (hx, cx) = self.lstm1(x14)
        x22 = hx[1]
       
        x = F.relu(self.fc5(x22))
        x = F.relu(self.fc6(x))
      
        output = self.fc7(x).view(-1) 
        output = torch.sigmoid(output)
     
        return output