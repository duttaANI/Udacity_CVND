import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)  # since we dont want to apply gradient descent here requires_grad = False
        
        modules = list(resnet.children())[:-1] #removing last nn.linear layer
        self.resnet = nn.Sequential(*modules)
        #print(self.resnet)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        #nn.LSTM take your full sequence (rather than chunks), automatically initializes the hidden and cell states to zeros,       
        #runs the lstm over your full sequence … 
        
    def forward(self, features, captions):
        #----------------------------------------------------------------------------------------------------------------
        # Some imp points
        #1) The pre-processed images from the batch in Step 2 of this notebook are then passed through the encoder, 
        #   and the output is stored in features.
        #2) features is a PyTorch tensor with shape [batch_size, embed_size].
        #3) your decoder should be written to accept an arbitrary batch (of embedded image features and pre-processed 
        #   captions [where all captions have the same length]) as input. 
        #   (features.shape[0]==batch_size) & (features.shape[1]==embed_size)
        #4) hidden = (torch.randn(self.num_layers, features.shape[0], self.hidden_size), torch.randn(self.num_layers, 
        #   features.shape[0], self.hidden_size)) 
        #5) `outputs.shape[1]` and `captions.shape[1]` represent the length of the vector of words in the predicted caption of a 
        #    given image. 
        #-----------------------------------------------------------------------------------------------------------------
        
        # the first value returned by LSTM is all of the hidden states throughout
        # the sequence. the second is just the most recent hidden state
        # Add the extra 2nd dimension to features using .unsqueeze(1)
        
        captions = captions[:, :-1]                             # <end> token is not given to input
        
        embed = self.word_embeddings(captions)                  # (64,seq_length+1(for <start>),512)
        #print("embed.shape : ",embed.shape)
        inputs = torch.cat((features.unsqueeze(1), embed), 1)   # (64,seq_length+2,512)
        #print("inputs.shape : ",inputs.shape)
        
         
        out, hidden = self.lstm(inputs)
        #print("out.shape : ",out.shape)
        #print("hidden[0].shape : ",hidden[0].shape)
        #print("hidden[1].shape : ",hidden[1].shape)
        x = self.fc(out)
        #print("x.shape : ",x.shape)                             # (64,seq_length+2,vocab_size)
        """
        embed.shape :  torch.Size([64, 13, 512])
        inputs.shape :  torch.Size([64, 14, 512])
        out.shape :  torch.Size([64, 14, 512])
        hidden[0].shape :  torch.Size([1, 64, 512])
        hidden[1].shape :  torch.Size([1, 64, 512])
        x.shape :  torch.Size([64, 14, 8855])
        type(outputs): <class 'torch.Tensor'>
        outputs.shape: torch.Size([64, 14, 8855])
        """

        return x

    

    def sample(self, inputs, states=None, max_len=20):
        # (Do not write the sample method yet - you will work with this method when you reach 3_Inference.ipynb.)
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        """
        assert (type(output)==list), "Output needs to be a Python list" 
        assert all([type(x)==int for x in output]), "Output should be a list of integers." 
        assert all([x in data_loader.dataset.vocab.idx2word for x in output]), "Each entry in the output needs to correspond to 
        an integer that indicates a token in the vocabulary."
        """
        ids = []
        #print("inputs.shape : ",inputs.shape)              o/p=>inputs.shape :      torch.Size([1, 1, 512])
        
        for i in range(max_len):
        
        # after each step, give its output word as input word to next step
            out, states = self.lstm(inputs,states)          
            #print("out.shape : ",out.shape)                #o/p=>out.shape :        torch.Size([1, 1, 512])
            #print("states[0].shape : ",states[0].shape)    #o/p=>states[0].shape :  torch.Size([1, 1, 512]) 
            #print("states[1].shape : ",states[1].shape)    #o/p=>states[1].shape :  torch.Size([1, 1, 512])
            out = self.fc(out)                             
            #print("out.shape : ",out.shape)                #o/p=>out.shape :        torch.Size([1, 1, 8855])
            
            out_word_value,out_word_id = torch.max(out,2)   # we used 2 for 3rd dimension maximum
            
            #print("out_word_id.shape : ",out_word_id.shape)#o/p=>out_word_id.shape :torch.Size([1, 1])
            #print("out_word_id.item() : ",out_word_id.item())
            ids.append(out_word_id.item())
            inputs = self.word_embeddings(out_word_id)
            #print("inputs.shape : ",inputs.shape)          #o/p=>inputs.shape :      torch.Size([1, 1, 512])
            
            

        
        return ids
