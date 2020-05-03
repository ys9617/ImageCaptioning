import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()      
        self.hidden_size = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.hidden2vocab = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(0.4)

    def forward(self, features, captions):
        embeds = self.word_embeddings(captions)
        features = features.unsqueeze(1)
        embeds = torch.cat((features, embeds[:, :-1,:]), dim=1)

        lstm_out, hidden = self.lstm(embeds)
        
        out = self.hidden2vocab(lstm_out)

        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        words = []

        hidden = (torch.randn(1, 1, 512).to(inputs.device),
                  torch.randn(1, 1, 512).to(inputs.device))
        
        for i in range(max_len):
            lstm_out, hidden = self.lstm(inputs, hidden)
            out = self.hidden2vocab(lstm_out.squeeze(1))

            _, predicted = out.max(1)
            
            words.append(predicted.item())

            inputs = (self.word_embeddings(predicted)).unsqueeze(1)

        return words