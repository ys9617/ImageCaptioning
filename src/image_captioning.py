
import os
import numpy as np
import skimage.io as io
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import nltk
nltk.download('punkt')

from pycocotools.coco import COCO
from torchvision import transforms
from PIL import Image
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN

transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

data_loader = get_loader(transform=transform_train, 
                         mode='test')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_size = 512
hidden_size = 512

vocab_size = len(data_loader.dataset.vocab)

encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()

encoder_file = 'encoder-3.pkl'
decoder_file = 'decoder-3.pkl'

encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file)))
decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file)))

encoder.to(device)
decoder.to(device)


# prediction
def clean_sentence(output):
    sentence = ""

    for idx in output:
        if idx == 0:
            continue
        elif idx == 1:
            break
        else:
            word = data_loader.dataset.vocab.idx2word[idx]
            sentence = sentence + word + " "

    return sentence.capitalize()

def get_prediction(img_path=None):
    matplotlib.use('TkAgg')
    
    if img_path == None:
        orig_image, image = next(iter(data_loader))
    else:
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        PIL_image = Image.open(img_path).convert('RGB')
        orig_image = np.array(PIL_image)
        image = transform_test(PIL_image)
        image = image.unsqueeze(0)

    image = image.to(device)
    features = encoder(image).unsqueeze(1)

    output = decoder.sample(features)
    sentence = clean_sentence(output)

    print('\n')
    print(sentence)
    print('\n')

    plt.imshow(np.squeeze(orig_image))
    plt.title('Sample Image')
    plt.show()

get_prediction('D:/Udacity/Computer Vision Nanodegree/Image Captioning/img/test2.jpg')
