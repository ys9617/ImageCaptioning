[image1]: ./img/image-captioning.png "Image Captioning Model"
[image2]: ./img/coco-examples.jpg "Sample Dataset Example"
[image3]: ./img/ResNet50-architecture.png "ResNet50"
[image4]: ./img/COCO_sample.png "COCO Sample"
[image5]: ./img/decoder.png "Decoder"
[image6]: ./img/train_result.png "Training Result"
[image7]: ./img/test_image.png "Test Image"


# Image Captioning

Automatically generate captions from images.

![Image Captioning Model][image1]



## Overview

1. Initialize COCO API and obataining batches by using the data loader
2. Defining CNN encoder and RNN decoder architecture
3. Image Captioning result


## Data (MS COCO dataset)

The Microsoft C*ommon *Objects in COntext (MS COCO) dataset is a large-scale dataset for scene understanding. The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms.

![Sample Dataset Example][image2]

You can read more about the dataset on the [website](http://cocodataset.org/#home) or in the [research paper](https://arxiv.org/pdf/1405.0312.pdf).


## Initialize COCO API and obataining batches by using the data loader

#### Initialize COCO API

```python
data_dir = './cocoapi'
data_type = 'val2014'

instances_ann_file = os.path.join(data_dir, 'annotations/instances_{}.json'.format(data_type))
coco = COCO(instances_ann_file)

captions_ann_file = os.path.join(data_dir, 'annotations/captions_{}.json'.format(data_type))
coco_caps = COCO(captions_ann_file)

ids = list(coco.anns.keys())

ann_id = np.random.choice(ids)
img_id = coco.anns[ann_id]['image_id']
img = coco.loadImgs(img_id)[0]
url = img['coco_url']

I = io.imread(url)
matplotlib.use('TkAgg')
plt.axis('off')
plt.imshow(I)
plt.show()
```

![COCO sample][image4]


#### Obataining batches by using the data loader

```python
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

vocab_threshold = 5
batch_size = 10

data_loader = get_loader(transform=transform_train, 
                         mode='train', batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=False)
```

## Defining CNN encoder and RNN decoder architecture

#### CNN encoder architecture

The encoder uses the pre-trained ResNet-50 architecture (with the final fully connected layer removed) to extract features from a batch of pre-processing images. The output is then flattend to a vector, before being passed through a linear layer to transform the feature vector to have the same size as the word embedding.

![ResNet50][image3]


#### RNN decoder architecture
![Decoder][image5]


## Training
```python
f = open(log_file, 'w')

for epoch in range(1, num_epochs+1):
    for i_step in range(1, total_step+1):
        # Randomly sample a caption length, and sample indices with that length.
        indices = data_loader.dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader.batch_sampler.sampler = new_sampler
        
        # Obtain the batch.
        images, captions = next(iter(data_loader))

        # Move batch of images and captions to GPU if CUDA is available.
        images = images.to(device)
        captions = captions.to(device)

        # Zero the gradients.
        decoder.zero_grad()
        encoder.zero_grad()
        
        # Pass the inputs through the CNN-RNN model.
        features = encoder(images)
        outputs = decoder(features, captions)
        
        # Calculate the batch loss.
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        
        # Backward pass.
        loss.backward()
        
        # Update the parameters in the optimizer.
        optimizer.step()
            
        # Get training statistics.
        stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (epoch, num_epochs, i_step, total_step, loss.item(), np.exp(loss.item()))
        
        # Print training statistics (on same line).
        print('\r' + stats, end="")
        sys.stdout.flush()
        
        # Print training statistics to file.
        f.write(stats + '\n')
        f.flush()
        
        # Print training statistics (on different line).
        if i_step % print_every == 0:
            print('\r' + stats)
            
    # Save the weights.
    if epoch % save_every == 0:
        torch.save(decoder.state_dict(), os.path.join('./models', 'decoder-%d.pkl' % epoch))
        torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-%d.pkl' % epoch))

# Close the training log file.
f.close()
```

![Training Result][image6]



## Image Captioning Result
![Training Result][image7]



## References

* [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/pdf/1411.4555.pdf) by Oriol Vinyals et al



