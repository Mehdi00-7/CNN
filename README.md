  
The curves for the evolution of loss and for the evolution of training and test accuracies:

![][image1.png]

**Diagram shows the evolution of Train and Test losses** 

\*The model shows good learning with minimal overfitting, as both train and test loss steadily decrease and stabilise at around 100 epochs when the loss gets to its lowest value.

![][image2.png]

**Diagram shows the evolution of Train and Test Accuracies** 

\*The model achieves strong generalization, with training accuracy steadily rising and test accuracy plateauing around 90% without overfitting

The model that I trained had different parts and parameters that could be changed to get a better performance and higher precision,i will start by briefing the architecture of the model :  
Stem as the first layer of the network which applies a 3\*3 convolution over the 3 input channels (RGB)for an image and outputs 32 feature map.I then used ReLU as an activation function to introduce non linearity.

Each backbone block in the model has two parts: an expert branch and a convolutional branch. The expert branch uses adaptive average pooling followed by two fully connected layers to create a set of k  weights. The first FC layer reduces the input size by a factor of r, and the second expands it to size k. After applying softmax, these weights decide how much each of the k convolutional paths should contribute. The convolutional branch has k parallel 3×3 conv layers (each with batch norm), and their outputs are combined using the weights from the expert branch. Basically, k controls how many paths the model can choose from, and r controls how much the feature info is compressed before making that choice.After the conv branch output is calculated, the model adds it to a skip connection (Identity or a 1×1 Conv) then applies a ReLU just like the residual block we learnt in ResNet

These are the values of k and r i used for my different blocks:  
I started by small values of k and r then increased them later on as the features become more   complex giving the model more paths to choose from. block1 \=\>(input=32, output=64, k=4, r=8)  
block2 \=\>(input=64, output=128, k=6, r=4)  
block3 \=\>(input=128, output=256, k=6, r=4)  
block4 \=\>(input=256, output=512, k=6, r=4)

Finally the classifier ,starts with a global average pooling as was given to us in the diagram,then mlp of 3 fully connected layers with relu and dropout and final output of size 10 for the 10 different classes.

The hyperparameters i used are:

| Hyperparameter | Value | Explanation |
| ----- | ----- | ----- |
| Initial learning rate | 0.001 | For stable and fast training,standard for Adam,tried 0.1 but 0.01 works better |
| Weight Decay | 0.0005 | Added it to prevent overfitting by regularising the weights |
| Optimizer | Adam | Better than SGM for adapting the learning rate in this specific model as i tried both |
| LR scheduler | Cosine Annealing | Improved convergence and reduces LR smoothly after each epoch |
| Loss Function | CrossEntropyLoss | Standard for classification,label smoothing for generalisation |
| Batch Size | 256 | Worked well for me given the size of the dataset |
| Epochs | 100 | Just enough time for my model to reach 90% accuracy before overfitting  |
| k | 4 then 6,6,6 | Start with few paths then expand later |
| r | 8 then 4,4,4 | Compress early then less in deep layers |

To improve model training, I used Kaiming initialization for all Conv2d and Linear layers, which is designed for ReLU activations. This helps the model start with better weight distributions.I also added Dropout to the classifier of 0.3 and 0.4 to avoid overfitting.I applied batchNorm in every conv branch to normalise activations for faster and better training.  
For data augmentation, I used RandomHorizentalFlip and random crop in transformers to improve generalisation.

