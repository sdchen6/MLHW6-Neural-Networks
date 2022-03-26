import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Please read the free response questions before starting to code.
#
# Note: Avoid using nn.Sequential here, as it prevents the test code from
# correctly checking your model architecture and will cause your code to
# fail the tests.

class Digit_Classifier(nn.Module):
    """
    This is the class that creates a neural network for classifying handwritten digits
    from the MNIST dataset.

    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size 128 nodes
    - Second hidden layer: fully connected layer of size 64 nodes
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers

    """
    def __init__(self):
        super(Digit_Classifier, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        #self.conv1 = nn.Conv2d(1, 6, 5)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.hidden1_layer = nn.Linear(784, 128)  
        self.hidden2_layer = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 10)

    def forward(self, inputs):
        output = F.relu(self.hidden1_layer(inputs))
        output = F.relu(self.hidden2_layer(output))
        output = self.output_layer(output)
        return output


class Dog_Classifier_FC(nn.Module):
    """
    This is the class that creates a fully connected neural network for classifying dog breeds
    from the DogSet dataset.

    Network architecture:
    - Input layer
    - First hidden layer: fully connected layer of size 128 nodes
    - Second hidden layer: fully connected layer of size 64 nodes
    - Output layer: a linear layer with one node per class (in this case 10)

    Activation function: ReLU for both hidden layers

    """

    def __init__(self):
        super(Dog_Classifier_FC, self).__init__()

        self.hidden1_layer = nn.Linear(12288, 128)  
        self.hidden2_layer = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 10)

    def forward(self, inputs):
        inputs = inputs.view(-1, 64*64*3)
        output = F.relu(self.hidden1_layer(inputs))
        output = F.relu(self.hidden2_layer(output))
        output = self.output_layer(output)
        return output


class Dog_Classifier_Conv(nn.Module):
    """
    This is the class that creates a convolutional neural network for classifying dog breeds
    from the DogSet dataset.
    Network architecture:
    - Input layer
    - First hidden layer: convolutional layer of size (select kernel size and stride)
    - Second hidden layer: convolutional layer of size (select kernel size and stride)
    - Output layer: a linear layer with one node per class (in this case 10)
    Activation function: ReLU for both hidden layers
    There should be a maxpool after each convolution.
    The sequence of operations looks like this:
        1. Apply convolutional layer with stride and kernel size specified
            - note: uses hard-coded in_channels and out_channels
            - read the problems to figure out what these should be!
        2. Apply the activation function (ReLU)
        3. Apply 2D max pooling with a kernel size of 2
    Inputs:
    kernel_size: list of length 2 containing kernel sizes for the two convolutional layers
                 e.g., kernel_size = [(3,3), (3,3)]
    stride: list of length 2 containing strides for the two convolutional layers
            e.g., stride = [(1,1), (1,1)]
    """

    def __init__(self, kernel_size, stride):
        super(Dog_Classifier_Conv, self).__init__()
        #conv
        h_in = 64
        w_in = 64
        padding = 0
        dilation = 1

        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size[0], stride=stride[0])

        h_in = (h_in + 2 * padding - dilation * (kernel_size[0][0] - 1) -1) // stride[0][0] + 1
        w_in = (w_in + 2 * padding - dilation * (kernel_size[0][1] - 1) -1) // stride[0][1] + 1
        #adjust for pooling
        h_in = h_in // 2
        w_in = w_in // 2


           
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size[1], stride=stride[1])
        
        #calculate sizes after second layer
        h_in = (h_in + 2 * padding - dilation * (kernel_size[1][0] - 1) -1) // stride[1][0] + 1
        w_in = (w_in + 2 * padding - dilation * (kernel_size[1][1] - 1) -1) // stride[1][1] + 1
        #adjust for pooling
        h_out = h_in // 2
        w_out = w_in // 2

        self.output_layer = nn.Linear(32*h_out*w_out, 10)
        self.pool = nn.MaxPool2d(2,2)

    def forward(self, inputs):
        #inputs = inputs.view(-1, 64643)
        inputs = inputs.view(-1,3,64,64)
        output = self.pool(F.relu(self.conv1(inputs)))
        conv2_out = self.pool(F.relu(self.conv2(output)))
        conv2_out_flat = torch.flatten(conv2_out,1)
        output = self.output_layer(conv2_out_flat)
        return output


