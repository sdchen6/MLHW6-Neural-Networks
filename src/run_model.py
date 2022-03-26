from cmath import inf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Please read the free response questions before starting to code.

def run_model(model,running_mode='train', train_set=None, valid_set=None, test_set=None,
    batch_size=1, learning_rate=0.01, n_epochs=1, stop_thr=1e-4, shuffle=True):
    """
    This function either trains or evaluates a model.

    training mode: the model is trained and evaluated on a validation set, if provided.
                   If no validation set is provided, the training is performed for a fixed
                   number of epochs.
                   Otherwise, the model should be evaluted on the validation set
                   at the end of each epoch and the training should be stopped based on one
                   of these two conditions (whichever happens first):
                   1. The validation loss stops improving.
                   2. The maximum number of epochs is reached.

    testing mode: the trained model is evaluated on the testing set

    Inputs:

    model: the neural network to be trained or evaluated
    running_mode: string, 'train' or 'test'
    train_set: the training dataset object generated using the class MyDataset
    valid_set: the validation dataset object generated using the class MyDataset
    test_set: the testing dataset object generated using the class MyDataset
    batch_size: number of training samples fed to the model at each training step
    learning_rate: determines the step size in moving towards a local minimum
    n_epochs: maximum number of epoch for training the model
    stop_thr: if the validation loss from one epoch to the next is less than this
              value, stop training
    shuffle: determines if the shuffle property of the DataLoader is on/off

    Outputs when running_mode == 'train':

    model: the trained model
    loss: dictionary with keys 'train' and 'valid'
          The value of each key is a list of loss values. Each loss value is the average
          of training/validation loss over one epoch.
          If the validation set is not provided just return an empty list.
    acc: dictionary with keys 'train' and 'valid'
         The value of each key is a list of accuracies (percentage of correctly classified
         samples in the dataset). Each accuracy value is the average of training/validation
         accuracies over one epoch.
         If the validation set is not provided just return an empty list.

    Outputs when running_mode == 'test':

    loss: the average loss value over the testing set.
    accuracy: percentage of correctly classified samples in the testing set.

    Summary of the operations this function should perform:
    1. Use the DataLoader class to generate trainin, validation, or test data loaders
    2. In the training mode:
       - define an optimizer (we use SGD in this homework)
       - call the train function (see below) for a number of epochs untill a stopping
         criterion is met
       - call the test function (see below) with the validation data loader at each epoch
         if the validation set is provided

    3. In the testing mode:
       - call the test function (see below) with the test data loader and return the results

    """
    #load data
    if train_set:
      train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    else:
      train_dataloader=None
    if test_set:
      test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
    else:
      test_dataloader=None
    if valid_set:
      valid_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=shuffle)
    else:
      valid_dataloader=None


    if running_mode == 'train':
      #SGD optimizer
      opt = optim.SGD(model.parameters(), learning_rate)

      #train for a number of epochs.
      count = 0
      old_loss = np.inf
      new_loss = 0
      cur_model = model
      loss = {
        "train": [],
        "valid": []
      }
      acc = {
        "train": [],
        "valid": []
      }
      epochs = []

      while (count < n_epochs) and abs(old_loss-new_loss) > stop_thr:
        old_loss = new_loss
        trained_model, train_loss, train_accuracy = _train(cur_model,train_dataloader,opt)
        new_loss = train_loss
        loss['train'] = loss['train'] + [train_loss]
        acc['train'] = acc['train'] + [train_accuracy]
        if valid_dataloader:
          valid_loss, valid_accuracy = _test(trained_model, valid_dataloader)
          loss['valid'] = loss['valid'] + [valid_loss]
          acc['valid'] = acc['valid'] + [valid_accuracy]
      
        cur_model = trained_model
        count += 1
        print(count)
        


      return cur_model, loss, acc
    else:
      return _test(model,test_dataloader)

        





def _train(model,data_loader,optimizer,device=torch.device('cpu')):

    """
    This function implements ONE EPOCH of training a neural network on a given dataset.
    Example: training the Digit_Classifier on the MNIST dataset
    Use nn.CrossEntropyLoss() for the loss function


    Inputs:
    model: the neural network to be trained
    data_loader: for loading the netowrk input and targets from the training dataset
    optimizer: the optimiztion method, e.g., SGD
    device: we run everything on CPU in this homework

    Outputs:
    model: the trained model
    train_loss: average loss value on the entire training dataset
    train_accuracy: average accuracy on the entire training dataset
    """
    correct = 0
    epoch_loss = 0
    
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        #dont really get why it needs to convert to long
        target = target.long()

        #optimizer stuff
        optimizer.zero_grad()
        #forward pass
        data = data.float()
        #loss = nn.CrossEntropyLoss()
        output = model(data)
        EntropyLoss = nn.CrossEntropyLoss()
        loss = EntropyLoss(output, target)
        
        loss.backward()
        optimizer.step()
        
        

        epoch_loss += loss.item()
        #how accurately was this data predicted?
        predicted_model = torch.max(output.data, 1)[1]
        model_vs_targs = (predicted_model == target)
        match = 0
        for i in model_vs_targs:
          if i:
            match += 1
        correct += match / len(model_vs_targs)

    #get the accuracy after seeing how many data points were correctly guessed

    #epochs_loss.append(epoch_loss/len(data_loader))
    train_loss = epoch_loss / len(data_loader)
    train_accuracy = correct / len(data_loader) * 100
    #train_acc = torch.sum(target == model(data))

    return model, train_loss, train_accuracy


def _test(model, data_loader, device=torch.device('cpu')):
    """
    This function evaluates a trained neural network on a validation set
    or a testing set.
    Use nn.CrossEntropyLoss() for the loss function

    Inputs:
    model: trained neural network
    data_loader: for loading the netowrk input and targets from the validation or testing dataset
    device: we run everything on CPU in this homework

    Output:
    test_loss: average loss value on the entire validation or testing dataset
    test_accuracy: percentage of correctly classified samples in the validation or testing dataset
    """
    correct = 0
    epoch_loss = 0
    
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        #dont really get why it needs to convert to long
        target = target.long()

        #forward pass
        data = data.float()
        #loss = nn.CrossEntropyLoss()
        output = model(data)
        EntropyLoss = nn.CrossEntropyLoss()
        loss = EntropyLoss(output, target)
        
        
        

        epoch_loss += loss.item()
        #how accurately was this data predicted?
        predicted_model = torch.max(output.data, 1)[1]
        model_vs_targs = (predicted_model == target)
        match = 0
        for i in model_vs_targs:
          if i:
            match += 1
        correct += match / len(model_vs_targs)

    #get the accuracy after seeing how many data points were correctly guessed

    #epochs_loss.append(epoch_loss/len(data_loader))
    train_loss = epoch_loss / len(data_loader)
    train_accuracy = correct / len(data_loader) * 100
    #train_acc = torch.sum(target == model(data))

    return train_loss, train_accuracy

