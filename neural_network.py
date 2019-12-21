import numpy as np
import torch
import itertools
import torch.nn.functional as F

from torch import nn
from torch import optim
from torchvision import datasets, transforms


def createNN(_inputSize):
    input_size = _inputSize
    hidden_sizes = [450, 250] # 450 nodes in first hidden layer -> 250 in second
    output_size = 29 # Number of possible outputs

    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], output_size))
    return model

def convert2tensor(x):
    x = torch.FloatTensor(x)
    return x

def convert2long(x):
    x = torch.LongTensor(x)
    return x

def switchLoader(e,it1,it2,it3):
    switcher={
        0:it1,
        1:it2,
        2:it3
        }
    return switcher.get(e,"Invalid Iterator")


def TrainNN(model,t1,t2,t3):
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 3
    print_every = 1000
    steps = 0
    
    for e in range(epochs):
        running_loss = 0
            
        loaderForData = switchLoader(e,t1,t2,t3)    

        for images, labels in iter(loaderForData):
            steps += 1
            
            images = convert2tensor(images)
            
            labels = [labels,]
            labels = convert2long(labels)
            labels = torch.LongTensor(labels)
            
            optimizer.zero_grad() # Clear the gradients as gradients are accumulated
        
            # Forward and backward passes
            output = model.forward(images)
            output = F.softmax(output, dim=0)
            
            output = output.unsqueeze(dim=0)
            loss = criterion(output, labels) # Calculate the loss
            
            loss.backward() # backpropagate to get values of the new weights
            optimizer.step() # Take a step to update the newly calculated weights
        
            running_loss += loss.item()
            
            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every))
        
        running_loss = 0 #After Each Epoch
        print("Ended Epoch.",str(e+1))
    #Saving the model after training:
    PATH = 'trained_model.pth'
    torch.save(model.state_dict(), PATH)
    

def TestNN(model,dev,testloader):

    images, labels = next(iter(testloader))

    img = images[0].view(1, 784)
    label = labels[0]

    img = img.to(dev) # Sending image to GPU
    label = label.to(dev) # Sending Label to GPU
    logits = model.forward(img)

    # Output of the network are logits, need to take softmax for probabilities
    ps = F.softmax(logits, dim=1)

    print('Actual Number is: {}'.format(label))
    
    
def load_checkpoint(filepath):
    model = torch.load('trained_model.pth')
    return model

