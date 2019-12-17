import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms

def createNN(_inputSize):
    input_size = _inputSize
    hidden_sizes = [400, 200, 100] # 400 nodes in first hidden layer -> 200 in second -> 100 in third
    output_size = 29 # Number of possible outputs

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # Defining our GPU device
    model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                          nn.ReLU(),
                          nn.Linear(hidden_sizes[2], output_size)).to(dev)
    model.cuda(dev) # Making our model run on the GPU
    print(model)
    return model,dev

#Example:
#createNN(40)

def TrainNN(model,dev,trainloader):
    # Using Cross Entropy For Loss Function and Stochastic Gradient Descent or Adam for Optimizing Weights:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 3
    print_every = 40
    steps = 0
    for e in range(epochs):
        running_loss = 0
        for images, labels in iter(trainloader):
            steps += 1
            images = images.to(dev)
            labels = labels.to(dev)
            # Flatten MNIST images into a 784 long vector
            images.resize_(images.size()[0], images[0].shape[1]*images[0].shape[2])
        
            optimizer.zero_grad() # Clear the gradients as gradients are accumulated
        
            # Forward and backward passes
        
            output = model.forward(images)
            loss = criterion(output, labels) # Calculate the loss
            loss.backward() # backpropagate to get values of the new weights
            optimizer.step() # Take a step to update the newly calculated weights
        
            running_loss += loss.item()
        

    #Saving the model after training:
    PATH = './trained_model.pth'
    torch.save(model.state_dict(), PATH)
    

def TestNN(model,dev,testloader):
    
    #testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    images, labels = next(iter(testloader))

    img = images[0].view(1, 784)
    label = labels[0]

    img = img.to(dev) # Sending image to GPU
    label = label.to(dev) # Sending Label to GPU
    logits = model.forward(img)

    # Output of the network are logits, need to take softmax for probabilities
    ps = F.softmax(logits, dim=1)

    print('Actual Number is: {}'.format(label))
    #helper.view_classify(img.view(1, 28, 28).cpu(), ps.cpu()) # Return Image and Tensor to CPU because numpy doesn't support GPU yet