import numpy as np
import torch
import torch.nn.functional as F
import dataset_creator as DC
from torch import nn
from torch import optim


def createNN(_inputSize):
    input_size = _inputSize
    hidden_sizes = 12 # 12 nodes in first hidden layer
    output_size = 29 # Number of possible outputs

    model = nn.Sequential(nn.Linear(input_size, hidden_sizes),
                          nn.ReLU(),
                          #nn.Dropout(0.5),
                          #nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                          #nn.ReLU(),
                          #nn.Dropout(0.3),
                          nn.Linear(hidden_sizes, output_size))
    return model

def convert2tensor(x):
    x = torch.cuda.FloatTensor(x)
    return x

def convert2long(x):
    x = torch.cuda.LongTensor(x)
    return x

def switchLoader(e,it1,it2,it3,it4,it5):
    switcher={
        0:it1,
        1:it2,
        2:it3,
        3:it4,
        4:it5
        }
    return switcher.get(e,"Invalid Iterator")

def TrainNN(model,t1,t2,t3,t4,t5):
    
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    epochs = 5
    print_every = 1000
    steps = 0
    correct_train = 0        
        
    for e in range(epochs):
        running_loss = 0
        loaderForData = switchLoader(e,t1,t2,t3,t4,t5)    

        for images, labels in iter(loaderForData):
            steps += 1
            
            images = convert2tensor(images)
            actual_label = labels
            labels = [labels,]
            labels = convert2long(labels)
            labels = torch.cuda.LongTensor(labels)
            
            optimizer.zero_grad() # Clear the gradients as gradients are accumulated
        
            # Forward and backward passes
            output = model.forward(images)
            output = F.softmax(output, dim=0)
            output = output.unsqueeze(dim=0)
            loss = criterion(output, labels) # Calculate the loss
            
            loss.backward() # backpropagate to get values of the new weights
            optimizer.step() # Take a step to update the newly calculated weights
            _, predicted = torch.max(output.data, 1)
            
            correct_train += predicted.eq(labels.data).sum().item()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every))
        
        print("Ended Epoch.",str(e+1))
    #Saving the model after training:
    train_accuracy = 100 * correct_train / 5000
    print("Train Accuracy on 1000 Elements: {}%".format(train_accuracy))
    PATH = 'trained_model.pth'
    torch.save(model.state_dict(), PATH)
    

def TestNN(model,testloader):
    images = torch.FloatTensor(testloader[:56])
    logits = model.forward(images)

    ps = F.softmax(logits, dim=0)
    ps = ps.data.numpy().squeeze()
    prediction = np.argmax(ps)
    D = DC.returnToArabicDictionary()
    return D[prediction]

def load_checkpoint(filepath):
    model = torch.load('trained_model.pth')
    return model

