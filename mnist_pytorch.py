import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.utils as utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets

## ---------------------------- Load data ---------------------------- ##

batch_size_train = 50
batch_size_test = 10000
train_loader = utils.data.DataLoader(
    datasets.MNIST('data/', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = utils.data.DataLoader(
    datasets.MNIST('data/', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ])),
    batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(example_data.shape)
torch.normal(mean=torch.zeros([10000]), std=0.1)


## -------------------------- Network Model -------------------------- ##

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(16*64, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #x = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x))), 2)
        x = x.view(-1, 16*64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


## --------------------------- Parameters --------------------------- ##

n_epochs = 20
batch_size_train = 50
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)

network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)


## --------------------------- Plot graph --------------------------- ##

train_losses = []
train_counter = []
accuracy = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def show_prediction():
    ## Evaluate output for example data
    with torch.no_grad():
        output = network(example_data)
    ## PLot figure
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def plot_loss():
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()
    fig = plt.figure()
    plt.plot(test_counter, accuracy, color='blue')
    plt.show()


## ------------------------------ Main ------------------------------ ##

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data) # y = f(x)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #    epoch, batch_idx * len(data), len(train_loader.dataset),
            #    100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
            ## Save training state
            torch.save(network.state_dict(), 'pytorch_state/model.pth')
            torch.save(optimizer.state_dict(), 'pytorch_state/optimizer.pth')

def test(epoch):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader: # x, y_
            output = network(data) # y = f(x)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    accuracy.append(correct / len(test_loader.dataset))
    print('\nTrain Epoch: {}, Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

test(0)

for epoch in range(1, n_epochs + 1):
    train(epoch)
    test(epoch)

show_prediction()
plot_loss()


## ------------------------ Continue Training ------------------------ ##

#continued_network = Net()
#continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate,
#                                momentum=momentum)

#network_state_dict = torch.load(model.pth)
#continued_network.load_state_dict(network_state_dict)

#optimizer_state_dict = torch.load(optimizer.pth)
#continued_optimizer.load_state_dict(optimizer_state_dict)

#for i in range(n_epoch + 1, 100):
#    test_counter.append(i*len(train_loader.dataset))
#    train(i)
#    test()

#show_prediction()
#plot_loss()