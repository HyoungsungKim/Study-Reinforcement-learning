from torchvision import datasets, transforms
import torch
import torch.nn as nn

import numpy as np
import layer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# * If want to save datas in parents driectory
# * data_dir = '../Datasets'
data_dir = './Datasets'
composed = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=composed)
validation_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=composed)

# ! validation batch : Classes of validation set have to uniform distribtion
train_batch_size = 32
validation_batch_size = 3000
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=validation_batch_size, shuffle=True)

subset_dataset = torch.utils.data.Subset(train_dataset, range(50))
subset_loader = torch.utils.data.DataLoader(dataset=subset_dataset, batch_size=train_batch_size, shuffle=True)


class SL:
    def __init__(self):
        self.loss_func = nn.CrossEntropyLoss()

    # * train function train only as much as train_batch_size
    def train(self, model, optimizer):
        model.train()
        train_loader_iter = iter(subset_loader)
        data, target = next(train_loader_iter)
        '''
        train_loader_iter = iter(train_loader)
        data, target = next(train_loader_iter)
        '''
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        z = model(data)
        loss = self.loss_func(z, target)
        loss.backward()
        optimizer.step()

    def get_distribution(self, model):
        model.eval()
        validation_loader_iter = iter(validation_loader)
        data, target = next(validation_loader_iter)

        # * torch.no_grad() speeds up validation(Prevent memory wasting)
        with torch.no_grad():
            data, target = data.to(device), target.to(device)
            # * z : Distribution
            # * yhat : Result of classification
            distributions = model(data)

        return distributions, data, target

    def test(self, model):
        test_loss = 0
        correct = 0
        z, _, target = self.get_distribution(model)
        # * max_val, argmax = torch.max(...)
        val, yhat = torch.max(z.data, 1)
        test_loss = (yhat == target)
        correct += test_loss.sum().item()
        accuracy = correct / validation_batch_size

        return accuracy


def model_test(path):
    model = layer.CNN(shape=[1, 28, 28], number_of_classes=10).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    sl = SL()
    accuracy = np.array([])
    for i in range(30):
        accuracy = np.append(accuracy, sl.test(model))

    return np.mean(accuracy)


if __name__ == '__main__':
    # For test saved model
    
    # model_test("./best_4.pth")
    
    model_test("./model/accuracy70.pth")
    
    '''    
    # * For training
    model = layer.CNN(shape=[1, 28, 28], number_of_classes=10).to(device)

    # ! Have to use optimizer which can avoid local optimal
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    sl = SL()
    target_accuracy = 0.60
    loop_counter = 0
    while True:
        sl.train(model, optimizer)
        accuracy = sl.test(model)
        loop_counter += 1
        if loop_counter % 10 == 0:
            print('Total trained data : {}, accuracy : {}' .format(loop_counter * train_batch_size, accuracy))
        if accuracy > target_accuracy:
            target_accuracy += 0.1
            torch.save(model.state_dict(), './model/' + 'accuracy' + str(int(accuracy * 100)) + '.pth')
            print('Total trained data : {}, accuracy : {}' .format(loop_counter * train_batch_size, accuracy))
            print("Target accuracy update")

        if accuracy > 0.95:
            torch.save(model.state_dict(), './model/' + 'accuracy' + str(int(accuracy * 100)) + '.pth')
            print("Done")
            break
    '''