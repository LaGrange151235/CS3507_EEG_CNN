import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import dataloader
import resnet18

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def get_data(data):
    return data["train_data"], data["train_label"], data["test_data"], data["test_label"]

def raw_reshape(raw_data_list):
    data_list = []
    for raw_data in raw_data_list:
        mat = np.zeros(shape=(5,9,9), dtype=float)
        for i in range(5):
            mat[i][0][0] = 0
            mat[i][0][1] = 0
            mat[i][0][2] = 0
            mat[i][0][3] = raw_data[0][i]   # FP1
            mat[i][0][4] = raw_data[1][i]   # FPZ
            mat[i][0][5] = raw_data[2][i]   # FP2
            mat[i][0][6] = 0
            mat[i][0][7] = 0
            mat[i][0][8] = 0
    
            mat[i][1][0] = 0
            mat[i][1][1] = 0
            mat[i][1][2] = raw_data[3][i]   # AF3
            mat[i][1][3] = 0
            mat[i][1][4] = 0
            mat[i][1][5] = 0
            mat[i][1][6] = raw_data[4][i]   # AF4
            mat[i][1][7] = 0
            mat[i][1][8] = 0
    
            for j in range(2, 7):
                for k in range(9):
                    mat[i][j][k] = raw_data[9*(j-2)+k+5][i]
            
            mat[i][7][0] = raw_data[50][i]  # PO7
            mat[i][7][1] = raw_data[51][i]  # PO5
            mat[i][7][2] = raw_data[52][i]  # PO3
            mat[i][7][3] = 0
            mat[i][7][4] = raw_data[53][i]  # POZ
            mat[i][7][5] = 0
            mat[i][7][6] = raw_data[54][i]  # PO4
            mat[i][7][7] = raw_data[55][i]  # PO6
            mat[i][7][8] = raw_data[56][i]  # PO8
    
            mat[i][8][0] = 0
            mat[i][8][1] = raw_data[57][i]     # CB1
            mat[i][8][2] = 0
            mat[i][8][3] = raw_data[58][i]     # O1
            mat[i][8][4] = raw_data[59][i]     # OZ
            mat[i][8][5] = raw_data[60][i]     # O2
            mat[i][8][6] = 0
            mat[i][8][7] = raw_data[61][i]     # CB2
            mat[i][8][8] = 0
        data_list.append(mat)
    return data_list

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=5,
                               out_channels=32,
                               kernel_size=5,
                               padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, 
                               out_channels=96, 
                               kernel_size=3,
                               padding=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.LeakyReLU(negative_slope=0.5)
        self.fc2 = nn.Flatten()
        self.fc3 = nn.Linear(96*9*9, 128)
        self.fc4 = nn.Linear(128, 4)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.fc1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc1(x)
        x = self.fc4(x)
        # output = F.log_softmax(x, dim=1)
        output = x
        return output

class my_CNN(nn.Module):
    def __init__(self):
        super(my_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=5,
                               out_channels=32,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               padding_mode='zeros')
        self.conv2 = nn.Conv2d(in_channels=32, 
                               out_channels=64, 
                               kernel_size=3, 
                               stride=1, 
                               padding=1, 
                               padding_mode='zeros')
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc = nn.LeakyReLU(negative_slope=0.5)
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.conv1(x)
        # x = F.relu(x)
        x = self.fc(x)
        x = self.conv2(x)
        # x = F.relu(x)
        x = self.fc(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = F.relu(x)
        x = self.fc(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(model, train_data_list, train_label_list, optimizer):
    model.train()
    data = train_data_list
    label = train_label_list
    optimizer.zero_grad()
    output = model(data)
    loss = F.cross_entropy(output, label.long())
    loss.backward()
    optimizer.step()
    return loss

def test(model, test_data_list, test_label_list):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(): 
        data = test_data_list
        label = test_label_list
        output = model(data)
        test_loss = F.cross_entropy(output, label.long(), reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.long().view_as(pred)).sum().item()
    # print(correct, len(test_data_list))
    test_loss /= len(test_data_list)
    acc = correct / float(len(test_data_list))
    return acc

def dependent_train(data):
    train_acc_list = []
    test_acc_list = []
    for experiment_id in range(len(data)):
        for session_id in range(len(data[experiment_id])):
            print("Experiment_id: %d, Session_id: %d" %(experiment_id, session_id))
            raw_train_data, raw_train_label, raw_test_data, raw_test_label = get_data(data[experiment_id][session_id])
            raw_train_data = normalization(standardization(raw_train_data))
            raw_test_data = normalization(standardization(raw_test_data))
            train_data = raw_reshape(raw_train_data)
            train_label = raw_train_label
            test_data = raw_reshape(raw_test_data)
            test_label = raw_test_label
            
            device = torch.device("cuda")
            # model = CNN().to(device)
            # model = my_CNN().to(device)
            model = resnet18.ResNet18_().to(device)
            # optimizer = optim.Adadelta(model.parameters(), lr=0.1)
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            # optimizer = optim.SGD(model.parameters(), lr=1e-2)
            train_data_gpu = torch.tensor(np.array(train_data), dtype=torch.float32).to(device)
            train_label_gpu = torch.tensor(np.array(train_label), dtype=torch.float32).to(device)
            test_data_gpu = torch.tensor(np.array(test_data), dtype=torch.float32).to(device)
            test_label_gpu = torch.tensor(np.array(test_label), dtype=torch.float32).to(device)

            model_train_acc_list = []
            model_test_acc_list = []
            for epoch in range(20):
              loss = train(model, train_data_gpu, train_label_gpu, optimizer)
              train_acc = test(model, train_data_gpu, train_label_gpu)
              test_acc = test(model, test_data_gpu, test_label_gpu)
              model_train_acc_list.append(train_acc)
              model_test_acc_list.append(test_acc)
              print("epoch: %d, loss: %.4f, train_acc: %.4f, test_acc: %.4f" % (epoch, loss, train_acc, test_acc))

            # train_acc = test(model, train_data_gpu, train_label_gpu)
            # test_acc = test(model, test_data_gpu, test_label_gpu)
            train_acc = max(model_train_acc_list)
            test_acc = max(model_test_acc_list)
                     
            print("train_acc: %.4f, test_acc: %.4f" % (train_acc, test_acc))
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

    print("avg_train_acc: ", sum(train_acc_list)/float(len(train_acc_list)))       
    print("avg_test_acc: ", sum(test_acc_list)/float(len(test_acc_list)))


if __name__=="__main__":
    data = dataloader.load_all_data()
    dependent_train(data)