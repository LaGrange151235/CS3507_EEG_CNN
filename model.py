import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import dataloader

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
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
#        print("input: ", x.shape)
        x = self.conv1(x)
#        print("conv1 output: ", x.shape)
        x = F.relu(x)
#        print("relu output: ", x.shape)
        x = self.conv2(x)
#        print("conv2 output: ", x.shape)
        x = F.relu(x)
#        print("relu output: ", x.shape)
        x = F.max_pool2d(x, 2)
#        x = self.dropout1(x)
#        print("max_pool2d output: ", x.shape)
        x = torch.flatten(x, 1)
#        print("flatten output: ", x.shape)
        x = self.fc1(x)
#        print("fc1 output: ", x.shape)
        x = F.relu(x)
#        x = self.dropout2(x)
#        print("relu output: ", x.shape)
        x = self.fc2(x)
#        print("fc2 output: ", x.shape)
        output = F.log_softmax(x, dim=1)
#        print("final output: ", x.shape, "\n")
        return output

def train(model, device, train_data_list, train_label_list, optimizer):
    model.train()
    train_data_list = torch.tensor(np.array(train_data_list), dtype=torch.float32).to(device)
    train_label_list = torch.tensor(np.array(train_label_list), dtype=torch.float32).to(device)
    
    data = train_data_list
    label = train_label_list
    
    for i in range(1000):
#        print("iter: ", i)
        optimizer.zero_grad()
        output = model(data)
#        print(output.shape, label.shape)
        loss = F.cross_entropy(output, label.long())
        loss.backward()
        optimizer.step()

def test(model, device, test_data_list, test_label_list):
    model.eval()
    test_loss = 0
    correct = 0
    test_data_list = torch.tensor(np.array(test_data_list), dtype=torch.float32).to(device)
    test_label_list = torch.tensor(np.array(test_label_list), dtype=torch.float32).to(device)
    with torch.no_grad(): 
        data = test_data_list
        label = test_label_list
        output = model(data)
        test_loss = F.cross_entropy(output, label.long(), reduction='sum').item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_data_list)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_data_list),
        100. * correct / len(test_data_list)))
    acc = correct / float(len(test_data_list))
    return acc

if __name__=="__main__":
    data = dataloader.load_all_data()
    train_acc_list = []
    acc_list = []
    for experiment_id in range(len(data)):
        for session_id in range(len(data[experiment_id])):
            print(experiment_id, session_id)
            raw_train_data = data[experiment_id][session_id]["train_data"]
            raw_train_label = data[experiment_id][session_id]["train_label"]
            raw_test_data = data[experiment_id][session_id]["test_data"]
            raw_test_label = data[experiment_id][session_id]["test_label"]

            train_data = raw_reshape(raw_train_data)
            train_label = raw_train_label
            test_data = raw_reshape(raw_test_data)
            test_label = raw_test_label
            
            device = torch.device("cuda")
            model = CNN().to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=0.1)
            train(model, device, train_data, train_label, optimizer)
            train_acc = test(model, device, train_data, train_label)
            acc = test(model, device, test_data, test_label)
            train_acc_list.append(train_acc)
            acc_list.append(acc)
    print("avg_train_acc: ", sum(train_acc_list)/float(len(train_acc_list)))       
    print("avg_test_acc: ", sum(acc_list)/float(len(acc_list)))

