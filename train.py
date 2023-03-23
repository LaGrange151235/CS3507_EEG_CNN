from math import exp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
import torch.backends.cudnn as backends_cudnn
import numpy as np
import random
import argparse

from ResNet18 import *
import dataloader as my_dataloader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    backends_cudnn.deterministic = True

def logging(string):
    print(string)
    with open("./record.txt", "a+", encoding="utf-8") as f:
        f.write(string+"\n")

def get_data(data):
    return data["train_data"], data["train_label"], data["test_data"], data["test_label"]

def test(model, test_data_list, test_label_list):
    model.eval()
    correct = 0
    with torch.no_grad(): 
        data = test_data_list
        label = test_label_list
        outputs = model(data)
        _, pred = torch.max(outputs.data, 1)
        correct = (pred == label).sum().item()
        acc = correct / float(len(label))
    return acc

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

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    args = parser.parse_args()
    
    # Define training hyperparameters
    learning_rate = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    setup_seed(0)
    
    # Load data
    data = my_dataloader.load_all_data()
    
    # Define model, loss function, and optimizer
    device = torch.device("cuda")
    model = ResNet18(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    train_acc_list = []
    test_acc_list = []
    for experiment_id in range(len(data)):
        for session_id in range(len(data[experiment_id])):
            logging("Experiment_id: %d, Session_id: %d" %(experiment_id, session_id))
            raw_train_data, raw_train_label, raw_test_data, raw_test_label = get_data(data[experiment_id][session_id])
            train_data = raw_reshape(raw_train_data)
            train_label = raw_train_label
            test_data = raw_reshape(raw_test_data)
            test_label = raw_test_label
    
            train_data_gpu = torch.tensor(np.array(train_data), dtype=torch.float32).to(device)
            train_label_gpu = torch.tensor(np.array(train_label), dtype=torch.float32).to(device)
            test_data_gpu = torch.tensor(np.array(test_data), dtype=torch.float32).to(device)
            test_label_gpu = torch.tensor(np.array(test_label), dtype=torch.float32).to(device)
    
            train_dataset = my_dataloader.my_dataset(train_data, train_label)
#            test_dataset = my_dataloader.my_dataset(test_data, test_label) 
            train_dataloader = torch_data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
#            test_dataloader = torch_data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    
            model_train_acc = []
            model_test_acc = []
            for epoch in range(num_epochs):
                for i, data_portion in enumerate(train_dataloader):
                    train_data, train_label = data_portion
                    train_data = train_data.float().to(device)
                    train_label = train_label.float().to(device)
                    model.train()
                    # Forward pass
                    outputs = model(train_data)
                    loss = criterion(outputs, train_label.long())
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                outputs = model(train_data_gpu)
                loss = criterion(outputs, train_label_gpu.long())
                train_acc = test(model, train_data_gpu, train_label_gpu)
                test_acc = test(model, test_data_gpu, test_label_gpu)
                model_train_acc.append(train_acc)
                model_test_acc.append(test_acc)
                if epoch % (num_epochs/10) == 0:
                    logging('Epoch [%d/%d], Loss: %.4f, Train_acc: %.4f, Test_acc: %.4f' % (epoch, num_epochs, loss.item(), train_acc, test_acc))
                   
            train_acc = max(model_train_acc)
            test_acc = max(model_test_acc)
            logging("Experiment_id: %d, Session_id: %d, Train_acc: %.4f, Test_acc: %.4f" % (experiment_id, session_id, train_acc, test_acc))
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
    
        avg_train_acc = sum(train_acc_list)/len(train_acc_list)
        avg_test_acc = sum(test_acc_list)/len(test_acc_list)
        logging("Avg_train_acc: %.4f, Avg_test_acc: %.4f" % (avg_train_acc, avg_test_acc))
