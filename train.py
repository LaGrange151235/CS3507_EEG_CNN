from math import exp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torch_data
import torch.backends.cudnn as backends_cudnn
import numpy as np
import random
import argparse
import time

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

def dependent_train(batch_size, num_epochs, data, device, model, criterion, optimizer):
    # Train model
    train_acc_list = []
    test_acc_list = []
    start_time = time.time()
    print("Start_time: %.4f, Args: %s" % (start_time, args))
    for experiment_id in range(len(data)):
        for session_id in range(len(data[experiment_id])):
            logging("Experiment_id: %d, Session_id: %d" %(experiment_id, session_id))
            raw_train_data, raw_train_label, raw_test_data, raw_test_label = my_dataloader.get_data(data[experiment_id][session_id])
            train_data = my_dataloader.raw_reshape(raw_train_data)
            train_label = raw_train_label
            test_data = my_dataloader.raw_reshape(raw_test_data)
            test_label = raw_test_label
    
            train_data_gpu = torch.tensor(np.array(train_data), dtype=torch.float32).to(device)
            train_label_gpu = torch.tensor(np.array(train_label), dtype=torch.float32).to(device)
            test_data_gpu = torch.tensor(np.array(test_data), dtype=torch.float32).to(device)
            test_label_gpu = torch.tensor(np.array(test_label), dtype=torch.float32).to(device)
    
            train_dataset = my_dataloader.my_dataset(train_data, train_label)
            train_dataloader = torch_data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
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

    end_time = time.time()    
    avg_train_acc = sum(train_acc_list)/len(train_acc_list)
    avg_test_acc = sum(test_acc_list)/len(test_acc_list)
    logging("Avg_train_acc: %.4f, Avg_test_acc: %.4f" % (avg_train_acc, avg_test_acc))
    logging("Total_training_time: %.4f" % (end_time-start_time))

def independent_train(batch_size, num_epochs, data, device, model, criterion, optimizer):
    # Train model
    train_acc_list = []
    test_acc_list = []
    start_time = time.time()
    print("Start_time: %.4f, Args: %s" % (start_time, args))
    for experiment_id in range(len(data)):
        for session_id in range(len(data[experiment_id])):
            logging("Experiment_id: %d, Session_id: %d" %(experiment_id, session_id))
            raw_train_data, raw_train_label, raw_test_data, raw_test_label = my_dataloader.get_data(data[experiment_id][session_id])
            train_data = my_dataloader.raw_reshape(raw_train_data)
            train_label = raw_train_label
            test_data = my_dataloader.raw_reshape(raw_test_data)
            test_label = raw_test_label
    
            train_data_gpu = torch.tensor(np.array(train_data), dtype=torch.float32).to(device)
            train_label_gpu = torch.tensor(np.array(train_label), dtype=torch.float32).to(device)
            test_data_gpu = torch.tensor(np.array(test_data), dtype=torch.float32).to(device)
            test_label_gpu = torch.tensor(np.array(test_label), dtype=torch.float32).to(device)
    
            train_dataset = my_dataloader.my_dataset(train_data, train_label)
            train_dataloader = torch_data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
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

    end_time = time.time()    
    avg_train_acc = sum(train_acc_list)/len(train_acc_list)
    avg_test_acc = sum(test_acc_list)/len(test_acc_list)
    logging("Avg_train_acc: %.4f, Avg_test_acc: %.4f" % (avg_train_acc, avg_test_acc))
    logging("Total_training_time: %.4f" % (end_time-start_time))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--mode", type=str, default="dependent")
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
    
    if args.mode == "dependent":
        dependent_train(batch_size, num_epochs, data, device, model, criterion, optimizer)
    if args.mode == "independent":
        independent_train(batch_size, num_epochs, data, device, model, criterion, optimizer)
