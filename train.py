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
import os

from ResNet18 import *
from NaiveCNN import *
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

def test(model, device, dataloader):
    correct = 0
    test_times = 0
    for i, data_portion in enumerate(dataloader):
        data, label = data_portion
        data = data.float().to(device)
        label = label.float().to(device)
        model.eval()
        with torch.no_grad(): 
            outputs = model(data)
            _, pred = torch.max(outputs.data, 1)
            correct += (pred == label).sum().item()
            test_times += len(data)
    acc = correct/test_times
    return acc

def independent_train(batch_size, num_epochs, data, args):
    # Train model
    train_acc_list = []
    test_acc_list = []

    test_data_lists = []
    test_label_lists = []
    train_data_lists = []
    train_label_lists = []
    for test_session_id in range(len(data[0])):
        test_data_list = []
        test_label_list = []
        train_data_list = []
        train_label_list = []
        for experiment_id in range(len(data)):
            raw_train_data, raw_train_label, raw_test_data, raw_test_label = my_dataloader.get_data(data[experiment_id][test_session_id])
            test_data_list.append(raw_test_data)
            test_label_list.append(raw_test_label)
            for session_id in range(len(data[experiment_id])):
                if session_id == experiment_id:
                    continue
                else:
                    raw_train_data, raw_train_label, raw_test_data, raw_test_label = my_dataloader.get_data(data[experiment_id][session_id])
                    train_data_list.append(raw_train_data)
                    train_label_list.append(raw_train_label)

        test_data_list = np.concatenate(test_data_list)
        test_label_list = np.concatenate(test_label_list)
        test_data_lists.append(test_data_list)
        test_label_lists.append(test_label_list)

        train_data_list = np.concatenate(train_data_list)
        train_label_list = np.concatenate(train_label_list)
        train_data_lists.append(train_data_list)
        train_label_lists.append(train_label_list)
    
    start_time = time.time()
    print("Start_time: %.4f, Args: %s" % (start_time, args))
    for session_id in range(len(data[0])):
        # Define model, loss function, and optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.model == "ResNet18":
            model = ResNet18(num_classes=4).to(device)
        if args.model == "NaiveCNN":
             model = CNN().to(device)
        else:
            model = ResNet18(num_classes=4).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)


        logging("Test_session_id: %d" %(session_id))
        raw_train_data = train_data_lists[session_id]
        raw_train_label = train_label_lists[session_id]
        raw_test_data = test_data_lists[session_id]
        raw_test_label = test_label_lists[session_id]
        train_data = my_dataloader.raw_reshape(raw_train_data)
        train_label = raw_train_label
        test_data = my_dataloader.raw_reshape(raw_test_data)
        test_label = raw_test_label

        train_dataset = my_dataloader.my_dataset(train_data, train_label)
        train_dataloader = torch_data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = my_dataloader.my_dataset(test_data, test_label)
        test_dataloader = torch_data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    
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
            outputs = model(train_data)
            loss = criterion(outputs, train_label.long())
            train_acc = test(model, device, train_dataloader)
            test_acc = test(model, device, test_dataloader)
            model_train_acc.append(train_acc)
            model_test_acc.append(test_acc)
            if epoch % (num_epochs/10) == 0:
                logging('Epoch [%d/%d], Loss: %.4f, Train_acc: %.4f, Test_acc: %.4f' % (epoch, num_epochs, loss.item(), train_acc, test_acc))

        train_acc = max(model_train_acc)
        test_acc = max(model_test_acc)
        logging("Test_session_id: %d, Train_acc: %.4f, Test_acc: %.4f" % (session_id, train_acc, test_acc))
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        save_dir=("./model/model_%s/mode_%s/lr_%s_bs_%s" % (str(args.model), str(args.mode), str(args.lr), str(args.batch_size)))
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)
        save_path=("%s/%d.pt" % (save_dir, session_id))
        torch.save(model, save_path)


    end_time = time.time()    
    avg_train_acc = sum(train_acc_list)/len(train_acc_list)
    avg_test_acc = sum(test_acc_list)/len(test_acc_list)
    logging("Avg_train_acc: %.4f, Avg_test_acc: %.4f" % (avg_train_acc, avg_test_acc))
    logging("Total_training_time: %.4f" % (end_time-start_time))

def dependent_train(batch_size, num_epochs, data, args):
    # Train model
    train_acc_list = []
    test_acc_list = []
    start_time = time.time()
    print("Start_time: %.4f, Args: %s" % (start_time, args))
    for experiment_id in range(len(data)):
        for session_id in range(len(data[experiment_id])):
            # Define model, loss function, and optimizer
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if args.model == "ResNet18":
                model = ResNet18(num_classes=4).to(device)
            if args.model == "NaiveCNN":
                model = CNN().to(device)
            else:
                model = ResNet18(num_classes=4).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)


            logging("Experiment_id: %d, Session_id: %d" %(experiment_id, session_id))
            raw_train_data, raw_train_label, raw_test_data, raw_test_label = my_dataloader.get_data(data[experiment_id][session_id])
            train_data = my_dataloader.raw_reshape(raw_train_data)
            train_label = raw_train_label
            test_data = my_dataloader.raw_reshape(raw_test_data)
            test_label = raw_test_label

            train_dataset = my_dataloader.my_dataset(train_data, train_label)
            train_dataloader = torch_data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            test_dataset = my_dataloader.my_dataset(test_data, test_label)
            test_dataloader = torch_data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

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
                outputs = model(train_data)
                loss = criterion(outputs, train_label.long())
                train_acc = test(model, device, train_dataloader)
                test_acc = test(model, device, test_dataloader)
                model_train_acc.append(train_acc)
                model_test_acc.append(test_acc)
                if epoch % (num_epochs/10) == 0:
                    logging('Epoch [%d/%d], Loss: %.4f, Train_acc: %.4f, Test_acc: %.4f' % (epoch, num_epochs, loss.item(), train_acc, test_acc))
                   
            train_acc = max(model_train_acc)
            test_acc = max(model_test_acc)
            logging("Experiment_id: %d, Session_id: %d, Train_acc: %.4f, Test_acc: %.4f" % (experiment_id, session_id, train_acc, test_acc))
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            
            save_dir=("./model/model_%s/mode_%s/lr_%s_bs_%s" % (str(args.model), str(args.mode), str(args.lr), str(args.batch_size)))
            if os.path.exists(save_dir) == False:
                os.makedirs(save_dir)
            save_path=("%s/%d_%d.pt" % (save_dir, experiment_id, session_id))
            torch.save(model, save_path)

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
    parser.add_argument("--model", type=str, default="ResNet18")
    args = parser.parse_args()
    
    # Define training hyperparameters
    learning_rate = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    setup_seed(0)
    
    # Load data
    data = my_dataloader.load_all_data()
       
    if args.mode == "dependent":
        dependent_train(batch_size, num_epochs, data, args)
    if args.mode == "independent":
        independent_train(batch_size, num_epochs, data, args)
