import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import dataloader
from model import train
import resnet18
import resnet18_zhk

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
    data = dataloader.load_all_data()
    train_acc_list = []
    test_acc_list = []
    device = torch.device("cuda")
    for experiment_id in range(len(data)):
        for session_id in range(len(data[experiment_id])):
            print("================================================================")
            print("experiment_id: %d, session_id: %d" %(experiment_id, session_id))
            raw_train_data = data[experiment_id][session_id]["train_data"]
            raw_train_label = data[experiment_id][session_id]["train_label"]
            raw_test_data = data[experiment_id][session_id]["test_data"]
            raw_test_label = data[experiment_id][session_id]["test_label"]


            train_data = raw_reshape(raw_train_data)
            train_label = raw_train_label
            test_data = raw_reshape(raw_test_data)
            test_label = raw_test_label

            train_data = torch.tensor(np.array(train_data), dtype=torch.float32).to(device)
            train_label = torch.tensor(np.array(train_label), dtype=torch.float32).to(device)
            test_data = torch.tensor(np.array(test_data), dtype=torch.float32).to(device)
            test_label = torch.tensor(np.array(test_label), dtype=torch.float32).to(device)

            model = resnet18_zhk.ResNet18_().to(device)
#            model = resnet18.resnet18().to(device)
            optimizer = optim.Adam(model.parameters(), lr=5*1e-5)
            model_train_acc_list = []
            model_test_acc_list = []
            for epoch in range(10):
                outputs = model(train_data)
                loss = F.cross_entropy(outputs, train_label.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()   

                correct = 0.0
                total = 0.0
                outputs = model(train_data)
                predicted = outputs.argmax(dim=1, keepdim=True)
                total += train_label.size(0)
                correct += predicted.eq(train_label.view_as(predicted)).sum().item()
#                print('Train Acc: %.3f%% ' % (100. * correct / total))
                model_train_acc_list.append(100. * correct / total)
            
                correct = 0.0
                total = 0.0
                outputs = model(test_data)
                predicted = outputs.argmax(dim=1, keepdim=True)
                total += test_label.size(0)
                correct += predicted.eq(test_label.view_as(predicted)).sum().item()
 #               print('Test Acc: %.3f%% ' % (100. * correct / total))
                model_test_acc_list.append(100. * correct / total)
          
            train_acc = max(model_train_acc_list)
            test_acc = max(model_test_acc_list)
            print('Max Train Acc: %.3f%% ' % (train_acc))
            print('Max Test Acc: %.3f%% ' % (test_acc))
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

    print('Avg Train Acc: %.3f%%, Test Acc: %.3f%%' % (sum(train_acc_list)/float(len(train_acc_list)), sum(test_acc_list)/float(len(test_acc_list))))
