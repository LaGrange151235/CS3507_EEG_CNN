import torch.utils.data as Data
import numpy as np
import os

data_path = "./SEED-IV/SEED-IV"

class my_dataset(Data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label
    def __len__(self):
        return len(self.data)

def load_all_data():
    data_list = []
    for experiment_id in os.listdir(data_path):
        experiment_path = data_path+"/"+experiment_id
        experiment_data_list = []
        for session_id in os.listdir(experiment_path):
            session_path = experiment_path+"/"+session_id
            session_data = {
                    "train_data":   np.load(session_path+"/train_data.npy"),
                    "train_label":  np.load(session_path+"/train_label.npy"),
                    "test_data":    np.load(session_path+"/test_data.npy"),
                    "test_label":   np.load(session_path+"/test_label.npy")
                    }
            experiment_data_list.append(session_data)
        data_list.append(experiment_data_list)
    return data_list

def load_data(require_experiment_id, require_session_id):
    i = 0
    for experiment_id in os.listdir(data_path):
        if require_experiment_id == i:
            experiment_path = data_path+"/"+experiment_id
            j = 0
            for session_id in os.listdir(experiment_path):
                if require_session_id == j:
                    session_path = experiment_path+"/"+session_id
                    session_data = {
                            "train_data":   np.load(session_path+"/train_data.npy"),
                            "train_label":  np.load(session_path+"/train_label.npy"),
                            "test_data":    np.load(session_path+"/test_data.npy"),
                            "test_label":   np.load(session_path+"/test_label.npy")
                            }
                    return session_data
                j += 1
        i += 1
                
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

def get_data(data):
    return data["train_data"], data["train_label"], data["test_data"], data["test_label"]
