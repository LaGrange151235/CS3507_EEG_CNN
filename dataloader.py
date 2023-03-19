import numpy as np
import os

data_path = "./SEED-IV/SEED-IV"

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
                

