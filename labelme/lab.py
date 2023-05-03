import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class LAB(Dataset):

    def __init__(self, data_dir, crowd_dir):


        features = np.load(data_dir + '/data_vgg16.npy')   # 任务特征 (10000, 4, 4, 512)
        features = features.reshape(features.shape[0], -1)   # 10000 * 8192

        f_open = open(data_dir + crowd_dir, 'r')
        reader = f_open.readlines()
        reader = [line.strip("\n") for line in reader]
        workers_num = []
        classes = []
        for line in reader:
            task, worker, label = line.split('\t')
            if worker not in workers_num:
                workers_num.append(worker)
            if label not in classes:
                classes.append(label)

        self.num_users = len(workers_num)  # 工人总数

        self.num_classes = len(classes)

        self.input_dims = features.shape[1]

        # 带特征的所有数据
        crowd_data = pd.read_csv(data_dir + crowd_dir, sep="\t", header=None)
        crowd_np = crowd_data.values
        crowd_data_total = np.zeros((len(crowd_data), features.shape[1]+3), dtype='float32')
        crowd_data_total[:, 0:3] = crowd_np
        features_np = np.array(features)
        for i in range(len(crowd_np)):
            current_features = features_np[crowd_np[i][1]]
            crowd_data_total[i, 3:] = current_features
        crowd_data_total = pd.DataFrame(crowd_data_total)


        task_id = crowd_data_total.iloc[:, 0]
        annotator_id = crowd_data_total.iloc[:, 1]
        label = crowd_data_total.iloc[:, 2]
        task_features = crowd_data_total.iloc[:, 3:]

        self.task_id = task_id
        # print(self.task_id)

        # annotator id 处理 onthot  得到 annotator_inputs
        annotator_id_np = annotator_id.to_numpy().astype(np.int64).reshape(-1, 1)
        annotator_tensor = torch.from_numpy(annotator_id_np)
        annotator_onehot = F.one_hot(annotator_tensor, self.num_users).resize_(len(annotator_tensor), self.num_users)
        annotator_inputs = annotator_onehot.type(torch.float32)
        self.annotator_inputs = annotator_inputs
        # print(annotator_inputs, annotator_inputs.shape)

        # label
        label_np = label.to_numpy()
        label_tensor = torch.from_numpy(label_np).type(torch.float32)
        self.label_tensor = label_tensor.type(torch.long)
        # print(label_tensor, label_tensor.shape)

        # task features 得到 task_inputs
        task_features_np = task_features.to_numpy()
        task_tensor = torch.from_numpy(task_features_np)
        task_inputs = task_tensor.type(torch.float32)
        self.task_inputs = task_inputs
        # print(task_inputs, task_inputs.shape)


    def __len__(self):
        return len(self.annotator_inputs)
    def __getitem__(self, idx):
        return idx, self.task_id[idx], self.annotator_inputs[idx], self.label_tensor[idx], self.task_inputs[idx]

