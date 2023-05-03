
# Load the data
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from relation import Relation
from utils import get_batch


def val(data_path, data_file):
    data = Relation(data_dir="./datasets",
                    data_path=data_path,
                    data_file=data_file)
    # Batchify the data
    loader = torch.utils.data.DataLoader(dataset=data, batch_size=1, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mymodel = torch.load('./model/mymodel2')
    mymodel.to(device)
    mymodel.eval()

    count = 0
    num = 0
    # TP, TN, FP, FN = 0, 0, 0, 0
    states = mymodel.t_vae.init_hidden(1)
    esp = 1e-6
    with torch.no_grad():

        sum = 0
        for iteration, batch in enumerate(loader):

            annotator_id, answer, task, target, task_lengths = get_batch(batch)
            # 工人
            annotator_id = np.array(annotator_id).astype(dtype=int)
            annotator_id = torch.from_numpy(annotator_id).to(device)
            annotator_inputs = F.one_hot(annotator_id, 154).type(torch.float32)  # 工人input
            # 工人vae
            a_output, a_mean, a_logv, a_z = mymodel.a_vae(annotator_inputs)
            # 任务
            task = task.to(device)
            # 任务vae
            t_output, t_mean, t_logv, t_z, states = mymodel.t_vae(task, task_lengths, states)
            states = states[0].detach(), states[1].detach()
            z = torch.cat((a_z, np.squeeze(t_z, axis=1)), 1)
            dev_label = mymodel(z)
            label_tensor = torch.from_numpy(np.array(batch['answer']).astype(dtype=float)).type(
                torch.LongTensor).to(device)

            prediction = torch.max(F.softmax(dev_label), 1)[1]
            pred_label = prediction.cpu().data.numpy().squeeze()
            target_label = label_tensor.cpu().data.numpy()

            # 0为正类, 1为负类
            if pred_label == target_label[0]:
                count += 1
            num += len(annotator_id)

    ACC = count/num

    print("acc:", ACC)

if __name__ == '__main__':
    val("/train_data/domain_relation_train.csv", "/train_data/domain_relation_task_train.json")
    val("/test_data/domain_relation_test.csv", "/test_data/domain_relation_task_test.json")


