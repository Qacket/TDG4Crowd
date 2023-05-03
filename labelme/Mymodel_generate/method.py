import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from lab import LAB

class Mymodel_generate:
    def __init__(self, crowd_file, truth_file, **kwargs):
        self.crowd_file = crowd_file
        self.truth_file = truth_file
        e2wl, w2el, label_set = self.gete2wlandw2el()
        self.e2wl = e2wl
        self.w2el = w2el
        self.workers = self.w2el.keys()
        self.examples = self.e2wl.keys()
        self.label_set = label_set

    def redundancy_distribution(self, count):
        sum = 0
        for item in count:
            sum += item
        miu = sum / len(count)
        # print(miu)
        s = 0
        for item in count:
            s += (item - miu) ** 2
        sigma = math.sqrt(s / len(count))
        # print(sigma)
        return miu, sigma
    def gete2wlandw2el(self):
        e2wl = {}
        w2el = {}
        label_set = []

        f = open(self.crowd_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]

        for line in reader:
            example, worker, label = line.split('\t')
            if example not in e2wl:
                e2wl[example] = []
            e2wl[example].append([worker,label])

            if worker not in w2el:
                w2el[worker] = []
            w2el[worker].append([example,label])

            if label not in label_set:
                label_set.append(label)

        e2s = {}
        for example in e2wl:
            e2s[example] = len(e2wl[example])

        example_miu, example_sigma = self.redundancy_distribution(list(e2s.values()))
        self.example_miu = example_miu
        self.example_sigma = example_sigma

        return e2wl, w2el, label_set

    def run(self):
        pass

    def generate_fixed_task(self, exist_task, generate_file, train_loader):

        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        mymodel = torch.load('./model/mymodel')
        mymodel.to(device)
        f_save = open(generate_file, 'w')
        for idx, data in enumerate(train_loader):
            task_id = data[1].numpy().astype(np.int64)  # 遍历得到的current_task
            annotator_id = np.argmax(data[2].numpy(), axis=1)
            label_np = data[3].numpy()
            if task_id in list(map(int, exist_task)):  # current_task在集合中 直接保存
                f_save.write(str(task_id[0]) + '\t' + str(annotator_id[0]) + '\t' + str(label_np[0]) + '\n')
            else:
                annotator_inputs = data[2].to(device)
                task_inputs = data[4].to(device)
                a_output, a_mean, a_logv, a_z = mymodel.a_vae(annotator_inputs)  # 获得 工人能力z1  生成的工人^ dev_annotator
                t_output, t_mean, t_logv, t_z = mymodel.t_vae(task_inputs)  # 获得 任务能力z2  生成的任务^ dev_task
                z = torch.cat((a_z, t_z), 1)  # z1 z2 结合
                dev_label = mymodel(z)  # 获得生成的标注^dev_label

                p = F.softmax(dev_label, dim=1).detach().cpu().numpy()
                label_new = np.random.choice(np.array(range(8)), p=p.ravel())
                f_save.write(str(task_id[0]) + '\t' + str(annotator_id[0]) + '\t' + str(label_new) + '\n')
        f_save.close()
    def generate_fixed_annotator(self, exist_annotator, generate_file, train_loader):
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        mymodel = torch.load('./model/mymodel')
        mymodel.to(device)
        f_save = open(generate_file, 'w')
        for idx, data in enumerate(train_loader):
            task_id = data[1].numpy().astype(np.int64)
            annotator_id = np.argmax(data[2].numpy(), axis=1)  # 遍历得到的current_annotator_id
            label_np = data[3].numpy()
            if annotator_id in list(map(int, exist_annotator)):  # current_annotator在集合中 直接保存
                f_save.write(str(task_id[0]) + '\t' + str(annotator_id[0]) + '\t' + str(label_np[0]) + '\n')
            else:
                annotator_inputs = data[2].to(device)
                task_inputs = data[4].to(device)

                a_output, a_mean, a_logv, a_z = mymodel.a_vae(annotator_inputs)  # 获得 工人能力z1  生成的工人^ dev_annotator
                t_output, t_mean, t_logv, t_z = mymodel.t_vae(task_inputs)  # 获得 任务能力z2  生成的任务^ dev_task
                z = torch.cat((a_z, t_z), 1)  # z1 z2 结合
                dev_label = mymodel(z)  # 获得生成的标注^dev_label

                p = F.softmax(dev_label, dim=1).detach().cpu().numpy()
                label_new = np.random.choice(np.array(range(8)), p=p.ravel())
                f_save.write(str(task_id[0]) + '\t' + str(annotator_id[0]) + '\t' + str(label_new) + '\n')
        f_save.close()
    def generate(self, sample_file, generate_file, train_loader):

        f_open = open(sample_file, 'r')
        reader = f_open.readlines()
        reader = [line.strip("\n") for line in reader]
        examples = []
        for line in reader:
            example, worker, label = line.split('\t')
            if example not in examples:
                examples.append(example)
        f_open.close()

        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        mymodel = torch.load('./model/mymodel')
        mymodel.to(device)
        f_save = open(generate_file, 'w')
        for idx, data in enumerate(train_loader):
            task_id = data[1].numpy().astype(np.int64)
            annotator_id = np.argmax(data[2].numpy(), axis=1)  # 遍历得到的current_annotator_id
            label_np = data[3].numpy()
            if task_id in list(map(int, examples)):
                # task_id = data[1].to(device)
                annotator_inputs = data[2].to(device)
                # label_tensor = data[3].to(device)
                task_inputs = data[4].to(device)

                a_output, a_mean, a_logv, a_z = mymodel.a_vae(annotator_inputs)  # 获得 工人能力z1  生成的工人^ dev_annotator
                t_output, t_mean, t_logv, t_z = mymodel.t_vae(task_inputs)  # 获得 任务能力z2  生成的任务^ dev_task
                z = torch.cat((a_z, t_z), 1)  # z1 z2 结合
                dev_label = mymodel(z)  # 获得生成的标注^dev_label

                p = F.softmax(dev_label, dim=1).detach().cpu().numpy()
                label_new = np.random.choice(np.array(range(8)), p=p.ravel())
                f_save.write(str(task_id[0]) + '\t' + str(annotator_id[0]) + '\t' + str(label_new) + '\n')
        f_save.close()
    def generate_replenish(self, exist_task, generate_file, train_loader):
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        mymodel = torch.load('./model/mymodel')
        mymodel.to(device)

        taskidx2task = {}
        annotatoridx2onehot = {}
        annotator_id_list = []
        for iteration, data in enumerate(train_loader):

            task_id = data[1].numpy().astype(np.int64)[0]
            annotator_id = np.argmax(data[2].numpy(), axis=1)[0]
            annotator_inputs = data[2]
            task_inputs = data[4]
            if task_id not in taskidx2task:
                taskidx2task[task_id] = task_inputs

            if annotator_id not in annotatoridx2onehot:
                annotatoridx2onehot[annotator_id] = annotator_inputs

            if annotator_id not in annotator_id_list:
                annotator_id_list.append(annotator_id)

        t2wl = {}
        for iteration, data in enumerate(train_loader):
            task_id = data[1].numpy().astype(np.int64)[0]
            annotator_id = np.argmax(data[2].numpy(), axis=1)[0]
            label_np = data[3].numpy()[0]

            if task_id not in t2wl:
                t2wl[task_id] = {}
                for annotator in annotator_id_list:
                    t2wl[task_id][annotator] = -1

            t2wl[task_id][annotator_id] = label_np
        f_save = open(generate_file, 'w')
        for task_id, w2l in t2wl.items():
            for annotator_id, answer in w2l.items():
                if task_id in list(map(int, exist_task)):
                    if answer != -1:
                        f_save.write(str(task_id) + '\t' + str(annotator_id) + '\t' + str(answer) + '\n')

                else:
                    if answer == -1:
                        annotator_inputs = annotatoridx2onehot[annotator_id].to(device)
                        task_inputs = taskidx2task[task_id].to(device)
                        a_output, a_mean, a_logv, a_z = mymodel.a_vae(annotator_inputs)  # 获得 工人能力z1  生成的工人^ dev_annotator
                        t_output, t_mean, t_logv, t_z = mymodel.t_vae(task_inputs)  # 获得 任务能力z2  生成的任务^ dev_task
                        z = torch.cat((a_z, t_z), 1)  # z1 z2 结合
                        dev_label = mymodel(z)  # 获得生成的标注^dev_label
                        p = F.softmax(dev_label, dim=1).detach().cpu().numpy()
                        label_new = np.random.choice(np.array(range(8)), p=p.ravel())
                        f_save.write(str(task_id) + '\t' + str(annotator_id) + '\t' + str(label_new) + '\n')
                    else:
                        f_save.write(str(task_id) + '\t' + str(annotator_id) + '\t' + str(answer) + '\n')
        f_save.close()




    def generate_for_distribution(self, generate_file, generate_truth_file, number, workers_number):

        e2d = {}  # 任务 to 难度
        e2r = {}  # 任务 to 冗余
        e2gt = {}  # 任务 to 真值
        w2a = {}

        train_dataset = Dataset()
        trn_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mymodel = torch.load('/mnt/4T/scj/labelme/Mymodel_generate/mymodel_labelme').to(device)


        new_task = random.sample(range(0, number), number)
        new_worker = random.sample(range(0, workers_number), workers_number)

        gt2p = {}
        for label in self.label_set:
            gt2p[label] = 1 / len(self.label_set)

        p = np.array(list(gt2p.values()))


        for example in new_task:
            if example not in e2d:
                e2d[example] = torch.from_numpy(np.random.normal(0, 1, size=50).reshape((1, 50))).float().to(device)
            if example not in e2r:
                r = -1
                while(r<=0):
                    r = int(random.normalvariate(self.example_miu, self.example_sigma))
                e2r[example] = r
            if example not in e2gt:
                e2gt[example] = np.random.choice(list(gt2p.keys()), p=p.ravel())

        for worker in new_worker:
            if worker not in w2a:
                w2a[worker] = torch.from_numpy(np.random.normal(0, 1, size=8).reshape((1, 8))).float().to(device)


        f_crowd_save = open(generate_file, 'w')
        f_truth_save = open(generate_truth_file, 'w')

        for example in new_task:
            f_truth_save.write(str(example) + '\t' + str(e2gt[example]) + '\n')

            difficulty = e2d[example]
            redundancy = e2r[example]
            worker_id_list = random.sample(list(w2a.keys()), redundancy)
            for i in range(redundancy):
                worker_id = worker_id_list[i]
                worker_ability = w2a[worker_id]
                z = torch.cat((worker_ability, difficulty), 1)  # z1 z2 结合
                dev_label = mymodel(z)  # 获得生成的标注^dev_label

                p = F.softmax(dev_label, dim=1).detach().cpu().numpy()
                label_new = np.random.choice(np.array(range(8)), p=p.ravel())

                f_crowd_save.write(str(example) + '\t' + str(worker_id) + '\t' + str(label_new) + '\n')
        f_crowd_save.close()
        f_truth_save.close()