import math
import random
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils import idx_to_word, get_batch


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

    def generate_fixed_task(self, exist_task, generate_file, test_loader):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mymodel = torch.load('./model/mymodel')
        mymodel.to(device)

        f_save = open(generate_file, 'w')
        states = mymodel.t_vae.init_hidden(1)

        mymodel.eval()
        for iteration, batch in enumerate(test_loader):

            annotator_id, answer, task, target, task_lengths = get_batch(batch)

            task_id = np.array(batch["task_id"]).astype(dtype=int)
            annotator_id = np.array(annotator_id).astype(dtype=int)
            label_np = np.array(answer).astype(dtype=int)


            if task_id in list(map(int, exist_task)):  # current_task在集合中 直接保存
                f_save.write(str(task_id[0]) + '\t' + str(annotator_id[0]) + '\t' + str(label_np[0]) + '\n')
            else:
                annotator_tensor = torch.from_numpy(annotator_id).to(device)
                annotator_inputs = F.one_hot(annotator_tensor, 154).type(torch.float32)  # 工人input

                # 工人vae
                a_output, a_mean, a_logv, a_z = mymodel.a_vae(annotator_inputs)
                task = task.to(device)
                # 任务vae
                t_output, t_mean, t_logv, t_z, states = mymodel.t_vae(task, task_lengths, states)
                states = states[0].detach(), states[1].detach()

                # label
                z = torch.cat((a_z, np.squeeze(t_z, axis=1)), 1)
                dev_label = mymodel(z)


                p = F.softmax(dev_label, dim=1).detach().cpu().numpy()
                label_new = np.random.choice(np.array(range(13)), p=p.ravel())

                f_save.write(str(task_id[0]) + '\t' + str(annotator_id[0]) + '\t' + str(label_new) + '\n')
        f_save.close()

    def generate_fixed_annotator(self, exist_annotator, generate_file, test_loader):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mymodel = torch.load('./model/mymodel')
        mymodel.to(device)
        mymodel.eval()

        f_save = open(generate_file, 'w')

        states = mymodel.t_vae.init_hidden(1)
        for iteration, batch in enumerate(test_loader):

            annotator_id, answer, task, target, task_lengths = get_batch(batch)

            task_id = np.array(batch["task_id"]).astype(dtype=int)
            annotator_id = np.array(annotator_id).astype(dtype=int)
            label_np = np.array(answer).astype(dtype=int)

            if annotator_id in list(map(int, exist_annotator)):  # current_annotator在集合中 直接保存
                f_save.write(str(task_id[0]) + '\t' + str(annotator_id[0]) + '\t' + str(label_np[0]) + '\n')
            else:
                annotator_tensor = torch.from_numpy(annotator_id).to(device)
                annotator_inputs = F.one_hot(annotator_tensor, 154).type(torch.float32)  # 工人input


                # 工人vae
                a_output, a_mean, a_logv, a_z = mymodel.a_vae(annotator_inputs)
                task = task.to(device)
                # 任务vae
                t_output, t_mean, t_logv, t_z, states = mymodel.t_vae(task, task_lengths, states)
                states = states[0].detach(), states[1].detach()

                # label
                z = torch.cat((a_z, np.squeeze(t_z, axis=1)), 1)
                dev_label = mymodel(z)


                p = F.softmax(dev_label, dim=1).detach().cpu().numpy()
                label_new = np.random.choice(np.array(range(13)), p=p.ravel())

                f_save.write(str(task_id[0]) + '\t' + str(annotator_id[0]) + '\t' + str(label_new) + '\n')
        f_save.close()

    def generate(self, sample_file, generate_file, test_loader):

        f_open = open(sample_file, 'r')
        reader = f_open.readlines()
        reader = [line.strip("\n") for line in reader]
        examples = []
        for line in reader:
            example, worker, label = line.split('\t')
            if example not in examples:
                examples.append(example)
        f_open.close()


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mymodel = torch.load('./model/mymodel')
        mymodel.to(device)
        mymodel.eval()

        f_save = open(generate_file, 'w')

        states = mymodel.t_vae.init_hidden(1)




        for iteration, batch in enumerate(test_loader):

            annotator_id, answer, task, target, task_lengths = get_batch(batch)

            task_id = np.array(batch["task_id"]).astype(dtype=int)
            annotator_id = np.array(annotator_id).astype(dtype=int)
            label_np = np.array(answer).astype(dtype=int)

            if task_id in list(map(int, examples)):  # current_task在集合中 直接保存
                annotator_tensor = torch.from_numpy(annotator_id).to(device)
                annotator_inputs = F.one_hot(annotator_tensor, 154).type(torch.float32)  # 工人input


                # 工人vae
                a_output, a_mean, a_logv, a_z = mymodel.a_vae(annotator_inputs)
                task = task.to(device)
                # 任务vae
                t_output, t_mean, t_logv, t_z, states = mymodel.t_vae(task, task_lengths, states)
                states = states[0].detach(), states[1].detach()

                # label
                z = torch.cat((a_z, np.squeeze(t_z, axis=1)), 1)
                dev_label = mymodel(z)


                p = F.softmax(dev_label, dim=1).detach().cpu().numpy()
                label_new = np.random.choice(np.array(range(13)), p=p.ravel())

                f_save.write(str(task_id[0]) + '\t' + str(annotator_id[0]) + '\t' + str(label_new) + '\n')
        f_save.close()


    def generate_replenish(self, exist_task, generate_file, test_loader):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mymodel = torch.load('./model/mymodel')
        mymodel.to(device)
        mymodel.eval()

        task2length = {}
        taskidx2task = {}
        annotator_id_list = []


        for iteration, batch in enumerate(test_loader):

            annotator_id, answer, task, target, task_lengths = get_batch(batch)

            task_id = batch["task_id"][0]
            annotator_id = annotator_id[0]

            if task_id not in taskidx2task:
                taskidx2task[task_id] = task

            if task_id not in task2length:
                task2length[task_id] = task_lengths

            if annotator_id not in annotator_id_list:
                annotator_id_list.append(annotator_id)


        t2wl = {}
        for iteration, batch in enumerate(test_loader):

            annotator_id = batch["annotator_id"][0]
            task_id = batch["task_id"][0]
            answer = batch["answer"][0]

            if task_id not in t2wl:
                t2wl[task_id] = {}
                for annotator in annotator_id_list:
                    t2wl[task_id][annotator] = '-1'

            t2wl[task_id][annotator_id] = answer


        f_save = open(generate_file, 'w')
        states = mymodel.t_vae.init_hidden(1)


        for task_id, w2l in t2wl.items():
            for annotator_id, answer in w2l.items():
                if task_id in exist_task:
                    if answer != '-1':
                        f_save.write(str(task_id) + '\t' + str(annotator_id) + '\t' + str(answer) + '\n')
                else:
                    if answer == '-1':

                        annotator_id = np.array(annotator_id).astype(dtype=int)
                        annotator_tensor = torch.from_numpy(annotator_id).unsqueeze(dim=0).to(device)
                        annotator_inputs = F.one_hot(annotator_tensor, 154).type(torch.float32)  # 工人input

                        a_output, a_mean, a_logv, a_z = mymodel.a_vae(annotator_inputs)

                        task_inputs = taskidx2task[task_id].to(device)
                        task_lengths = task2length[task_id]

                        t_output, t_mean, t_logv, t_z, states = mymodel.t_vae(task_inputs, task_lengths, states)
                        states = states[0].detach(), states[1].detach()

                        # label

                        z = torch.cat((a_z, np.squeeze(t_z, axis=1)), 1)
                        dev_label = mymodel(z)

                        p = F.softmax(dev_label, dim=1).detach().cpu().numpy()
                        label_new = np.random.choice(np.array(range(13)), p=p.ravel())

                        f_save.write(str(task_id) + '\t' + str(annotator_id) + '\t' + str(label_new) + '\n')

                    else:
                        f_save.write(str(task_id) + '\t' + str(annotator_id) + '\t' + str(answer) + '\n')
        f_save.close()
