import copy
import math
import csv
import random
import sys

import numpy as np


class DS_generate:
    def __init__(self, crowd_file, truth_file, **kwargs):
        self.crowd_file = crowd_file
        self.truth_file = truth_file
        e2wl, w2el, label_set = self.gete2wlandw2el()
        self.e2wl = e2wl   # {t0:[w0:l0], t1:[w1:l1], ...}
        self.w2el = w2el    # {w0:[t0:l0], w1:[t1:l1], ...}
        self.workers = self.w2el.keys()
        self.label_set = label_set
        self.initalquality = 0.7
        for kw in kwargs:
            setattr(self, kw, kwargs[kw])

    # E-step
    def Update_e2lpd(self):
        self.e2lpd = {}
        for example, worker_label_set in self.e2wl.items():
            lpd = {}
            total_weight = 0
            for tlabel, prob in self.l2pd.items():
                weight = prob
                for (w, label) in worker_label_set:
                    weight *= self.w2cm[w][tlabel][label]

                lpd[tlabel] = weight
                total_weight += weight
            for tlabel in lpd:
                if total_weight == 0:
                    # uniform distribution
                    lpd[tlabel] = 1.0 / len(self.label_set)
                else:
                    lpd[tlabel] = lpd[tlabel] * 1.0 / total_weight

            self.e2lpd[example] = lpd

        # print(self.e2lpd)  # 推断
    # M-step

    def Update_l2pd(self):
        for label in self.l2pd:
            self.l2pd[label] = 0
        for _, lpd in self.e2lpd.items():
            for label in lpd:
                self.l2pd[label] += lpd[label]

        for label in self.l2pd:
            self.l2pd[label] *= 1.0 / len(self.e2lpd)
        # print(self.l2pd)  # 更新先验
    def Update_w2cm(self):

        for w in self.workers:
            for tlabel in self.label_set:
                for label in self.label_set:
                    self.w2cm[w][tlabel][label] = 0

        w2lweights = {}
        for w in self.w2el:
            w2lweights[w] = {}
            for label in self.label_set:
                w2lweights[w][label] = 0
            for example, _ in self.w2el[w]:
                for label in self.label_set:
                    w2lweights[w][label] += self.e2lpd[example][label]

            for tlabel in self.label_set:

                if w2lweights[w][tlabel] == 0:
                    for label in self.label_set:
                        if tlabel == label:
                            self.w2cm[w][tlabel][label] = self.initalquality
                        else:
                            self.w2cm[w][tlabel][label] = (1 - self.initalquality) * 1.0 / (len(self.label_set) - 1)

                    continue

                for example, label in self.w2el[w]:
                    self.w2cm[w][tlabel][label] += self.e2lpd[example][tlabel] * 1.0 / w2lweights[w][tlabel]


        # for w in self.workers:
        #     for tlabel in self.label_set:
        #         a = 0
        #         for label in self.label_set:
        #             if tlabel == label:
        #                 a = round((1 - self.w2cm[w][tlabel][label]) * 1.0 / (len(self.label_set) - 1), 10)
        #         for label in self.label_set:
        #             if tlabel != label:
        #                 self.w2cm[w][tlabel][label] = a

        return self.w2cm

    # initialization
    def Init_l2pd(self):
        # uniform probability distribution
        l2pd = {}
        for label in self.label_set:
            l2pd[label] = 1.0 / len(self.label_set)
        return l2pd

    def Init_w2cm(self):
        w2cm = {}
        for worker in self.workers:
            w2cm[worker] = {}
            for tlabel in self.label_set:
                w2cm[worker][tlabel] = {}
                for label in self.label_set:
                    if tlabel == label:
                        w2cm[worker][tlabel][label] = self.initalquality
                    else:
                        w2cm[worker][tlabel][label] = (1 - self.initalquality) / (len(self.label_set) - 1)

        return w2cm

    def run(self, iter=50):

        self.l2pd = self.Init_l2pd()   # {l0:p0, l1:p1, l2:p2, l3:p3}
        self.w2cm = self.Init_w2cm()
                                        # {'78': {'1': {'1': 0.7, '3': 0.1, '4': 0.1, '2': 0.1},
                                        #         '3': {'1': 0.1, '3': 0.7, '4': 0.1, '2': 0.1},
                                        #         '4': {'1': 0.1, '3': 0.1, '4': 0.7, '2': 0.1},
                                        #         '2': {'1': 0.1, '3': 0.1, '4': 0.1, '2': 0.7}
                                        #         }
                                        #  '2': {'1': {'1': 0.7, '3': 0.1, '4': 0.1, '2': 0.1},
                                        #        '3': {'1': 0.1, '3': 0.7, '4': 0.1, '2': 0.1},
                                        #        '4': {'1': 0.1, '3': 0.1, '4': 0.7, '2': 0.1},
                                        #        '2': {'1': 0.1, '3': 0.1, '4': 0.1, '2': 0.7}
                                        #        }
                                        #  }

        while iter > 0:
            # E-step
            self.Update_e2lpd()
            # M-step
            self.Update_l2pd()
            self.Update_w2cm()
            # compute the likelihood
            # print self.computelikelihood()
            iter -= 1

        return self.e2lpd, self.w2cm

    def computelikelihood(self):

        lh = 0

        for _, worker_label_set in self.e2wl.items():
            temp = 0
            for tlabel, prior in self.l2pd.items():
                inner = prior
                for worker, label in worker_label_set:
                    inner *= self.w2cm[worker][tlabel][label]
                temp += inner

            lh += math.log(temp)

        return lh

    ###################################
    # The above is the EM method (a class)
    # The following are several external functions
    ###################################

    def get_accuracy(self):
        e2truth = {}
        f = open(self.truth_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]

        for line in reader:
            example, truth = line.split('\t')
            e2truth[example] = truth

        tcount = 0
        count = 0
        e2lpd = self.e2lpd
        for e in e2lpd:
            if e not in e2truth:
                continue
            temp = 0
            for label in e2lpd[e]:
                if temp < e2lpd[e][label]:
                    temp = e2lpd[e][label]
            candidate = []
            for label in e2lpd[e]:
                if temp == e2lpd[e][label]:
                    candidate.append(label)
            truth = random.choice(candidate)
            count += 1
            if truth == e2truth[e]:
                tcount += 1

        return tcount * 1.0 / count

    def generate(self, sample_file, generate_file, test_loader):

        e2truth = {}
        f = open(self.truth_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]
        for line in reader:
            example, truth = line.split('\t')
            e2truth[example] = truth
        f.close()
        f_open = open(sample_file, 'r')
        reader = f_open.readlines()
        reader = [line.strip("\n") for line in reader]
        f_save = open(generate_file, 'w')
        for line in reader:
            example, worker, label = line.split('\t')
            p = np.array(list(self.w2cm[worker][e2truth[example]].values()))
            label = np.random.choice(list(self.w2cm[worker][e2truth[example]].keys()), p=p.ravel())
            f_save.write(example + '\t' + worker + '\t' + label + '\n')
        f_open.close()
        f_save.close()

    def generate_fixed_annotator(self, exist_annotator, generate_file, test_loader):

        e2truth = {}
        f = open(self.truth_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]

        for line in reader:
            example, truth = line.split('\t')
            e2truth[example] = truth
        f.close()
        f_open = open(self.crowd_file, 'r')
        reader = f_open.readlines()
        reader = [line.strip("\n") for line in reader]
        f_save = open(generate_file, 'w')
        for line in reader:
            example, worker, label = line.split('\t')
            if worker in exist_annotator:
                f_save.write(example + '\t' + worker + '\t' + label + '\n')
            else:
                p = np.array(list(self.w2cm[worker][e2truth[example]].values()))
                label = np.random.choice(list(self.w2cm[worker][e2truth[example]].keys()), p=p.ravel())
                f_save.write(example + '\t' + worker + '\t' + label + '\n')

        f_open.close()
        f_save.close()

    def generate_fixed_task(self, exist_task, generate_file, test_loader):

        e2truth = {}
        f = open(self.truth_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]
        for line in reader:
            example, truth = line.split('\t')
            e2truth[example] = truth
        f.close()

        f_open = open(self.crowd_file, 'r')
        reader = f_open.readlines()
        reader = [line.strip("\n") for line in reader]
        f_save = open(generate_file, 'w')
        for line in reader:
            example, worker, label = line.split('\t')
            if example in exist_task:
                f_save.write(example + '\t' + worker + '\t' + label + '\n')
            else:
                p = np.array(list(self.w2cm[worker][e2truth[example]].values()))
                new_label = np.random.choice(list(self.w2cm[worker][e2truth[example]].keys()), p=p.ravel())
                f_save.write(example + '\t' + worker + '\t' + new_label + '\n')
        f_open.close()
        f_save.close()

    def generate_replenish(self, exist_task, generate_file, test_loader):
        e2truth = {}
        f = open(self.truth_file, 'r')
        reader = f.readlines()
        reader = [line.strip("\n") for line in reader]
        for line in reader:
            example, truth = line.split('\t')
            e2truth[example] = truth
        f.close()

        f_open = open(self.crowd_file, 'r')
        reader = f_open.readlines()
        reader = [line.strip("\n") for line in reader]

        worker_id_list = []
        example_id_list = []
        for line in reader:
            example, worker, label = line.split('\t')
            if worker not in worker_id_list:
                worker_id_list.append(worker)
            if example not in example_id_list:
                example_id_list.append(example)

        e2wl = {}
        for line in reader:
            example, worker, label = line.split('\t')

            if example not in e2wl:
                e2wl[example] = {}
                for worker_id in worker_id_list:
                    e2wl[example][worker_id] = -1

            e2wl[example][worker] = label

        f_open.close()

        f_save = open(generate_file, 'w')
        for example, w2l in e2wl.items():
            for worker, label in w2l.items():
                if example in exist_task:
                    if label != -1:
                        f_save.write(example + '\t' + worker + '\t' + label + '\n')
                else:
                    if label == -1:   # 未答的 生成label
                        p = np.array(list(self.w2cm[worker][e2truth[example]].values()))
                        new_label = np.random.choice(list(self.w2cm[worker][e2truth[example]].keys()), p=p.ravel())
                        f_save.write(example + '\t' + worker + '\t' + new_label + '\n')
                    else:     #  作答的 保留label
                        f_save.write(example + '\t' + worker + '\t' + label + '\n')

        f_save.close()
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
            e2wl[example].append([worker, label])

            if worker not in w2el:
                w2el[worker] = []
            w2el[worker].append([example, label])

            if label not in label_set:
                label_set.append(label)

        e2s = {}
        for example in e2wl:
            e2s[example] = len(e2wl[example])

        example_miu, example_sigma = self.redundancy_distribution(list(e2s.values()))
        self.example_miu = example_miu
        self.example_sigma = example_sigma

        return e2wl, w2el, label_set

    def generate_for_distribution(self, generate_file, generate_truth_file, number, workers_number):
        ability_miu, ability_sigma = self.get_ability_distribution()
        difficulty_miu, difficulty_sigma = self.get_difficulty_distribution()

        e2d = {}  # 任务 to 难度
        e2r = {}  # 任务 to 冗余
        e2gt = {}  # 任务 to 真值
        w2a = {}

        new_task = random.sample(range(0, number), number)
        new_worker = random.sample(range(0, workers_number), workers_number)

        gt2p = {}
        for label in self.label_set:
            gt2p[label] = 1 / len(self.label_set)

        p = np.array(list(gt2p.values()))


        for example in new_task:
            if example not in e2d:
                e2d[example] = random.normalvariate(difficulty_miu, difficulty_sigma)
            if example not in e2r:
                r = -1
                while(r<=0):
                    r = int(random.normalvariate(self.example_miu, self.example_sigma))
                e2r[example] = r
            if example not in e2gt:
                e2gt[example] = np.random.choice(list(gt2p.keys()), p=p.ravel())

        for worker in new_worker:
            if worker not in w2a:
                w2a[worker] = random.normalvariate(ability_miu, ability_sigma)

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

                l2p = {}
                for label in self.label_set:
                    if label == e2gt[example]:
                        l2p[label] = self.irt(worker_ability - self.expbeta(difficulty))
                    else:
                        l2p[label] = (1 - self.irt(worker_ability - self.expbeta(difficulty))) / (len(self.label_set) - 1)

                p = np.array(list(l2p.values()))
                label = np.random.choice(list(l2p.keys()), p=p.ravel())
                f_crowd_save.write(str(example) + '\t' + str(worker_id) + '\t' + str(label) + '\n')
        f_crowd_save.close()
        f_truth_save.close()
    def get_ability_distribution(self):
        alpha = list(self.alpha.values())
        sum = 0
        for item in alpha:
            sum += item
        miu = sum/len(alpha)
        # print(miu)
        s = 0
        for item in alpha:
            s += (item-miu)**2
        sigma = math.sqrt(s/len(alpha))
        # print(sigma)
        return miu, sigma
    def get_difficulty_distribution(self):
        beta = list(self.beta.values())
        sum = 0
        for item in beta:
            sum += item
        miu = sum/len(beta)
        # print(miu)
        s = 0
        for item in beta:
            s += (item-miu)**2
        sigma = math.sqrt(s/len(beta))
        # print(sigma)
        return miu, sigma
    def redundancy_distribution(self, count):
        sum = 0
        for item in count:
            sum += item
        miu = sum/len(count)
        # print(miu)
        s = 0
        for item in count:
            s += (item-miu)**2
        sigma = math.sqrt(s/len(count))
        # print(sigma)
        return miu, sigma









