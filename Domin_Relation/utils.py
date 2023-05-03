from collections import Counter, OrderedDict

import numpy as np
import scipy.stats
from scipy import stats
import torch.nn.functional as F

def pre_deal(crowd_file, truth_file):

    gt2t = {}
    f_truth_open = open(truth_file, 'r')
    reader = f_truth_open.readlines()
    reader = [line.strip("\n") for line in reader]
    for line in reader:
        task, gt = line.split('\t')
        gt = int(gt)
        if gt not in gt2t:
            gt2t[gt] = []
        gt2t[gt].append(task)

    f_truth_open.close()
    gt_sum = len(gt2t.keys())
    gt2t = sorted(gt2t.items(), key=lambda item: item[0], reverse=False)
    classification_task = []

    for i in range(len(gt2t)):
        classification_task.append(gt2t[i][1])
    data_list = []
    f_crowd_open = open(crowd_file, 'r')
    reader = f_crowd_open.readlines()
    reader = [line.strip("\n") for line in reader]
    for i in range(gt_sum):
        data = []
        for line in reader:
            task, worker, label = line.split('\t')
            if task in classification_task[i]:
                data.append(int(label))
        data_list.append(data)

    f_crowd_open.close()
    return data_list


def all_list(arr):
    x = {}
    for i in set(arr):
        x[i] = arr.count(i)
    for i in range(13):
        if i not in x.keys():
            x[i] = 0

    x = sorted(x.items(), key=lambda item: item[0], reverse=False)

    result = []
    for i in range(len(x)):
        result.append(x[i][1])
    return result

def calculate_kl_ks(x, y):

    # KS
    (ks_stat, ks_pval) = stats.ks_2samp(x, y)


    # KL
    x = all_list(x)
    y = all_list(y)

    print(x)
    print(y)
    x = [i+1 for i in x]   # Laplace smoothing
    y = [i+1 for i in y]   # Laplace smoothing
    px = x / np.sum(x)
    py = y / np.sum(y)

    kl = scipy.stats.entropy(px, py)
    print(kl)
    return kl, ks_stat, ks_pval



def idx_to_word(x, vocab, eos_idx, pad_idx):
    words = []
    for i in x:
        word = vocab.itos[i]
        if '<' not in word:
            words.append(word)
        if i == eos_idx or i == pad_idx:
            break
    words = " ".join(words)
    return words


def get_batch(batch):
    annotator_id = batch["annotator_id"]
    answer = batch["answer"]
    sentences = batch["input"]
    target = batch["target"]
    sentences_length = batch["length"]
    return annotator_id, answer, sentences, target, sentences_length



class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)



def cal_acc(original_file, generate_file):
    f_ori = open(original_file, 'r')
    reader_ori = f_ori.readlines()
    reader_ori = [line.strip("\n") for line in reader_ori]

    f_gen = open(generate_file, 'r')
    reader_gen = f_gen.readlines()
    reader_gen = [line.strip("\n") for line in reader_gen]

    ori_lable = []
    for line in reader_ori:
        example, worker, label = line.split('\t')
        ori_lable.append(label)

    gen_lable = []
    for line in reader_gen:
        example, worker, label = line.split('\t')
        gen_lable.append(label)

    count = 0
    for i in range(len(ori_lable)):
        if ori_lable[i] == gen_lable[i]:
            count += 1
    acc = count / len(ori_lable)

    return acc


