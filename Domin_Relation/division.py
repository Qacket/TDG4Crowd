cimport math
import random

import numpy as np
import pandas as pd

seed = 1
random.seed(seed)
np.random.seed(seed)

df = pd.read_csv('./datasets/total_data/total_domain_relation.csv')
# print(df)
gt = df.iloc[:, [0, 2, 3]]
gt = gt.drop_duplicates(subset=["task_id"], keep='first', inplace=False)
gt.index = range(len(gt))
# print(gt)

w2t = {}
for i in range(len(df)):
    if df.iloc[i, 1] not in w2t:
        w2t[df.iloc[i, 1]] = {}
    w2t[df.iloc[i, 1]][df.iloc[i, 0]] = df.iloc[i, 4]




for i in range(len(w2t)):
    item = list(w2t[i].items())
    random.shuffle(item)
    w2t[i] = item


print(len(w2t))
s = []
for i in range(len(w2t)):
    s.append(len(w2t[i]))

print(s)
print(len(s))
w2t_train = w2t.copy()
w2t_validation = w2t.copy()
w2t_test = w2t.copy()



print("---------------------train-------------------------------------")


# 训练集
for i in range(len(w2t_train)):

    w2t_train[i] = w2t_train[i][0:int(math.ceil(len(w2t_train[i])/2))]

print(len(w2t_train))
s = []
for i in range(len(w2t_train)):
    s.append(len(w2t_train[i]))
print(s)
f_save = open('./datasets/train_data/domain_relation_train.txt', 'w')
for worker, t2l in w2t_train.items():
    for k in range(len(t2l)):
        task_id = t2l[k][0]
        label = t2l[k][1]
        task = gt.iloc[task_id, 1]
        ground_truth = gt.iloc[task_id, 2]
        f_save.write(str(task_id) + '\t' + str(worker) + '\t' + str(task) + '\t' + str(ground_truth) + '\t' + str(label) + '\n')
f_save.close()

df = pd.read_csv("./datasets/train_data/domain_relation_train.txt", delimiter="\t", header=None)
df.to_csv("./datasets/train_data/domain_relation_train.csv", encoding='utf-8', index=False, header=["task_id", "annotator_id", "task", "ground_truth", "answer"])





# # 验证集
# for i in range(len(w2t_validation)):
#     w2t_validation[i] = w2t_validation[i][int(len(w2t_validation[i])/3)+1: (int(len(w2t_validation[i])/3)+1)*2]
#
# s = []
# for i in range(len(w2t_validation)):
#     s.append(len(w2t_validation[i]))
# print(s)
# f_save = open('./datasets/validation_data/sentiment_validation.txt', 'w')
# for worker, t2l in w2t_train.items():
#     for k in range(len(t2l)):
#         task_id = t2l[k][0]
#         label = t2l[k][1]
#         task = gt.iloc[task_id, 1]
#         ground_truth = gt.iloc[task_id, 2]
#         f_save.write(str(task_id) + '\t' + str(worker) + '\t' + str(task) + '\t' + str(ground_truth) + '\t' + str(label) + '\n')
# f_save.close()
# df = pd.read_csv("./datasets/validation_data/sentiment_validation.txt", delimiter="\t", header=None)
# df.to_csv("./datasets/validation_data/sentiment_validation.csv", encoding='utf-8', index=False, header=["task_id", "annotator_id", "task", "ground_truth", "answer"])


print("---------------------test-------------------------------------")
# 测试集
for i in range(len(w2t_test)):
    w2t_test[i] = w2t_test[i][int(math.ceil(len(w2t_test[i])/2)):]
print(len(w2t_test))
s = []
for i in range(len(w2t_test)):
    s.append(len(w2t_test[i]))
print(s)
f_save = open('./datasets/test_data/domain_relation_test.txt', 'w')
for worker, t2l in w2t_test.items():
    for k in range(len(t2l)):
        task_id = t2l[k][0]
        label = t2l[k][1]
        task = gt.iloc[task_id, 1]
        ground_truth = gt.iloc[task_id, 2]
        f_save.write(str(task_id) + '\t' + str(worker) + '\t' + str(task) + '\t' + str(ground_truth) + '\t' + str(label) + '\n')
f_save.close()

df = pd.read_csv("./datasets/test_data/domain_relation_test.txt", delimiter="\t", header=None)
df.to_csv("./datasets/test_data/domain_relation_test.csv", encoding='utf-8', index=False, header=["task_id", "annotator_id", "task", "ground_truth", "answer"])






# 获得sentiment_crowd_train.txt

df = pd.read_csv("./datasets/train_data/domain_relation_train.csv")
print(df)

gt = df.iloc[:, [0, 3]]
gt = gt.drop_duplicates(subset=["task_id"], keep='first', inplace=False)
gt.index = range(len(gt))
print(gt)
f_save = open('./datasets/train_data/domain_relation_truth_train.txt', 'w')
for i in range(len(gt)):
    f_save.write(str(gt.iloc[i, 0]) + '\t' + str(gt.iloc[i, 1]) + '\n')
f_save.close()

crowd = df.iloc[:, [0, 1, 4]]
f_save = open('./datasets/train_data/domain_relation_crowd_train.txt', 'w')
for i in range(len(crowd)):
    f_save.write(str(crowd.iloc[i, 0]) + '\t' + str(crowd.iloc[i, 1]) + '\t' + str(crowd.iloc[i, 2]) + '\n')
f_save.close()



# # 获得sentiment_crowd_validation.txt
#
# df = pd.read_csv("./datasets/validation_data/sentiment_validation.csv")
# print(df)
#
# gt = df.iloc[:, [0, 3]]
# gt = gt.drop_duplicates(subset=["task_id"], keep='first', inplace=False)
# gt.index = range(len(gt))
# print(gt)
# f_save = open('./datasets/validation_data/sentiment_truth_validation.txt', 'w')
# for i in range(len(gt)):
#     f_save.write(str(gt.iloc[i, 0]) + '\t' + str(gt.iloc[i, 1]) + '\n')
# f_save.close()
#
# crowd = df.iloc[:, [0, 1, 4]]
# f_save = open('./datasets/validation_data/sentiment_crowd_validation.txt', 'w')
# for i in range(len(crowd)):
#     f_save.write(str(crowd.iloc[i, 0]) + '\t' + str(crowd.iloc[i, 1]) + '\t' + str(crowd.iloc[i, 2]) + '\n')
# f_save.close()




# 获得sentiment_crowd_test.txt

df = pd.read_csv("./datasets/test_data/domain_relation_test.csv")
print(df)

gt = df.iloc[:, [0, 3]]
gt = gt.drop_duplicates(subset=["task_id"], keep='first', inplace=False)
gt.index = range(len(gt))
print(gt)
f_save = open('./datasets/test_data/domain_relation_truth_test.txt', 'w')
for i in range(len(gt)):
    f_save.write(str(gt.iloc[i, 0]) + '\t' + str(gt.iloc[i, 1]) + '\n')
f_save.close()

crowd = df.iloc[:, [0, 1, 4]]
f_save = open('./datasets/test_data/domain_relation_crowd_test.txt', 'w')
for i in range(len(crowd)):
    f_save.write(str(crowd.iloc[i, 0]) + '\t' + str(crowd.iloc[i, 1]) + '\t' + str(crowd.iloc[i, 2]) + '\n')
f_save.close()

