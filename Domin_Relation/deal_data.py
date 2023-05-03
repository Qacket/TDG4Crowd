import pandas as pd


def deal_data():



    # task = pd.read_csv('./datasets/original_data/aggregated_sentences.csv')
    # task = task.iloc[:1601:, :9]
    # task.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
    # task = task.drop(columns=['1', '2', '3', '4', '5', '7'])
    # idx_task_id_list = list(task.iloc[:, 0])
    # idx2taskid = {}
    #
    # for i in range(len(idx_task_id_list)):
    #     idx2taskid[idx_task_id_list[i]] = i
    #
    # task.columns = ['task_id', 'ground_truth', 'task']
    # task = task.set_index('task_id').to_dict("index")
    #
    #
    # data = pd.read_csv('./datasets/original_data/worker_judgments.csv')
    # data = data.iloc[:24032, 6:]
    # sums = (data == 1).astype(int).sum(axis=1)
    # sums_result = list(sums[sums > 1].index)
    # data = data.drop(sums_result, axis=0)
    # data = data.drop(columns=['none', 'per:cause_of_death', 'per:charges', 'per:origin'])
    #
    # sums = (data == 1).astype(int).sum(axis=1)
    # sums_result = list(sums[sums == 0].index)
    # data = data.drop(sums_result, axis=0)
    # data.reset_index(drop=True)
    # label = []
    # for i in range(len(data)):
    #     for key, value in dict(data.iloc[i, 2:]).items():
    #         if int(value) == 1:
    #             label.append(key)
    # data['answer'] = label
    #
    # data = data[['unit', 'worker', 'answer']]
    # data.columns = ['task_id', 'annotator_id', 'answer']
    # print(data)
    #
    # gt_list = []
    # tk_list = []
    # for i in range(len(data)):
    #     task_id = data.iloc[i, 0]
    #     ground_truth = task[task_id]['ground_truth']
    #     gt_list.append(ground_truth)
    #     task_text = task[task_id]['task']
    #     tk_list.append(task_text)
    # data['ground_truth'] = gt_list
    # data['task'] = tk_list
    #
    # print(data)
    # data.to_csv('./datasets/total_data/deal_domain_relation.csv', index=False, header=True)






    data = pd.read_csv('./datasets/total_data/deal_domain_relation.csv')
    print(data)

    task2idx = {}
    task_idx = 0
    annotator2idx = {}
    annotator_idx = 0
    label2idx = {}
    label_idx = 0
    for i in range(len(data)):
        if data.iloc[i, 0] not in task2idx:
            task2idx[data.iloc[i, 0]] = task_idx
            task_idx += 1
        if data.iloc[i, 1] not in annotator2idx:
            annotator2idx[data.iloc[i, 1]] = annotator_idx
            annotator_idx += 1
        if data.iloc[i, 2] not in label2idx:
            label2idx[data.iloc[i, 2]] = label_idx
            label_idx += 1
        if data.iloc[i, 3] not in label2idx:
            label2idx[data.iloc[i, 3]] = label_idx
            label_idx += 1

        data.iloc[i, 0] = cc[data.iloc[i, 0]]
        data.iloc[i, 1] = annotator2idx[data.iloc[i, 1]]
        data.iloc[i, 2] = label2idx[data.iloc[i, 2]]
        data.iloc[i, 3] = label2idx[data.iloc[i, 3]]



    data[['answer', 'task']] = data[['task', 'answer']]
    data.columns = ['task_id', 'annotator_id', 'task', 'ground_truth', 'answer']

    print(data)
    data.to_csv('./datasets/total_data/total_domain_relation.csv', index=False, header=True)





    crowd = data.drop(columns=['task', 'ground_truth'], inplace=False)
    crowd.to_csv('./datasets/total_data/domain_relation_crowd.txt', sep='\t', index=False, header=False)

    ground_truth = data.drop_duplicates(subset=['task_id'], keep='first', inplace=False). \
        drop(columns=['annotator_id', 'task', 'answer'], inplace=False)

    ground_truth.to_csv('./datasets/total_data/domain_relation_truth.txt', sep='\t', index=False, header=False)



deal_data()