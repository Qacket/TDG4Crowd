c




from collections import Counter


def statistical(crowd_file, truth_file):
    f = open(truth_file, 'r')
    reader = f.readlines()
    reader = [line.strip("\n") for line in reader]
    e2truth = {}
    truth_list = []
    for line in reader:
        example, truth = line.split('\t')
        e2truth[example] = truth
        truth_list.append(truth)

    result = Counter(truth_list)
    result = sorted(result.items(), key=lambda item: item[0])

    f.close()
    total_example_sum = len(e2truth)


    item = 0
    f = open(crowd_file, 'r')
    reader = f.readlines()
    reader = [line.strip("\n") for line in reader]
    workers = []
    e2wl = {}
    for line in reader:
        example, worker, label = line.split('\t')
        item += 1
        if worker not in workers:
            workers.append(worker)

        if example not in e2wl:
            e2wl[example] = []
        e2wl[example].append([worker, label])

    workers_sum = len(workers)
    e2ws = {}  # 任务：工人人数
    for example, w2l in e2wl.items():
        e2ws[example] = len(w2l)
    List = list(e2ws.values())
    redundant = {}
    for i in List:
        if List.count(i) > 1:
            redundant[i] = List.count(i)
    a = 0
    b = 0
    for red, example_sum in redundant.items():
        a += red * example_sum
        b += example_sum
    ave_redun = a / b
    redundant = sorted(redundant.items(), key=lambda x: x[0])
    f.close()

    return result, total_example_sum, workers_sum, item, redundant, ave_redun


if __name__ == '__main__':
     # real data
    total_data_crowd_file = './datasets/total_data/domain_relation_crowd.txt'
    total_data_truth_file = './datasets/total_data/domain_relation_truth.txt'
    label_count, total_example_sum, workers_sum, item, redundant, ave_redun = statistical(total_data_crowd_file, total_data_truth_file)
    print('real_data', 'label num:', label_count, 'total_example_sum', total_example_sum, 'workers_sum:', workers_sum, 'item_sum:', item,
          'redundant:', redundant, 'average_redundancy:', ave_redun)

    # train_data
    train_data_crowd_file = './datasets/train_data/domain_relation_crowd_train.txt'
    train_data_truth_file = './datasets/train_data/domain_relation_truth_train.txt'
    label_count, total_example_sum, workers_sum, item, redundant, ave_redun = statistical(train_data_crowd_file, train_data_truth_file)
    print('train_data', 'label num:', label_count, 'total_example_sum', total_example_sum, 'workers_sum:', workers_sum, 'item_sum:', item,
          'redundant:', redundant, 'average_redundancy:', ave_redun)

    # # validation_data
    # validation_data_crowd_file = './datasets/validation_data/sentiment_crowd_validation.txt'
    # validation_data_truth_file = './datasets/validation_data/sentiment_truth_validation.txt'
    # pos, neg, total_example_sum, workers_sum, item, redundant, ave_redun = statistical(validation_data_crowd_file,
    #                                                                                     validation_data_truth_file)
    # print('validation_data', 'pos:', pos, 'neg:', neg, 'total_example_sum', total_example_sum, 'workers_sum:', workers_sum,
    #        'item_sum:', item,
    #        'redundant:', redundant, 'average_redundancy:', ave_redun)

    # test_data
    test_data_crowd_file = './datasets/test_data/domain_relation_crowd_test.txt'
    test_data_truth_file = './datasets/test_data/domain_relation_truth_test.txt'
    label_count, total_example_sum, workers_sum, item, redundant, ave_redun = statistical(test_data_crowd_file, test_data_truth_file)
    print('test_data','label num:', label_count, 'total_example_sum', total_example_sum, 'workers_sum:', workers_sum, 'item_sum:', item,
          'redundant:', redundant, 'average_redundancy:', ave_redun)
