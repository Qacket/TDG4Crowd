import random

import torch
from scipy import stats
import pandas as pd
import scipy.stats
import numpy as np
from multiprocessing import Pool

from converge.converge_data import converging
from generate.generate_data import generating_fixed_task, generating, generating_replenish
from sample.sample_data import sampling

from relation import Relation
from utils import pre_deal, cal_acc, calculate_kl_ks, all_list

if __name__ == '__main__':
    main_dir = r'./datasets'
    crowd_file = main_dir + '/test_data/domain_relation_crowd_test.txt'
    truth_file = main_dir + '/test_data/domain_relation_truth_test.txt'
    generate_methods = ['DS_generate', 'HDS_generate', 'MV_generate', 'GLAD_generate', 'RDG_generate', 'Mymodel_generate']
    converge_methods = ['DS', 'HDS', 'MV', 'GLAD']
    data_scale_range = range(300, 1350, 150)
    max_sampling_time = 5
    max_generate_sum = 5
    pool = Pool()

    # # Load the data
    # test_data = Relation(data_dir="./datasets",
    #                  data_path="/test_data/domain_relation_test.csv",
    #                  data_file="/test_data/domain_relation_task_test.json")
    # # Batchify the data
    # test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    #
    #
    # for number in data_scale_range:  # 采样
    #     for i in range(max_sampling_time):
    #         exist_task = sampling(crowd_file, number).run_fixed_task()
    #         print(exist_task)
    #         for generate_method_name in generate_methods:  # 一个样本文件在 每个生成方法下生成数据
    #             for j in range(max_generate_sum):
    #                 generate_file = main_dir + '/domain_relation_4/' + str(number) + '/' + generate_method_name + '/' + str(i) + '_' + str(j) + '.txt'
    #                 if generate_method_name == 'Mymodel_generate':
    #                     generating_replenish(crowd_file, truth_file, exist_task, generate_method_name, generate_file, test_loader)  # 生成20个数据集
    #                 else:
    #                     pool.apply_async(generating_replenish,
    #                                      args=(crowd_file, truth_file, exist_task, generate_method_name, generate_file, test_loader))
    #
    # pool.close()
    # pool.join()




    # # KL and KS
    # original_data_list = pre_deal(crowd_file, truth_file)
    #
    #
    # for z in range(13):
    #     print(z, "-------------------------------------------------------------------------------------")
    #     mean_KL_list = []
    #     mean_KL_path = main_dir + '/domain_relation_4/KL/KL_mean_' + str(z) + '.csv'
    #     std_KL_list = []
    #     std_KL_path = main_dir + '/domain_relation_4/KL/KL_std_' + str(z) + '.csv'
    #
    #     mean_KS_stat_list = []
    #     mean_KS_stat_path = main_dir + '/domain_relation_4/KS/KS_stat_mean_' + str(z) + '.csv'
    #     std_KS_stat_list = []
    #     std_KS_stat_path = main_dir + '/domain_relation_4/KS/KS_stat_std_' + str(z) + '.csv'
    #
    #     mean_KS_pval_list = []
    #     mean_KS_pval_path = main_dir + '/domain_relation_4/KS/KS_pavl_mean_' + str(z) + '.csv'
    #     std_KS_pval_list = []
    #     std_KS_pval_path = main_dir + '/domain_relation_4/KS/KS_pavl_std_' + str(z) + '.csv'
    #
    #     for number in data_scale_range:
    #
    #         for generate_method_name in generate_methods:  # 生成方法 5
    #
    #             KL_list = []
    #             KL_path = main_dir + '/domain_relation_4/' + str(number) + '/' + generate_method_name + '/KL_total_' + str(z) + '.csv'
    #
    #             KS_stat_list = []
    #             KS_stat_path = main_dir + '/domain_relation_4/' + str(number) + '/' + generate_method_name + '/KS_stat_total_' + str(z) + '.csv'
    #             KS_pval_list = []
    #             KS_pval_path = main_dir + '/domain_relation_4/' + str(number) + '/' + generate_method_name + '/KS_pval_total_' + str(z) + '.csv'
    #             for i in range(max_sampling_time):
    #                 for j in range(max_generate_sum):
    #                     generate_file = main_dir + '/domain_relation_4/' + str(number) + '/' + generate_method_name + '/' + str(i) + '_' + str(j) + '.txt'
    #                     generate_data_list = pre_deal(generate_file, truth_file)
    #
    #                     kl, ks_stat, ks_pval = calculate_kl_ks(original_data_list[z], generate_data_list[z])
    #
    #                     KL_list.append(kl)
    #                     KS_stat_list.append(ks_stat)
    #                     KS_pval_list.append(ks_pval)
    #
    #             KL_list = np.array(KL_list).reshape(-1, 1)
    #             KL_pd = pd.DataFrame(data=KL_list)
    #             KL_pd.to_csv(KL_path, index=False)
    #
    #             KS_stat_list = np.array(KS_stat_list).reshape(-1, 1)
    #             KS_stat_pd = pd.DataFrame(data=KS_stat_list)
    #             KS_stat_pd.to_csv(KS_stat_path, index=False)
    #
    #             KS_pval_list = np.array(KS_pval_list).reshape(-1, 1)
    #             KS_pval_pd = pd.DataFrame(data=KS_pval_list)
    #             KS_pval_pd.to_csv(KS_pval_path, index=False)
    #
    #             mean_KL = np.mean(KL_list)
    #             std_KL = np.std(KL_list, ddof=1)
    #
    #             mean_KL_list.append(mean_KL)
    #             std_KL_list.append(std_KL)
    #
    #             mean_KS_stat = np.mean(KS_stat_list)
    #             std_KS_stat = np.std(KS_stat_list, ddof=1)
    #
    #             mean_KS_stat_list.append(mean_KS_stat)
    #             std_KS_stat_list.append(std_KS_stat)
    #
    #             mean_KS_pval = np.mean(KS_pval_list)
    #             std_KS_pval = np.std(KS_pval_list, ddof=1)
    #
    #             mean_KS_pval_list.append(mean_KS_pval)
    #             std_KS_pval_list.append(std_KS_pval)
    #
    #     mean_KL_np = np.array(mean_KL_list).reshape(-1, 6)
    #     mean_KL_pd = pd.DataFrame(data=mean_KL_np)
    #     mean_KL_pd.to_csv(mean_KL_path, index=False)
    #
    #     std_KL_np = np.array(std_KL_list).reshape(-1, 6)
    #     std_KL_pd = pd.DataFrame(data=std_KL_np)
    #     std_KL_pd.to_csv(std_KL_path, index=False)
    #
    #     mean_KS_stat_np = np.array(mean_KS_stat_list).reshape(-1, 6)
    #     mean_KS_stat_pd = pd.DataFrame(data=mean_KS_stat_np)
    #     mean_KS_stat_pd.to_csv(mean_KS_stat_path, index=False)
    #
    #     std_KS_stat_np = np.array(std_KS_stat_list).reshape(-1, 6)
    #     std_KS_stat_pd = pd.DataFrame(data=std_KS_stat_np)
    #     std_KS_stat_pd.to_csv(std_KS_stat_path, index=False)
    #
    #     mean_KS_pval_np = np.array(mean_KS_pval_list).reshape(-1, 6)
    #     mean_KS_pval_pd = pd.DataFrame(data=mean_KS_pval_np)
    #     mean_KS_pval_pd.to_csv(mean_KS_pval_path, index=False)
    #
    #     std_KS_pval_np = np.array(std_KS_pval_list).reshape(-1, 6)
    #     std_KS_pval_pd = pd.DataFrame(data=std_KS_pval_np)
    #     std_KS_pval_pd.to_csv(std_KS_pval_path, index=False)






    # # 汇聚
    # for number in data_scale_range:
    #     # 生成数据汇聚
    #     for generate_method_name in generate_methods:   # 生成方法 5
    #         acc_path = main_dir + '/domain_relation_4/' + str(number) + '/' + generate_method_name + '/accuracy.csv'
    #         acc_applyResult_list = []
    #         for i in range(max_sampling_time):
    #             for j in range(max_generate_sum):
    #                 for converge_method_name in converge_methods:  # 汇聚方法 4
    #                     generate_file = main_dir + '/domain_relation_4/' + str(number) + '/' + generate_method_name + '/' + str(i) + '_' + str(j) + '.txt'
    #                     acc = pool.apply_async(converging, args=(generate_file, truth_file, converge_method_name))
    #                     acc_applyResult_list.append(acc)
    #         acc_list = []
    #         for acc_applyResult in acc_applyResult_list:
    #             acc_list.append(acc_applyResult.get())
    #
    #         acc_list = np.array(acc_list).reshape(-1, 4)
    #         acc_pd = pd.DataFrame(data=acc_list)
    #         acc_pd.to_csv(acc_path, index=False)
    #
    # pool.close()
    # pool.join()




    # 汇聚排序
    original_acc_list = []
    for converge_method_name in converge_methods:
        acc = converging(crowd_file, truth_file, converge_method_name)
        original_acc_list.append(acc)
    original_acc_list = np.array(original_acc_list).reshape(-1, 4)
    print(original_acc_list)
    original_acc_mean_list = np.mean(original_acc_list, axis=0).tolist()
    original_acc_std_list = np.std(original_acc_list, axis=0, ddof=1).tolist()
    print(original_acc_mean_list)
    for number in data_scale_range:  # 汇聚排序

        mean_total_path = main_dir + '/domain_relation_4/' + str(number) + '/mean.csv'
        mean_total_list = []

        std_total_path = main_dir + '/domain_relation_4/' + str(number) + '/std.csv'
        std_total_list = []

        mean_total_list = mean_total_list + original_acc_mean_list
        std_total_list = std_total_list + original_acc_std_list

        for generate_method_name in generate_methods:  # 生成方法 5
            generate_acc_path = main_dir + '/domain_relation_4/' + str(
                number) + '/' + generate_method_name + '/accuracy.csv'
            generate_acc_total = pd.read_csv(generate_acc_path)
            generate_acc_mean = generate_acc_total.mean(axis=0)
            generate_acc_std = generate_acc_total.std(axis=0)

            generate_acc_mean_list = list(generate_acc_mean)
            generate_acc_std_list = list(generate_acc_std)

            mean_total_list = mean_total_list + generate_acc_mean_list
            std_total_list = std_total_list + generate_acc_std_list

        mean_total_list = np.array(mean_total_list).reshape(-1, 4)
        mean_total_pd = pd.DataFrame(data=mean_total_list)
        mean_total_pd.to_csv(mean_total_path, index=False)

        std_total_list = np.array(std_total_list).reshape(-1, 4)
        std_total_pd = pd.DataFrame(data=std_total_list)
        std_total_pd.to_csv(std_total_path, index=False)
