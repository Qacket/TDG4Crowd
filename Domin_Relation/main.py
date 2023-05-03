import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import My_Model, A_VAE, T_VAE
from relation import Relation
from train import train_mymodel, train_a_vae, train_t_vae


def main(args):


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # # 构建完整的任务vab
    # total_data = Relation(
    #     data_dir='./datasets',
    #     data_path='/total_data/total_domain_relation.csv',
    #     data_file='/total_data/domain_relation_task.json',
    #     create_vocab=True
    # )
    # total_task_vab = total_data.vocab_size


    # Load the train data
    train_data = Relation(
        data_dir=args.data_dir,
        data_path=args.train_data_path,
        data_file=args.train_data_file
    )
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True
    )

    # Load the test data
    test_data = Relation(
        data_dir=args.data_dir,
        data_path='/test_data/domain_relation_test.csv',
        data_file='/test_data/domain_relation_task_test.json'
    )
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=64,
        shuffle=True
    )


    # a_params = dict(
    #     E_in=args.annotator_dim,
    #     middle_size=20,
    #     hidden_size=20,
    #     latent_size=8,
    #     D_out=args.annotator_dim,
    #     device=device
    # )
    # # 初始化工人vae
    # a_vae = A_VAE(**a_params)
    # # 训练工人vae
    # train_a_vae(args, device, train_loader, a_vae)
    #
    #
    # t_params = dict(
    #     train_data=train_data,
    #     vocab_size=total_task_vab,
    #     embedding_size=args.embedding_size,
    #     hidden_size=args.hidden_size,
    #     latent_size=args.latent_size,
    #     num_layers=args.num_layers,
    #     embedding_dropout=args.embedding_dropout,
    #     device=device
    # )
    #
    # # 初始化任务vae
    # t_vae = T_VAE(**t_params)
    # # 训练任务vae
    # train_t_vae(args, device, train_loader, t_vae)




    a_vae = torch.load(args.model_dir + args.annotator_vae_name)
    t_vae = torch.load(args.model_dir + args.task_vae_name)
    a_vae.trainable = False
    t_vae.trainable = False

    mymodel = My_Model(a_vae, t_vae)
cc


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./datasets')
    parser.add_argument('--train_data_path', type=str, default='/train_data/domain_relation_train.csv')
    parser.add_argument('--train_data_file', type=str, default='/train_data/domain_relation_task_train.json')


    parser.add_argument('--model_dir', type=str, default='./model')

    parser.add_argument('-ep', '--epochs', type=int, default=500)
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-a_dim', '--annotator_dim', type=int, default=154)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-hs', '--hidden_size', type=int, default=128)
    parser.add_argument('-ls', '--latent_size', type=int, default=50)

    parser.add_argument('-taw', '--annotator_writer', type=str, default='logs/a_vae')
    parser.add_argument('-avn', '--annotator_vae_name', type=str, default='/a_vae')

    parser.add_argument('-ttw', '--task_writer', type=str, default='logs/t_vae')
    parser.add_argument('-tvn', '--task_vae_name', type=str, default='/t_vae')

    parser.add_argument('-mw', '--mymodel_writer', type=str, default='logs/mymodel2')
    parser.add_argument('-mn', '--mymodel_name', type=str, default='/mymodel2')

    args = parser.parse_args()

    main(args)

