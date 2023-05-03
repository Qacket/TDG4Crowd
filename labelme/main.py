import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from lab import LAB
from model import VAE, My_Model
from train import train_mymodel, train_a_vae, train_t_vae

seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def main(args):

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    train_data = LAB(
        data_dir=args.data_dir,
        crowd_dir=args.crowd_dir
    )
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
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
    # a_vae = VAE(**a_params)
    # # 训练工人vae
    # train_a_vae(args, device, train_loader, a_vae)

    # t_params = dict(
    #     E_in=args.task_dim,
    #     middle_size=100,
    #     hidden_size=100,
    #     latent_size=50,
    #     D_out=args.task_dim,
    #     device=device
    # )
    # # 初始化任务vae
    # t_vae = VAE(**t_params)
    # # 训练任务vae
    # train_t_vae(args, device, train_loader, t_vae)

    a_vae = torch.load(args.model_dir + args.annotator_vae_name)
    t_vae = torch.load(args.model_dir + args.task_vae_name)

    a_vae.trainable = False
    t_vae.trainable = False

    mymodel = My_Model(a_vae, t_vae)
    train_mymodel(args, device, train_loader, mymodel)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./datasets')
    parser.add_argument('--model_dir', type=str, default='./model')
    parser.add_argument('--crowd_dir', type=str, default='/train_data/labelme_crowd_train.txt')
    parser.add_argument('--truth_dir', type=str, default='/train_data/labelme_truth_train.txt')

    parser.add_argument('-ep', '--epochs', type=int, default=1000)
    parser.add_argument('-bs', '--batch_size', type=int, default=512)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)


    parser.add_argument('-a_dim', '--annotator_dim', type=int, default=59)
    parser.add_argument('-t_dim', '--task_dim', type=int, default=8192)

    parser.add_argument('-taw', '--annotator_writer', type=str, default='logs/a_vae')
    parser.add_argument('-avn', '--annotator_vae_name', type=str, default='/a_vae')

    parser.add_argument('-ttw', '--task_writer', type=str, default='logs/t_vae')
    parser.add_argument('-tvn', '--task_vae_name', type=str, default='/t_vae')


    parser.add_argument('-mw', '--mymodel_writer', type=str, default='logs/mymodel')
    parser.add_argument('-mn', '--mymodel_name', type=str, default='/mymodel')

    args = parser.parse_args()

    main(args)