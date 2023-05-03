import random
from collections import Counter
from multiprocessing import cpu_count
from torch.autograd import Variable
import numpy as np
import torch
from torch import optim, nn
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from loss import T_VAE_Loss, A_VAE_Loss
from model import T_VAE, A_VAE
from utils import get_batch

seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def train_a_vae(args, device, train_loader, a_vae):

    a_vae = a_vae.to(device)
    A_loss = A_VAE_Loss()
    optimizer = optim.Adam(a_vae.parameters(), lr=args.learning_rate)


    torch.set_grad_enabled(True)
    a_vae.train()
    writer = SummaryWriter(args.annotator_writer)

    for epoch in range(args.epochs):

        for batch in tqdm(train_loader):

            annotator_id, answer, task, target, task_lengths = get_batch(batch)
            optimizer.zero_grad()
            # 工人
            annotator_id = np.array(annotator_id).astype(dtype=int)
            annotator_id = torch.from_numpy(annotator_id).to(device)
            annotator_inputs = F.one_hot(annotator_id, args.annotator_dim).type(torch.float32)  # 工人input
            # 工人vae
            a_output, a_mean, a_logv, a_z = a_vae(annotator_inputs)

            # 工人 loss
            a_KL_loss, a_recon_loss = A_loss(mu=a_mean, log_var=a_logv, recon_x=a_output, x=annotator_inputs)

            a_mloss = a_KL_loss + a_recon_loss

            a_mloss.backward()

            optimizer.step()


        writer.add_scalar(tag='a_KL_loss', scalar_value=a_KL_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='a_recon_loss', scalar_value=a_recon_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='a_mloss', scalar_value=a_mloss.data.item(), global_step=epoch)
        print(epoch, a_mloss.data.item())

    torch.save(a_vae, args.model_dir + args.annotator_vae_name)


def train_t_vae(args, device, train_loader, t_vae):

    t_vae = t_vae.to(device)
    T_loss = T_VAE_Loss()
    optimizer = optim.Adam(t_vae.parameters(), lr=args.learning_rate)

    torch.set_grad_enabled(True)
    t_vae.train()
    writer = SummaryWriter(args.task_writer)

    for epoch in range(args.epochs):
        states = t_vae.init_hidden(args.batch_size)
        for batch in tqdm(train_loader):

            annotator_id, answer, task, target, task_lengths = get_batch(batch)
            optimizer.zero_grad()

            # 任务
            task = task.to(device)
            target = target.to(device)
            # 任务vae
            t_output, t_mean, t_logv, t_z, states = t_vae(task, task_lengths, states)
            states = states[0].detach(), states[1].detach()

            # 任务loss
            t_KL_loss, t_recon_loss = T_loss(mu=t_mean, log_var=t_logv, x_hat_param=t_output, x=target)

            t_mloss = t_KL_loss + t_recon_loss

            t_mloss.backward()

            optimizer.step()

        writer.add_scalar(tag='t_KL_loss', scalar_value=t_KL_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='t_recon_loss', scalar_value=t_recon_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='t_mloss', scalar_value=t_mloss.data.item(), global_step=epoch)
        print(epoch, t_KL_loss.data.item(), t_recon_loss.data.item(), t_mloss.data.item())

    torch.save(t_vae, args.model_dir + args.task_vae_name)


def train_mymodel(args, device, train_loader, test_loader, mymodel):

    mymodel = mymodel.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(mymodel.parameters(), lr=0.001)
    writer = SummaryWriter(args.mymodel_writer)

    for epoch in range(30):
        states = mymodel.t_vae.init_hidden(args.batch_size)

        # train_TP, train_TN, train_FP, train_FN = 0, 0, 0, 0
        # validation_TP, validation_TN, validation_FP, validation_FN = 0, 0, 0, 0

        esp = 1e-6
        mymodel.train()
        count = 0
        num = 0
        for iteration, batch in enumerate(train_loader):

            optimizer.zero_grad()

            annotator_id, answer, task, target, task_lengths = get_batch(batch)
            # 工人
            annotator_id = np.array(annotator_id).astype(dtype=int)
            annotator_id = torch.from_numpy(annotator_id).to(device)
            annotator_inputs = F.one_hot(annotator_id, args.annotator_dim).type(torch.float32)  # 工人input
            # 工人vae
            a_output, a_mean, a_logv, a_z = mymodel.a_vae(annotator_inputs)
            # 任务
            task = task.to(device)
            # 任务vae
            t_output, t_mean, t_logv, t_z, states = mymodel.t_vae(task, task_lengths, states)
            states = states[0].detach(), states[1].detach()
            z = torch.cat((a_z, t_z.squeeze()), 1)
            dev_label = mymodel(z)
            label_tensor = torch.from_numpy(np.array(batch['answer']).astype(dtype=float)).type(torch.LongTensor).to(device)

            # 监督loss
            train_loss = loss_fn(dev_label, label_tensor)

            train_loss.backward()
            optimizer.step()

            prediction = torch.max(F.softmax(dev_label), 1)[1]
            pred_label = prediction.cpu().data.numpy().squeeze()
            target_label = label_tensor.cpu().data.numpy()

            for i in range(len(annotator_id)):
                if pred_label[i] == target_label[i]:
                    count += 1
            num += len(annotator_id)

        #     # 0为正类, 1为负类
        #     for i in range(len(annotator_id)):
        #         if pred_label[i] == 0 and target_label[i] == 0:
        #             train_TP += 1
        #         if pred_label[i] == 1 and target_label[i] == 1:
        #             train_TN += 1
        #         if pred_label[i] == 0 and target_label[i] == 1:
        #             train_FP += 1
        #         if pred_label[i] == 1 and target_label[i] == 0:
        #             train_FN += 1
        #
        # TRAIN_ACC = (train_TP+train_TN)/(train_TP+train_TN+train_FP+train_FN+esp)
        # TRAIN_PRE = train_TP/(train_TP+train_FP+esp)
        # TRAIN_REC = train_TP/(train_TP+train_FN+esp)

        ACC = count/num

        writer.add_scalar(tag='train_loss', scalar_value=train_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='train_acc', scalar_value=ACC, global_step=epoch)
        # writer.add_scalar(tag='train_pre', scalar_value=TRAIN_PRE, global_step=epoch)
        # writer.add_scalar(tag='train_rec', scalar_value=TRAIN_REC, global_step=epoch)
        print("--------------------------TRAIN----------------------------------")
        print("epoch", epoch, "train_loss", train_loss.data.item())
        print("train_acc:", ACC)
        # print("train_acc:", TRAIN_ACC, "train_pre:", TRAIN_PRE, "train_rec：", TRAIN_REC)


        count = 0
        num = 0
        mymodel.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(test_loader):

                annotator_id, answer, task, target, task_lengths = get_batch(batch)
                # 工人
                annotator_id = np.array(annotator_id).astype(dtype=int)
                annotator_id = torch.from_numpy(annotator_id).to(device)
                annotator_inputs = F.one_hot(annotator_id, args.annotator_dim).type(torch.float32)  # 工人input
                # 工人vae
                a_output, a_mean, a_logv, a_z = mymodel.a_vae(annotator_inputs)
                # 任务
                task = task.to(device)
                # 任务vae
                t_output, t_mean, t_logv, t_z, states = mymodel.t_vae(task, task_lengths, states)
                states = states[0].detach(), states[1].detach()
                z = torch.cat((a_z, t_z.squeeze()), 1)
                dev_label = mymodel(z)
                label_tensor = torch.from_numpy(np.array(batch['answer']).astype(dtype=float)).type(
                    torch.LongTensor).to(device)

                # 监督loss
                validation_loss = loss_fn(dev_label, label_tensor)

                prediction = torch.max(F.softmax(dev_label), 1)[1]
                pred_label = prediction.cpu().data.numpy().squeeze()
                target_label = label_tensor.cpu().data.numpy()

                for i in range(len(annotator_id)):
                    if pred_label[i] == target_label[i]:
                        count += 1
                num += len(annotator_id)

        #         # 0为正类, 1为负类
        #         for i in range(len(annotator_id)):
        #             if pred_label[i] == 0 and target_label[i] == 0:
        #                 validation_TP += 1
        #             if pred_label[i] == 1 and target_label[i] == 1:
        #                 validation_TN += 1
        #             if pred_label[i] == 0 and target_label[i] == 1:
        #                 validation_FP += 1
        #             if pred_label[i] == 1 and target_label[i] == 0:
        #                 validation_FN += 1
        #
        #
        #
        # VALIDATION_ACC = (validation_TP+validation_TN)/(validation_TP+validation_TN+validation_FP+validation_FN+esp)
        # VALIDATION_PRE = validation_TP/(validation_TP+validation_FP+esp)
        # VALIDATION_REC = validation_TP/(validation_TP+validation_FN+esp)

        ACC = count/num

        writer.add_scalar(tag='validation_loss', scalar_value=validation_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='validation_acc', scalar_value=ACC, global_step=epoch)
        # writer.add_scalar(tag='validation_pre', scalar_value=VALIDATION_PRE, global_step=epoch)
        # writer.add_scalar(tag='validation_rec', scalar_value=VALIDATION_REC, global_step=epoch)

        print("---------------------------VALIDATION---------------------------------")
        print(epoch, validation_loss.data.item())
        print("validation_acc:", ACC)
        # print("validation_acc:", VALIDATION_ACC, "validation_pre:", VALIDATION_PRE, "validation_rec：", VALIDATION_REC)






    torch.save(mymodel, args.model_dir + args.mymodel_name)



