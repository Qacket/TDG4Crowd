from torch import optim, nn
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from loss import VAE_Loss
def train_a_vae(args, device, train_loader, a_vae):

    a_vae = a_vae.to(device)
    vae_loss = VAE_Loss()
    optimizer = optim.Adam(a_vae.parameters(), lr=args.learning_rate)


    torch.set_grad_enabled(True)
    a_vae.train()
    writer = SummaryWriter(args.annotator_writer)

    for epoch in range(args.epochs):

        for iteration, data in enumerate(train_loader):

            task_id = data[1].to(device)
            annotator_inputs = data[2].to(device)
            label_tensor = data[3].to(device)
            task_inputs = data[4].to(device)

            optimizer.zero_grad()

            a_output, a_mean, a_logv, a_z = a_vae(annotator_inputs)  # 获得 工人能力z1  生成的工人^ dev_annotator

            # 工人 loss
            a_KL_loss, a_recon_loss = vae_loss(mu=a_mean, log_var=a_logv, recon_x=a_output, x=annotator_inputs)

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
    vae_loss = VAE_Loss()
    optimizer = optim.Adam(t_vae.parameters(), lr=args.learning_rate)

    torch.set_grad_enabled(True)
    t_vae.train()
    writer = SummaryWriter(args.task_writer)

    for epoch in range(args.epochs):

        for iteration, data in enumerate(train_loader):

            task_id = data[1].to(device)
            annotator_inputs = data[2].to(device)
            label_tensor = data[3].to(device)
            task_inputs = data[4].to(device)

            optimizer.zero_grad()

            t_output, t_mean, t_logv, t_z = t_vae(task_inputs)  # 获得 任务能力z2  生成的任务^ dev_task

            t_KL_loss, t_recon_loss = vae_loss(mu=t_mean, log_var=t_logv, recon_x=t_output, x=task_inputs)

            t_mloss = t_KL_loss + t_recon_loss

            t_mloss.backward()

            optimizer.step()

        writer.add_scalar(tag='t_KL_loss', scalar_value=t_KL_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='t_recon_loss', scalar_value=t_recon_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='t_mloss', scalar_value=t_mloss.data.item(), global_step=epoch)
        print(epoch, t_KL_loss.data.item(), t_recon_loss.data.item(), t_mloss.data.item())

    torch.save(t_vae, args.model_dir + args.task_vae_name)


def train_mymodel(args, device, train_loader, mymodel):

    mymodel = mymodel.to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(mymodel.parameters(), lr=args.learning_rate)

    torch.set_grad_enabled(True)
    mymodel.train()
    writer = SummaryWriter(args.mymodel_writer)

    for epoch in range(args.epochs):
        count = 0
        sum = 0
        for iteration, data in enumerate(train_loader):
            task_id = data[1].to(device)
            annotator_inputs = data[2].to(device)
            label_tensor = data[3].to(device)
            task_inputs = data[4].to(device)

            optimizer.zero_grad()

            a_output, a_mean, a_logv, a_z = mymodel.a_vae(annotator_inputs)  # 获得 工人能力z1  生成的工人^ dev_annotator
            t_output, t_mean, t_logv, t_z = mymodel.t_vae(task_inputs)  # 获得 任务能力z2  生成的任务^ dev_task
            z = torch.cat((a_z, t_z), 1)  # z1 z2 结合
            dev_label = mymodel(z)  # 获得生成的标注^dev_label

            # a_KL_loss, a_recon_loss = vae_loss(mu=a_mean, log_var=a_logv, recon_x=a_output, x=annotator_inputs)

            # t_KL_loss, t_recon_loss = vae_loss(mu=t_mean, log_var=t_logv, recon_x=t_output, x=task_inputs)


            sup_loss = loss_fn(dev_label, label_tensor)

            # loss = t_KL_loss + t_recon_loss + sup_loss
            loss = sup_loss

            loss.backward()
            optimizer.step()

            prediction = torch.max(F.softmax(dev_label), 1)[1]
            pred_label = prediction.cpu().data.numpy().squeeze()
            target_label = label_tensor.cpu().data.numpy()
            for i in range(len(task_id)):
                if pred_label[i] == target_label[i]:
                    count += 1
            sum += len(task_id)
        acc = count / sum
        writer.add_scalar(tag='sup_loss', scalar_value=sup_loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='loss', scalar_value=loss.data.item(), global_step=epoch)
        writer.add_scalar(tag='acc', scalar_value=acc, global_step=epoch)
        print(epoch, sup_loss.data.item(), loss.data.item(), "ACC:" + str(acc))

    torch.save(mymodel, args.model_dir + args.mymodel_name)

