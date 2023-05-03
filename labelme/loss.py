import torch


class VAE_Loss(torch.nn.Module):
    def __init__(self):
        super(VAE_Loss, self).__init__()
        self.mseloss = torch.nn.MSELoss()

    def KL_loss(self, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    def forward(self, mu, log_var, recon_x, x):

        KLD = self.KL_loss(mu, log_var)
        BCE = self.mseloss(recon_x, x)
        return KLD, BCE
