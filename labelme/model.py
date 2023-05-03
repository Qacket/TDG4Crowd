import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class VAE(torch.nn.Module):

    def __init__(self, E_in, middle_size, hidden_size, latent_size, D_out, device):
        super(VAE, self).__init__()
        self.E_in = E_in
        self.D_out = D_out
        self.middle_size = middle_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.device = device

        self.E_linear1 = torch.nn.Linear(self.E_in, self.middle_size)
        self.E_linear2 = torch.nn.Linear(self.middle_size, self.hidden_size)

        self.hidden2mean = torch.nn.Linear(self.hidden_size, self.latent_size)
        self.hidden2logv = torch.nn.Linear(self.hidden_size, self.latent_size)

        self.latent2hidden = torch.nn.Linear(self.latent_size, self.hidden_size)
        self.D_linear = torch.nn.Linear(self.hidden_size, self.D_out)


    def forward(self, input):

        batch_size = input.size(0)
        # ENCODER
        e_middle = torch.nn.functional.relu(self.E_linear1(input))
        hidden = torch.nn.functional.relu(self.E_linear2(e_middle))

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)
        z = torch.randn([batch_size, self.latent_size]).to(self.device)
        z = z * std + mean

        # DECODER
        hidden = self.latent2hidden(z)
        ouput = torch.nn.functional.relu(self.D_linear(hidden))

        return ouput, mean, logv, z



class My_Model(torch.nn.Module):
    def __init__(self, a_vae, t_vae):
        super(My_Model, self).__init__()
        self.a_vae = a_vae
        self.t_vae = t_vae
        self.hidden = torch.nn.Linear(58, 16)
        self.output = torch.nn.Linear(16, 8)

    def forward(self, x):
        x = torch.nn.functional.relu(self.hidden(x))
        x = torch.nn.functional.relu(self.output(x))
        return x