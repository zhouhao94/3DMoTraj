import os
import math
import random
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils import data
from torch.autograd import Variable

import datetime, shutil, argparse, logging, sys

import utils
import time
from trajectory_augmenter import TrajectoryAugmenter

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from thop import profile


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--gpu_deterministic', type=bool, default=False, help='set cudnn in deterministic mode (slow)')
    parser.add_argument("--data_scale", default=60, type=float)
    parser.add_argument("--dec_size", default=[1024, 512, 1024], type=list)
    parser.add_argument("--enc_dest_size", default=[256, 128], type=list)
    parser.add_argument("--enc_latent_size", default=[256, 512], type=list)
    parser.add_argument("--enc_past_size", default=[512, 256], type=list)
    parser.add_argument("--predictor_hidden_size", default=[1024, 512, 256], type=list)
    parser.add_argument("--non_local_theta_size", default=[256, 128, 64], type=list)
    parser.add_argument("--non_local_phi_size", default=[256, 128, 64], type=list)
    parser.add_argument("--non_local_g_size", default=[256, 128, 64], type=list)
    parser.add_argument("--non_local_dim", default=128, type=int)
    parser.add_argument("--fdim", default=16, type=int)
    parser.add_argument("--future_length", default=12, type=int)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--kld_coeff", default=0.5, type=float)
    parser.add_argument("--future_loss_coeff", default=1, type=float)
    parser.add_argument("--dest_loss_coeff", default=2, type=float)
    parser.add_argument("--learning_rate", default=0.0005, type=float)
    parser.add_argument("--lr_decay_step_size", default=4, type=int)
    parser.add_argument("--lr_decay_gamma", default=0.5, type=float)
    parser.add_argument("--mu", default=0, type=float)
    parser.add_argument("--n_values", default=20, type=int)
    parser.add_argument("--nonlocal_pools", default=3, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--past_length", default=8, type=int)
    parser.add_argument("--sigma", default=1.3, type=float)
    parser.add_argument("--zdim", default=16, type=int)
    parser.add_argument("--print_log", default=6, type=int)
    parser.add_argument("--sub_goal_indexes", default=[2, 5, 8, 11], type=list)


    parser.add_argument('--e_prior_sig', type=float, default=2, help='prior of ebm z')
    parser.add_argument('--e_init_sig', type=float, default=2, help='sigma of initial distribution')
    parser.add_argument('--e_activation', type=str, default='lrelu', choices=['gelu', 'lrelu', 'swish', 'mish'])
    parser.add_argument('--e_activation_leak', type=float, default=0.2)
    parser.add_argument('--e_energy_form', default='identity', choices=['identity', 'tanh', 'sigmoid', 'softplus'])
    parser.add_argument('--e_l_steps', type=int, default=20, help='number of langevin steps')
    parser.add_argument('--e_l_steps_pcd', type=int, default=20, help='number of langevin steps')
    parser.add_argument('--e_l_step_size', type=float, default=0.4, help='stepsize of langevin')
    parser.add_argument('--e_l_with_noise', default=True, type=bool, help='noise term of langevin')
    parser.add_argument('--e_sn', default=False, type=bool, help='spectral regularization')
    parser.add_argument('--e_lr', default=0.00003, type=float)
    parser.add_argument('--e_is_grad_clamp', type=bool, default=False, help='whether doing the gradient clamp')
    parser.add_argument('--e_max_norm', type=float, default=25, help='max norm allowed')
    parser.add_argument('--e_decay', default=1e-4, help='weight decay for ebm')
    parser.add_argument('--e_gamma', default=0.998, help='lr decay for ebm')
    parser.add_argument('--e_beta1', default=0.9, type=float)
    parser.add_argument('--e_beta2', default=0.999, type=float)
    parser.add_argument('--memory_size', default=200000, type=int)


    parser.add_argument('--dataset_name', type=str, default='eth')
    parser.add_argument('--dataset_folder', type=str, default='dataset')
    parser.add_argument('--obs',type=int,default=8)
    parser.add_argument('--preds',type=int,default=12)
    parser.add_argument('--delim',type=str,default='\t')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--val_size',type=int, default=0)
    parser.add_argument('--batch_size',type=int,default=70)

    parser.add_argument('--ny', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='saved_models/lbebm3D_scene1.pt')
    parser.add_argument('--test_mode', default=False, action='store_true', help='Bool type')
    parser.add_argument('--vis', type=bool, default=False, help='whether visualize predicted results')
    parser.add_argument('--lstm_layers', type=int, default=1)
    parser.add_argument('--state_layers', type=int, default=3)

    return parser.parse_args()


def set_gpu(gpu):
    torch.cuda.set_device('cuda:{}'.format(gpu))

def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]

def get_output_dir(exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger

def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        if not deterministic:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def main():

    exp_id = get_exp_id(__file__)
    output_dir = get_output_dir(exp_id)
    copy_source(__file__, output_dir)

    args = parse_args()
    set_gpu(args.device)
    set_cuda(deterministic=args.gpu_deterministic)
    set_seed(args.seed)
    args.way_points = list(set(list(range(args.future_length))) - set(args.sub_goal_indexes))

    logger = setup_logging('job{}'.format(0), output_dir, console=True)
    logger.info(args)

    if not args.test_mode:
        if args.val_size==0:
            train_dataset, _ = utils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=True, verbose=True)
            val_dataset, _ = utils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=False, verbose=True)
        else:
            train_dataset, val_dataset = utils.create_dataset(args.dataset_folder, args.dataset_name, args.val_size,args.obs, args.preds, delim=args.delim, train=True, verbose=args.verbose)
    else:
        test_dataset, _ =  utils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs,args.preds,delim=args.delim,train=False,eval=True, verbose=True)

    if not args.test_mode:
        tr_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size*10, shuffle=False, num_workers=args.num_workers)
    else:
        test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size*10, shuffle=False, num_workers=args.num_workers)
    
    trajaugmenter = TrajectoryAugmenter()

    def initial_pos(traj_batches):
        batches = []
        for b in traj_batches:
            starting_pos = b[:,7,:].copy()/1000
            batches.append(starting_pos)
        return batches

    def sample_p_0(n, nz=16):
        return args.e_init_sig * torch.randn(*[n, nz]).double().cuda()

    def calculate_loss_3dtraj_3d_offset(dest, dest_recon_x, dest_recon_y, dest_recon_z, mean, log_var, criterion, future, \
                interpolated_future_x, interpolated_future_y, interpolated_future_z, sub_goal_indexes, offset_x, offset_y, offset_z):
        dest_trans = dest.view(-1, dest.shape[1]//3, 3)
        dest_x = dest_trans[:,:,0]
        dest_y = dest_trans[:,:,1]
        dest_z = dest_trans[:,:,2]
        
        dest_x_loss = criterion(dest_x, dest_recon_x)
        dest_y_loss = criterion(dest_y, dest_recon_y)
        dest_z_loss = criterion(dest_z, dest_recon_z)

        future_trans = future.view(-1, future.shape[1]//3, 3)
        future_x = future_trans[:,:,0]
        future_y = future_trans[:,:,1]
        future_z = future_trans[:,:,2]

        future_x_loss = criterion(future_x, interpolated_future_x)
        future_y_loss = criterion(future_y, interpolated_future_y)
        future_z_loss = criterion(future_z, interpolated_future_z)

        s = len(offset_x)
        offset_loss_avg_x = 0
        offset_loss_avg_y = 0
        offset_loss_avg_z = 0
        pred_offset_x = torch.zeros_like(offset_x[0])
        pred_offset_y = torch.zeros_like(offset_x[0])
        pred_offset_z = torch.zeros_like(offset_x[0])
        
        for i in range(s):
            offset_xi = offset_x[i]
            offset_yi = offset_y[i]
            offset_zi = offset_z[i]
            pred_offset_x += offset_xi
            pred_offset_y += offset_yi
            pred_offset_z += offset_zi

            offset_x_loss = criterion(future_x, interpolated_future_x+pred_offset_x)
            offset_y_loss = criterion(future_y, interpolated_future_y+pred_offset_y)
            offset_z_loss = criterion(future_z, interpolated_future_z+pred_offset_z)

            offset_loss_avg_x += offset_x_loss
            offset_loss_avg_y += offset_y_loss
            offset_loss_avg_z += offset_z_loss
        offset_loss_avg_x /= s
        offset_loss_avg_y /= s
        offset_loss_avg_z /= s

        kl = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        interpolated_future_x = interpolated_future_x.view(dest.size(0), future.size(1)//3, 1)[:, sub_goal_indexes, :].view(dest.size(0), -1)
        interpolated_future_y = interpolated_future_y.view(dest.size(0), future.size(1)//3, 1)[:, sub_goal_indexes, :].view(dest.size(0), -1)
        interpolated_future_z = interpolated_future_z.view(dest.size(0), future.size(1)//3, 1)[:, sub_goal_indexes, :].view(dest.size(0), -1)

        subgoal_x_reg = criterion(dest_recon_x, interpolated_future_x)
        subgoal_y_reg = criterion(dest_recon_y, interpolated_future_y)
        subgoal_z_reg = criterion(dest_recon_z, interpolated_future_z)

        return dest_x_loss, dest_y_loss, dest_z_loss, future_x_loss, future_y_loss, future_z_loss, kl, subgoal_x_reg, subgoal_y_reg, subgoal_z_reg, offset_x_loss, offset_y_loss, offset_z_loss

    class MLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
            super(MLP, self).__init__()
            dims = []
            dims.append(input_dim)
            dims.extend(hidden_size)
            dims.append(output_dim)
            self.layers = nn.ModuleList()
            for i in range(len(dims)-1):
                self.layers.append(nn.Linear(dims[i], dims[i+1]))

            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'sigmoid':
                self.activation = nn.Sigmoid()

            self.sigmoid = nn.Sigmoid() if discrim else None
            self.dropout = dropout

        def forward(self, x):
            for i in range(len(self.layers)):
                x = self.layers[i](x)
                if i != len(self.layers)-1:
                    x = self.activation(x)
                    if self.dropout != -1:
                        x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
                elif self.sigmoid:
                    x = self.sigmoid(x)
            return x


    class ReplayMemory(object):
        def __init__(self, capacity):
            self.capacity = capacity
            self.memory = []
            self.position = 0

        def push(self, input_memory):
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = input_memory
            self.position = (self.position + 1) % self.capacity

        def sample(self, n=100):
            samples = random.sample(self.memory, n)
            return torch.cat(samples)

        def __len__(self):
            return len(self.memory)


    class LBEBM3D(nn.Module):
        def __init__(self, 
                    enc_past_size, 
                    enc_dest_size, 
                    enc_latent_size, 
                    dec_size, 
                    predictor_size, 
                    fdim, 
                    zdim, 
                    sigma, 
                    past_length, 
                    future_length):
            super(LBEBM3D, self).__init__()
            self.zdim = zdim
            self.sigma = sigma
            self.nonlocal_pools = args.nonlocal_pools
            self.lstm_layers = args.lstm_layers
            self.state_layers = args.state_layers
            non_local_dim = args.non_local_dim
            non_local_phi_size = args.non_local_phi_size
            non_local_g_size = args.non_local_g_size
            non_local_theta_size = args.non_local_theta_size

            self.encoder_past = MLP(input_dim=past_length*3, output_dim=fdim, hidden_size=enc_past_size)
            self.encoder_dest = MLP(input_dim=len(args.sub_goal_indexes)*3, output_dim=fdim, hidden_size=enc_dest_size)
            self.encoder_latent = MLP(input_dim=2*fdim, output_dim=2*zdim, hidden_size=enc_latent_size)

            # prediction for z axi trajectory
            self.decoder_z = MLP(input_dim=fdim+zdim, output_dim=len(args.sub_goal_indexes), hidden_size=dec_size)
            self.predictor_z = MLP(input_dim=2*fdim, output_dim=1*(future_length), hidden_size=predictor_size)

            # prediction for x aix trajectory
            self.decoder_x = MLP(input_dim=fdim+zdim, output_dim=len(args.sub_goal_indexes), hidden_size=dec_size)
            self.predictor_x = MLP(input_dim=2*fdim, output_dim=1*(future_length), hidden_size=predictor_size)

            self.decoder_y = MLP(input_dim=fdim+zdim, output_dim=len(args.sub_goal_indexes), hidden_size=dec_size)
            self.predictor_y = MLP(input_dim=2*fdim, output_dim=1*(future_length), hidden_size=predictor_size)

            self.non_local_theta = MLP(input_dim = fdim, output_dim = non_local_dim, hidden_size=non_local_theta_size)
            self.non_local_phi = MLP(input_dim = fdim, output_dim = non_local_dim, hidden_size=non_local_phi_size)
            self.non_local_g = MLP(input_dim = fdim, output_dim = fdim, hidden_size=non_local_g_size)

            self.EBM = nn.Sequential(
                nn.Linear(zdim + fdim, 200),
                nn.GELU(),
                nn.Linear(200, 200),
                nn.GELU(),
                nn.Linear(200, args.ny),
                )
                        
            self.replay_memory = ReplayMemory(args.memory_size)
            
            self.encoder_futurex = MLP(input_dim=3, output_dim=fdim, hidden_size=[128, 64])
            self.encoder_futurey = MLP(input_dim=3, output_dim=fdim, hidden_size=[128, 64])
            self.encoder_futurez = MLP(input_dim=3, output_dim=fdim, hidden_size=[128, 64])
            self.encoder_futures = MLP(input_dim=2*fdim, output_dim=fdim, hidden_size=[128, 64])            
            self.sc_lstm = SC_LSTM(fdim, fdim*4, 4, num_state=self.state_layers) # B x seq x N
            
            self.decoder_offsetx = nn.Linear(fdim*4, 1)
            self.decoder_offsety = nn.Linear(fdim*4, 1)
            self.decoder_offsetz = nn.Linear(fdim*4, 1)

            self.scale_weight_x = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.scale_weight_y = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
            self.scale_weight_z = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

            self.scale_weight_x.data.fill_(15)
            self.scale_weight_y.data.fill_(15)
            self.scale_weight_z.data.fill_(15)

        def forward(self, x, dest=None, mask=None, iteration=1, y=None):
            
            ftraj = self.encoder_past(x)

            if mask:
                for _ in range(self.nonlocal_pools):
                    ftraj = self.non_local_social_pooling(ftraj, mask)

            if self.training:
                pcd = True if len(self.replay_memory) == args.memory_size else False
                if pcd:
                    z_e_0 = self.replay_memory.sample(n=ftraj.size(0)).clone().detach().cuda()
                else:
                    z_e_0 = sample_p_0(n=ftraj.size(0), nz=self.zdim)
                z_e_k, _ = self.sample_langevin_prior_z(Variable(z_e_0), ftraj, pcd=pcd, verbose=(iteration % 1000==0))
                for _z_e_k in z_e_k.clone().detach().cpu().split(1):
                    self.replay_memory.push(_z_e_k)
            else:
                z_e_0 = sample_p_0(n=ftraj.size(0), nz=self.zdim)
                z_e_k, _ = self.sample_langevin_prior_z(Variable(z_e_0), ftraj, pcd=False, verbose=(iteration % 1000==0), y=y)                        
            z_e_k = z_e_k.double().cuda()


            if self.training:
                dest_features = self.encoder_dest(dest)
                features = torch.cat((ftraj, dest_features), dim=1)
                latent =  self.encoder_latent(features)
                mu = latent[:, 0:self.zdim]
                logvar = latent[:, self.zdim:]

                var = logvar.mul(0.5).exp_()
                eps = torch.DoubleTensor(var.size()).normal_().cuda()
                z_g_k = eps.mul(var).add_(mu)
                z_g_k = z_g_k.double().cuda()

            if self.training:
                decoder_input = torch.cat((ftraj, z_g_k), dim=1)
            else:
                decoder_input = torch.cat((ftraj, z_e_k), dim=1)

            # prediction for z axi trajectory
            generated_dest_z = self.decoder_z(decoder_input) # (batch, 1x4)

            # prediction for x axi trajectory
            generated_dest_x = self.decoder_x(decoder_input) # (batch, 1x4)

            # prediction for z axi trajectory
            generated_dest_y = self.decoder_y(decoder_input) # (batch, 1x4)

            # combine generated destination
            generated_dest_x_trans = generated_dest_x.unsqueeze(2)
            generated_dest_y_trans = generated_dest_y.unsqueeze(2)
            generated_dest_z_trans = generated_dest_z.unsqueeze(2)
            generated_dest_xyz = torch.cat((generated_dest_x_trans, generated_dest_y_trans, generated_dest_z_trans), dim=-1)
            generated_dest_xyz = generated_dest_xyz.view(-1, generated_dest_z.shape[1]*3)

            if self.training:
                generated_dest_features = self.encoder_dest(generated_dest_xyz)
                prediction_features = torch.cat((ftraj, generated_dest_features), dim=1)

                # predict future coordinates
                pred_future_z = self.predictor_z(prediction_features)
                pred_future_x = self.predictor_x(prediction_features)
                pred_future_y = self.predictor_y(prediction_features)

                en_pos = self.ebm(z_g_k, ftraj).mean()
                en_neg = self.ebm(z_e_k.detach().clone(), ftraj).mean()
                cd = en_pos - en_neg

                offset_x_ms = []
                offset_y_ms = []
                offset_z_ms = []
                pred_future_x_refine = pred_future_x.clone().detach()
                pred_future_y_refine = pred_future_y.clone().detach()
                pred_future_z_refine = pred_future_z.clone().detach()
                for n in range(self.lstm_layers):
                    if n != 0:
                        pred_future_x_refine += offset_x
                        pred_future_y_refine += offset_y
                        pred_future_z_refine += offset_z
                    #  construct x, y, z
                    trans_x = x.reshape(-1, x.shape[-1]//3, 3)
                    tem_x = torch.cat((trans_x[:,:,0], pred_future_x_refine),dim=-1)
                    tem_y = torch.cat((trans_x[:,:,1], pred_future_y_refine),dim=-1)
                    tem_z = torch.cat((trans_x[:,:,2], pred_future_z_refine),dim=-1)
                    tem_vx = tem_x[:,1:]-tem_x[:,:-1]
                    tem_vy = tem_y[:,1:]-tem_y[:,:-1]
                    tem_vz = tem_z[:,1:]-tem_z[:,:-1]

                    # coordinates
                    future_c_x = tem_x[:,2:]
                    future_c_y = tem_y[:,2:]
                    future_c_z = tem_z[:,2:]

                    # velocity
                    future_v_x = tem_vx[:,1:]
                    future_v_y = tem_vy[:,1:]
                    future_v_z = tem_vz[:,1:]

                    # acceleration
                    future_a_x = tem_vx[:,1:] - tem_vx[:,:-1]
                    future_a_y = tem_vx[:,1:] - tem_vx[:,:-1]
                    future_a_z = tem_vx[:,1:] - tem_vx[:,:-1]

                    cva_x = torch.cat((future_c_x.unsqueeze(2), future_v_x.unsqueeze(2), future_a_x.unsqueeze(2)), dim=-1) # B x seq x 3
                    cva_y = torch.cat((future_c_y.unsqueeze(2), future_v_y.unsqueeze(2), future_a_y.unsqueeze(2)), dim=-1)
                    cva_z = torch.cat((future_c_z.unsqueeze(2), future_v_z.unsqueeze(2), future_a_z.unsqueeze(2)), dim=-1)

                    cva_featurex = self.encoder_futurex(cva_x)
                    cva_featurey = self.encoder_futurey(cva_y)
                    cva_featurez = self.encoder_futurez(cva_z)

                    cva_feat = torch.cat((cva_featurex.unsqueeze(3), cva_featurey.unsqueeze(3), cva_featurez.unsqueeze(3)), dim=-1) # B x seq x fdim x 3

                    cva_maxfeat, _ = torch.max(cva_feat, dim=-1) # B x seq x fdim
                    cva_meanfeat = torch.mean(cva_feat, dim=-1) # B x seq x fdim
                    cva_features = torch.cat((cva_maxfeat, cva_meanfeat), dim=-1) # B x seq x 2*fdim
                    cva_features = self.encoder_futures(cva_features)
                    h_c, _, h_x, _, h_y, _, h_z, _ = self.sc_lstm(cva_featurex, cva_featurey, cva_featurez, cva_features)

                    length = pred_future_x.shape[1]
                    offset_x = self.scale_weight_x * torch.sigmoid(self.decoder_offsetx(h_x)[:,-length:,0])#[:,:,0]
                    offset_y = self.scale_weight_y * torch.sigmoid(self.decoder_offsety(h_y)[:,-length:,0])#[:,:,0]
                    offset_z = self.scale_weight_z * torch.sigmoid(self.decoder_offsetz(h_z)[:,-length:,0])#[:,:,0]

                    offset_x_ms.append(offset_x)
                    offset_y_ms.append(offset_y)
                    offset_z_ms.append(offset_z)

                return generated_dest_x, generated_dest_y, generated_dest_z, mu, logvar, pred_future_x, pred_future_y, pred_future_z, cd, en_pos, en_neg, pcd, offset_x_ms, offset_y_ms, offset_z_ms

            return generated_dest_xyz

        def ebm(self, z, condition, cls_output=False):
            condition_encoding = condition.detach().clone()
            z_c = torch.cat((z, condition_encoding), dim=1)
            conditional_neg_energy = self.EBM(z_c)
            assert conditional_neg_energy.shape == (z.size(0), args.ny)
            if cls_output:
                return - conditional_neg_energy
            else:
                return - conditional_neg_energy.logsumexp(dim=1)
        
        def sample_langevin_prior_z(self, z, condition, pcd=False, verbose=False, y=None):
            z = z.clone().detach()
            z.requires_grad = True
            _e_l_steps = args.e_l_steps_pcd if pcd else args.e_l_steps
            _e_l_step_size = args.e_l_step_size
            for i in range(_e_l_steps):
                if y is None:
                    en = self.ebm(z, condition)
                else:
                    en = self.ebm(z, condition, cls_output=True)[range(z.size(0)), y]
                z_grad = torch.autograd.grad(en.sum(), z)[0]

                z.data = z.data - 0.5 * _e_l_step_size * _e_l_step_size * (z_grad + 1.0 / (args.e_prior_sig * args.e_prior_sig) * z.data)
                if args.e_l_with_noise:
                    z.data += _e_l_step_size * torch.randn_like(z).data

                if (i % 5 == 0 or i == _e_l_steps - 1) and verbose:
                    if y is None:
                        print('Langevin prior {:3d}/{:3d}: energy={:8.3f}'.format(i+1, _e_l_steps, en.sum().item()))
                    else:
                        logger.info('Conditional Langevin prior {:3d}/{:3d}: energy={:8.3f}'.format(i + 1, _e_l_steps, en.sum().item()))

                z_grad_norm = z_grad.view(z_grad.size(0), -1).norm(dim=1).mean()

            return z.detach(), z_grad_norm


        def predict(self, past, generated_dest):
            ftraj = self.encoder_past(past)
            generated_dest_features = self.encoder_dest(generated_dest)
            prediction_features = torch.cat((ftraj, generated_dest_features), dim=1)

            interpolated_future_z = self.predictor_z(prediction_features)
            interpolated_future_x = self.predictor_x(prediction_features)
            interpolated_future_y = self.predictor_y(prediction_features)

            for n in range(self.lstm_layers):
                if n != 0:
                    interpolated_future_x = interpolated_future_x + offset_x
                    interpolated_future_y = interpolated_future_y + offset_y
                    interpolated_future_z = interpolated_future_z + offset_z
                
                #  construct x, y, z
                trans_x = past.reshape(-1, past.shape[-1]//3, 3)
                tem_x = torch.cat((trans_x[:,:,0], interpolated_future_x),dim=-1)
                tem_y = torch.cat((trans_x[:,:,1], interpolated_future_y),dim=-1)
                tem_z = torch.cat((trans_x[:,:,2], interpolated_future_z),dim=-1)
                tem_vx = tem_x[:,1:]-tem_x[:,:-1]
                tem_vy = tem_y[:,1:]-tem_y[:,:-1]
                tem_vz = tem_z[:,1:]-tem_z[:,:-1]

                # coordinates
                future_c_x = tem_x[:,2:]
                future_c_y = tem_y[:,2:]
                future_c_z = tem_z[:,2:]

                # velocity
                future_v_x = tem_vx[:,1:]
                future_v_y = tem_vy[:,1:]
                future_v_z = tem_vz[:,1:]

                # acceleration
                future_a_x = tem_vx[:,1:] - tem_vx[:,:-1]
                future_a_y = tem_vx[:,1:] - tem_vx[:,:-1]
                future_a_z = tem_vx[:,1:] - tem_vx[:,:-1]

                cva_x = torch.cat((future_c_x.unsqueeze(2), future_v_x.unsqueeze(2), future_a_x.unsqueeze(2)), dim=-1) # B x seq x 3
                cva_y = torch.cat((future_c_y.unsqueeze(2), future_v_y.unsqueeze(2), future_a_y.unsqueeze(2)), dim=-1)
                cva_z = torch.cat((future_c_z.unsqueeze(2), future_v_z.unsqueeze(2), future_a_z.unsqueeze(2)), dim=-1)

                cva_featurex = self.encoder_futurex(cva_x)
                cva_featurey = self.encoder_futurey(cva_y)
                cva_featurez = self.encoder_futurez(cva_z)

                cva_feat = torch.cat((cva_featurex.unsqueeze(3), cva_featurey.unsqueeze(3), cva_featurez.unsqueeze(3)), dim=-1) # B x seq x fdim x 3

                cva_maxfeat, _ = torch.max(cva_feat, dim=-1) # B x seq x fdim
                cva_meanfeat = torch.mean(cva_feat, dim=-1) # B x seq x fdim
                cva_features = torch.cat((cva_maxfeat, cva_meanfeat), dim=-1) # B x seq x 2*fdim
                cva_features = self.encoder_futures(cva_features)
                h_c, _, h_x, _, h_y, _, h_z, _ = self.sc_lstm(cva_featurex, cva_featurey, cva_featurez, cva_features)

                length = interpolated_future_x.shape[1]
                offset_x = self.scale_weight_x * torch.sigmoid(self.decoder_offsetx(h_x)[:,-length:,0])   
                offset_y = self.scale_weight_y * torch.sigmoid(self.decoder_offsety(h_y)[:,-length:,0])
                offset_z = self.scale_weight_z * torch.sigmoid(self.decoder_offsetz(h_z)[:,-length:,0])

            interpolated_future_x += offset_x
            interpolated_future_y += offset_y
            interpolated_future_z += offset_z
            
            interpolated_future_x_trans = interpolated_future_x.unsqueeze(2)
            interpolated_future_y_trans = interpolated_future_y.unsqueeze(2)
            interpolated_future_z_trans = interpolated_future_z.unsqueeze(2)
            interpolated_future = torch.cat((interpolated_future_x_trans, interpolated_future_y_trans, interpolated_future_z_trans), dim=-1)
            interpolated_future = interpolated_future.view(-1, interpolated_future_z.shape[1]*3)

            return interpolated_future

        def non_local_social_pooling(self, feat, mask):
            theta_x = self.non_local_theta(feat)
            phi_x = self.non_local_phi(feat).transpose(1,0)
            f = torch.matmul(theta_x, phi_x)
            f_weights = F.softmax(f, dim = -1)
            f_weights = f_weights * mask
            f_weights = F.normalize(f_weights, p=1, dim=1)
            pooled_f = torch.matmul(f_weights, self.non_local_g(feat))

            return pooled_f + feat


    class SC_LSTM(nn.Module):
        def __init__(self, input_sz, hidden_sz, reduce_sz, peephole=False, num_state=1):
            super(SC_LSTM, self).__init__()
            self.input_sz = input_sz
            self.hidden_size = hidden_sz
            self.peephole = peephole
            self.num_iter = num_state

            self.W_c = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
            self.U_c = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
            self.bias_c = nn.Parameter(torch.Tensor(hidden_sz * 4))
            self.W_x = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
            self.U_x = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
            self.bias_x = nn.Parameter(torch.Tensor(hidden_sz * 4))
            self.W_y = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
            self.U_y = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
            self.bias_y = nn.Parameter(torch.Tensor(hidden_sz * 4))
            self.W_z = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
            self.U_z = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
            self.bias_z = nn.Parameter(torch.Tensor(hidden_sz * 4))

            # state corelation
            self.W_rc1 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz//reduce_sz))
            self.W_rx1 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz//reduce_sz))
            self.W_ry1 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz//reduce_sz))
            self.W_rz1 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz//reduce_sz))

            self.W_cx = nn.Parameter(torch.Tensor(hidden_sz//reduce_sz*2, hidden_sz))
            self.W_cy = nn.Parameter(torch.Tensor(hidden_sz//reduce_sz*2, hidden_sz))
            self.W_cz = nn.Parameter(torch.Tensor(hidden_sz//reduce_sz*2, hidden_sz))
            self.bias_cx = nn.Parameter(torch.Tensor(hidden_sz))
            self.bias_cy = nn.Parameter(torch.Tensor(hidden_sz))
            self.bias_cz = nn.Parameter(torch.Tensor(hidden_sz))
            self.W_cxyz = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))

            # state aggregation
            self.W_rc2 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz//reduce_sz))
            self.W_rx2 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz//reduce_sz))
            self.W_ry2 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz//reduce_sz))
            self.W_rz2 = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz//reduce_sz))

            self.W_tcc = nn.Parameter(torch.Tensor(hidden_sz//reduce_sz*4, hidden_sz//reduce_sz))
            self.W_tcx = nn.Parameter(torch.Tensor(hidden_sz//reduce_sz*4, hidden_sz//reduce_sz))
            self.W_tcy = nn.Parameter(torch.Tensor(hidden_sz//reduce_sz*4, hidden_sz//reduce_sz))
            self.W_tcz = nn.Parameter(torch.Tensor(hidden_sz//reduce_sz*4, hidden_sz//reduce_sz))

            self.W_xc = nn.Parameter(torch.Tensor(hidden_sz//reduce_sz, hidden_sz))
            self.W_yc = nn.Parameter(torch.Tensor(hidden_sz//reduce_sz, hidden_sz))
            self.W_zc = nn.Parameter(torch.Tensor(hidden_sz//reduce_sz, hidden_sz))
            self.bias_xc = nn.Parameter(torch.Tensor(hidden_sz))
            self.bias_yc = nn.Parameter(torch.Tensor(hidden_sz))
            self.bias_zc = nn.Parameter(torch.Tensor(hidden_sz))
            self.W_xyzc = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))

            self.W_q = nn.Parameter(torch.Tensor(hidden_sz//reduce_sz, hidden_sz//reduce_sz))
            self.W_k = nn.Parameter(torch.Tensor(hidden_sz//reduce_sz, hidden_sz//reduce_sz))

            self.init_weights()
                    
        def init_weights(self):
            stdv = 1.0 / math.sqrt(self.hidden_size)
            for weight in self.parameters():
                weight.data.uniform_(-stdv, stdv)

        def state_extraction(self, h_tc, h_tx, h_ty, h_tz, c_tx, c_ty, c_tz):
            r_tc = h_tc @ self.W_rc1
            r_tx = h_tx @ self.W_rx1
            r_ty = h_ty @ self.W_ry1
            r_tz = h_tz @ self.W_rz1

            gate_tcx = torch.sigmoid(torch.cat((r_tc, r_tx), dim=-1) @ self.W_cx + self.bias_cx)
            gate_tcy = torch.sigmoid(torch.cat((r_tc, r_ty), dim=-1) @ self.W_cy + self.bias_cy)
            gate_tcz = torch.sigmoid(torch.cat((r_tc, r_tz), dim=-1) @ self.W_cz + self.bias_cz)

            cc_tx = c_tx + (h_tc * gate_tcx) @ self.W_cxyz
            cc_ty = c_ty + (h_tc * gate_tcy) @ self.W_cxyz
            cc_tz = c_tz + (h_tc * gate_tcz) @ self.W_cxyz

            return cc_tx, cc_ty, cc_tz
        
        def state_aggregation(self, h_tc, h_tx, h_ty, h_tz, c_tc):
            r_tc = h_tc @ self.W_rc2
            r_tx = h_tx @ self.W_rx2
            r_ty = h_ty @ self.W_ry2
            r_tz = h_tz @ self.W_rz2

            state_tc = torch.cat((r_tc, r_tx, r_ty, r_tz), dim=-1) @ self.W_tcc
            state_tx = torch.cat((r_tc, r_tx, r_ty, r_tz), dim=-1) @ self.W_tcx
            state_ty = torch.cat((r_tc, r_tx, r_ty, r_tz), dim=-1) @ self.W_tcy
            state_tz = torch.cat((r_tc, r_tx, r_ty, r_tz), dim=-1) @ self.W_tcz

            gate_txc = state_tx @ self.W_xc + self.bias_xc
            gate_tyc = state_ty @ self.W_yc + self.bias_yc
            gate_tzc = state_tz @ self.W_zc + self.bias_cz

            state_xyz =  torch.cat((state_tx.unsqueeze(1), state_ty.unsqueeze(1), state_tz.unsqueeze(1)),dim=1) # B x 3 x N
            query_tc = state_tc @ self.W_q # B x N
            key_txyz = state_xyz @ self.W_k # B x 3 x N

            att = torch.bmm(key_txyz, query_tc.unsqueeze(2)).squeeze() # B x 3
            att = torch.softmax(att, dim=-1) # B x 3
            if len(att.shape) == 1:
                att = att.unsqueeze(0)
            
            cc_tc = att[:,0:1] * (h_tx * gate_txc) + att[:,1:2] * (h_ty * gate_tyc) + att[:,2:3] * (h_tz * gate_tzc)
            cc_tc = cc_tc @ self.W_xyzc

            return cc_tc

        def forward(self, x, y, z, c, init_states_c=None, init_states_x=None, init_states_y=None, init_states_z=None):
            """Assumes x is of shape (batch, sequence, feature)"""
            bs, seq_sz, _ = x.size()
            hidden_seq_c = []
            hidden_seq_x = []
            hidden_seq_y = []
            hidden_seq_z = []
            if init_states_c is None:
                h_tc, c_tc = (torch.zeros(bs, self.hidden_size, dtype=torch.double).to(x.device), 
                            torch.zeros(bs, self.hidden_size, dtype=torch.double).to(x.device))
            else:
                h_tc, c_tc = init_states_c

            if init_states_x is None:
                h_tx, c_tx = (torch.zeros(bs, self.hidden_size, dtype=torch.double).to(x.device), 
                            torch.zeros(bs, self.hidden_size, dtype=torch.double).to(x.device))
            else:
                h_tx, c_tx = init_states_x

            if init_states_y is None:
                h_ty, c_ty = (torch.zeros(bs, self.hidden_size, dtype=torch.double).to(x.device), 
                            torch.zeros(bs, self.hidden_size, dtype=torch.double).to(x.device))
            else:
                h_ty, c_ty = init_states_y

            if init_states_z is None:
                h_tz, c_tz = (torch.zeros(bs, self.hidden_size, dtype=torch.double).to(x.device), 
                            torch.zeros(bs, self.hidden_size, dtype=torch.double).to(x.device))
            else:
                h_tz, c_tz = init_states_z

            HS = self.hidden_size
            for t in range(seq_sz):
                c_t = c[:, t, :]
                x_t = x[:, t, :]
                y_t = y[:, t, :]
                z_t = z[:, t, :]
                # batch the computations into a single matrix multiplication
                
                if self.peephole:
                    gates_c = c_t @ self.W_c + c_tc @ self.U_c + self.bias_c
                    gates_x = x_t @ self.W_x + c_tx @ self.U_x + self.bias_x
                    gates_y = y_t @ self.W_y + c_ty @ self.U_y + self.bias_y
                    gates_z = z_t @ self.W_z + c_tz @ self.U_z + self.bias_z
                else:
                    gates_c = c_t @ self.W_c + c_tc @ self.U_c + self.bias_c
                    gates_x = x_t @ self.W_x + c_tx @ self.U_x + self.bias_x
                    gates_y = y_t @ self.W_y + c_ty @ self.U_y + self.bias_y
                    gates_z = z_t @ self.W_z + c_tz @ self.U_z + self.bias_z
                    g_tc = torch.tanh(gates_c[:, HS*2:HS*3])
                    g_tx = torch.tanh(gates_x[:, HS*2:HS*3])
                    g_ty = torch.tanh(gates_y[:, HS*2:HS*3])
                    g_tz = torch.tanh(gates_z[:, HS*2:HS*3])
                
                i_tc, f_tc, o_tc = (
                    torch.sigmoid(gates_c[:, :HS]), # input
                    torch.sigmoid(gates_c[:, HS:HS*2]), # forget
                    torch.sigmoid(gates_c[:, HS*3:]), # output
                )
                i_tx, f_tx, o_tx = (
                    torch.sigmoid(gates_x[:, :HS]), # input
                    torch.sigmoid(gates_x[:, HS:HS*2]), # forget
                    torch.sigmoid(gates_x[:, HS*3:]), # output
                )
                i_ty, f_ty, o_ty = (
                    torch.sigmoid(gates_y[:, :HS]), # input
                    torch.sigmoid(gates_y[:, HS:HS*2]), # forget
                    torch.sigmoid(gates_y[:, HS*3:]), # output
                )
                i_tz, f_tz, o_tz = (
                    torch.sigmoid(gates_z[:, :HS]), # input
                    torch.sigmoid(gates_z[:, HS:HS*2]), # forget
                    torch.sigmoid(gates_z[:, HS*3:]), # output
                )
                
                if self.peephole:
                    c_tc = f_tc * c_tc + i_tc * torch.sigmoid(c_t @ self.W_c + self.bias_c)[:, HS*2:HS*3]
                    h_tc = torch.tanh(o_tc * c_tc)
                    c_tx = f_tx * c_tx + i_tx * torch.sigmoid(x_t @ self.W_x + self.bias_x)[:, HS*2:HS*3]
                    h_tx = torch.tanh(o_tx * c_tx)
                    c_ty = f_ty * c_ty + i_ty * torch.sigmoid(y_t @ self.W_y + self.bias_y)[:, HS*2:HS*3]
                    h_ty = torch.tanh(o_ty * c_ty)
                    c_tz = f_tz * c_tz + i_tz * torch.sigmoid(z_t @ self.W_z + self.bias_z)[:, HS*2:HS*3]
                    h_tz = torch.tanh(o_tz * c_tz)
                else:
                    c_tc = f_tc * c_tc + i_tc * g_tc
                    h_tc = o_tc * torch.tanh(c_tc)
                    c_tx = f_tx * c_tx + i_tx * g_tx
                    h_tx = o_tx * torch.tanh(c_tx)
                    c_ty = f_ty * c_ty + i_ty * g_ty
                    h_ty = o_ty * torch.tanh(c_ty)
                    c_tz = f_tz * c_tz + i_tz * g_tz
                    h_tz = o_tz * torch.tanh(c_tz)

                for i in range(self.num_iter):
                    cc_tx, cc_ty, cc_tz = self.state_extraction(h_tc, h_tx, h_ty, h_tz, c_tx, c_ty, c_tz)
                    cc_tc = self.state_aggregation(h_tc, h_tx, h_ty, h_tz, c_tc)
                    c_tc, c_tx, c_ty, c_tz = cc_tc, cc_tx, cc_ty, cc_tz
                
                    if self.peephole:
                        h_tc = torch.tanh(o_tc * c_tc)
                        h_tx = torch.tanh(o_tx * c_tx)
                        h_ty = torch.tanh(o_ty * c_ty)
                        h_tz = torch.tanh(o_tz * c_tz)
                    else:
                        h_tc = o_tc * torch.tanh(c_tc)
                        h_tx = o_tx * torch.tanh(c_tx)
                        h_ty = o_ty * torch.tanh(c_ty)
                        h_tz = o_tz * torch.tanh(c_tz)
                
                hidden_seq_c.append(h_tc.unsqueeze(0))
                hidden_seq_x.append(h_tx.unsqueeze(0))
                hidden_seq_y.append(h_ty.unsqueeze(0))
                hidden_seq_z.append(h_tz.unsqueeze(0))
                
            hidden_seq_c = torch.cat(hidden_seq_c, dim=0)
            hidden_seq_x = torch.cat(hidden_seq_x, dim=0)
            hidden_seq_y = torch.cat(hidden_seq_y, dim=0)
            hidden_seq_z = torch.cat(hidden_seq_z, dim=0)
            # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
            hidden_seq_c = hidden_seq_c.transpose(0, 1).contiguous()
            hidden_seq_x = hidden_seq_x.transpose(0, 1).contiguous()
            hidden_seq_y = hidden_seq_y.transpose(0, 1).contiguous()
            hidden_seq_z = hidden_seq_z.transpose(0, 1).contiguous()
            
            return hidden_seq_c, (h_tc, c_tc), hidden_seq_x, (h_tx, c_tx), hidden_seq_y, (h_ty, c_ty), hidden_seq_z, (h_tz, c_tz)

    def train(model, optimizer, epoch, sub_goal_indexes):
        model.train()
        train_loss, total_dest_loss, total_future_loss, total_offset_loss = 0, 0, 0, 0
        criterion = nn.MSELoss()

        for i, trajx in enumerate(tr_dl):
            x = trajx['src'][:, :, :3] # n x 8 x 3
            y = trajx['trg'][:, :, :3] # n x 12 x 3
            x, y = x.unsqueeze(0).permute(0, 1, 3, 2).contiguous(), y.unsqueeze(0).permute(0, 1, 3, 2).contiguous()
            x, y = trajaugmenter.augment(x, y) # input: 1 x num x 3 x time_step
            x, y = x.squeeze(0).permute(0, 2, 1).contiguous(), y.squeeze(0).permute(0, 2, 1).contiguous()

            init_point = x[:, -1:, :3]
            x = x - init_point
            y = y - init_point

            x *= args.data_scale
            y *= args.data_scale
            x = x.double().cuda()
            y = y.double().cuda()

            x = x.view(-1, x.shape[1]*x.shape[2])
            dest = y[:, sub_goal_indexes, :].detach().clone().view(y.size(0), -1)
            future = y.view(y.size(0),-1)

            torch.autograd.set_detect_anomaly(True)

            dest_recon_x, dest_recon_y, dest_recon_z, mu, var, interpolated_future_x, interpolated_future_y, interpolated_future_z, \
            cd, en_pos, en_neg, pcd, offset_x, offset_y, offset_z = model.forward(x, dest=dest, mask=None, iteration=i)
            
            optimizer.zero_grad()

            # strategy 3 + offset x, y, z
            dest_x_loss, dest_y_loss, dest_z_loss, future_x_loss, future_y_loss, future_z_loss, kld, subgoal_x_reg, subgoal_y_reg, subgoal_z_reg, offset_x_loss, offset_y_loss, offset_z_loss \
                = calculate_loss_3dtraj_3d_offset(dest, dest_recon_x, dest_recon_y, dest_recon_z, mu, var, criterion, future, interpolated_future_x, interpolated_future_y, interpolated_future_z, \
                sub_goal_indexes, offset_x, offset_y, offset_z)
            
            loss = args.dest_loss_coeff * (dest_x_loss + dest_y_loss + dest_z_loss) + args.future_loss_coeff * (future_x_loss + future_y_loss + future_z_loss) \
                + args.kld_coeff * kld + cd + (subgoal_x_reg + subgoal_y_reg + subgoal_z_reg) + (offset_x_loss + offset_y_loss + offset_z_loss)

            loss.backward()

            train_loss += loss.item()
            total_dest_loss += (dest_x_loss.item()+dest_y_loss.item()+dest_z_loss.item())
            total_future_loss += (future_x_loss.item()+future_y_loss.item()+future_z_loss.item())
            total_offset_loss += (offset_x_loss.item()+offset_y_loss.item()+offset_z_loss.item())
            optimizer.step()

            if (i+1) % args.print_log == 0:
                logger.info('{:5d}/{:5d} '.format(i, epoch) +
                            'dest_loss={:8.6f} '.format(dest_x_loss.item()+dest_y_loss.item()+dest_z_loss.item()) +
                            'future_loss={:8.6f} '.format(future_x_loss.item()+future_y_loss.item()+future_z_loss.item()) +
                            'offset_loss={:8.6f} '.format(offset_x_loss.item()+offset_y_loss.item()+offset_z_loss.item()) + 
                            'kld={:8.6f} '.format(kld.item()) +
                            'cd={:8.6f} '.format(cd.item()) +
                            'en_pos={:8.6f} '.format(en_pos.item()) +
                            'en_neg={:8.6f} '.format(en_neg.item()) +
                            'pcd={} '.format(pcd) +
                            'subgoal_reg={} '.format((subgoal_x_reg+subgoal_y_reg+subgoal_z_reg).detach().cpu().numpy()) 
                )

        return train_loss, total_dest_loss, total_future_loss

    def test(model, dataloader, dataset, sub_goal_indexes, best_of_n=20):
        model.eval()


        total_dest_err = 0.
        total_overall_err = 0.

        for i, trajx in enumerate(dataloader):
            x = trajx['src'][:, :, :3]
            y = trajx['trg'][:, :, :3]
            x = x - trajx['src'][:, -1:, :3]
            y = y - trajx['src'][:, -1:, :3]

            x *= args.data_scale
            y *= args.data_scale
            x = x.double().cuda()
            y = y.double().cuda()

            y = y.cpu().numpy()

            x = x.view(-1, x.shape[1]*x.shape[2])

            plan = y[:, sub_goal_indexes, :].reshape(y.shape[0],-1)
            all_plan_errs = []
            all_plans = []
            for _ in range(best_of_n):
                plan_recon = model.forward(x, mask=None)
                plan_recon = plan_recon.detach().cpu().numpy()
                all_plans.append(plan_recon)
                plan_err = np.linalg.norm(plan_recon - plan, axis=-1)
                all_plan_errs.append(plan_err)

            all_plan_errs = np.array(all_plan_errs) 
            all_plans = np.array(all_plans) 
            indices = np.argmin(all_plan_errs, axis=0)
            best_plan = all_plans[indices, np.arange(x.shape[0]),  :]

            # FDE
            best_dest_err = np.linalg.norm(best_plan[:, -3:] - plan[:, -3:], axis=1).sum()

            best_plan = torch.DoubleTensor(best_plan).cuda()
            interpolated_future = model.predict(x, best_plan)
            interpolated_future = interpolated_future.detach().cpu().numpy() # batch x 12*3

            # ADE
            predicted_future = np.reshape(interpolated_future, (-1, args.future_length, 3))
            overall_err = np.linalg.norm(y - predicted_future, axis=-1).mean(axis=-1).sum()

            overall_err /= args.data_scale
            best_dest_err /= args.data_scale

            total_overall_err += overall_err
            total_dest_err += best_dest_err            

        total_overall_err /= len(dataset)
        total_dest_err /= len(dataset)

        return total_overall_err, total_dest_err


    def run_training(args):
        model = LBEBM3D(
            args.enc_past_size,
            args.enc_dest_size,
            args.enc_latent_size,
            args.dec_size,
            args.predictor_hidden_size,
            args.fdim,
            args.zdim,
            args.sigma,
            args.past_length,
            args.future_length)
        
        model = model.double().cuda()

        def get_param_groups(model, special_lr_layers):
            special_params = []
            base_params = []
            
            for name, param in model.named_parameters():
                if any(layer_name in name for layer_name in special_lr_layers):
                    special_params.append(param)
                else:
                    base_params.append(param)
            
            return special_params, base_params

        special_lr_layers = ['scale_weight_x','scale_weight_y','scale_weight_z']

        special_params, base_params = get_param_groups(model, special_lr_layers)

        optimizer = optim.Adam([
            {'params': special_params, 'lr': args.learning_rate * 100},
            {'params': base_params, 'lr': args.learning_rate}
        ])

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_gamma)

        best_val_ade = 50
        best_val_fde = 50
        best_test_ade = 50
        best_test_fde = 50

        for epoch in range(args.num_epochs):
            # Start the timer for the epoch
            start_time = time.time()
            train_loss, dest_loss, overall_loss = train(model, optimizer, epoch, args.sub_goal_indexes)
            overall_err, dest_err = test(model, val_dl, val_dataset, args.sub_goal_indexes, args.n_values)

            if best_val_ade > overall_err:
                patience_epoch = 0
                best_val_ade = overall_err
                best_val_fde = dest_err

                torch.save({
						'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict()
						}, args.model_path)
                print("Saved model to:\n{}".format(args.model_path))
            # End the timer for the epoch
            end_time = time.time()
            epoch_time = end_time - start_time
            print(f"Time taken for dataset {args.dataset_name} epoch {epoch}: {epoch_time:.2f} seconds")
            logger.info("Train Loss {}".format(train_loss))
            logger.info("Overall Loss {}".format(overall_loss))
            logger.info("Dest Loss {}".format(dest_loss))
            logger.info("Val ADE {}".format(overall_err))
            logger.info("Val FDE {}".format(dest_err))
            logger.info("Val Best ADE {}".format(best_val_ade))
            logger.info("Val Best FDE {}".format(best_val_fde))
            logger.info("----->learning rate {}".format(optimizer.param_groups[0]['lr'])) 

            scheduler.step()

    def run_eval(args):
        model = LBEBM3D(
            args.enc_past_size,
            args.enc_dest_size,
            args.enc_latent_size,
            args.dec_size,
            args.predictor_hidden_size,
            args.fdim,
            args.zdim,
            args.sigma,
            args.past_length,
            args.future_length)

        model = model.double().cuda()
        
        ckpt = torch.load(args.model_path, map_location=torch.device('cuda'))
        model.load_state_dict(ckpt['model_state_dict'])

        overall_err, dest_err = test(model, test_dl, test_dataset, args.sub_goal_indexes, args.n_values)
        logger.info("{}, test ADE {}".format(args.dataset_name, overall_err))
        logger.info("{}, test FDE {}".format(args.dataset_name, dest_err))

    if not args.test_mode:
        run_training(args)
    else:
        run_eval(args)


if __name__ == '__main__':
    main()