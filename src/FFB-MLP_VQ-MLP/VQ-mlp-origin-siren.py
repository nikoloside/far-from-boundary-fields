# 基礎
import argparse
import os
import numpy as np
import math
import sys
import h5py
import glob
from tqdm import tqdm
import scipy
from pysdf import SDF
import igl
from scipy.ndimage import zoom
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import datetime

from mpl_toolkits.mplot3d import axes3d
from io import BytesIO
from PIL import Image

from animator3d import MedicalImageAnimator

# Deep Library
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

#region Manual Argsz
os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=80000, help="number of epochs of training")
parser.add_argument("--no_epochs", type=float, default=0.000001, help="loss while ending the training")
parser.add_argument("--start_epochs", type=int, default=2000, help="loss while ending the training")
parser.add_argument('--continue_train', action='store_false', help='default true, if true, load the model')

parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--ndf", type=int, default=32, help="ndf default 64")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

parser.add_argument("--realTimeSampling", type=str, default='pre-sample', help="run-time, hybrid, pre-sample this three mode for sampling")
parser.add_argument("--z_latent_dim", type=int, default=120, help="dimensionality of the z latent space, 0 represent no latent dim")
parser.add_argument("--pos_encode_dim", type=int, default=128, help="default 128 dimensionality of the x, y, z pos encode")
parser.add_argument("--noisy_dim", type=int, default=8, help="default 128 dimensionality of the x, y, z pos encode")
parser.add_argument("--mlp_dim", type=int, default=512, help="dimensionality of the mlp")
parser.add_argument("--mlp_sample_num", type=int, default=12000, help="number of 1 seg map x,y,z sample num")
parser.add_argument("--data_shape", type=int, default=64, help="dimensionality of data shape(shape, shape, shape)")
parser.add_argument("--mlp_batches", type=int, default=100, help="sample batch")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between showing image samples")
parser.add_argument("--save_interval", type=int, default=1000, help="interval between saving parameter")
parser.add_argument("--resample_interval", type=int, default=2000, help="interval between saving parameter")

# parser.add_argument("--channels", type=int, default=1, help="number of image channels")
# parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
# parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")

parser.add_argument("--dataroot", type=str, default="/data/data-mlp/_out_squirrel/", help="datapath")
parser.add_argument("--phase", type=str, default="train", help="phase")
parser.add_argument('--save_bool', action='store_false', help='if false, save the model')
parser.add_argument('--save_path', type=str, default="/nashome/yhuang/yhuang/Workspace/data-mlp/mlp-result/", help="savepath")
parser.add_argument('--projName', type=str, default="squirrel", help="projName")
parser.add_argument('--runName', type=str, default="squirrel-run-presample-vq-mlp-ffbdf-300/", help="Run Name")

parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--train_dataset_size', type=int, default=350, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
parser.add_argument('--test_dataset_size', type=int, default=1, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

opt = parser.parse_args()
print(opt)
#endregion

#region Helper Function

import nibabel as nib

def save_as_nib(file_name, variable):
    ni_img = nib.Nifti1Image(variable, affine=np.eye(4))
    nib.save(ni_img, file_name)

# def exp(x):
#     s=0.1
#     r = (1.0 - np.exp(-x**2/2/s/s) - 0.5) * 2.0
#     return r


def drawGraph(decoder, encoder, featureEncoder, dataset, opt, save_path, epoch, feature_List):
    avg_c = []
    decoder.eval(); encoder.eval(); featureEncoder.eval()
    with torch.no_grad():
        # 每次drawGraph时，重新clear tested indices
        tested_indices = set()
        for i, (pos, dir, imp, coords, sdfs, latents, idx) in enumerate(dataset):

            for i in range(len(latents)):
                # 只生成前5个 test case, 不重复
                current_idx = idx[i].item()
                if current_idx in tested_indices or len(tested_indices) >= 5:
                    continue
                tested_indices.add(current_idx)

                before = datetime.datetime.now()

                pos = pos.cuda()
                dir = dir.cuda()
                imp = imp.cuda()

                feature_z = featureEncoder(pos[i], dir[i], imp[i])

                latents = latents.cuda()

                # VQ-PoC
                latent_z = torch.cuda.FloatTensor(1, opt.noisy_dim)
                init.xavier_normal_(latent_z)
                print("start", idx[i])
                query = torch.concat((feature_z, latent_z[0]), -1)

                min_index = idx[i]
                minDist = torch.tensor(10000)
                min_feature = feature_z

                for (ind_n, feature_n) in feature_List:
                    # print(feature_n, dataset.dataset.latent_vectors[ind_n].cuda())
                    cookbook = torch.concat((feature_n, dataset.dataset.latent_vectors[ind_n].cuda()[0]), -1)

                    dist = torch.linalg.norm(cookbook - query)

                    print("search %d,%f,%f" % (min_index, minDist, dist))

                    if minDist.item() > dist.item():
                        minDist = dist
                        min_index = ind_n
                        min_feature = feature_n

                # output = decoder.predict(min_feature, dataset.latent_vectors[min_index]).squeeze().squeeze()

                # 64x64x64
                point_grid = np.mgrid[-1:1:(opt.data_shape * 1j), -1:1:(opt.data_shape * 1j), -1:1:(opt.data_shape * 1j)]
                point_grid = np.ascontiguousarray(point_grid.reshape(3, -1).transpose())
                
                point_grid = torch.from_numpy(point_grid).to(torch.float32)
                point_grid = point_grid.cuda().unsqueeze(0)

                feature_coords = encoder(point_grid)
                predicted_sdf = decoder.forward(min_feature.unsqueeze(0), dataset.dataset.latent_vectors[min_index].cuda(), feature_coords)
                predicted_sdf = predicted_sdf.reshape((opt.data_shape,opt.data_shape,opt.data_shape))#.transpose().squeeze().squeeze()                                 

                after = datetime.datetime.now()
                c = before - after
                print("Inference Time:", c.microseconds)
                avg_c.append(c.microseconds)

                gif_path = os.path.join(save_path, 'test_epoch_%d_ind_%d.gif' % (epoch, idx[i].numpy()))
                animator = MedicalImageAnimator(predicted_sdf.to('cpu').detach().numpy().copy(), [], 0, save=True)
                animate = animator.run(gif_path)

                vox_path = os.path.join(save_path, 'test_epoch_%d_ind_%d.nii' % (epoch, idx[i].numpy()))
                save_as_nib(vox_path, predicted_sdf.to('cpu').detach().numpy().copy())

                # 256x256x256
                point_grid = np.mgrid[-1:1:(opt.data_shape * 4j), -1:1:(opt.data_shape * 4j), -1:1:(opt.data_shape * 4j)]
                point_grid = np.ascontiguousarray(point_grid.reshape(3, -1).transpose())

                point_grids = np.vsplit(point_grid,  64)
                predicted_sdfs = []
                
                for grid in point_grids:
                    torch_grid = torch.from_numpy(grid).to(torch.float32)
                    torch_grid = torch_grid.cuda().unsqueeze(0)

                    feature_coords = encoder(torch_grid)

                    latents = latents.cuda()

                    predicted_sdf = decoder.forward(min_feature.unsqueeze(0), dataset.dataset.latent_vectors[min_index].cuda(), feature_coords)
                    predicted_sdfs.append(predicted_sdf.to('cpu').detach().numpy().copy())

                    # del torch_grid
                    # torch.cuda.empty_cache()

                sdfs = np.concatenate(predicted_sdfs)
                sdfs = sdfs.reshape((opt.data_shape * 4,opt.data_shape * 4,opt.data_shape * 4))#.transpose().squeeze().squeeze()
                
                gif_path = os.path.join(save_path, 'test_epoch_%d_ind_%d_vq_%d_256.gif' % (epoch, idx[i].numpy(), min_index))
                animator = MedicalImageAnimator(sdfs, [], 0, save=True)
                animate = animator.run(gif_path)

                vox_path = os.path.join(save_path, 'test_epoch_%d_ind_%d_vq_%d_256.nii' % (epoch, idx[i].numpy(), min_index))
                save_as_nib(vox_path, sdfs)

                # break

                print("Avg Inference Time: ", np.mean(avg_c))
    #endregion

#region Data Loader
import json
from siren_pytorch import Siren, Sine
class InputMapDataset(Dataset):
    def __init__(self):
        super(InputMapDataset, self).__init__()

    def name(self):
        return 'InputMapDataset'

    def __len__(self):
        return min(len(self.AB_paths), self.opt.train_dataset_size)
    
    def getPos(self, jsonName):
        maxInx = 0
        max = 0
        impList = []
        dirList = []
        posList = []
        maxImpulse = 304527
        with open(jsonName) as f:
            jsonInput = json.load(f)
            ind = 0
            for imp in jsonInput:
                Imp = imp["collImpulse"]
                impList.append(np.array([Imp / maxImpulse], dtype=np.float32))
                dirList.append(np.array([imp["collDirections"][0], imp["collDirections"][1],imp["collDirections"][2]],dtype = np.float32))
                posList.append(np.array([imp["collPoints"][0],imp["collPoints"][1],imp["collPoints"][2]],dtype = np.float32))

                if Imp > max:
                    max = Imp
                    maxInx = ind

                ind += 1
        return posList[maxInx], dirList[maxInx], impList[maxInx]



    def __getitem__(self, idx):
        expTime = os.path.basename(self.AB_paths[idx]).split(".")[0]
        jsonName = self.workspacePath + "impact/" + str(expTime) + ".txt"

        pos, dir, imp = self.getPos(jsonName)

        pos = torch.from_numpy(pos).to(torch.float32)
        dir = torch.from_numpy(dir).to(torch.float32)
        imp = torch.from_numpy(imp).to(torch.float32)
        # idx = torch.from_numpy(idx).to(torch.int32)

        poisson_grid_points = []
        sdf_values = []

        if self.realTimeSampling == "real-time":
            objName = self.workspacePath + "obj/" + str(expTime) + ".obj"
            pd_sampler = scipy.stats.qmc.PoissonDisk(d=3, radius = 0.01)
            vertices, faces = igl.read_triangle_mesh(objName)
            poisson_grid_points = pd_sampler.random(self.mlp_sample_num)#, workers=4)
            poisson_grid_points = 2 * poisson_grid_points - 1
            sdf_values = SDF(vertices, faces)(poisson_grid_points)
        # sdf_values_igl, indices, closest = igl.signed_distance(poisson_grid_points, vertices, faces)
        else:
            npyName = self.workspacePath + "npz/" + str(expTime) + ".npz"
            data = np.load(npyName)
            poisson_grid_points = data['poisson_grid_points']
            sdf_values = data['sdf_values']
            
        # sdf_values = exp(sdf_values)
        N = len(sdf_values)  # total number of items
        limited_num = 130000
        indices = np.random.choice(N, size=limited_num, replace=False)
        sdf_values = sdf_values[indices]
        poisson_grid_points = poisson_grid_points[indices]
        # print(limited_num, len(sdf_values), len(poisson_grid_points))

        poisson_grid_points = torch.from_numpy(poisson_grid_points).to(torch.float32)
        sdf_values = torch.from_numpy(sdf_values).to(torch.float32)
        indices = torch.from_numpy(np.array([idx])).to(torch.int)
        
        # print("sampler grid: ", poisson_grid_points.shape, torch.min(poisson_grid_points), torch.max(poisson_grid_points))
        # print("sampler sdf: ", torch.min(sdf_values), torch.max(sdf_values))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        return pos, dir, imp, poisson_grid_points, sdf_values, self.latent_vectors[idx], indices

    def collate_fn(self, x):
        min_sample_len = min([len(i[3]) for i in x])
        # print("sampler: ", min_sample_len, len(x), len(x[0]), len(x[0][3]))
        pos_vals = [i[0][:min_sample_len].unsqueeze(0) for i in x]
        dir_vals = [i[1][:min_sample_len].unsqueeze(0) for i in x]
        imp_vals = [i[2][:min_sample_len].unsqueeze(0) for i in x]
        x_vals = [i[3][:min_sample_len].unsqueeze(0) for i in x]
        y_vals = [i[4][:min_sample_len].unsqueeze(0) for i in x]
        vec = [i[5].unsqueeze(0) for i in x]
        indices = [i[6].unsqueeze(0) for i in x]
        return torch.cat(pos_vals, dim=0), torch.cat(dir_vals, dim=0), torch.cat(imp_vals, dim=0), torch.cat(x_vals, dim=0), torch.cat(y_vals, dim=0), torch.cat(vec, dim=0), torch.cat(indices, dim=0)

    def initialize(self,opt):
        self.opt = opt
        self.root = opt.dataroot
        self.pos_encode_dim = opt.pos_encode_dim
        self.mlp_sample_num = opt.mlp_sample_num
        self.mlp_batches = opt.mlp_batches
        self.realTimeSampling = opt.realTimeSampling

        # mi
        self.projectPath = "_out_" + opt.projName + "/"
        self.workspacePath = "/data/data-mlp/" + self.projectPath

        self.AB_paths = [x.split('.')[0] for x in glob.glob(self.workspacePath + 'npz/*.npz')]
        self.AB_paths = sorted(self.AB_paths)
        # self.file_paths = [os.path.join(dataset_path, i) for i in fnames]
        latent_sd = 0.01
        latent_mean = 0.0
        self.latent_vectors = torch.randn(len(self.AB_paths), opt.noisy_dim)
        self.latent_vectors = (self.latent_vectors * latent_sd) + latent_mean
        self.latent_vectors.requires_grad = True


        transform_list = transforms.Compose([
            # transforms.Resize((opt.img_height, opt.img_width, opt.img_depth), Image.BICUBIC),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.transforms = transform_list

class DataLoader():
    def __init__(self):
        pass
    
    def initialize(self, opt):
        self.opt = opt
        self.dataset = InputMapDataset()
        print("dataset [%s] was created" % (self.dataset.name()))
        self.dataset.initialize(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            collate_fn=self.dataset.collate_fn,
            # num_workers=int(opt.nThreads)
            )
    
    def load_data(self):
        return self.dataloader


    def __len__(self):
        return min(len(self.dataset), self.opt.train_dataset_size)

#endregion

#region Network
import torch.nn.functional as F
import torch.nn.init as init

from siren_pytorch import Siren
class MultiLatentEncoder(nn.Module):
    def __init__(self, opt):
        super(MultiLatentEncoder, self).__init__()

        self.neuron_input = Siren(
            dim_in = 7,
            dim_out = opt.z_latent_dim
        )

    def forward(self, pos, direct, imp):
        input_encoded = torch.concat((pos, direct, imp), -1)
        output = self.neuron_input(input_encoded)
        return output

class PosEncoder(nn.Module):
    def __init__(self, opt):
        super(PosEncoder, self).__init__()

        self.neuron_input = Siren(
            dim_in = 3,
            dim_out = opt.pos_encode_dim
        )

    def forward(self, pos):
        output = self.neuron_input(pos)
        return output


# 我们现在在P1 也就是说，需要一个 mlp to 东西
class ImplicitFunction(nn.Module):
    def __init__(self, opt, dropout_p = 0.2):
        super(ImplicitFunction, self).__init__()

        self.dropout_p = dropout_p
        # self.mlp_stage1 = nn.Sequential(
        #     nn.Linear(opt.z_latent_dim + opt.pos_encode_dim + opt.noisy_dim, opt.mlp_dim),
        #     nn.ReLU(True),
        #     nn.Linear(opt.mlp_dim, opt.mlp_dim),
        #     nn.ReLU(True),
        #     nn.Linear(opt.mlp_dim, opt.mlp_dim),
        #     nn.ReLU(True),
        #     nn.Linear(opt.mlp_dim, opt.mlp_dim),
        #     nn.ReLU(True),
        #     nn.Linear(opt.mlp_dim, opt.mlp_dim))
        
        # self.mlp_stage2 = nn.Sequential(
        #     nn.Linear(opt.mlp_dim + opt.z_latent_dim + opt.pos_encode_dim + opt.noisy_dim, opt.mlp_dim),
        #     nn.ReLU(True),
        #     nn.Linear(opt.mlp_dim, opt.mlp_dim),
        #     nn.ReLU(True),
        #     nn.Linear(opt.mlp_dim, opt.mlp_dim),
        #     nn.ReLU(True),
        #     nn.Linear(opt.mlp_dim, opt.mlp_dim),
        #     nn.ReLU(True),
        #     nn.Linear(opt.mlp_dim, 1),
        #     nn.Tanh())

        self.input_layer = self.create_layer_block(opt.z_latent_dim + opt.pos_encode_dim + opt.noisy_dim, opt.mlp_dim)
        self.layer2 = self.create_layer_block(opt.mlp_dim, opt.mlp_dim)
        self.layer3 = self.create_layer_block(opt.mlp_dim, opt.mlp_dim)
        self.layer4 = self.create_layer_block(opt.mlp_dim, opt.mlp_dim - opt.z_latent_dim - opt.pos_encode_dim - opt.noisy_dim)
        self.layer5 = self.create_layer_block(opt.mlp_dim, opt.mlp_dim)
        self.layer6 = self.create_layer_block(opt.mlp_dim, opt.mlp_dim)
        self.layer7 = self.create_layer_block(opt.mlp_dim, opt.mlp_dim)
        # self.layer8 = nn.linear(opt.mlp_dim, 1)
        self.layer8 = self.create_layer_block(opt.mlp_dim, 1)

    def create_layer_block(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.SiLU(),
            # nn.ReLU(True),
            # nn.Dropout(self.dropout_p)
        )
        # return Siren(
        #     dim_in = input_size,
        #     dim_out = output_size
        # )


    def forward(self, feature_z, latent_vec, coords):
        latent_vec = latent_vec.unsqueeze(1).repeat(1, coords.shape[1], 1)
        feature_vec = feature_z.unsqueeze(1).repeat(1, coords.shape[1], 1)

        # print("forwarding:" , feature_vec.shape, latent_vec.shape, coords.shape)
        x = torch.cat([feature_vec, latent_vec, coords], dim = -1)
        skip_x = x

        x = self.input_layer(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(torch.cat([x, skip_x], dim = -1)) # skip connection
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        # return has shape [batch_size, num_coords], where each element is the SDF
        # at the given input coordinate
        return x.squeeze(-1).tanh()
        
#endregion



def clamp(delta, x):
    return torch.clamp(x, min=-delta, max=delta)

def mse(outputs, targets):
    return ((outputs - targets) ** 2).sum() #taking sum just to track the progress

class DeepSDFLoss:

    def __init__(self, delta, sd):
        self.mae = nn.L1Loss()
        self.delta = delta
        self.sd = sd

    def __call__(self, yhat, y, latent):
        l = self.mae(torch.clamp(yhat, -self.delta, self.delta), torch.clamp(y, -self.delta, self.delta))
        latent_norm = torch.pow(latent, 2).sum(dim=-1).mean()  * (1 / (self.sd ** 2))
        return l + latent_norm

crit = DeepSDFLoss(1.0, 0.01)

#region Main Code Initialize

data_loader = DataLoader()
data_loader.initialize(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

save_path = opt.save_path + opt.runName
fileExist = glob.glob(save_path)
if len(fileExist) <= 0:
    os.mkdir(save_path)
    os.mkdir(save_path + "networks/")

cuda = True if torch.cuda.is_available() else False

featureEncoder = MultiLatentEncoder(opt)
encoder = PosEncoder(opt)
decoder = ImplicitFunction(opt)

log_file = open(save_path + "log.txt", "w")

if cuda:
    decoder.cuda()
    encoder.cuda()
    featureEncoder.cuda()
    # discriminator.cuda()
    print(cuda)

if opt.continue_train or opt.phase == "test":
    codeName = save_path + opt.projName + "-codes.npz"
    data = np.load(codeName)
    dataset.dataset.latent_vectors =  torch.from_numpy(data[data.files[0]]).to(torch.float32)
    decoder = torch.load(save_path + opt.projName + "-decoder.pt")
    encoder = torch.load(save_path + opt.projName + "-encoder.pt")
    featureEncoder = torch.load(save_path + opt.projName + "-featureEncoder.pt")

if opt.phase == "train":
    # recusntruction loss
    criterion = nn.MSELoss()#nn.BCEWithLogitsLoss()

    learning_rate = 2e-5
    optimizer = torch.optim.Adam([dataset.dataset.latent_vectors] + list(decoder.parameters()), lr=opt.lr * opt.batch_size)
    # optimizer = torch.optim.Adam(decoder.parameters(), lr=opt.lr)

    decoder.train()
    feature_List = []

    epoch = opt.start_epochs
    while True:

        # もし一定の数を超えればであれば、Resampling
        if opt.realTimeSampling == "hybrid" and epoch % opt.resample_interval == 0 :
        
            for path in dataset.dataset.AB_paths:
                expTime = os.path.basename(path).split(".")[0]

                print("start load:", path)
                objName = dataset.dataset.workspacePath + "obj/" + str(expTime) + ".obj"
                pd_sampler = scipy.stats.qmc.PoissonDisk(d=3, radius = 0.005)
                vertices, faces = igl.read_triangle_mesh(objName)
                poisson_grid_points = pd_sampler.random(opt.mlp_sample_num)#, workers=4)
                poisson_grid_points = 2 * poisson_grid_points - 1
                sdf_values = SDF(vertices, faces)(poisson_grid_points)

                npyName = dataset.dataset.workspacePath + "npz/" + str(expTime) + ".npz"

                print("start save:", npyName)

                np.savez(npyName, poisson_grid_points=poisson_grid_points, sdf_values=sdf_values)

                # data = np.load(npyName)
                # print(data["poisson_grid_points"], data["sdf_values"])
        epoch_loss = 0
        # epoch_mse = 0

        loop = tqdm(dataset, total=len(dataset))
        loop.set_description('Epoch {}'.format(epoch))
        
        for pos, dir, imp, coords, sdfs, latents, idx in loop:

            optimizer.zero_grad()

            pos = pos.cuda()
            dir = dir.cuda()
            imp = imp.cuda()

            feature_z = featureEncoder(pos, dir, imp)

            for i in range(len(idx)):
                if idx[i] > opt.train_dataset_size:
                    continue
                feature_List.append((idx[i], feature_z[i]))

            coords = coords.cuda()

            feature_coords = encoder(coords)

            sdfs = sdfs.cuda()
            latents = latents.cuda()

            # print("input data:", feature_z.shape, latents.shape, feature_coords.shape)
            predicted_sdf = decoder.forward(feature_z, latents, feature_coords)
            loss = crit(predicted_sdf, sdfs, latents)
            
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss='{:.10f}'.format(loss.item()))#, mse = '{:.10f}'.format(mean_mse))
            loop.update()
                
            log_file.write('Epoch {}, loss={:.10f}\n'.format(epoch, loss.item()))
            epoch_loss += loss.item()
            # break

        epoch_loss /= len(dataset)
        # epoch_mse /= len(dataset)
        
        log_file.write('Epoch {}, epoch_loss={:.10f}\n'.format(epoch, epoch_loss))

        if epoch % opt.sample_interval == 0:
            decoder.eval()
            drawGraph(decoder, encoder, featureEncoder, dataset, opt, save_path, epoch, feature_List)
            decoder.train()

        if opt.save_bool:
            codeName = save_path + opt.projName + "-codes.npz"
            np.savez(codeName, dataset.dataset.latent_vectors.detach().numpy())
            torch.save(decoder, save_path + opt.projName + "-decoder.pt")
            torch.save(encoder, save_path + opt.projName + "-encoder.pt")
            torch.save(featureEncoder, save_path + opt.projName + "-featureEncoder.pt")

            if epoch % opt.save_interval == 0:
                codeName = save_path + "networks/" + opt.projName + str(epoch) + "-codes.npz"
                np.savez(codeName, dataset.dataset.latent_vectors.detach().numpy())
                torch.save(decoder, save_path + "networks/" + opt.projName + str(epoch) + "-decoder.pt")
                torch.save(encoder, save_path + "networks/" + opt.projName + str(epoch) + "-encoder.pt")
                torch.save(featureEncoder, save_path + "networks/" + opt.projName + str(epoch) + "-featureEncoder.pt")


        epoch += 1


        if (opt.n_epochs > 0 and epoch > opt.n_epochs):
            break

    decoder.eval()
    drawGraph(decoder, encoder, featureEncoder, dataset, opt, save_path, epoch, feature_List)

else:
    decoder.eval()
    feature_List = []
    for i, (pos, dir, imp, coords, sdfs, latents, idx) in enumerate(dataset):
        
        if i >= opt.train_dataset_size:
            continue

        pos = pos.cuda()
        dir = dir.cuda()
        imp = imp.cuda()

        feature_z = featureEncoder(pos, dir, imp)

        for i in range(len(idx)):
            feature_List.append((idx[i], feature_z[i]))

    drawGraph(decoder, encoder, featureEncoder, dataset, opt, save_path, 999, feature_List)

#endregion