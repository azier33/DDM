import os
import sys
sys.path.append(os.path.abspath('/raid'))
gpu_devices = [0]
gpu_devices = ','.join([str(id) for id in gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
import torch
import numpy as np
import sys
import torch.distributed as dist
from Final_diffusion.modules.sevir import SevirDataset
from Final_diffusion.modules.moving_mnist import MovingMNIST
from Final_diffusion.logs.MM.Diffusion_BeiBei_gdl import Unet3D, GaussianDiffusion
# from Final_diffusion.modules.Diffusion_BeiBei  import Unet3D, GaussianDiffusion
from Final_diffusion.logs.MM.Diffusion_pred_no_diff import Unet3D as Unet_ori, GaussianDiffusion as GaussianDiffusion_ori
from PIL import Image
import argparse
import imageio
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Diffusion")
parser.add_argument("--random_seed", default=1234)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--max_epoch', default=1400, type=int)
parser.add_argument("--log_dir", type=str, default='/raid/Final_diffusion/logs/Sevir/5to5/D20241231T2347', help="Where to save logs of the model.")
parser.add_argument("--ori_log_dir", type=str, default='/raid/Final_diffusion/logs/Sevir/5to5/D20241231T2347', help="Where to save logs of the model.")
# parser.add_argument('--data_dir', default="/raid/DataSet/Moving_mnist", type=str) #sevir dataset: /raid/DataSet/SevirLr/data/vil
parser.add_argument('--data_dir', default="/raid/DataSet/SevirLr/data/vil", type=str)
parser.add_argument("--world_size", default=4)#单机多卡：代表有几块GPU
parser.add_argument("--out_dim", default=5) #输出维度
parser.add_argument("--ori_out_dim", default=5) #输出维度
parser.add_argument("--cond_dim", default=5) #条件维度
parser.add_argument('--num_workers', type=int, default=4, help='')#设置为显卡数量的2-4倍
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--dist_backend', default='nccl', type=str, help='')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:5223', type=str, help='')
parser.add_argument("--num_repeats", default=5, type=int)
parser.add_argument("--timesteps", default=1000, type=int)
parser.add_argument("--sample_timesteps", default=500, type=int)
parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument("--restore_from", default=None, type=str)
parser.add_argument('--save-img-freq', default=5000, type=int,  metavar='N', help='save image frequency')
parser.add_argument('--save-vid-freq', default=1000, type=int,  metavar='N', help='save video frequency')
parser.add_argument('--set_start', default=False, type=bool)
parser.add_argument('--print_freq', default=1000, type=int,  metavar='N', help='printing frequency')
parser.add_argument('--save_model_fre_step', default=500, type=int,  metavar='N', help='save model frequency')
parser.add_argument('--save_model_fre_epoch', default=100, type=int,  metavar='N', help='update model frequency')
parser.add_argument('--epoch_milestones', type=int, nargs='+', default = [80, 100])
args = parser.parse_args()

def sample_img(rec_img, idx=0):
    rec_img = rec_img.data.cpu().numpy().copy()
    rec_img[rec_img < 0] = 0
    rec_img[rec_img > 1] = 1
    rec_img *= 255
    #print(np.array(rec_img, np.uint8))
    return np.array(rec_img, np.uint8)

model = Unet3D(
    dim = args.img_dim,
    channels = 1,
    out_dim= args.out_dim,
    cond_dim = args.cond_dim,
    dim_mults = (1, 2, 4, 4)
    )
model_ori = Unet_ori(
    dim = args.img_dim,
    out_dim= args.ori_out_dim,
    channels = 1, # input_dim (output) + cond_dim
    cond_dim = args.cond_dim,
    dim_mults = (1, 2, 4, 4)
    )
model_ori.to(device)
model.to(device)
diffusion = GaussianDiffusion(
    model,
    objective = 'pred_noise',
    image_size = args.img_dim,
    timesteps = args.timesteps,    # number of steps
    sampling_timesteps= args.sample_timesteps
    )
diffusion.load_state_dict(torch.load(os.path.join(args.log_dir,'modelshots/MM_diff_epoch100.pt'), map_location='cuda:0'))
diffusion_ori = GaussianDiffusion_ori(
    model_ori,
    objective = 'pred_noise',
    image_size = args.img_dim,
    timesteps = args.timesteps,     # number of steps
    sampling_timesteps= args.sample_timesteps
)
diffusion_ori.load_state_dict(torch.load(os.path.join(args.log_dir,'modelshots/MM_diff_epoch100.pt'), map_location='cuda:0'))
diffusion.to(device)
diffusion_ori.to(device)
print("检查一下device:",device)

new_im = Image.new('L', (args.img_dim, args.img_dim))
new_im_diff = Image.new('L', (args.img_dim, args.img_dim))


#dataloader setting
if args.data_dir == "/raid/DataSet/Moving_mnist":
    datasets = MovingMNIST(root=args.data_dir, train=True, download=False)
else:
    datasets = SevirDataset(root=args.data_dir, train=True, download=False)
dataloader = torch.utils.data.DataLoader(datasets, batch_size= args.batch_size, shuffle=False, num_workers=2)
num = 0
#parameter for lpips、 ssim、 psnr``s calculation
#~~~~~~~~~~~~~~~~~


for step, (frames, diff_frames) in enumerate(dataloader):
    frames = frames.to(device)
    frames = frames.permute(0, 3, 1, 2) #使用MM数据集时，可能需要注销掉 RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 25 but got size 128 for tensor number 1 in the list.
    bs = 0
    nf = 0
    # while bs < frames.shape[0]:
    # cond_frames = frames[:args.batch_size, :args.cond_dim, :, :]
    # generated_images_list = []
    # for gnf in range(args.out_dim):
    #     generated_images = diffusion.sample(cond_frames, batch_size=args.batch_size) #min(-1), max(1)
    #     generated_images_list.append(generated_images)
    #     cond_frames = torch.cat([cond_frames[:, 1:], generated_images], dim=1)
    #     # cond_frames = torch.cat([cond_frames[:, 1:], frames[:,(args.cond_dim+gnf):(args.cond_dim+gnf+1)]], dim=1)
    # generated_images = torch.stack(generated_images_list,dim=1)
    # generated_images = torch.squeeze(generated_images, dim=2)
    # print(generated_images.shape)
    # generated_images_ori = cond_frames
    generated_images = diffusion.sample(frames[:args.batch_size, :args.cond_dim, :, :], batch_size=args.batch_size)
    # generated_images_ori = diffusion_ori.sample(frames[:args.batch_size, :args.cond_dim, :, :], batch_size=args.batch_size) #min(-1), max(1)
    generated_images_ori = generated_images #min(-1), max(1)
    
    # generated_images = sample_img(generated_images)
    new_im_arr_list = []
    new_im_arr_list_ori = []
    x_start = frames[0, (args.cond_dim - 1), :, :] #(0, 1)
    # x_start = generated_images[0, 0, :, :] #(0, 1)
    for nf in range(generated_images.shape[1]):
        new_im = Image.new('L', (args.img_dim, args.img_dim))               
        new_im.paste(Image.fromarray(sample_img(generated_images[1, nf, :, :]), 'L'), (0, 0))  
        new_im_arr_list.append(new_im)


        in_im = Image.new('L', (args.img_dim, args.img_dim))               
        in_im.paste(Image.fromarray(sample_img(generated_images_ori[1, nf, :, :]), 'L'), (0, 0))  
        new_im_arr_list_ori.append(in_im)

    print("Done!")
    
    new_vid_name = 'imgshots/'+'Gen_Diff' + '_S' + format(step, "06d") + '_' + ".gif"
    new_vid_name_ori = 'imgshots/'+'Gen_Ori' + '_S' + format(step, "06d") + '_' + ".gif"
    new_vid_file = os.path.join(args.log_dir, new_vid_name)
    new_vid_file_ori = os.path.join(args.log_dir, new_vid_name_ori)
    imageio.mimsave(new_vid_file, new_im_arr_list)
    imageio.mimsave(new_vid_file_ori, new_im_arr_list_ori)