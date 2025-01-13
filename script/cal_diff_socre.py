import os
import sys
sys.path.append(os.path.abspath('/raid'))
gpu_devices = [6]
gpu_devices = ','.join([str(id) for id in gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
import torch
import numpy as np
import sys
import torch.distributed as dist
from Final_diffusion.modules.sevir import SevirDataset
from Final_diffusion.modules.moving_mnist import MovingMNIST
# from Final_diffusion.logs.MM.Diffusion_BeiBei_gdl import Unet3D, GaussianDiffusion
from Final_diffusion.modules.Diffusion_BeiBei import Unet3D, GaussianDiffusion
# from Final_diffusion.modules.evaluation import BRISUQE, temporal_consistancy,FIDEvaluation
from Final_diffusion.modules.metrics import Evaluator
from Final_diffusion.logs.MM.Diffusion_BeiBei_gdl import Unet3D as Unet_ori, unnormalize_to_zero_to_one as unnormalize
from Final_diffusion.logs.MM.Diffusion_BeiBei_gdl import GaussianDiffusion as GaussianDiffusion_ori
from PIL import Image
import argparse
import imageio
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Diffusion")
parser.add_argument("--random_seed", default=1234)
parser.add_argument('--batch_size', default=5, type=int)
parser.add_argument('--max_epoch', default=1400, type=int)
parser.add_argument("--log_dir", type=str, default='/raid/Final_diffusion/logs/Sevir/5to5/D20241231T2347', help="Where to save logs of the model.")
parser.add_argument("--ori_log_dir", type=str, default='/raid/Final_diffusion/logs/Sevir/5to5/D20241225T1911', help="Where to save logs of the model.")
# parser.add_argument('--data_dir', default="/raid/DataSet/Moving_mnist", type=str) #sevir dataset: /raid/DataSet/SevirLr/data/vil
parser.add_argument('--data_dir', default="/raid/DataSet/SevirLr/data/vil", type=str)
parser.add_argument("--world_size", default=4)#单机多卡：代表有几块GPU
parser.add_argument("--ori_out_dim", default=5) #输出维度
parser.add_argument("--out_dim", default=5) #输出维度
parser.add_argument("--cond_dim", default=5) #条件维度
parser.add_argument('--num_workers', type=int, default=4, help='')#设置为显卡数量的2-4倍
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--dist_backend', default='nccl', type=str, help='')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:5223', type=str, help='')
parser.add_argument("--num_repeats", default=5, type=int)
parser.add_argument("--timesteps", default=1000, type=int)
parser.add_argument('--img_dim', default=128, type=int)
parser.add_argument('--ori_img_dim', default=128, type=int)
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

    #rec_img += np.array(MEAN)/255.0
    # rec_img[rec_img < 0] = 0
    # rec_img[rec_img > 1] = 1
    rec_img *= 255
    return np.array(rec_img, np.uint8)
model = Unet3D(
    dim = args.ori_img_dim,
    channels = 1,
    out_dim= args.out_dim,
    cond_dim = args.cond_dim,
    dim_mults = (1, 2, 4, 4)
    )
evaluator = Evaluator(seq_len=args.out_dim, value_scale=255.0)
evaluator_ori = Evaluator(seq_len=args.ori_out_dim, value_scale=255.0)
model_ori = Unet_ori(
    dim = args.ori_img_dim,
    channels = 1, # input_dim (output) + cond_dim
    out_dim= args.ori_out_dim,
    cond_dim = args.cond_dim,
    dim_mults = (1, 2, 4, 4)
)
model.to(device)
diffusion = GaussianDiffusion(
    model,
    objective = 'pred_noise',
    image_size = args.ori_img_dim,
    timesteps = args.timesteps    # number of steps
    )
diffusion.load_state_dict(torch.load(os.path.join(args.log_dir,'modelshots/MM_diff_epoch100.pt'), map_location='cuda:0'))
diffusion_ori = GaussianDiffusion_ori(
    model_ori,
    objective = 'pred_noise',
    image_size = args.ori_img_dim,
    timesteps = args.timesteps     # number of steps
)
diffusion_ori.load_state_dict(torch.load(os.path.join(args.ori_log_dir,'modelshots/MM_diff_epoch100.pt'), map_location='cuda:0'))
diffusion_ori.to(device)
diffusion.to(device)
diffusion_ori.to(device)
print("检查一下device:",device)

new_im = Image.new('L', (args.img_dim, args.img_dim))
new_im_diff = Image.new('L', (args.img_dim, args.img_dim))


#dataloader setting
if args.data_dir == "/raid/DataSet/Moving_mnist":
    datasets = MovingMNIST(root=args.data_dir, train=False, download=False)
else:
    datasets = SevirDataset(root=args.data_dir, train=False, download=False)
dataloader = torch.utils.data.DataLoader(datasets, batch_size= args.batch_size, shuffle=False, num_workers=2)
num = 0
#parameter for lpips、 ssim、 psnr``s calculation
#~~~~~~~~~~~~~~~~~
lpips_value_sum = 0
lpips_value_ori_sum = 0
ssim_ori_sum = 0
psnr_ori_sum = 0
ssim_sum = 0
psnr_sum = 0
avg_c_sum = 0
avg_f_sum = 0
avg_p_sum = 0
avg_h_sum = 0
avg_c_sum_ori = 0
avg_f_sum_ori = 0
avg_p_sum_ori = 0
avg_h_sum_ori = 0


for step, (frames, diff_frames) in enumerate(dataloader):
    frames = frames.to(device)
    frames = frames.permute(0, 3, 1, 2) #使用MM数据集时，可能需要注销掉 RuntimeError: Sizes of tensors must match except in dimension 1. Expected size 25 but got size 128 for tensor number 1 in the list.
    bs = 0
    # while bs < frames.shape[0]:
    generated_images = diffusion.sample(frames[:args.batch_size, :args.cond_dim, :, :], batch_size=args.batch_size) #min(-1), max(1)
    # generated_images = unnormalize(diffusion.decoder(frames[:args.batch_size, args.cond_dim - 1:args.cond_dim, :, :], generated_images))
    # generated_images_ori = diffusion_ori.sample(frames[:args.batch_size, :args.cond_dim, :, :], batch_size=args.batch_size) #min(-1), max(1)
    generated_images_ori = generated_images
    # frame = torch.round(frame*255.0).byte()
    new_im_arr_list = []
    nf = 0
    x_start = torch.rand_like(generated_images)
    x_start[:, 0:1, :, :] = frames[:args.batch_size, (args.cond_dim -1):args.cond_dim, :, :] + generated_images[:, nf:nf+1, : , :]

    # test_list = torch.stack(new_im_arr_list, dim = 1)#(6,128,128)
    # test_list = torch.squeeze(test_list, dim = 2) # [5, 20, 128, 128]
    print(generated_images.shape)
    # imageio.mimsave("/raid/Final_diffusion/logs/MM/5to20/beibei_gdl_predx0/imgshots/test.gif", sample_img(generated_images[4,:]))
    # imageio.mimsave("/raid/Final_diffusion/logs/MM/5to20/beibei_gdl_predx0/imgshots/gen_ori.gif", sample_img(generated_images_ori[4,:]))
    # imageio.mimsave("/raid/Final_diffusion/logs/MM/5to20/beibei_gdl_predx0/imgshots/ori.gif", sample_img(frames[4, args.cond_dim:args.cond_dim + args.out_dim,:,:]))
    print(generated_images_ori.shape)
    evaluator.evaluate(frames[:args.batch_size, args.cond_dim:args.cond_dim + args.out_dim,:,:], generated_images)
    avg_csi, avg_far, avg_pod, avg_hss, avg_csi44, avg_csi16, mses, mass, rmses, psnrs, ssims, crpss, lpipss = evaluator.done()

    psnr_sum += psnrs
    ssim_sum += ssims
    lpips_value_sum +=lpipss
    avg_c_sum += avg_csi
    avg_f_sum += avg_far
    avg_p_sum += avg_pod
    avg_h_sum += avg_hss

    evaluator_ori.evaluate(frames[:args.batch_size, args.cond_dim:args.cond_dim + args.ori_out_dim,:,:], generated_images_ori)
    avg_csi_ori, avg_far_ori, avg_pod_ori, avg_hss_ori, avg_csi44_ori, avg_csi16_ori, mses_ori, mass_ori, rmses_ori, psnrs_ori, ssims_ori, crpss_ori, lpipss_ori = evaluator_ori.done()
    
    psnr_ori_sum += psnrs_ori
    ssim_ori_sum += ssims_ori
    lpips_value_ori_sum +=lpipss_ori
    avg_c_sum_ori += avg_csi_ori
    avg_f_sum_ori += avg_far_ori
    avg_p_sum_ori += avg_pod_ori
    avg_h_sum_ori += avg_hss_ori


    print("ssim_ori:",ssims_ori, "ssim:",ssims)
    print("psnr_ori:",psnrs_ori, "psnr:",psnrs)
    print("lpipss_ori:",lpipss_ori, "lpipss:",lpipss)
    print("avg_c_ori:",avg_csi_ori, "avg_c:",avg_csi)
    print("avg_f_ori:",avg_far_ori, "avg_f:",avg_far)
    print("avg_p_ori:",avg_pod_ori, "avg_p:",avg_pod)
    print("avg_h_ori:",avg_hss_ori, "avg_h:",avg_hss)


    num+=1
    if step%300 == 0:
            with open(os.path.join(args.log_dir, 'test_score.txt'), 'a') as file:
                # 写入标题
                file.write("############################################################ \n************************************************************")
                file.write("Gen Gen Gen Gen Gen Gen Gen Gen Gen Gen \n")
                # 写入计算后的分数
                file.write(f"{step}:lpips_score: {lpipss}\n")
                file.write(f"{step}:ssim_score: {ssims}\n")
                file.write(f"{step}:psnr_score: {psnrs}\n")
                file.write(f"{step}:avg_c_score: {avg_csi}\n")
                file.write(f"{step}:avg_f_score: {avg_far}\n")
                file.write(f"{step}:avg_p_score: {avg_pod}\n")
                file.write(f"{step}:avg_h_score: {avg_hss}\n")
                
                # 写入第二组标题
                file.write("Ori Ori Ori Ori Ori Ori Ori Ori Ori Ori \n")
                # 写入计算后的分数
                file.write(f"{step}:lpips_ori_score: {lpipss_ori}\n")
                file.write(f"{step}:ssim_ori_score: {ssims_ori}\n")
                file.write(f"{step}:psnr_ori_score: {psnrs_ori}\n")
                file.write(f"{step}:avg_c_score: {avg_csi_ori}\n")
                file.write(f"{step}:avg_f_score: {avg_far_ori}\n")
                file.write(f"{step}:avg_p_score: {avg_pod_ori}\n")
                file.write(f"{step}:avg_h_score: {avg_hss_ori}\n")


# The Score of lpips  psnr  and ssim
print("Gen Gen Gen Gen Gen Gen Gen Gen Gen Gen ")
print("lpips_score:",lpips_value_sum/num)
print("ssim_score:",ssim_sum/num)
print("psnr_score:",psnr_sum/num)
print("avg_c_score:",avg_c_sum/num)
print("avg_f_score:",avg_f_sum/num)
print("avg_p_score:",avg_p_sum/num)
print("avg_h_score:",avg_h_sum/num)

# The Score of ori  lpips  psnr  and ssim
print("Ori Ori Ori Ori Ori Ori Ori Ori Ori Ori ")
print("lpips_ori_score:",lpips_value_ori_sum/num)
print("ssim_ori_score:",ssim_ori_sum/num)
print("psnr_ori_score:",psnr_ori_sum/num)
print("avg_c_ori_score:",avg_c_sum_ori/num)
print("avg_f_ori_score:",avg_f_sum_ori/num)
print("avg_p_ori_score:",avg_p_sum_ori/num)
print("avg_h_ori_score:",avg_h_sum_ori/num)


with open(os.path.join(args.log_dir, 'test_score.txt'), 'a') as file:
    # 写入标题
    file.write("############################################################ \n************************************************************")
    file.write("Gen Gen Gen Gen Gen Gen Gen Gen Gen Gen \n")
    # 写入计算后的分数
    file.write(f"final——lpips_score: {lpips_value_sum/num}\n")
    file.write(f"final——ssim_score: {ssim_sum/num}\n")
    file.write(f"final——psnr_score: {psnr_sum/num}\n")
    file.write(f"final——avg_c_score: {avg_c_sum/num}\n")
    file.write(f"final——avg_f_score: {avg_f_sum/num}\n")
    file.write(f"final——avg_p_score: {avg_p_sum/num}\n")
    file.write(f"final——avg_h_score: {avg_h_sum/num}\n")
    
    # 写入第二组标题
    file.write("Ori Ori Ori Ori Ori Ori Ori Ori Ori Ori \n")
    # 写入计算后的分数
    file.write(f"final——lpips_ori_score: {lpips_value_ori_sum/num}\n")
    file.write(f"final——ssim_ori_score: {ssim_ori_sum/num}\n")
    file.write(f"final——psnr_ori_score: {psnr_ori_sum/num}\n")
    file.write(f"final——avg_c_score: {avg_c_sum_ori/num}\n")
    file.write(f"final——avg_f_score: {avg_f_sum_ori/num}\n")
    file.write(f"final——avg_p_score: {avg_p_sum_ori/num}\n")
    file.write(f"final——avg_h_score: {avg_h_sum_ori/num}\n")