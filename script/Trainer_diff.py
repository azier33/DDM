import sys
sys.path.append('/raid')
import os
import argparse
import gc
import numpy as np
import random
gpu_devices = [6, 7]
gpu_devices = ','.join([str(id) for id in gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
import imageio
from PIL import Image
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.utils.data import RandomSampler
from Final_diffusion.modules.utils import Logger
from Final_diffusion.modules.sevir import SevirDataset
from Final_diffusion.modules.moving_mnist import MovingMNIST
from Final_diffusion.modules.metrics import Evaluator
# from Final_diffusion.modules.Diffusion_BeiBei import Unet3D, GaussianDiffusion, unnormalize_to_zero_to_one as unnormalize
from Final_diffusion.logs.MM.Diffusion_BeiBei_gdl import Unet3D, GaussianDiffusion, unnormalize_to_zero_to_one as unnormalize
def main():
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")
    parser = argparse.ArgumentParser(description="Diffusion")
    parser.add_argument("--random_seed", default=1234)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--max_epoch', default=1000, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    parser.add_argument("--log_dir", type=str, default='/raid/Final_diffusion/logs/Sevir/5to5/D20241231T2347', help="Where to save logs of the model.")
    # parser.add_argument('--data_dir', default="/raid/DataSet/Moving_mnist", type=str)
    parser.add_argument('--data_dir', default="/raid/DataSet/SevirLr/data/vil", type=str)
    parser.add_argument("--world_size", default=2)#单机多卡：代表有几块GPU
    parser.add_argument("--out_dim", default=5) #输出维度
    parser.add_argument("--cond_dim", default=5) #条件维度
    parser.add_argument('--num_workers', type=int, default=4, help='')#设置为显卡数量的2-4倍
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:5123', type=str, help='')
    parser.add_argument("--timesteps", default=1000, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument("--restore_from", default=None, type=str)
    parser.add_argument('--save_validate_sample_fre_step', default=1000, type=int,  metavar='N', help='save model frequency')
    parser.add_argument('--save_model_fre_epoch', default=100, type=int,  metavar='N', help='update model frequency')
    parser.add_argument('--save_train_sample_fre_step', default=1000, type=int,  metavar='N', help='update model frequency')
    
    parser.add_argument('--epoch_milestones', type=int, nargs='+', default = [80, 100])
    args = parser.parse_args()

    #randmo seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = True

    #gpu setting
    ngpus_per_node = torch.cuda.device_count()
    print("using number of gpus:", ngpus_per_node)
    args.world_size = ngpus_per_node
    
    #setup main function by mp
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))#nprocs: 进程数量，即：world_size


def sample_img(rec_img, idx=0):
    rec_img = rec_img.data.cpu().numpy().copy()
    rec_img[rec_img < 0] = 0
    rec_img[rec_img > 1] = 1
    rec_img *= 255
    #print(np.array(rec_img, np.uint8))
    return np.array(rec_img, np.uint8)

def add_memory(title, memory, args):
    with open(os.path.join(args.log_dir, 'Memory_logs.txt'), 'a') as file:
        # 写入计算后的分数
        file.write(f"allocate_memory: {memory}_____")
        file.write(f"{title}\n")

def main_worker(gpu, ngpus_per_node, args):
    #summarywriter log dir setting
    log_txt_dir = os.path.join(args.log_dir, 'logs')
    os.makedirs(args.log_dir, exist_ok=True)
    log_txt = os.path.join(log_txt_dir,"B"+str(args.batch_size)+"E"+str(args.max_epoch)+".log")

    args.img_dir = os.path.join(args.log_dir, 'imgshots')
    args.train_vid_dir = os.path.join(args.log_dir, 'vidshots', 'train')
    args.val_vid_dir = os.path.join(args.log_dir, 'vidshots', 'val')
    args.snapshot_dir = os.path.join(args.log_dir, 'modelshots')

    os.makedirs(log_txt_dir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)
    os.makedirs(args.train_vid_dir, exist_ok=True)
    os.makedirs(args.val_vid_dir, exist_ok=True)
    os.makedirs(args.snapshot_dir, exist_ok=True)
    
    
    sys.stdout = Logger(log_txt, sys.stdout)
    logger = SummaryWriter(log_dir=log_txt_dir)


    #multi gpu setting
    args.rank = args.rank * ngpus_per_node + gpu
    print("Use GPU: {} for training".format(args.rank)) 
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    
    #dataloader setting
    if args.data_dir == "/raid/DataSet/Moving_mnist":
        datasets = MovingMNIST(root=args.data_dir, train=True, download=False)
        img_dim = 64
    else:
        img_dim = 128
        datasets = SevirDataset(root=args.data_dir, train=True, download=False)
    data_sampler = torch.utils.data.distributed.DistributedSampler(datasets)
    args.num_workers = ngpus_per_node * 2
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=data_sampler, drop_last=True)
    #(2244, 128, 128, 25)
    #将time维度和channel维度合并
    model = Unet3D(
        dim = img_dim,
        channels = 1,
        out_dim= args.out_dim,
        cond_dim = args.cond_dim,
        dim_mults = (1, 2, 4, 4)
        )
    grad_vars = list(model.parameters())
    # model.load_state_dict(torch.load(os.path.join(args.log_dir, "modelshots/MM_diff_final.pt"), map_location='cuda:0'))
    model.cuda(args.rank)
    evaluator = Evaluator(seq_len= args.out_dim , value_scale=255.0)
    diffusion = GaussianDiffusion(
        model,
        objective = 'pred_noise',
        image_size = img_dim,
        timesteps = args.timesteps    # number of steps
        )
    
    grad_vars += list(diffusion.parameters())
    diffusion.load_state_dict(torch.load("/raid/Final_diffusion/logs/Sevir/5to5/D20241225T1911/modelshots/MM_diff_epoch100.pt"))
    diffusion.cuda(args.rank)

    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1000, gamma=0.1)
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    diffusion.train()

    global_step = 0
    epoch = args.start_epoch
    for epoch in range(args.max_epoch):
        gc.collect()
        torch.cuda.empty_cache()
        for step, (frames, diff_frames) in enumerate(dataloader):
            global_step += 1
            if img_dim == 128:
                frames = frames.permute(0, 3, 1, 2)
                diff_frames = diff_frames.permute(0, 3, 1, 2)
            optimizer.zero_grad()
            diff_frames = diff_frames.cuda(args.rank)#shape(B,19,W,H)
            frames = frames.cuda(args.rank)           
            loss, gframes = diffusion(frames[:, 0:args.cond_dim, :, :], diff_frames[:, (args.cond_dim - 1):(args.cond_dim + args.out_dim - 1), :, :], frames[:, args.cond_dim:(args.cond_dim+args.out_dim), :, :])             
            loss.backward()            
            optimizer.step()           
            lr_scheduler.step()
            logger.add_scalar('loss', loss.item(), global_step)
            
            # torch.cuda.empty_cache()
            # torch.no_grad()
            #保存训练过程中的生成样本
            if global_step % args.save_train_sample_fre_step == 0:
                with torch.no_grad():
                    print('Epoch: ', epoch, 'global_step: ', global_step, 'loss: ', loss.item())
                    new_im_arr_list = []
                    in_im_arr_list = []
                    for nf in range(gframes.shape[1]):
                        new_im = Image.new('L', (img_dim, img_dim))               
                        new_im.paste(Image.fromarray(sample_img(gframes[0, nf, :, :]), 'L'), (0, 0))  
                        new_im_arr_list.append(new_im)
                        in_im = Image.new('L', (img_dim, img_dim))
                        in_im.paste(Image.fromarray(sample_img(frames[0, args.cond_dim+nf, :, :]), 'L'), (0, 0))  
                        in_im_arr_list.append(in_im)

                    new_vid_name = 'train_'+'B' + format(epoch, "04d") + '_S' + format(global_step, "06d") + '_pred' + ".gif"
                    new_vid_file = os.path.join(args.train_vid_dir, new_vid_name)
                    imageio.mimsave(new_vid_file, new_im_arr_list)

                    in_vid_name = 'train_' + 'B' + format(epoch, "04d") + '_S' + format(global_step, "06d") + '_gt' + ".gif"
                    in_vid_file = os.path.join(args.train_vid_dir, in_vid_name)
                    imageio.mimsave(in_vid_file, in_im_arr_list)
            #保存验证过程的生成样本
            
           
                
            # if global_step % args.save_validate_sample_fre_step == 0:    
                
            #     # diffusion.eval() 
            #     with torch.no_grad():                               
            #         gframes = diffusion.sample(frames[:, 0:args.cond_dim, :, :], batch_size=args.batch_size)                               
            #         new_im_arr_list = []
            #         for nf in range(gframes.shape[1]):
            #             new_im = Image.new('L', (img_dim, img_dim))
            #             new_im.paste(Image.fromarray(sample_img(gframes[0, nf, :, :]), 'L'), (0, 0))  
            #             new_im_arr_list.append(new_im)

            #             in_im = Image.new('L', (img_dim, img_dim))
            #             in_im.paste(Image.fromarray(sample_img(frames[0, args.cond_dim+nf, :, :]), 'L'), (0, 0))  
            #             in_im_arr_list.append(in_im)
                    
            #         new_vid_name = 'val_' + 'B' + format(epoch, "04d") + '_S' + format(global_step, "06d") + '_pred' + ".gif"
            #         new_vid_file = os.path.join(args.val_vid_dir, new_vid_name)
            #         imageio.mimsave(new_vid_file, new_im_arr_list)
            #         in_vid_name = 'val_' + 'B' + format(epoch, "04d") + '_S' + format(global_step, "06d") + '_gt' + ".gif"
            #         in_vid_file = os.path.join(args.val_vid_dir, in_vid_name)
            #         imageio.mimsave(in_vid_file, in_im_arr_list)
                # add_memory(f"after_save_infering_E{epoch}_S{step}\n", torch.cuda.memory_allocated(0), args)

            #         '''
            #         evaluator.evaluate(frames[:args.batch_size ,args.cond_dim:args.cond_dim + args.out_dim,:,:], gframes)
            #         avg_csi_ori, avg_far_ori, avg_pod_ori, avg_hss_ori, avg_csi44_ori, avg_csi16_ori, mses_ori, mass_ori, rmses_ori, psnrs_ori, ssims_ori, crpss_ori, lpipss_ori = evaluator.done()
            #         with open(os.path.join(args.log_dir, 'Metircs_Score.txt'), 'a') as file:
            #             # 写入标题
            #             file.write("Unet3D \n ****************Unet3D****************\n***********Unet3D*************\n")
            #             # 写入计算后的分数
            #             file.write(f"final——lpips_score: {lpipss_ori}\n")
            #             file.write(f"final——ssim_score: {ssims_ori}\n")
            #             file.write(f"final——psnr_score: {psnrs_ori}\n")
            #             file.write(f"final——avg_c_score: {avg_csi_ori}\n")
            #             file.write(f"final——avg_f_score: {avg_far_ori}\n")
            #             file.write(f"final——avg_p_score: {avg_pod_ori}\n")
            #             file.write(f"final——avg_h_score: {avg_hss_ori}\n")
            #             file.write(f"final——avg_csi44_score: {avg_csi44_ori}\n")
            #             file.write(f"final——csi16_score: {avg_csi16_ori}\n\n")
            #         '''
            del loss, gframes, frames, diff_frames
            gc.collect()
            torch.cuda.empty_cache() 
        if epoch % args.save_model_fre_epoch==0: 
            save_name = 'modelshots/MM_diff_epoch' + str(epoch) + '.pt'
            torch.save(diffusion.state_dict(), os.path.join(args.log_dir, save_name))
    torch.save(diffusion.state_dict(),os.path.join(args.log_dir,'modelshots/final.pt'))





if __name__=='__main__':
    main()
