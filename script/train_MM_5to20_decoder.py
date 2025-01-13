import sys
sys.path.append('/raid')
import os
import argparse
import numpy as np
import random
gpu_devices = [4, 5]
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
from Final_diffusion.logs.MM.Diffusion_3d_WODE_2loss import Unet3D, GaussianDiffusion, Decoder, unnormalize_to_zero_to_one as unnormalize


def main():
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")
    parser = argparse.ArgumentParser(description="Diffusion")
    parser.add_argument("--random_seed", default=1234)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_epoch', default=1400, type=int)
    parser.add_argument("--log_dir", type=str, default='/raid/Final_diffusion/logs/MM/5to20/wo_Decoder_twoloss', help="Where to save logs of the model.")
    parser.add_argument('--data_dir', default="/raid/DataSet/Moving_mnist", type=str)
    parser.add_argument("--world_size", default=4)#单机多卡：代表有几块GPU
    parser.add_argument("--out_dim", default=15) #输出维度
    parser.add_argument("--cond_dim", default=5) #条件维度
    parser.add_argument('--num_workers', type=int, default=4, help='')#设置为显卡数量的2-4倍
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:2333', type=str, help='')
    parser.add_argument("--num_repeats", default=5, type=int)
    parser.add_argument("--timesteps", default=500, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument("--restore_from", default=None, type=str)
    parser.add_argument('--save-img-freq', default=5000, type=int,  metavar='N', help='save image frequency')
    parser.add_argument('--save-vid-freq', default=1000, type=int,  metavar='N', help='save video frequency')
    parser.add_argument('--set_start', default=False, type=bool)
    parser.add_argument('--print_freq', default=1000, type=int,  metavar='N', help='printing frequency')
    parser.add_argument('--save_model_fre_step', default=500, type=int,  metavar='N', help='save model frequency')
    parser.add_argument('--save_model_fre_epoch', default=100, type=int,  metavar='N', help='update model frequency')
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
    rec_img *= 255
    return np.array(rec_img, np.uint8)

def main_worker(gpu, ngpus_per_node, args):
    #summarywriter log dir setting
    log_txt_dir = os.path.join(args.log_dir, 'logs')
    os.makedirs(args.log_dir, exist_ok=True)
    log_txt = os.path.join(log_txt_dir,"B"+str(args.batch_size)+"E"+str(args.max_epoch)+".log")

    args.img_dir = os.path.join(args.log_dir, 'imgshots')
    args.vid_dir = os.path.join(args.log_dir, 'vidshots')
    args.snapshot_dir = os.path.join(args.log_dir, 'modelshots')
    os.makedirs(log_txt_dir, exist_ok=True)
    os.makedirs(args.img_dir, exist_ok=True)
    os.makedirs(args.vid_dir, exist_ok=True)
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
        datasets_test = MovingMNIST(root=args.data_dir, train=False, download=False)
        img_dim = 64
    else:
        img_dim = 128
        datasets = SevirDataset(root=args.data_dir, train=True, download=False)
        datasets_test = SevirDataset(root=args.data_dir, train=False, download=False)
    data_sampler = torch.utils.data.distributed.DistributedSampler(datasets)
    data_sampler_test = RandomSampler(datasets_test)
    args.num_workers = ngpus_per_node * 2
    dataloader = torch.utils.data.DataLoader(datasets, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=data_sampler)
    dataloader_test = torch.utils.data.DataLoader(datasets_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, sampler=data_sampler_test)
    #(2244, 128, 128, 25)
    #将time维度和channel维度合并
    decoder = Decoder(
        dim= 64, 
        out_dim= args.out_dim,
        con_dim= None
        )
    model = Unet3D(
        dim = 64,
        channels = 1,
        out_dim= args.out_dim,
        cond_dim = args.cond_dim,
        dim_mults = (1, 2, 4, 8)
        )
    # model.load_state_dict(torch.load(os.path.join(args.log_dir, "modelshots/MM_diff_final.pt"), map_location='cuda:0'))
    model.cuda(args.rank)
    evaluator = Evaluator(seq_len= args.out_dim , value_scale=255.0)
    diffusion = GaussianDiffusion(
        model,
        decoder,
        objective = 'pred_noise',
        image_size = img_dim,
        timesteps = args.timesteps    # number of steps
        )
    diffusion.load_state_dict(torch.load('/raid/Final_diffusion/logs/MM/5to20/Decoder_Block/modelshots/MM_diff_epoch600.pt', map_location='cuda:0'))
    diffusion.cuda(args.rank)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=5e-5, rho=0.9, eps=1e-06, weight_decay=0)
    numstep = 0
    for epoch in range(args.max_epoch):
        print('Epoch: ', epoch+1)
        for step, (frames, diff_frames) in enumerate(dataloader):
            if img_dim == 128:
                frames = frames.permute(0, 3, 1, 2)
            optimizer.zero_grad()
            diff_frames = diff_frames.cuda(args.rank)#shape(B,19,W,H)
            frames = frames.cuda(args.rank)
            loss = diffusion(frames[:, 0:args.cond_dim, :, :], diff_frames[:, (args.cond_dim - 1):(args.cond_dim + args.out_dim - 1), :, :], frames[:, args.cond_dim:(args.cond_dim+args.out_dim), :, :])
            logger.add_scalar('loss', loss, numstep)
            numstep += 1

            if step % args.save_model_fre_step == 0:
                print("epoch:", epoch, "Step: ", step+1, "Loss:", loss.item())
                print(loss.device)
            if step % args.save_model_fre_step == 0 and epoch % args.save_model_fre_epoch ==0:
                (test_frames, _) = next(iter(dataloader_test))
                test_frames = test_frames.cuda(args.rank)
                test_list = diffusion.sample(test_frames[:args.batch_size, 0:args.cond_dim, :, :], batch_size=args.batch_size)
                
                # generated_images = sample_img(generated_images)
                new_im_arr_list = []
                for nf in range(test_list.shape[1]):
                    new_im_arr_list.append(sample_img(test_list[2, nf, :, :]))
                
                evaluator.evaluate(test_frames[:args.batch_size ,args.cond_dim:args.cond_dim + args.out_dim,:,:], test_list)
                avg_csi_ori, avg_far_ori, avg_pod_ori, avg_hss_ori, avg_csi44_ori, avg_csi16_ori, mses_ori, mass_ori, rmses_ori, psnrs_ori, ssims_ori, crpss_ori, lpipss_ori = evaluator.done()
                with open(os.path.join(args.log_dir, 'Metircs_Score.txt'), 'a') as file:
                    # 写入标题
                    file.write("Unet3D \n ****************Unet3D****************\n***********Unet3D*************\n")
                    # 写入计算后的分数
                    file.write(f"final——lpips_score: {lpipss_ori}\n")
                    file.write(f"final——ssim_score: {ssims_ori}\n")
                    file.write(f"final——psnr_score: {psnrs_ori}\n")
                    file.write(f"final——avg_c_score: {avg_csi_ori}\n")
                    file.write(f"final——avg_f_score: {avg_far_ori}\n")
                    file.write(f"final——avg_p_score: {avg_pod_ori}\n")
                    file.write(f"final——avg_h_score: {avg_hss_ori}\n")
                    file.write(f"final——avg_csi44_ori: {avg_csi44_ori}\n")
                    file.write(f"final——csi16: {avg_csi16_ori}\n\n")
                new_vid_name = 'B' + format(epoch, "04d") + '_S' + format(step, "06d") + '_' + ".gif"
                new_vid_file = os.path.join(args.vid_dir, new_vid_name)
                imageio.mimsave(new_vid_file, new_im_arr_list)

                
            loss.backward()
            optimizer.step()

        if epoch%50==0: 
            # imageio.mimsave(new_vid_file, new_im_arr_list)
            save_name = 'modelshots/MM_diff_epoch' + str(epoch) + '.pt'
            torch.save(diffusion.state_dict(), os.path.join(args.log_dir, save_name))

    torch.save(diffusion.state_dict(),os.path.join(args.log_dir,'modelshots/MM_diff_final.pt'))





if __name__=='__main__':
    main()