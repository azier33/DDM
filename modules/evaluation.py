import math
import os
import numpy as np
import torch
from einops import rearrange, repeat
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import math
from scipy.special import gamma
import numpy as np
from typing import Type, Any, Callable, Union, List, Optional
import skimage.io
import scipy.io
from tqdm import trange
from sklearn.model_selection import GridSearchCV,cross_validate,ShuffleSplit
from sklearn.svm import SVR
from scipy import stats

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 定义一个感知损失类，使用预训练的 AlexNet
class PerceptualLossAlexNet(nn.Module):
    def __init__(self, pretrained=True):
        super(PerceptualLossAlexNet, self).__init__()
        
        # 加载预训练的 ResNet-50 模型
        resnet = models.resnet50(pretrained=pretrained)
        
        # 修改第一层卷积层，适应输入 15 通道的图像
        resnet.conv1 = nn.Conv2d(15, resnet.conv1.out_channels, kernel_size=resnet.conv1.kernel_size,
                                 stride=resnet.conv1.stride, padding=resnet.conv1.padding, bias=resnet.conv1.bias)
        
        # ResNet-50 的特征提取部分是去掉了最后的全连接层
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # 去掉最后的fc层和avgpool层
        
        # 冻结所有层的参数，不进行更新
        for param in self.resnet.parameters():
            param.requires_grad = False
    
    def forward(self, x, y):
        # 提取 x 和 y 的 ResNet 特征
        x_feat = self.resnet(x)
        y_feat = self.resnet(y)
        
        # 计算感知损失：L2 损失（均方误差）
        return F.mse_loss(x_feat, y_feat)


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr
# PSNR computation
def calculate_ssim_psnr(img1, img2):
    img1 = img1.to('cuda')
    img2 = img2.to('cuda')

    # 自定义 SSIM 实现
    ssim_value = ssim(img1, img2, window_size=11, val_range=1.0)

    # 计算 PSNR
    mse = F.mse_loss(img1, img2)
    # print("####################mse###################:", mse)
    epsilon = 1e-10
    psnr = 10 * torch.log10(1 / (mse + epsilon))

    return ssim_value.item(), psnr.item()

class FIDEvaluation:
    def __init__(
        self,
        dl,
        sampler,
        channels=3,
        accelerator=None,
        stats_dir="./results",
        device="cuda",
        num_fid_samples=50000,
        inception_block_idx=2048,
    ):
        self.n_samples = num_fid_samples
        self.device = device
        self.channels = channels
        self.dl = dl
        self.sampler = sampler
        self.stats_dir = stats_dir
        self.print_fn = print if accelerator is None else accelerator.print
        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        self.inception_v3 = InceptionV3([block_idx]).to(device)
        self.dataset_stats_loaded = False

    def calculate_inception_features(self, samples):
        if self.channels == 1:
            samples = repeat(samples, "b 1 ... -> b c ...", c=3)
        #samples(shape[1,3,64,64])

        self.inception_v3.eval()
        features = self.inception_v3(samples)[0]#features(shape[1,2048,1,1])
        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        return features

    def load_or_precalc_dataset_stats(self):
        path = os.path.join(self.stats_dir, "dataset_stats")
        try:
            ckpt = np.load(path + ".npz")
            self.m2, self.s2 = ckpt["m2"], ckpt["s2"]
            self.print_fn("Dataset stats loaded from disk.")
            ckpt.close()
        except OSError:
            num_batches = int(math.ceil(self.n_samples / self.batch_size))
            stacked_real_features = []
            self.print_fn(
                f"Stacking Inception features for {self.n_samples} samples from the real dataset."
            )
            for _ in tqdm(range(num_batches)):
                try:
                    real_samples = next(self.dl)
                except StopIteration:
                    break
                real_samples = real_samples.to(self.device)
                real_features = self.calculate_inception_features(real_samples)
                stacked_real_features.append(real_features)
            stacked_real_features = (
                torch.cat(stacked_real_features, dim=0).cpu().numpy()
            )
            m2 = np.mean(stacked_real_features, axis=0)
            s2 = np.cov(stacked_real_features, rowvar=False)
            np.savez_compressed(path, m2=m2, s2=s2)
            self.print_fn(f"Dataset stats cached to {path}.npz for future use.")
            self.m2, self.s2 = m2, s2
        self.dataset_stats_loaded = True

    @torch.inference_mode()
    def fid_score(self,image_d, image_r):
        # self.load_or_precalc_dataset_stats()
        # self.print_fn(
        #     f"Stacking Inception features for {self.n_samples} generated samples."
        # )
        # image_d = torch.unsqueeze(image_d, dim=1)
        # image_r = torch.unsqueeze(image_r, dim=1)
        image_d = torch.tensor(image_d, dtype=torch.float32)
        image_r = torch.tensor(image_r, dtype=torch.float32)
        fake_features = self.calculate_inception_features(image_d)
        m1 = torch.mean(fake_features, dim =1, keepdim = True)
        s1 = torch.cov(fake_features)
        fake_features_r = self.calculate_inception_features(image_r)
        m2 = torch.mean(fake_features_r, dim =1, keepdim = True)
        s2 = torch.cov(fake_features_r)

        m1_cpu = m1.cpu().detach().numpy()
        s1_cpu = s1.cpu().detach().numpy()
        m2_cpu = m2.cpu().detach().numpy()
        s2_cpu = s2.cpu().detach().numpy()        
        # 然后使用转换后的 NumPy 数组计算 Frechet Distance
        fid_score = calculate_frechet_distance(m1_cpu, s1_cpu, m2_cpu, s2_cpu)
        return fid_score

class BRISUQE:
    def __init__(
        self,
        batch_size = 5,
        channels=3,
        accelerator=None,
        stats_dir="./results",
        device="cuda",
        num_fid_samples=50000,
        inception_block_idx=2048,
    ):
        self.batch_size = batch_size
        self.n_samples = num_fid_samples
        self.device = device
        self.channels = channels
        self.stats_dir = stats_dir
        self.print_fn = print if accelerator is None else accelerator.print
        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        self.inception_v3 = InceptionV3([block_idx]).to(device)
        self.dataset_stats_loaded = False
    def gaussian_2d_kernel(self, kernel_size, sigma):
        kernel = torch.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        if sigma == 0:
            sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        sum_val = 0
        for i in range(0, kernel_size):
            for j in range(0, kernel_size):
                x = i - center
                y = j - center
                kernel[i, j] = math.exp(-(x ** 2 + y ** 2) / s)
                sum_val += kernel[i, j]
        sum_val = 1 / sum_val
        return kernel * sum_val

    def estimate_GGD_parameters(self, vec):
        vec=vec.view(-1)
        gam =np.arange(0.2,10.0,0.001)
        r_gam = (gamma(1/gam)*gamma(3/gam))/((gamma(2/gam))**2)#根据候选的γ计算r(γ)
        sigma_sq=torch.mean((vec)**2).to('cpu').numpy()
        E=torch.mean(torch.abs(vec)).to('cpu').numpy()
        # print("检查一下E是什么玩意儿:",torch.abs(vec))
        r=sigma_sq/(E**2)#根据sigma^2和E计算r(γ)
        diff=np.abs(r-r_gam)
        gamma_param=gam[np.argmin(diff, axis=0)]
        return gamma_param,sigma_sq

    def estimate_AGGD_parameters(self, vec):
        vec = vec.to('cpu').numpy()
        alpha =np.arange(0.2,10.0,0.001)#产生候选的α
        r_alpha=((gamma(2/alpha))**2)/(gamma(1/alpha)*gamma(3/alpha))#根据候选的γ计算r(α)
        sigma_l=np.sqrt(np.mean(vec[vec<0]**2))
        sigma_r=np.sqrt(np.mean(vec[vec>0]**2))
        gamma_=sigma_l/sigma_r
        u2=np.mean(vec**2)
        m1=np.mean(np.abs(vec))
        r_=m1**2/u2
        R_=r_*(gamma_**3+1)*(gamma_+1)/((gamma_**2+1)**2)
        diff=(R_-r_alpha)**2
        alpha_param=alpha[np.argmin(diff, axis=0)]
        const1 = np.sqrt(gamma(1 / alpha_param) / gamma(3 / alpha_param))
        const2 = gamma(2 / alpha_param) / gamma(1 / alpha_param)
        eta =(sigma_r-sigma_l)*const1*const2
        return alpha_param,eta,sigma_l**2,sigma_r**2


    def brisque_feature(self, imdist:Type[Union[torch.Tensor,np.ndarray]],device='cuda'):

        #算法需要输入为灰度图像，像素值0-255
        if type(imdist)==np.ndarray:
            assert imdist.ndim==2 or imdist.ndim==3
            if imdist.ndim==2:
                imdist=torch.from_numpy(imdist).unsqueeze(0).unsqueeze(0)
            else:
                imdist = torch.from_numpy(imdist).unsqueeze(0)
        # input (Batch,1,H,W)
        assert imdist.dim()==4
        assert imdist.shape[1]==1 or imdist.shape[1]==3

        if torch.max(imdist)<=1:
            imdist = imdist * 255
        # RGB to Gray
        if imdist.shape[1]==3:
            imdist=imdist[:,0,:]*0.299+imdist[:,1,:]*0.587+imdist[:,2,:]*0.114
        # GPU is much much faster
        if 'cuda' in device:
            imdist=imdist.half().to(device)
        elif device=='cpu':
            imdist=imdist.float().to(device)
        else:
            raise ValueError('cpu or cuda',device)

        # 算法主体
        scalenum = 2
        window=self.gaussian_2d_kernel(7,7/6).unsqueeze(0).unsqueeze(0).float().to(device)
        if 'cuda' in device:
            window=window.half()

        feat=np.zeros((18*scalenum,))
        for i in range(scalenum):
            mu=F.conv2d(imdist,window,stride=1,padding=3)
            mu_sq=mu*mu
            sigma=torch.sqrt(torch.abs(F.conv2d(imdist*imdist,window,stride=1,padding=3)-mu_sq))
            structdis = (imdist - mu) / (sigma + 1)
            del mu, mu_sq,sigma
            feat[i*18],feat[i*18+1] = self.estimate_GGD_parameters(structdis)

            shifts = [(0,1),(1,0),(1,1),(-1,1)]
            for itr_shift in range(4):
                shifted_structdis=structdis.roll(shifts[itr_shift],(2,3))
                pair=structdis*shifted_structdis
                pair=pair.view(-1)
                feat[i*18+2+itr_shift*4],feat[i*18+3+itr_shift*4],feat[i*18+4+itr_shift*4],feat[i*18+5+itr_shift*4]=self.estimate_AGGD_parameters(pair)

            imdist=F.interpolate(imdist,scale_factor=(0.5,0.5),mode='bilinear')
        return feat
    
    def compute_score(self, img):
        # 训练
        X=[]
        Y=[]
        for img_name,img_dmos in train_set:
            img=skimage.io.imread(os.path.join(root_path,img_name),as_gray=True)#读图像
            feat=self.brisque_feature(img)#提特征
            X.append(feat)
            Y.append(img_dmos)
        X=np.array(X)
        Y=np.array(Y)
        svr = SVR(kernel='rbf', C=C, gamma=Gamma)#训练SVR
        svr.fit(X,Y)

        # 测试
        img=skimage.io.imread(os.path.join(test_img_path),as_gray=True)
        feat=self.brisque_feature(img)
        predicted_score=svr.predict(feat)


import clip
from PIL import Image
class temporal_consistancy():
    def __init__(self,
                 chanalls = 3,
                 batch_size = 2,
                 device = "cuda:0"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.to(self.device)
    # 定义函数来计算 CLIP 图像特征
    def compute_clip_features(self,frames):
        brisuqe = BRISUQE()
        features = []
        for num in range(frames.shape[1]):
            # image = frames[num, :, :]
            # image = torch.unsqueeze(image, dim = 0)
            # image = image.repeat(3, 1, 1)
            # frames shape(1, 5, 64, 64)
            with torch.no_grad():
                frames_in = frames[:, num, :, :]
                frames_in = torch.unsqueeze(frames_in, dim=1)
                feature = brisuqe.brisque_feature(imdist = frames_in)
                # feature(shape(36, ))
            features.append(feature)
        return features

    # 计算余弦相似度
    def compute_cosine_similarity(self, features):
        similarities = []
        for i in range(len(features) - 1):
            feature1 = features[i].squeeze()
            feature2 = features[i + 1].squeeze()
            feature1 = torch.from_numpy(feature1)
            feature2 = torch.from_numpy(feature2)
            similarity = torch.nn.functional.cosine_similarity(feature1, feature2, dim=0)
            similarities.append(similarity.item())
        return similarities

    # 计算视频的时序一致性
    def compute_temporal_consistency(self, video_frames):
        # 提取图像特征
        clip_features = self.compute_clip_features(video_frames)
        # 计算余弦相似度
        similarities = self.compute_cosine_similarity(clip_features)
        # 计算平均余弦相似度
        mean_similarity = np.mean(similarities)
        return mean_similarity

#Test of Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def sample_img(rec_img, idx=0):
    rec_img = rec_img.data.cpu().numpy().copy()

    #rec_img += np.array(MEAN)/255.0
    #rec_img[rec_img < 0] = 0
    #rec_img[rec_img > 1] = 1
    rec_img *= 255
    return np.array(rec_img, np.uint8)
fid = FIDEvaluation(
    channels=1,
    dl = 3,
    sampler = 1
)
temporal = temporal_consistancy(
    batch_size= 2,
    chanalls = 5,
    device = 2
)
# if __name__ == '__main__':
#     import sys
#     import torch.distributed as dist
#     sys.path.append('.')
#     # 添加项目根目录到 Python 路径
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     sys.path.append(project_root)
#     from DataSet.MovingMnist_.generator_mm import MovingMNISTAdvancedIterator
#         #dataloader setting
#     from sevir import SevirDataset
#     #multi gpu setting
#     nf_temporal = 0
#     nf_frames = 2000
#     nf = 0
#     fid_score_average = 0
#     mnist_generator = MovingMNISTAdvancedIterator(distractor_num=6)
#     SevirDataset.__init__(self=SevirDataset,root="/raid/DataSet/SevirLr/data/vil",train= True)
#     file_list = SevirDataset.get_file_list(self=SevirDataset,is_train=True)
#     frames, diff_frames = SevirDataset.load_data(self=SevirDataset, file_list=file_list)#frames(shape[7964, 128, 128, 25])
#     print("dasdasdas",frames.shape)
#     for nf in range(nf_frames):
#         # frames,_ = mnist_generator.sample(5, 5)#frames:shape(n,b,c=1,w,h)
#         frames = torch.from_numpy(frames)
#         frames = frames[:, :, 0, :, :]
#         frames = frames.permute(1, 0, 2, 3)
#         generated_images = sample_img(frames)
#         bs_temporal = 0
#         fid_ever_epoch = 0
#         for bs in range(generated_images.shape[0]):
#             tf_temporal = 0
#             frames_for_temporal = torch.from_numpy(generated_images[bs, :, :, :])
#             frames_for_temporal = frames_for_temporal.float()
#             m2 = torch.mean(frames_for_temporal[0, :, :],keepdim = True)
#             print("均值：",m2)
#             s2 = torch.cov(frames_for_temporal)
#             print("均值：",m2,"方差：",s2)
#             frames_for_temporal = torch.unsqueeze(frames_for_temporal, dim = 0)
#             frames_for_temporal = frames_for_temporal.to("cuda")
#             fid_every_batch = 0
#             for ff in range(frames_for_temporal.shape[1]-1):
#                 fid_score = fid.fid_score(frames_for_temporal[:,ff,:,:],frames_for_temporal[:,ff+1,:,:])
#                 fid_every_batch +=fid_score
#             fid_ever_epoch+=(fid_every_batch/(ff+1))
#             tf_temporal = temporal.compute_temporal_consistency(frames_for_temporal)
#             # brisuqe_score = evaluate.brisque_feature(tensor_niqe)
#             # print("one of video niqe score:",tf_niqe)
#             bs_temporal+=tf_temporal
#         bs_temporal = bs_temporal/(bs+1)
#         fid_ever_epoch=fid_ever_epoch/(bs+1)
#         print("batch of FID score:",fid_ever_epoch)
#         print("batch of video niqe score:",bs_temporal,"bs:",bs)
#         fid_score_average+=fid_ever_epoch
#         nf_temporal+=bs_temporal
#     fid_score_average = fid_score_average/(nf+1)
#     nf_temporal = nf_temporal/(nf+1)
#     print("average of all sequences temporal score:",nf_temporal)
#     print("average of all sequences fid score:",fid_score_average)

if __name__ == '__main__':
    import sys
    import torch.distributed as dist
    from JunYi_Modify.train_code.moving_mnist import MovingMNIST
    sys.path.append('.')
    # 添加项目根目录到 Python 路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(project_root)
    from DataSet.MovingMnist_.generator_mm import MovingMNISTAdvancedIterator
        #dataloader setting
    from sevir import SevirDataset
    #multi gpu setting
    nf_temporal = 0
    nf_frames = 2000
    nf = 0
    fid_score_average = 0
    # dataset = MovingMNIST(root="/raid/LFDM/dataset/Moving_mnist", train=False, download=False)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False, num_workers=2)
    # datas = []
    # for step, (frames, diff_frames) in enumerate(dataloader):  
    #     datas.append((frames))
    # frames_o = torch.cat(datas, dim=0) #frames(1000, 20, 64, 64)

    mnist_generator = MovingMNISTAdvancedIterator(distractor_num=6)
    SevirDataset(root="/raid/DataSet/SevirLr/data/vil",train= True)
    dataset = SevirDataset(root="/raid/DataSet/SevirLr/data/vil", train=True, download=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5, shuffle=False, num_workers=2)
    datas = []
    for step, (frames, diff_frames) in enumerate(dataloader):  
        datas.append((frames))
    frames_o = torch.cat(datas, dim=0) #frames[7964, 128, 128, 25]
    # file_list = SevirDataset.get_file_list(self=SevirDataset,is_train=True)
    # frames_o, diff_frames = SevirDataset.load_data(self=SevirDataset, file_list=file_list)#frames(shape[7964, 128, 128, 25])
    for nf in range(nf_frames):
        start_idx = np.random.randint(0, 994)
        frames = frames_o[start_idx:start_idx+5, :, :, :20]
        # frames,_ = mnist_generator.sample(5, 5)#frames:shape(n,b,c=1,w,h)
        frames = frames.permute(0, 3, 1, 2)
        #generated_images = sample_img(frames)
        generated_images = frames
        bs_temporal = 0
        fid_ever_epoch = 0
        for bs in range(generated_images.shape[0]):
            tf_temporal = 0
            #frames_for_temporal = torch.from_numpy(generated_images[bs, :, :, :])
            frames_for_temporal = generated_images[bs, :, :, :]
            frames_for_temporal = frames_for_temporal.float()
            # m2 = torch.mean(frames_for_temporal[0, :, :])
            # print("均值：",m2)
            # s2 = torch.var(frames_for_temporal[0, :, :])
            # print("均值：",m2,"方差：",s2)
            frames_for_temporal = torch.unsqueeze(frames_for_temporal, dim = 0)
            frames_for_temporal = frames_for_temporal.to("cuda")
            fid_every_batch = 0
            for ff in range(frames_for_temporal.shape[1]-1):
                # print("max:",torch.max(frames_for_temporal[:,ff,:,:]),"min:",torch.min(frames_for_temporal[:,ff,:,:]))
                fid_score = fid.fid_score(frames_for_temporal[:,ff,:,:],frames_for_temporal[:,ff+1,:,:])
                fid_every_batch +=fid_score
            fid_ever_epoch+=(fid_every_batch/(ff+1))
            tf_temporal = temporal.compute_temporal_consistency(frames_for_temporal)
            bs_temporal+=tf_temporal
        bs_temporal = bs_temporal/(bs+1)
        fid_ever_epoch=fid_ever_epoch/(bs+1)
        print("batch of FID score:",fid_ever_epoch)
        print("batch of video niqe score:",bs_temporal,"bs:",bs)
        fid_score_average+=fid_ever_epoch
        nf_temporal+=bs_temporal
    fid_score_average = fid_score_average/(nf+1)
    nf_temporal = nf_temporal/(nf+1)
    print("average of all sequences temporal score:",nf_temporal)
    print("average of all sequences fid score:",fid_score_average)