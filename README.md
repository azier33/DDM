<h1 align=center>
SemCity: Semantic Scene Generation 

with Triplane Diffusion
</h1>

![fig0](./figs/semcity.gif)

> Towards High-Fidelity and Temporal-Consistency Generation for Radar Echo Perception Via Differential Diffusion Model
> 
> Junyi Li, Beibei Jin, Xiaohui Song, Jianye Wang, JinDong Li, Pengfei Zhang 

[Paper](https://arxiv.org/abs/2403.07773) | [Project Page](https://sglab.kaist.ac.kr/SemCity)

## 📌 Setup
We test our code on Ubuntu 20.04 with a single RTX 4090 GPU.

### Environment 

    git clone https://github.com/zoomin-lee/SemCity.git
    conda create -n CCDM 
    conda activate CCDM
    conda install -r requirements.txt

### Datasets
We use the Sevirlr and MovingMnist datasets. See [dataset.md](./dataset/dataset.md) for detailed data structure.

Please adjust the `sequences` folder path in `dataset/path_manager.py`.

## 📌 Training
Train the Triplane Autoencoder and then the Triplane Diffusion.
You can set dataset using `--dataset kitti` or `--dataset carla`.
In/outpainting and semantic scene completion refinement are only possible with SemanticKITTI datasets.

### Triplane Autoencoder

    python scripts/train_ae_main.py --save_path exp/ae

When you are finished training the triplane autoencoder, save the triplane. 
The triplane is a proxy representation of the scene for triplane diffusion training.

    python scripts/save_triplane.py --data_name voxels --save_tail .npy --resume {ae.pt path}

If you want to train semantic scene completion refinement, also save the triplane of the result of the ssc method (e.g. monoscene).

    python scripts/save_triplane.py --data_name monoscene --save_tail _monoscene.npy --resume {ae.pt path}

### Triplane Diffusion

For training for semantic scene generation or in/outpainting,

    python scripts/train_diffusion_main.py --triplane_loss_type l2 --save_path exp/diff

For training semantic scene completion refinement,

    python scripts/train_diffusion_main.py --ssc_refine --refine_dataset monoscene --triplane_loss_type l1 --save_path exp/diff

## 📌 Sampling
In `dataset/path_manager.py`, adjust the triplane autoencoder and triplane diffusion `.pt` paths to `AE_PATH` and `DIFF_PATH`.

![fig1](./figs/semcity.png)

To generate 3D semantic scene like `fig(a)`,

    python sampling/generation.py --num_samples 10 --save_path exp/gen

For semantic scene completion refinement like `fig(b)`,

    python sampling/ssc_refine.py --refine_dataset monoscene --save_path exp/ssc_refine

Currently, we're only releasing the code to outpaint twice the original scene.

    python sampling/outpainting.py --load_path figs/000840.label --save_path exp/out

For inpainting, as in `fig(d)`, you can define the region (top right, top left, bottom right, bottom left) where you want to regenerate.

    python sampling/inpainting.py --load_path figs/000840.label --save_path exp/in

## 📌 Evaluation

We render our scene with [pyrender](https://pyrender.readthedocs.io/en/latest/index.html) and then evaluate it using [torch-fidelity](https://github.com/toshas/torch-fidelity). 

## Acknowledgement
The code is partly based on [guided-diffusion](https://github.com/openai/guided-diffusion), [Sin3DM](https://github.com/Sin3DM/Sin3DM) and [scene-scale-diffusion](https://github.com/zoomin-lee/scene-scale-diffusion). 

## Bibtex
If you find this code useful for your research, please consider citing our paper:

    @inproceedings{lee2024semcity,
        title={SemCity: Semantic Scene Generation with Triplane Diffusion},
        author={Lee, Jumin and Lee, Sebin and Jo, Changho and Im, Woobin and Seon, Juhyeong and Yoon, Sung-Eui},
        booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
        year={2024}
    }

## 📌 License

This project is released under the MIT License.
