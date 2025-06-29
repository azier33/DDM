<h1 align=center>
Towards High-Fidelity and Temporal-Consistency Generation for Radar Echo Perception Via 
    
Differential Diffusion Model
</h1>

![fig0](./figs/semcity.gif)

> Towards High-Fidelity and Temporal-Consistency Generation for Radar Echo Perception Via Differential Diffusion Model
> 
> Junyi Li, Beibei Jin, Xiaohui Song, Jianye Wang, JinDong Li, Pengfei Zhang 

[Paper]() | [Project Page](https://github.com/azier33/DDM/)

## 📌 Setup
We test our code on Ubuntu 22.04 with a single RTX 4090 GPU.

### Environment 

    git clone https://github.com/azier33/DDM.git
    conda create -n DDM 
    conda activate DDM
    conda install -r requirements.txt

### Datasets
We use the Sevirlr and MovingMnist datasets. See [dataset.md](./data/dataset.md) for detailed data structure.

Please adjust the `dataset` folder path in training and test scripts.

## 📌 Training
Train the DDM Diffusion.
You can set dataset using `--dataset /path/to/sevir` or `--dataset path/to/MM`.

### Trianing

    python script/Trainer_diff.py

If you want to train an no diff method ,you can user the command follow.

    python scripts/Trainer_wo_diff.py


### Evaluation

For evaluation for the diff generation,

    python scripts/cal_diff_score.py

For evaluation for the no_diff generation,

    python scripts/cal_score.py

## 📌 Visualizing
![fig1](./data/Sevir/train_B0109_S436000_pred.gif)
![fig2](./data/Sevir/train_B0119_S475000_pred.gif)
![fig3](./data/Sevir/train_B0120_S476000_pred.gif)
![fig4](./data/Sevir/train_B0121_S480000_pred.gif)

To visualize the generation which obtain from model that we pre-trained,

    python scripts/cal_score.py

## 📌 Dataset
You can download the dataset that we used in the DDM samed to the [Prediff](https://github.com/gaozhihan/PreDiff) used. 

## Acknowledgement
The code is partly based on [video-diffusion-pytorch](https://github.com/lucidrains/video-diffusion-pytorch), [Diffuser](https://github.com/huggingface/diffusers) and [Prediff](https://github.com/gaozhihan/PreDiff). 

## 📌 License

This project is released under the MIT License.
