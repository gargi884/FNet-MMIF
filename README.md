# $\ell_0$-Regularized Sparse Coding-based Interpretable Network for Multi-Modal Image Fusion

Accetped by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)

[Paper](https://ieeexplore.ieee.org/document/11301656) [arxiv](https://arxiv.org/abs/2411.04519)  [Supplementary](https://drive.google.com/file/d/1Y0sRXkg2p226I8sxAlx3V-nk-gBX0LD1/view?usp=drive_link)

![](figs/demo.gif)

![](figs/sinet.png)

## Dependencies
- Python 3.9
- PyTorch 2.0.1
- [NVIDIA GPU + CUDA](https://developer.nvidia.com/cuda-downloads)

## Create environment and install packages
```
conda create -n FNet python=3.9
conda activate FNet
pip install -r requirements.txt
```

## Training Dataset

[MSRS](https://github.com/Linfeng-Tang/MSRS) (J. Ma, L. Tang, F. Fan, J. Huang, X. Mei, and Y. Ma, “SwinFusion: Cross-domain long-range learning for general image fusion via swin transformer,” IEEE/CAA Journal of Automatica Sinica, vol. 9, no. 7,
pp. 1200–1217, 2022.) is utilized to train FNet.

## Training

- For stage-I training, run the following script.

```
python train.py --opt = options/FNet/train_stage1.json
```
- For stage-II training, run the following script.

```
python train.py --opt = options/FNet/train_stage2.json
```
## Testing

- To generate gray-scale images, run the following script.

```
python test.py 
```
- To generate color images, run the following script.

```
python test.py --color
```

 
## Citation

If you find the code helpful in your research or work, please cite the following paper.

```
@ARTICLE{11301656,
  author={Panda, Gargi and Kundu, Soumitra and Bhattacharya, Saumik and Routray, Aurobinda},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={$\ell _{0}$-Regularized Sparse Coding-based Interpretable Network for Multi-Modal Image Fusion}, 
  year={2025},
  volume={},
  number={},
  pages={1-18},
  keywords={Convolutional codes;Feature extraction;Encoding;Training;Image fusion;Estimation;Sparse approximation;Thresholding (Imaging);Image coding;Reviews; $\ell _{0}$ -regularized convolutional sparse coding;LZSC block;multi-modal image fusion;inverse fusion},
  doi={10.1109/TPAMI.2025.3643898}}

```
