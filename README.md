# MGCA-Net implementation

This repository provides the official PyTorch implementation of MGCA-Net (Multi-Graph Contextual Attention Network), a deep learning framework designed for robust two-view correspondence learning and outlier rejection. MGCA-Net leverages multi-graph modeling and contextual attention mechanisms to achieve accurate and reliable feature matching, even under challenging scenarios with high outlier ratios and complex geometric deformations.

If you find this project useful, please cite our work. Your support is greatly appreciated.


## Requirements

Please use Python 3.6, opencv-contrib-python (3.12.0) and Pytorch (>= 2.6.0). Other dependencies should be easily installed through pip or conda.

### Generate training and testing data
First, download the YFCC100M and SUN3D datasets from [this Google Drive link](https://drive.google.com/drive/folders/1RbBWKy-6QdbKofZGHGhTTRGNlv0rgtVi).

*Note: This download link is originally provided by the OANet project.*

You will need the following files:

- **YFCC100M**: `raw_data_yfcc.tar.gz` (parts 0–8)
- **SUN3D Test Set**: `raw_sun3d_test.tar.gz` (parts 0–2)
- **SUN3D Train Set**: `raw_sun3d_train.tar.gz` (parts 0–63)

Please ensure all parts are downloaded before extracting the datasets.


After downloading all parts, use the provided script to merge and extract the dataset files. For example, to merge YFCC100M:
### Merge and Extract YFCC100M
```bash
bash merge_chunks.sh raw_data_yfcc.tar merged_yfcc.tar.gz 0 8
tar -xzvf merged_yfcc.tar.gz
```

### Merge and Extract SUN3D
```bash
bash merge_chunks.sh raw_sun3d_train.tar.gz merged_sun3d_train.tar.gz 0 63
tar -xzvf merged_sun3d_train.tar.gz
bash merge_chunks.sh raw_sun3d_test.tar.gz merged_sun3d_test.tar.gz 0 2
tar -xzvf merged_sun3d_test.tar.gz
```


Then generate matches for YFCC100M and SUN3D with SIFT.
```bash
cd ../dump_match
python extract_feature.py
python yfcc.py
python extract_feature.py --input_path=../raw_data/sun3d_test
python sun3d.py
```
Generate SUN3D training data if you need by following the same procedure and uncommenting corresponding lines in `sun3d.py`.



### Test pretrained model

We provide pretrained MGCA-Net models trained on the YFCC100M and SUN3D datasets. You can use these models to quickly evaluate the network’s performance on benchmark test sets or your own data formatted in the same way.

```bash
python main.py --run_mode=test --model_path=../weights/yfcc100m/ --res_path=./results/yfcc/test
```
You can change the default settings for test in `./core/config.py`.

### Train model on YFCC100M or SUN3D

After generating dataset for YFCC100M, run the tranining script.
```bash
cd ./core 
python main.py
```

You can change the default settings for network structure and training process in `./core/config.py`.

### Training with Custom Local Features or Datasets
The pre-trained models in this repository are trained using SIFT features. If you wish to use MGCA-Net with other local features (e.g., RootSIFT, SuperPoint) or your own dataset, it is recommended to retrain the model for optimal performance.

To prepare your own data or local features, you can follow the example scripts provided in the ```./dump_match``` directory for dataset generation and formatting.

## Acknowledgement
This code is borrowed from [OANet](https://github.com/zjhthu/OANet), [BCLNet](https://github.com/guobaoxiao/BCLNet), and [MSGSA](https://github.com/shuyuanlin/MSGSA). If using the part of code related to data generation, testing and evaluation, please cite these papers.

```
@inproceedings{zhang2019learning,
  title={Learning two-view correspondences and geometry using order-aware network},
  author={Zhang, Jiahui and Sun, Dawei and Luo, Zixin and Yao, Anbang and Zhou, Lei and Shen, Tianwei and Chen, Yurong and Quan, Long and Liao, Hongen},
  booktitle={Proceedings of the IEEE International Conference on Computer Cision},
  pages={5845--5854},
  year={2019}
}
@InProceedings{miao2024bclnet,
  title={Bclnet: Bilateral consensus learning for two-view correspondence pruning},
  author={Miao, Xiangyang and Xiao, Guobao and Wang, Shiping and Yu, Jun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={5},
  pages={4225--4232},
  year={2024}
}
@Article{lin2024multi,
  author    = {Lin, Shuyuan and Chen, Xiao and Xiao, Guobao and Wang, Hanzi and Huang, Feiran and Weng, Jian},
  journal   = {IEEE Transactions on Image Processing},
  title     = {Multi-Stage Network with Geometric Semantic Attention for Two-View Correspondence Learning},
  year      = {2024},
  pages     = {3031--3046},
  volume    = {33},
  publisher = {IEEE},
}
```

## Citation
```
@article{lin2025mgcanet,
  title = {MGCA-Net: Multi-graph contextual attention network for two-view correspondence learning},
  author = {Shuyuan Lin, Mengtin Lo, Haosheng Chen, Yanjie Liang, Qiangqiang Wu},
  journal = {Proceedings of the International Joint Conference on Artificial Intelligence},
  pages = {1--9},
  year = {2025},
}
```
