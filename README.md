<div align="center">
<h2><center>👉 Three-Dimensional Trajectory Prediction with 3DMoTraj Dataset</h2>

[Hao Zhou](), [Xu Yang](), [Mingyu Fan](), [Lu Qi](), [Ming-Hsuan Yang](), [Fei Luo]()


<a href='https://openreview.net/forum?id=jkVH7nLzUR&noteId=E8xLdmkLSv'><img src='https://img.shields.io/badge/Conference-ICML-orange'></a> 

</div>

<div align=center>
<img src="architecture.pdf" alt="3DMoTraj" align="middle"/>
</div>

This repo is the official PyTorch implementation for ICML 2025 paper [**3DMoTraj**](https://openreview.net/forum?id=jkVH7nLzUR&noteId=E8xLdmkLSv).

##  Installation 🛠️
Download the repository:
```bash
git clone https://github.com/zhouhao94/3DMoTraj.git
```
and set up the environment:
```bash
torch==2.1.1
scipy==1.9.3
numpy==1.23.5
pandas==1.4.3
```

## Dataset 
Build a folder ```dataset```:
```bash
cd 3DMoTraj
mkdir dataset
```
download [3DMoTraj](xxx) dataset and place them as follows:
```bash
./
|-- dataset
-- |-- 3Dscene1
-- |-- 3Dscene2
-- |-- 3Dscene3
-- |-- 3Dscene4
-- |-- 3Dscene5
-- |-- 3Dscene6
-- |-- 3Dscene7
-- |-- 3Dscene8
   |   |-- train
   |   |   |-- *.txt
   |   |-- val
   |   |   |-- *.txt
   |   |-- test
   |   |   |-- *.txt
```

## Training
For training new models on 3DMoTraj dataset, execute the following command:
```bash
cd 3DMoTraj
bash train.sh
```
and the checkpoints will be saved in ```saved_models```.

## Evaluation
For evaluation the trained models on 3DMoTraj dataset, execute the follwing command:
```bash
cd 3DMoTraj
bash test.sh
```

## CheckPoints
For reproduction results on 3DMoTraj dataset, download our trained [checkpoint](xxx), put them in ```saved_models```, and execute the evaluation command.

## Bibtex 
🌟 If you find our work helpful, please leave us a star and cite our paper. Thank you!
```
@inproceedings{
zhou2025threedimensional,
title={Three-Dimensional Trajectory Prediction with 3{DM}oTraj Dataset},
author={Hao Zhou and Xu Yang and Mingyu Fan and Lu Qi and Xiangtai Li and Ming-Hsuan Yang and Fei Luo},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=jkVH7nLzUR}
}
```
## Acknowledgments
This work is based on [LBEBM](https://github.com/bpucla/lbebm). Thanks for their great work.
