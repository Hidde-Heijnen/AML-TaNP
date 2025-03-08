# TaNP

The source code is based on the paper: ”Task-adaptive Neural Process for User Cold-Start Recommendation" accepted in WWW 2021.

For the MPhil MLMI at the university of Cambridge, we've improved the code to support multiple datasets out of the box, and implemented the missing default decoder.

```
@inproceedings{lincsr2021,
  title={Task-adaptive Neural Process for User Cold-Start Recommendation},
  author={Lin, Xixun and Wu, Jia and Zhou, Chuan and Pan, Shirui and Cao, Yanan and Wang, Bin},
  booktitle={ACM International World Wide Web Conferences (WWW)},
  year={2021}
}
```

## Requirements

python == 3.6.3  
 torch == 1.1  
 numpy == 1.17  
 scipy == 1.3.1  
 scikit-learn == 0.21.3

## Datasets

MoviesLens-1M is provided by MeLU, and you can find it from [here](https://github.com/hoyeoplee/MeLU).  
 Last.FM is provided by MKR, and you can find it from [here](https://github.com/hwwang55/MKR).  
 Gowalla is provided by NGCF, and you can find it from [here](https://github.com/xiangwang1223/neural_graph_collaborative_filtering).

## How to use

The utils/loader is used for data preprocessing, and you can customize this part for your own data.  
 Use the following commands for running a sub-dataset:  
 unzip data.zip file
run:

```bash
chmod +x TaNP/clone_movielens_dataset.sh
TaNP/clone_movielens_dataset.sh
```

zsh train.sh # Training with default hyper-parameters.
