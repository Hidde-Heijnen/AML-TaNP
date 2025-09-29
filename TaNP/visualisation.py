import os
from datetime import datetime
import time
import numpy as np
import random
import argparse
import pickle
import torch
import uuid
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import json
from utils.loader import Preprocess
from TaNP import Trainer
from TaNP_training import training
from utils import helper
from eval import testing
from utils.movielens_dataprep import generate_movielens
from utils.lastfm_hetrec_dataprep import generate_lastfm_hetrec
import re
import json
import matplotlib.pyplot as plt
import ast
from matplotlib.ticker import FormatStrFormatter


def evaluating_model(opt, model_file_path):
    model_filename = model_file_path
    # dataset preprocessing
    dataset_dir = f"{opt['data_dir']}/{opt['dataset']}" 

    ## already generated dataset
    # if opt['dataset'] == 'ml-1m' and (opt['regenerate_movielens'] or not os.path.exists(os.path.join(dataset_dir, "warm_state"))):
        # print("Generating data from ml-1m")
        # generate_movielens(dataset_dir, opt)

    if opt['dataset'] == 'lastfm_20':
    #     print("Generating data from lastfm_20")
        preprocess = Preprocess(opt)
    #     print("Preprocess is done.")
        opt['uf_dim'] = preprocess.uf_dim
        opt['if_dim'] = preprocess.if_dim

    elif opt['dataset'] == 'lastfm_hetrec':
        uf_dim, if_dim = generate_lastfm_hetrec(dataset_dir, opt)
        opt['uf_dim'] = uf_dim
        opt['if_dim'] = if_dim
    #     print("LastFM-HetRec preprocess is done.")

    # create and load model
    print("Create model TaNP...")
    trainer = Trainer(opt)

    if opt['use_cuda']:
        trainer.cuda()

    # find the saved datasets according to the model.
    if opt['dataset'] == 'lastfm_20':
        training_subdir = "training/log"
        testing_subdir = "testing/log"
    elif opt['dataset'] == 'ml-1m':
        training_subdir = "warm_state"
        testing_subdir = "user_cold_state"
    elif opt['dataset'] == 'lastfm_hetrec':
        training_subdir = "training/log"
        testing_subdir = "testing/log"

    # split the dataset
    print('Creating the testing data...')
    # # training data
    # training_set_size = int(len(os.listdir(f"{dataset_dir}/{training_subdir}")) / 4)
    # supp_xs_s = []
    # supp_ys_s = []
    # query_xs_s = []
    # query_ys_s = []
    # for idx in range(training_set_size):
    #     supp_xs_s.append(pickle.load(open(f"{dataset_dir}/{training_subdir}/supp_x_{idx}.pkl", "rb")))
    #     supp_ys_s.append(pickle.load(open(f"{dataset_dir}/{training_subdir}/supp_y_{idx}.pkl", "rb")))
    #     query_xs_s.append(pickle.load(open(f"{dataset_dir}/{training_subdir}/query_x_{idx}.pkl", "rb")))
    #     query_ys_s.append(pickle.load(open(f"{dataset_dir}/{training_subdir}/query_y_{idx}.pkl", "rb")))
    # train_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))

    # del (supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)

    # testing data
    testing_set_size = int(len(os.listdir(f"{dataset_dir}/{testing_subdir}")) / 4)
    supp_xs_s = []
    supp_ys_s = []
    query_xs_s = []
    query_ys_s = []
    for idx in range(testing_set_size):
        supp_xs_s.append(pickle.load(open(f"{dataset_dir}/{testing_subdir}/supp_x_{idx}.pkl", "rb")))
        supp_ys_s.append(pickle.load(open(f"{dataset_dir}/{testing_subdir}/supp_y_{idx}.pkl", "rb")))
        query_xs_s.append(pickle.load(open(f"{dataset_dir}/{testing_subdir}/query_x_{idx}.pkl", "rb")))
        query_ys_s.append(pickle.load(open(f"{dataset_dir}/{testing_subdir}/query_y_{idx}.pkl", "rb")))
    test_dataset = list(zip(supp_xs_s, supp_ys_s, query_xs_s, query_ys_s))

    del (supp_xs_s, supp_ys_s, query_xs_s, query_ys_s)

    # load the trained params
    print("Load pre-trained model...")

    trained_state_dict = torch.load(model_filename, weights_only=True)
    trainer.load_state_dict(trained_state_dict)

    # evaluate
    print('Evaluating...')
    P5, NDCG5, MAP5, P7, NDCG7, MAP7, P10, NDCG10, MAP10 = testing(trainer, opt, test_dataset)

    print('Result for {} and {}'.format(opt['decoder'], opt['dataset']))
    print("precision5\tNDCG5\tMAP5\tprecision7\tNDCG7\tMAP7\tprecision10\tNDCG10\tMAP10")

    print(P5, NDCG5, MAP5, P7, NDCG7, MAP7, P10, NDCG10, MAP10)

    return [P5, NDCG5, MAP5, P7, NDCG7, MAP7, P10, NDCG10, MAP10]

def visualise_exp2(xs, ys, title, save_path, ymin, ymax):
    plt.figure(figsize=(8.5, 6))
    
    plt.bar(xs, ys, color = ["#A9A9A9","#E9967A","#CD853F","#CC9A1E","#DDA0DD","#ADD8E6","#66C6B9"])
    
    plt.title(title, fontsize=14)
    plt.xlabel('Model name', fontsize=12)
    plt.ylabel('P@10', fontsize=12)

    plt.ylim(ymin, ymax)

    plt.tight_layout()
    plt.savefig(save_path)



# def visualise_exp3():


def visualise_exp4(result_path):
    with open(result_path, 'r') as f:
        results = f.read()

    results = ast.literal_eval(results)

    def plot_metrics(results, x_key, title, xlabel, ylabel, save_path):
        plt.figure(figsize=(4, 6))

        for model in ['gating', 'film']:
            key = f"{model}_{x_key}"
            values = [item['value'] for item in results[key]]
            metrics = [round(item['metrics'] * 100, 1) for item in results[key]]

            indices = range(len(values))
            plt.plot(indices, metrics, marker='o', linestyle='--', label=f"{model.capitalize()}")

            plt.xticks(indices, [str(v) for v in values])

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        # if x_key == 'k':
        #     plt.xticks([10, 20, 30, 40, 50])
        # elif x_key == 'lambda':
        #     plt.xticks([0.01, 0.05, 0.1, 0.5, 1.0])

        plt.ylim(bottom=86.5)

        # save
        plt.savefig(save_path)
        # plt.show()
    
    # plot k values results
    plot_metrics(results, 'k', 'P@10 on MovieLens-1M', 'k Values', 'P@10 (%)', 'p10_vs_k.png')
    
    # plot lambda values results
    plot_metrics(results, 'lambda', 'P@10 on MovieLens-1M', 'λ Values', 'P@10 (%)', 'p10_vs_lambda.png')



def visualise_exp4_plaindata(xs, y1, y2, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(4, 6))

    indices = range(len(xs))

    plt.plot(indices, y2, marker='o', linestyle='--', label=f"FiLM")
    plt.plot(indices, y1, marker='o', linestyle='--', label=f"Gating-FiLM")

    plt.xticks(indices, [str(v) for v in xs])

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.ylim(bottom=86.5)
    plt.savefig(save_path)



'''
adjust code mode here..
'''

exp_part = 0  # plot or evaluate. plot = 0, evaluate = exp number
pic_flag = 2  # plot which exp

if exp_part == 2:

    base_dir = "save_model_dir"
    output_file = "results.json"

    results = []

    for dataset_folder in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset_folder)

        if not os.path.isdir(dataset_path):
            continue

        for model_folder in os.listdir(dataset_path):
            model_path = os.path.join(dataset_path, model_folder)

            if not os.path.isdir(model_path):
                continue

            # find config.json and model_best.pt
            config_path = os.path.join(model_path, "config.json")
            model_file = os.path.join(model_path, "model_best.pt")

            # read pretrained model and settings
            print('Loading the settings..')
            opt = helper.load_config(config_path)
            # helper.print_config(opt)

            '''
                Change Ns here.
            '''
            opt['support_size'] = 10

            metrics = evaluating_model(opt,model_file)

            results.append({
                "decoder": opt["decoder"],
                "dataset": opt["dataset"],
                "metrics": {
                    "P5": metrics[0], "NDCG5": metrics[1], "MAP5": metrics[2],
                    "P7": metrics[3], "NDCG7": metrics[4], "MAP7": metrics[5],
                    "P10": metrics[6], "NDCG10": metrics[7], "MAP10": metrics[8]}
            })

            print(results)

elif exp_part == 3:
    pass

elif exp_part == 4:
    # dataset: LastFM
    data_dir = 'save_model_dir/lastfm_20'
    
    results = {
    'gating_k': [], 'film_k': [],
    'gating_lambda': [], 'film_lambda': []
    }

    
    pattern = re.compile(r'(gating|film)_(k|lambda)(\d+(?:\.\d+)?)')

    for folder in os.listdir(data_dir):
        match = pattern.match(folder)
        if match:
            model_type, param_type, value = match.groups()
            value = float(value) if '.' in value else int(value)

            # find path
            config_path = os.path.join(data_dir, folder, 'config.json')
            model_path = os.path.join(data_dir, folder, 'model_best.pt')

            if os.path.isfile(config_path) and os.path.isfile(model_path):
                # loading opt
                opt = helper.load_config(config_path)

                # evaluate
                metrics = evaluating_model(opt, model_path)

                # add results
                key = f"{model_type}_{param_type}"
                results[key].append({'value': value, 'metrics': metrics[6]})

    # rank according to key
    for key in results:
        results[key].sort(key=lambda x: x['value'])

    print(results)


'''
Plot pic
'''
if exp_part == 0:
    if pic_flag == 2:
        # LastFM with Ns=10
        xs = ["MetaNLBA", "MeLU", "MetaCS", "MAMO", "TaNP(w/o tm)", "TaNP(FiLM)", "TaNP(Gating-FiLM)"]
        ys = [83.9, 86.0, 85.4, 86.1, 87.58, 89.09, 88.18]
        title = r"Last.FM with $N_{S_i}=10$"
        save_path = 'N10_lastFM.png'
        visualise_exp2(xs, ys, title, save_path, 82.5, 90)

        # LastFM with Ns=15
        ys = [84.9, 86.2, 86.3, 86.5, 87.8, 88.2, 89.1]
        title = r"Last.FM with $N_{S_i}=15$"
        save_path = 'N15_lastFM.png'
        visualise_exp2(xs, ys, title, save_path, 82.5, 90)

        # MovieLens-1M with Ns=10
        xs = ["MeLU","MetaCS","MetaHIN","MAMO","TaNP(w/o tm)","TaNP(FiLM)","TaNP(Gating-FiLM)"]
        ys = [57.2, 57.23, 57.6, 59.2, 62.2, 62.37, 62.25]
        title = r"MovieLens-1M with $N_{S_i}=10$"
        save_path = 'N10_MovieLens-1M.png'
        visualise_exp2(xs, ys, title, save_path, 56, 64.5)

        ys = [59.5, 59.4, 59.9, 60.0, 62.24, 62.40, 63.50]
        title = r"MovieLens-1M with $N_{S_i}=15$"
        save_path = 'N15_MovieLens-1M.png'
        visualise_exp2(xs, ys, title, save_path, 56, 64.5)


    elif pic_flag == 3:
        pass
    elif pic_flag == 4:
        # visualise_exp4("results_lambda_k.txt")

        ## plain plot for lambda
        xs = [0.01, 0.05, 0.1, 0.5, 1.0]  # lambda
        y1 = [88.0, 88.4, 88.6, 88.2, 87.9]  # Gating-FiLM
        y2 = [87.4, 88.0, 88.4, 87.8, 87.6]  # FiLM
        title = "P@10 on LastFM"
        path = "p10_vs_lambda_lastFM.png"
        visualise_exp4_plaindata(xs, y1, y2, title, "λ Values", "P@10", path)

        ## plain plot for k
        xs = [10, 20, 30, 40, 50]
        y1 = [88.0, 88.7, 88.2, 87.9, 87.8]
        y2 = [88.6, 88.2, 87.7, 87.6, 87.3]
        title = "P@10 on LastFM"
        path = "p10_vs_k_lastFM.png"
        visualise_exp4_plaindata(xs, y1, y2, title, "k Values", "P@10", path)
