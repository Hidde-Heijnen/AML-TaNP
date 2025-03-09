import os
import re
import json
import pickle
import random
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from datetime import datetime

states = ["warm_state", "user_cold_state", "item_cold_state", "user_and_item_cold_state"]

class Movielens_1m(object):
    def __init__(self, path):
        self.user_data, self.item_data, self.score_data = self.load()
        self.path = path

    def load(self):
        profile_data_path = os.path.join(self.path, "users.dat")
        score_data_path = os.path.join(self.path, "ratings.dat")
        item_data_path = os.path.join(self.path, "movies_extrainfos.dat")

        profile_data = pd.read_csv(
            profile_data_path, names=['user_id', 'gender', 'age', 'occupation_code', 'zip'],
            sep="::", engine='python'
        )
        item_data = pd.read_csv(
            item_data_path, names=['movie_id', 'title', 'year', 'rate', 'released', 'genre', 'director', 'writer', 'actors', 'plot', 'poster'],
            sep="::", engine='python', encoding="utf-8"
        )
        score_data = pd.read_csv(
            score_data_path, names=['user_id', 'movie_id', 'rating', 'timestamp'],
            sep="::", engine='python'
        )

        score_data['time'] = score_data["timestamp"].map(lambda x: datetime.fromtimestamp(x))
        score_data = score_data.drop(["timestamp"], axis=1)
        return profile_data, item_data, score_data

def item_converting(row, rate_list, genre_list, director_list, actor_list):
    rate_idx = torch.tensor([[rate_list.index(str(row['rate']))]]).long()
    genre_idx = torch.zeros(1, 25).long()
    for genre in str(row['genre']).split(", "):
        idx = genre_list.index(genre)
        genre_idx[0, idx] = 1
    director_idx = torch.zeros(1, 2186).long()
    for director in str(row['director']).split(", "):
        idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
        director_idx[0, idx] = 1
    actor_idx = torch.zeros(1, 8030).long()
    for actor in str(row['actors']).split(", "):
        idx = actor_list.index(actor)
        actor_idx[0, idx] = 1
    return torch.cat((rate_idx, genre_idx, director_idx, actor_idx), 1)

def user_converting(row, gender_list, age_list, occupation_list, zipcode_list):
    gender_idx = torch.tensor([[gender_list.index(str(row['gender']))]]).long()
    age_idx = torch.tensor([[age_list.index(str(row['age']))]]).long()
    occupation_idx = torch.tensor([[occupation_list.index(str(row['occupation_code']))]]).long()
    zip_idx = torch.tensor([[zipcode_list.index(str(row['zip'])[:5])]]).long()
    return torch.cat((gender_idx, age_idx, occupation_idx, zip_idx), 1)

def load_list(fname):
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_

def generate(master_path):

    # check if the master_path exists
    if not os.path.exists(master_path):
        raise FileNotFoundError(f"The master path {master_path} does not exist. You should clone the repository first using clone_movielends_dataset.sh")
    
    # Loading
    rate_list = load_list("{}/m_rate.txt".format(master_path))
    genre_list = load_list("{}/m_genre.txt".format(master_path))
    actor_list = load_list("{}/m_actor.txt".format(master_path))
    director_list = load_list("{}/m_director.txt".format(master_path))
    gender_list = load_list("{}/m_gender.txt".format(master_path))
    age_list = load_list("{}/m_age.txt".format(master_path))
    occupation_list = load_list("{}/m_occupation.txt".format(master_path))
    zipcode_list = load_list("{}/m_zipcode.txt".format(master_path))

    # creat warm_state/ 和 log/
    if not os.path.exists("{}/warm_state/".format(master_path)):
        print("warm_state doesn't exist")
        for state in states:
            os.mkdir("{}/{}/".format(master_path, state))
    if not os.path.exists("{}/log/".format(master_path)):
        os.mkdir("{}/log/".format(master_path))

    dataset = Movielens_1m(master_path)

    if not os.path.exists("{}/m_movie_dict.pkl".format(master_path)):
        movie_dict = {}
        for idx, row in dataset.item_data.iterrows():
            m_info = item_converting(row, rate_list, genre_list, director_list, actor_list)
            movie_dict[row['movie_id']] = m_info
        pickle.dump(movie_dict, open("{}/m_movie_dict.pkl".format(master_path), "wb"))
    else:
        movie_dict = pickle.load(open("{}/m_movie_dict.pkl".format(master_path), "rb"))

    if not os.path.exists("{}/m_user_dict.pkl".format(master_path)):
        user_dict = {}
        for idx, row in dataset.user_data.iterrows():
            u_info = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
            user_dict[row['user_id']] = u_info
        pickle.dump(user_dict, open("{}/m_user_dict.pkl".format(master_path), "wb"))
    else:
        user_dict = pickle.load(open("{}/m_user_dict.pkl".format(master_path), "rb"))

    # process the traing data
    for state in states:
        idx = 0
        if not os.path.exists("{}/{}/{}".format(master_path, "log", state)):
            os.mkdir("{}/{}/{}".format(master_path, "log", state))
        # load train.json, valid.json, test.json（MovieLens 用户观看历史）。
        with open("{}/{}.json".format(master_path, state), encoding="utf-8") as f:
            dataset = json.loads(f.read())
        # load train_y.json, valid_y.json, test_y.json（对应的评分标签）
        with open("{}/{}_y.json".format(master_path, state), encoding="utf-8") as f:
            dataset_y = json.loads(f.read())

        for _, user_id in tqdm(enumerate(dataset.keys())):
            u_id = int(user_id)
            seen_movie_len = len(dataset[str(u_id)])
            indices = list(range(seen_movie_len))
            # 过滤观看历史太短或太长的用户
            if seen_movie_len < 13 or seen_movie_len > 100:
                continue
            # 打乱观看顺序，then split: support_x_app（first n-10）query_x_app（last 10）
            random.shuffle(indices)
            tmp_x = np.array(dataset[str(u_id)])
            tmp_y = np.array(dataset_y[str(u_id)])
            # 拼接用户 & 电影特征 -> support_x_app
            support_x_app = None
            for m_id in tmp_x[indices[:-10]]:
                m_id = int(m_id)
                tmp_x_converted = torch.cat((movie_dict[m_id], user_dict[u_id]), 1)
                try:
                    support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
                except:
                    support_x_app = tmp_x_converted
            # query_x_app
            query_x_app = None
            for m_id in tmp_x[indices[-10:]]:
                m_id = int(m_id)
                u_id = int(user_id)
                tmp_x_converted = torch.cat((movie_dict[m_id], user_dict[u_id]), 1)
                try:
                    query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
                except:
                    query_x_app = tmp_x_converted
            support_y_app = torch.FloatTensor(tmp_y[indices[:-10]])
            query_y_app = torch.FloatTensor(tmp_y[indices[-10:]])

            # Save
            pickle.dump(support_x_app, open("{}/{}/supp_x_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(support_y_app, open("{}/{}/supp_y_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(query_x_app, open("{}/{}/query_x_{}.pkl".format(master_path, state, idx), "wb"))
            pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(master_path, state, idx), "wb"))
            with open("{}/log/{}/supp_x_{}_u_m_ids.txt".format(master_path, state, idx), "w") as f:
                for m_id in tmp_x[indices[:-10]]:
                    f.write("{}\t{}\n".format(u_id, m_id))
            with open("{}/log/{}/query_x_{}_u_m_ids.txt".format(master_path, state, idx), "w") as f:
                for m_id in tmp_x[indices[-10:]]:
                    f.write("{}\t{}\n".format(u_id, m_id))
            idx += 1