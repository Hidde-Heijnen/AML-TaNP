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
        self.path = path  # Fixed: Store path correctly
        self.user_data, self.item_data, self.score_data = self.load()  # Fixed: Call load with self

    def load(self):  # Fixed: No parameters needed since path is stored
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

def generate_movielens(master_path, opt):
    # Check if master_path exists
    if not os.path.exists(master_path):
        raise FileNotFoundError(f"The master path {master_path} does not exist. You should clone the repository first using clone_movielends_dataset.sh")
    
    # Loading feature lists
    rate_list = load_list(os.path.join(master_path, "m_rate.txt"))
    genre_list = load_list(os.path.join(master_path, "m_genre.txt"))
    actor_list = load_list(os.path.join(master_path, "m_actor.txt"))
    director_list = load_list(os.path.join(master_path, "m_director.txt"))
    gender_list = load_list(os.path.join(master_path, "m_gender.txt"))
    age_list = load_list(os.path.join(master_path, "m_age.txt"))
    occupation_list = load_list(os.path.join(master_path, "m_occupation.txt"))
    zipcode_list = load_list(os.path.join(master_path, "m_zipcode.txt"))

    # Create directories for states and logs
    for state in states:
        state_path = os.path.join(master_path, state)
        if not os.path.exists(state_path):
            os.makedirs(state_path)
    log_dir = os.path.join(master_path, "log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    support_size = opt['support_size']
    query_size = opt['query_size']

    # Initialize dataset
    dataset = Movielens_1m(master_path)  # Fixed: Pass master_path

    # Load or create movie dictionary
    movie_dict_path = os.path.join(master_path, "m_movie_dict.pkl")
    if not os.path.exists(movie_dict_path):
        movie_dict = {row['movie_id']: item_converting(row, rate_list, genre_list, director_list, actor_list) 
                      for _, row in dataset.item_data.iterrows()}
        pickle.dump(movie_dict, open(movie_dict_path, "wb"))
    else:
        movie_dict = pickle.load(open(movie_dict_path, "rb"))

    # Load or create user dictionary
    user_dict_path = os.path.join(master_path, "m_user_dict.pkl")
    if not os.path.exists(user_dict_path):
        user_dict = {row['user_id']: user_converting(row, gender_list, age_list, occupation_list, zipcode_list) 
                     for _, row in dataset.user_data.iterrows()}
        pickle.dump(user_dict, open(user_dict_path, "wb"))
    else:
        user_dict = pickle.load(open(user_dict_path, "rb"))

    # Process training data for each state
    for state in states:
        idx = 0
        log_state_path = os.path.join(master_path, "log", state)
        if not os.path.exists(log_state_path):
            os.makedirs(log_state_path)

        # Load dataset and ratings
        with open(os.path.join(master_path, f"{state}.json"), encoding="utf-8") as f:
            dataset_json = json.loads(f.read())
        with open(os.path.join(master_path, f"{state}_y.json"), encoding="utf-8") as f:
            dataset_y = json.loads(f.read())

        for user_id in tqdm(dataset_json.keys()):
            u_id = int(user_id)
            seen_movie_len = len(dataset_json[user_id])  # Keys are strings in JSON
            if seen_movie_len < (support_size + query_size) or seen_movie_len > 100:
                continue

            indices = list(range(seen_movie_len))
            random.shuffle(indices)
            tmp_x = np.array(dataset_json[user_id], dtype=int)  # Ensure integer type
            tmp_y = np.array(dataset_y[user_id])

            support_indices = indices[:support_size]
            query_indices = indices[support_size:]
            support_movie_ids = tmp_x[support_indices]
            query_movie_ids = tmp_x[query_indices]

            user_tensor = user_dict[u_id]  # Fetch once per user

            # Support set: Batch tensor operations
            support_movie_tensors = torch.cat([movie_dict[m_id] for m_id in support_movie_ids], 0)
            support_x_app = torch.cat((support_movie_tensors, user_tensor.repeat(len(support_indices), 1)), 1)
            support_y_app = torch.FloatTensor(tmp_y[support_indices])

            # Query set: Batch tensor operations
            query_movie_tensors = torch.cat([movie_dict[m_id] for m_id in query_movie_ids], 0)
            query_x_app = torch.cat((query_movie_tensors, user_tensor.repeat(len(query_indices), 1)), 1)
            query_y_app = torch.FloatTensor(tmp_y[query_indices])

            # Save data
            pickle.dump(support_x_app, open(os.path.join(master_path, state, f"supp_x_{idx}.pkl"), "wb"))
            pickle.dump(support_y_app, open(os.path.join(master_path, state, f"supp_y_{idx}.pkl"), "wb"))
            pickle.dump(query_x_app, open(os.path.join(master_path, state, f"query_x_{idx}.pkl"), "wb"))
            pickle.dump(query_y_app, open(os.path.join(master_path, state, f"query_y_{idx}.pkl"), "wb"))

            with open(os.path.join(log_state_path, f"supp_x_{idx}_u_m_ids.txt"), "w") as f:
                for m_id in support_movie_ids:
                    f.write(f"{u_id}\t{m_id}\n")
            with open(os.path.join(log_state_path, f"query_x_{idx}_u_m_ids.txt"), "w") as f:
                for m_id in query_movie_ids:
                    f.write(f"{u_id}\t{m_id}\n")

            idx += 1