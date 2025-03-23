import os
import re
import json
import pickle
import random
import numpy as np
import torch
from tqdm import tqdm

# Directory structure will match the lastfm_20 dataset
states = ["training/log", "validation/log", "testing/log"]
evidence_dirs = ["training/evidence", "validation/evidence", "testing/evidence"]

class LastFMHetrec(object):
    def __init__(self, path):
        self.path = path
        self.artists_data, self.user_artists_data = self.load()
        
    def load(self):
        artists_path = os.path.join(self.path, "artists.dat")
        user_artists_path = os.path.join(self.path, "user_artists.dat")
        
        # Check if files exist
        for file_path in [artists_path, user_artists_path]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} not found. Make sure the LastFM-HetRec dataset is properly extracted.")
        
        # Load artists data
        artists = {}
        with open(artists_path, 'r', encoding='utf-8') as f:
            # Skip header
            f.readline()
            for line in f:
                parts = line.strip().split('\t')
                artist_id = parts[0]
                artist_name = parts[1]
                artists[artist_id] = artist_name
        
        # Load user-artist data (plays/listens)
        user_artists = {}
        with open(user_artists_path, 'r', encoding='utf-8') as f:
            # Skip header
            f.readline()
            for line in f:
                parts = line.strip().split('\t')
                user_id = parts[0]
                artist_id = parts[1]
                weight = int(parts[2])  # listening count
                
                if user_id not in user_artists:
                    user_artists[user_id] = []
                
                user_artists[user_id].append((artist_id, weight))
        
        return artists, user_artists

def clear_generated_data(master_path, states, evidence_dirs):
    """
    Clear previously generated data files to ensure clean generation with new parameters.
    """
    # Remove generated pkl files in state directories
    for state in states:
        state_path = os.path.join(master_path, state)
        if os.path.exists(state_path):
            for file in os.listdir(state_path):
                if file.endswith('.pkl'):
                    os.remove(os.path.join(state_path, file))
            print(f"Cleared previously generated data in {state_path}")
    
    # Remove evidence files
    for evidence_dir in evidence_dirs:
        evidence_path = os.path.join(master_path, evidence_dir)
        if os.path.exists(evidence_path):
            for file in os.listdir(evidence_path):
                if file.endswith('.txt'):
                    os.remove(os.path.join(evidence_path, file))
            print(f"Cleared evidence files in {evidence_path}")
    
    # Remove pickle files in the master path
    for file in os.listdir(master_path):
        if file.endswith('.pkl'):
            os.remove(os.path.join(master_path, file))
    
    # Remove json files in the master path
    for file in ['interaction_dict_x.json', 'interaction_dict_y.json', 'user_list.json', 'item_list.json']:
        file_path = os.path.join(master_path, file)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed {file_path}")

def to_onehot_dict(item_list):
    """Convert a list to one-hot encoded dictionary"""
    one_hot = {}
    for i, item in enumerate(item_list):
        vec = torch.zeros(1, len(item_list))
        vec[0, i] = 1
        one_hot[item] = vec
    return one_hot

def generate_lastfm_hetrec(master_path, opt):
    """
    Generate LastFM-HetRec dataset files for TaNP model
    """
    if not os.path.exists(master_path):
        os.makedirs(master_path, exist_ok=True)
    
    print(f"Generating lastfm_hetrec data with support_size={opt['support_size']}, query_size={opt['query_size']}")
    
    # Clear existing generated data
    clear_generated_data(master_path, states, evidence_dirs)
    
    # Create directories for states and evidence
    for state in states:
        state_path = os.path.join(master_path, state)
        os.makedirs(state_path, exist_ok=True)
    
    for evidence_dir in evidence_dirs:
        evidence_path = os.path.join(master_path, evidence_dir)
        os.makedirs(evidence_path, exist_ok=True)
    
    # Initialize dataset
    dataset = LastFMHetrec(master_path)
    
    # Process data
    # 1. Convert to implicit feedback
    print("Converting to implicit feedback...")
    inter_dict_x = {}  # user -> list of artist ids
    inter_dict_y = {}  # user -> list of feedback values (0 or 1)
    
    user_list = []
    item_list = []
    
    # Use support_size + query_size as minimum threshold instead of fixed 40
    min_interactions = opt['support_size'] + opt['query_size']
    
    print("max length: ", opt['max_len'])
    print("min interactions: ", min_interactions)
    
    for user_id, artist_data in dataset.user_artists_data.items():
        # Skip users with too few or too many interactions
        if len(artist_data) < min_interactions or len(artist_data) > opt['max_len']:
            continue
        
        user_list.append(user_id)
        
        weights = [item[1] for item in artist_data]
        median_weight = np.median(weights)
        
        inter_dict_x[user_id] = []
        inter_dict_y[user_id] = []
        
        for artist_id, weight in artist_data:
            inter_dict_x[user_id].append(artist_id)
            item_list.append(artist_id)
            # Convert to implicit feedback (1 if weight >= median, 0 otherwise)
            feedback = 1 if weight >= median_weight else 0
            inter_dict_y[user_id].append(feedback)
    
    # Get unique items and users
    item_list = list(set(item_list))
    user_list = list(set(user_list))
    
    print(f"Number of users: {len(user_list)}")
    print(f"Number of items: {len(item_list)}")
    print(f"Number of interactions: {sum(len(items) for items in inter_dict_x.values())}")
    
    # Save the processed data to JSON files
    with open(os.path.join(master_path, 'interaction_dict_x.json'), 'w') as f:
        json.dump(inter_dict_x, f)
    
    with open(os.path.join(master_path, 'interaction_dict_y.json'), 'w') as f:
        json.dump(inter_dict_y, f)
    
    with open(os.path.join(master_path, 'user_list.json'), 'w') as f:
        json.dump(user_list, f)
    
    with open(os.path.join(master_path, 'item_list.json'), 'w') as f:
        json.dump(item_list, f)
    
    # 2. Split data into train/valid/test
    random.shuffle(user_list)
    train_size = int(len(user_list) * opt['train_ratio'])
    valid_size = int(len(user_list) * opt['valid_ratio'])
    
    train_users = user_list[:train_size]
    valid_users = user_list[train_size:train_size + valid_size]
    test_users = user_list[train_size + valid_size:]
    
    train_dict_x = {u: inter_dict_x[u] for u in train_users}
    train_dict_y = {u: inter_dict_y[u] for u in train_users}
    
    valid_dict_x = {u: inter_dict_x[u] for u in valid_users}
    valid_dict_y = {u: inter_dict_y[u] for u in valid_users}
    
    test_dict_x = {u: inter_dict_x[u] for u in test_users}
    test_dict_y = {u: inter_dict_y[u] for u in test_users}
    
    # Remove cold items from test set (items not seen in training)
    print("Removing cold items from test set...")
    print('Before removing cold items, test data has {} interactions.'.format(sum(len(v) for v in test_dict_x.values())))
    train_item_set = set()
    for items in train_dict_x.values():
        train_item_set.update(items)
    
    for user in test_dict_x.keys():
        items = test_dict_x[user]
        feedbacks = test_dict_y[user]
        filtered_items = [item for item, f in zip(items, feedbacks) if item in train_item_set]
        filtered_feedbacks = [f for item, f in zip(items, feedbacks) if item in train_item_set]
        test_dict_x[user] = filtered_items
        test_dict_y[user] = filtered_feedbacks
    print('After removing cold items, test data has {} interactions.'.format(sum(len(v) for v in test_dict_x.values())))
    
    # Save the split data to pickle files
    test_ratio = 1 - opt['train_ratio'] - opt['valid_ratio']
    pickle.dump(train_dict_x, open(os.path.join(master_path, f"training_dict_x_{opt['train_ratio']:.6f}.pkl"), "wb"))
    pickle.dump(train_dict_y, open(os.path.join(master_path, f"training_dict_y_{opt['train_ratio']:.6f}.pkl"), "wb"))
    pickle.dump(valid_dict_x, open(os.path.join(master_path, f"valid_dict_x_{opt['valid_ratio']:.6f}.pkl"), "wb"))
    pickle.dump(valid_dict_y, open(os.path.join(master_path, f"valid_dict_y_{opt['valid_ratio']:.6f}.pkl"), "wb"))
    pickle.dump(test_dict_x, open(os.path.join(master_path, f"test_dict_x_{test_ratio:.6f}.pkl"), "wb"))
    pickle.dump(test_dict_y, open(os.path.join(master_path, f"test_dict_y_{test_ratio:.6f}.pkl"), "wb"))
    
    # 3. Create one-hot encoding dictionaries
    user_dict = to_onehot_dict(user_list)
    item_dict = to_onehot_dict(item_list)
    
    # 4. Generate episodes
    def generate_episodes(dict_x, dict_y, category):
        """Generate episodes for the given category (training/validation/testing)"""
        print(f"Generating episodes for {category}...")
        
        log_dir = os.path.join(master_path, category, "log")
        evidence_dir = os.path.join(master_path, category, "evidence")
        
        support_size = opt['support_size']
        query_size = opt['query_size']
        max_len = opt['max_len']
        
        idx = 0
        for user_id in tqdm(dict_x.keys()):
            # Get user's interactions
            items = dict_x[user_id]
            feedbacks = dict_y[user_id]
            
            # Skip if too few or too many interactions
            if len(items) < (support_size + query_size) or len(items) > max_len:
                continue
            
            # Shuffle indices
            indices = list(range(len(items)))
            random.shuffle(indices)
            
            # Split into support and query sets
            support_indices = indices[:support_size]
            query_indices = indices[support_size:]
            
            # Prepare support set features and labels
            support_x_app = None
            for i in support_indices:
                item_id = items[i]
                tmp_x_converted = torch.cat((item_dict[item_id], user_dict[user_id]), 1)
                if support_x_app is None:
                    support_x_app = tmp_x_converted
                else:
                    support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
            
            # Prepare query set features and labels
            query_x_app = None
            for i in query_indices:
                item_id = items[i]
                tmp_x_converted = torch.cat((item_dict[item_id], user_dict[user_id]), 1)
                if query_x_app is None:
                    query_x_app = tmp_x_converted
                else:
                    query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
            
            # Convert labels to tensors
            support_y_app = torch.FloatTensor([feedbacks[i] for i in support_indices])
            query_y_app = torch.FloatTensor([feedbacks[i] for i in query_indices])
            
            # Save the episodes
            pickle.dump(support_x_app, open(os.path.join(log_dir, f"supp_x_{idx}.pkl"), "wb"))
            pickle.dump(support_y_app, open(os.path.join(log_dir, f"supp_y_{idx}.pkl"), "wb"))
            pickle.dump(query_x_app, open(os.path.join(log_dir, f"query_x_{idx}.pkl"), "wb"))
            pickle.dump(query_y_app, open(os.path.join(log_dir, f"query_y_{idx}.pkl"), "wb"))
            
            # Save user-item pairs for evidence (debugging/analysis)
            with open(os.path.join(evidence_dir, f"supp_x_{idx}_u_m_ids.txt"), "w") as f:
                for i in support_indices:
                    f.write(f"{user_id}\t{items[i]}\n")
            
            with open(os.path.join(evidence_dir, f"query_x_{idx}_u_m_ids.txt"), "w") as f:
                for i in query_indices:
                    f.write(f"{user_id}\t{items[i]}\n")
            
            idx += 1
        
        print(f"Generated {idx} episodes for {category}")
        return idx
    
    train_episodes = generate_episodes(train_dict_x, train_dict_y, "training")
    valid_episodes = generate_episodes(valid_dict_x, valid_dict_y, "validation")
    test_episodes = generate_episodes(test_dict_x, test_dict_y, "testing")
    
    print("Successfully prepared LastFM-HetRec dataset!")
    print(f"Train episodes: {train_episodes}")
    print(f"Validation episodes: {valid_episodes}")
    print(f"Test episodes: {test_episodes}")
    
    return len(user_list), len(item_list)
