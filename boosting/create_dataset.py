import pandas as pd
import numpy as np
import scipy
from tqdm import tqdm
import os

tqdm.pandas()


class Dataset():
    def __init__(self, path_train,path_val, path_test, dataset, history_len=100, basket_count_min=3,\
                                                         min_item_count = 5, basket_len = 30):
        self.basket_count_min = basket_count_min
        self.min_item_count = min_item_count
        self.history_len = history_len
        self.basket_len = basket_len
        
        self.train = pd.read_csv(path_train)
        self.val = pd.read_csv(path_val)
        self.test = pd.read_csv(path_test)
        
        self.model_name = 'boosting/' + 'data/' + dataset +'/'
        
        # используемые юзеры и айтемы
        basket_per_user = self.train[['user_id','basket_id']].drop_duplicates() \
            .groupby('user_id').agg({'basket_id':'count'}).reset_index()
        self.all_users = basket_per_user[basket_per_user['basket_id'] >= self.basket_count_min]['user_id'].tolist()
        print('Total users:', len(self.all_users))
        item_counts = self.train.groupby(['item_id']).size().to_frame(name = 'item_count').reset_index()
        self.all_items = item_counts[item_counts['item_count']>= self.min_item_count].item_id
        print('Total items:', len(self.all_items))
        
        ######################
        self.train_cleaned = self.train[ (self.train.user_id.isin(self.all_users)) & (self.train.item_id.isin(self.all_items))].\
                    sort_values(['user_id','date'],ascending=True)
        
        val_cleaned = self.val[ (self.val.user_id.isin(self.all_users)) & (self.val.item_id.isin(self.all_items))].\
                    sort_values(['user_id','date'],ascending=True)
        self.test_cleaned = self.test[ (self.test.user_id.isin(self.all_users)) & (self.test.item_id.isin(self.all_items))].\
                    sort_values(['user_id','date'],ascending=True)
        full_df = pd.concat([self.train_cleaned, val_cleaned, self.test_cleaned]).reset_index(drop=True)\
                                                    .sort_values(['user_id','date'],ascending=True)
        # необходимые аггрегаты
        user_baskets_train = self.train_cleaned[['user_id','date','basket_id']].drop_duplicates().\
                groupby(['user_id'])['basket_id'].apply(list).reset_index()
        self.user_baskets_train_dict = dict(zip(user_baskets_train['user_id'],user_baskets_train['basket_id']))
        
        user_baskets_val = val_cleaned[['user_id','basket_id']].drop_duplicates()
        self.user_baskets_val_dict = dict(zip(user_baskets_val['user_id'],user_baskets_val['basket_id']))
        
        user_baskets_test = self.test_cleaned[['user_id','basket_id']].drop_duplicates()
        self.user_baskets_test_dict = dict(zip(user_baskets_test['user_id'],user_baskets_test['basket_id']))
        
        
        basket_items = full_df.groupby(['basket_id'])['item_id'].apply(list).reset_index()
        self.basket_items_dict = dict(zip(basket_items['basket_id'],basket_items['item_id']))
        self.basket_items_dict['null'] = []
        
        basket_date = full_df[['basket_id','date']].drop_duplicates()
        self.basket_date_dict = dict(zip(basket_date['basket_id'],basket_date['date']))
        
        user_item_train = self.train_cleaned[['user_id','item_id']].drop_duplicates().\
                groupby(['user_id'])['item_id'].apply(list).reset_index()
        self.user_item_train_dict = dict(zip(user_item_train['user_id'],user_item_train['item_id']))
        
        ##########
        u_i_b = self.train_cleaned.groupby(['user_id', 'item_id'])['basket_id'].apply(list).reset_index()
        self.u_i_b_dict = dict(zip(zip(u_i_b.user_id, u_i_b.item_id), u_i_b.basket_id))
        
    def iterate_train_users(self, user):
        history, labels, user_id, item_id, basket_id, basket_date = [], [], [], [], [], []

        user_baskets = self.user_baskets_train_dict[user]
        user_items = self.user_item_train_dict[user]

        for item in user_items:

            baskets_with_item = self.u_i_b_dict[(user), (item)]

            first_date = self.basket_date_dict[baskets_with_item[0]]

            if len(baskets_with_item)>1:
            
                baskets_after_item = [b for b in user_baskets if self.basket_date_dict[b]>first_date]
     
                for i in range(max(0,len(baskets_after_item)-self.basket_len), len(baskets_after_item)):
                    basket = baskets_after_item[i]
                    prev_baskets =  baskets_after_item[:i]
                    dates_list = [first_date]

                    dates_list.extend([self.basket_date_dict[b] for b in prev_baskets if b in baskets_with_item]) 
                    curr_basket_date = self.basket_date_dict[basket]
                    hist = [curr_basket_date - elem for elem in dates_list]
                    
                    if len(hist)<self.history_len:
                        hist = np.pad(hist, [(self.history_len-len(hist), 0)], mode='constant')
                    else:
                        hist = hist[-self.history_len:]

                    lab = int(basket in baskets_with_item)

                    history.append(hist)
                    labels.append(lab)
                    user_id.append(user) 
                    item_id.append(item)
                    basket_id.append(basket) 
                    basket_date.append(curr_basket_date)
        
        return history, labels, user_id, item_id, basket_id, basket_date
        
    def iterate_val_test_user(self, user, mode):
        if mode=='test':
            test_baskets = self.user_baskets_test_dict
        elif mode=='val':
            test_baskets = self.user_baskets_val_dict

        history, labels, user_id, item_id, basket_id, basket_date = [], [], [], [], [], []

        user_baskets = self.user_baskets_train_dict[user]
        user_items = self.user_item_train_dict[user]

        current_basket = test_baskets[user]

        curr_basket_date = self.basket_date_dict[current_basket]
        for item in user_items:

            baskets_with_item = self.u_i_b_dict[(user), (item)]

            dates_list = [self.basket_date_dict[b] for b in baskets_with_item]


            curr_basket_date = self.basket_date_dict[current_basket]
            hist = [curr_basket_date - elem for elem in dates_list]

            if len(hist)<self.history_len:
                hist = np.pad(hist, [(self.history_len-len(hist), 0)], mode='constant')
            else:
                hist = hist[-self.history_len:]

            lab = int(item in self.basket_items_dict[current_basket])

            history.append(hist)
            labels.append(lab)
            user_id.append(user) 
            item_id.append(item)
            basket_id.append(current_basket) 
            basket_date.append(curr_basket_date)

        return history, labels, user_id, item_id, basket_id, basket_date
       
    
    def create_train_data(self):
        save_path = self.model_name + str(self.basket_len) + '_train.npz'
        print(save_path)
        if os.path.isfile(save_path):
            print('Done!')
            res_array_sparse = scipy.sparse.load_npz(save_path)
            data = pd.DataFrame.sparse.from_spmatrix(res_array_sparse)
            data.columns = list(np.arange(self.history_len)) + ['labels', 'user_id', 'item_id', 'basket_id', 'basket_date']
            return data[(data.iloc[:, self.history_len-1]!=0) & (data.iloc[:, self.history_len-2]!=0)]
        
        history, labels, user_id, item_id, basket_id, basket_date = [],[],[],[],[],[]
        for user in tqdm(self.all_users):
            history_u, labels_u, user_id_u, item_id_u, basket_id_u, basket_date_u = self.iterate_train_users(user)
            history.extend(history_u)
            labels.extend(labels_u)
            user_id.extend(user_id_u)
            item_id.extend(item_id_u)
            basket_id.extend(basket_id_u)
            basket_date.extend(basket_date_u)

        res_array = np.hstack([np.array(history), np.array(labels).reshape((-1, 1)),  np.array(user_id).reshape((-1, 1)),
            np.array(item_id).reshape((-1, 1)),  np.array(basket_id).reshape((-1, 1)),  np.array(basket_date).reshape((-1, 1))])
        
        res_array_sparse = scipy.sparse.csr_matrix(res_array)#.astype('int32'))
        scipy.sparse.save_npz(save_path,res_array_sparse)
        
        data = pd.DataFrame.sparse.from_spmatrix(res_array_sparse)
        data.columns = list(np.arange(self.history_len)) + ['labels', 'user_id', 'item_id', 'basket_id', 'basket_date']
        
        return data[(data.iloc[:, self.history_len-1]!=0) & (data.iloc[:, self.history_len-2]!=0)]

        
    def create_val_test_data(self, mode):
        save_path = self.model_name + str(self.basket_len) + f'_{mode}.npz'
        print(save_path)  
      
        if os.path.isfile(save_path):
            print('Done!')
            res_array_sparse = scipy.sparse.load_npz(save_path)
            data = pd.DataFrame.sparse.from_spmatrix(res_array_sparse)
            data.columns = list(np.arange(self.history_len)) + ['labels', 'user_id', 'item_id', 'basket_id', 'basket_date']
            return data[(data.iloc[:, self.history_len-1]!=0) & (data.iloc[:, self.history_len-2]!=0)]
        
        history, labels, user_id, item_id, basket_id, basket_date = [],[],[],[],[],[]
        
        if mode=='test':
            test_users = list(self.user_baskets_test_dict.keys())
        elif mode=='val':
            test_users = list(self.user_baskets_val_dict.keys())
        
        for user in tqdm(test_users):
            history_u, labels_u, user_id_u, item_id_u, basket_id_u, basket_date_u = self.iterate_val_test_user(user, mode=mode)
            history.extend(history_u)
            labels.extend(labels_u)
            user_id.extend(user_id_u)
            item_id.extend(item_id_u)
            basket_id.extend(basket_id_u)
            basket_date.extend(basket_date_u)

        res_array = np.hstack([np.array(history), np.array(labels).reshape((-1, 1)),  np.array(user_id).reshape((-1, 1)),
            np.array(item_id).reshape((-1, 1)),  np.array(basket_id).reshape((-1, 1)),  np.array(basket_date).reshape((-1, 1))])
        
        res_array_sparse = scipy.sparse.csr_matrix(res_array)#.astype('int32'))
        scipy.sparse.save_npz(save_path,res_array_sparse)
        
        data = pd.DataFrame.sparse.from_spmatrix(res_array_sparse)
        data.columns = list(np.arange(self.history_len)) + ['labels', 'user_id', 'item_id', 'basket_id', 'basket_date']
        return data[(data.iloc[:, self.history_len-1]!=0) & (data.iloc[:, self.history_len-2]!=0)]