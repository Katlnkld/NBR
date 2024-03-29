import numpy as np
from tqdm import tqdm


def recall_k(y_true, y_pred, k ):
    a = len(set(y_pred[:k]).intersection(set(y_true)))
    b = len(set(y_true))
    return a/b

def precision_k(y_true, y_pred, k ):
    a = len(set(y_pred[:k]).intersection(set(y_true)))
    return a/k

def hitrate_k(y_true, y_pred, k ):
    a = len(set(y_pred[:k]).intersection(set(y_true)))
    if a==0:
        return 0
    else:
        return 1
   

def ndcg_k(y_true, y_pred, k):
    a = 0
    for i,x in enumerate(y_pred[:k]):
        if x in y_true:
            a+= 1/np.log2(i+2)
    b = 0
    for i in range(k):#range(min(k,len(set(y_true)))):
        b +=1/np.log2(i+2)
    return a/b


def repeat_score_item(train_baskets):
    user_item = train_baskets.groupby(['user_id','item_id']).size().to_frame(name = 'item_count'). \
        reset_index()

    user_train_baskets_df = train_baskets[['user_id','basket_id']].drop_duplicates().groupby(['user_id'])\
        .size().reset_index()
    user_train_baskets_dict = dict(zip( user_train_baskets_df['user_id'],user_train_baskets_df[0]))


    user_baskets = train_baskets[['user_id','date','basket_id']].drop_duplicates(). \
        sort_values(['user_id','date'],ascending=True).groupby(['user_id'])['basket_id'].apply(list).reset_index()
    user_baskets_dict = dict(zip(user_baskets['user_id'],user_baskets['basket_id']))

    basket_items = train_baskets.groupby(['basket_id'])['item_id'].apply(list).reset_index()
    basket_items_dict = dict(zip(basket_items['basket_id'],basket_items['item_id']))

    rep_score = {}
    for user in tqdm(user_baskets_dict):
        baskets = user_baskets_dict[user]
        rep_score[user] = 0
        for i in range(len(baskets)):
            next_basket = baskets[i]
            history_baskets = baskets[:i]

            history_items = []
            for basket in history_baskets:
                for item in basket_items_dict[basket]:
                    history_items.append(item)
            history_items = set(history_items)
            score = 0
            for item in basket_items_dict[next_basket]:
                if item in history_items:
                    score +=1
            rep_score[user] += score/len(basket_items_dict[next_basket])
        rep_score[user]/=len(baskets)
    return rep_score

def repeat_score_user(train_baskets):
    user_baskets = train_baskets[['user_id','date','basket_id']].drop_duplicates().\
        sort_values(['user_id','date'],ascending=True).groupby(['user_id'])['basket_id'].apply(list).reset_index()
    user_baskets_dict = dict(zip(user_baskets['user_id'],user_baskets['basket_id']))

    basket_items = train_baskets.groupby(['basket_id'])['item_id'].apply(list).reset_index()
    basket_items_dict = dict(zip(basket_items['basket_id'],basket_items['item_id']))

    rep_score = {}
    for user in user_baskets_dict:
        baskets = user_baskets_dict[user]
        rep_score[user] = 0
        for i in range(len(baskets)):
            next_basket = baskets[i]
            history_baskets = baskets[:i]

            history_items = []
            for basket in history_baskets:
                for item in basket_items_dict[basket]:
                    history_items.append(item)
            history_items = set(history_items)
            score = 0
            for item in basket_items_dict[next_basket]:
                if item in history_items:
                    score +=1
            rep_score[user] += score/len(basket_items_dict[next_basket])
        rep_score[user]/=len(baskets)
    return rep_score

def repeat_rate_user(df):
    """Доля пользователей, которые покупали хоть один элемент повторно"""
    
    df_ui = df.groupby('user_id').item_id.nunique().reset_index().\
            merge(df.groupby('user_id').item_id.count().reset_index(), on='user_id')
    df_ui['repeated'] = (df_ui.item_id_y>df_ui.item_id_x).astype(int)
    
    return df_ui['repeated'].mean()
    
def repeat_rate_item(df):
    """Доля элементов, которые юзеры покупали повторно хоть один раз"""
    
    df_ui = df.groupby('item_id').user_id.nunique().reset_index().\
            merge(df.groupby('item_id').user_id.count().reset_index(), on='item_id')
    df_ui['repeated'] = (df_ui.user_id_y>df_ui.user_id_x).astype(int)
    
    return df_ui['repeated'].mean()