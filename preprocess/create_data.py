import pandas as pd
import numpy as np

def create_dunnhumby_cj_data(input_file_path):
    '''
    Reads the raw files, renames columns, last basket as test and the rest as train.
    No additional preprocessing steps.
    '''
    #input_file_path = 'dunnhumby_The-Complete-Journey/dunnhumby - The Complete Journey CSV/transaction_data.csv'
    train_baskets_file_path = 'data/dunnhumby_cj/baskets/train_baskets.csv'
    test_baskets_file_path = 'data/dunnhumby_cj/baskets/test_baskets.csv'
    valid_baskets_file_path = 'data/dunnhumby_cj/baskets/valid_baskets.csv'

    df = pd.read_csv(input_file_path)
    print(df.shape)
    df['date'] = df['DAY'].astype(int)
    df['basket_id'] = df['BASKET_ID']
    df['item_id'] = df['PRODUCT_ID'].astype(str)
    df['user_id'] = df['household_key'].astype(str)

    processed_df = df[['date','basket_id','user_id','item_id']].drop_duplicates()
    print(processed_df.shape)
    print(processed_df.nunique())
    last_baskets = processed_df[['user_id','basket_id','date']].drop_duplicates() \
        .groupby('user_id').apply(lambda grp: grp.nlargest(1, 'date'))
    last_baskets.index = last_baskets.index.droplevel()
    test_baskets = pd.merge(last_baskets, processed_df, how='left')


    train_baskets = pd.concat([processed_df,test_baskets]).drop_duplicates(keep=False)

    all_users = list(set(test_baskets['user_id'].tolist()))
    valid_indices = np.random.choice(range(len(all_users)),int(0.5*len(all_users)),
                                 replace=False)
    valid_users = [all_users[i] for i in valid_indices]

    valid_baskets = test_baskets[test_baskets['user_id'].isin(valid_users)]
    test_baskets = test_baskets[~test_baskets['user_id'].isin(valid_users)]

    print(valid_baskets.shape)
    print(test_baskets.shape)
    print(train_baskets.shape)

    print(valid_baskets.nunique())
    print(test_baskets.nunique())
    print(train_baskets.nunique())

    train_baskets.to_csv(train_baskets_file_path,index=False)
    test_baskets.to_csv(test_baskets_file_path,index=False)
    valid_baskets.to_csv(valid_baskets_file_path,index=False)

def create_instacart_data():
    prior_orders_file_path = 'instacart_2017_05_01/order_products__prior.csv'
    train_orders_file_path = 'instacart_2017_05_01/order_products__train.csv'
    orders_file_path = 'instacart_2017_05_01/orders.csv'
    train_baskets_file_path = 'data/instacart/train_baskets.csv'
    test_baskets_file_path = 'data/instacart/test_baskets.csv'
    valid_baskets_file_path = 'data/instacart/valid_baskets.csv'

    prior_orders = pd.read_csv(prior_orders_file_path)
    train_orders = pd.read_csv(train_orders_file_path)
    all_orders = pd.concat([prior_orders,train_orders])
    #print(all_orders.shape)
    #print(all_orders.nunique())

    order_info = pd.read_csv(orders_file_path)

    all_orders = pd.merge(order_info,all_orders,how='inner')
    print(all_orders.shape)
    print(all_orders.nunique())
    #print(all_orders.head())

    all_orders = all_orders.rename(columns={'order_id':'basket_id', 'product_id':'item_id'})

    #all_users = list(set(all_orders['user_id'].tolist()))
    #random_users_indices = np.random.choice(range(len(all_users)), 10000, replace=False)
    #random_users = [all_users[i] for i in range(len(all_users)) if i in random_users_indices]
    #all_orders = all_orders[all_orders['user_id'].isin(random_users)]

    last_baskets = all_orders[['user_id','basket_id','order_number']].drop_duplicates() \
        .groupby('user_id').apply(lambda grp: grp.nlargest(1, 'order_number'))
    last_baskets.index = last_baskets.index.droplevel()
    test_baskets = pd.merge(last_baskets, all_orders, how='left')
    train_baskets = pd.concat([all_orders,test_baskets]).drop_duplicates(keep=False)

    all_users = list(set(test_baskets['user_id'].tolist()))
    valid_indices = np.random.choice(range(len(all_users)),int(0.5*len(all_users)),
                                    replace=False)
    valid_users = [all_users[i] for i in valid_indices]

    valid_baskets = test_baskets[test_baskets['user_id'].isin(valid_users)]
    test_baskets = test_baskets[~test_baskets['user_id'].isin(valid_users)]

    print(valid_baskets.shape)
    print(test_baskets.shape)
    print(train_baskets.shape)

    print(valid_baskets.nunique())
    print(test_baskets.nunique())
    print(train_baskets.nunique())

    train_baskets.to_csv(train_baskets_file_path,index=False)
    test_baskets.to_csv(test_baskets_file_path,index=False)
    valid_baskets.to_csv(valid_baskets_file_path,index=False)


def create_valuedshopper_data():
    input_file_path = 'transactions/kaggle-acquire/transactions_sample.csv'
    train_baskets_file_path = 'data/valued_shopper_sample/train_baskets.csv'
    test_baskets_file_path = 'data/valued_shopper_sample/test_baskets.csv'
    valid_baskets_file_path = 'data/valued_shopper_sample/valid_baskets.csv'

    df = pd.read_csv(input_file_path)
    print(df.shape)
    df['date'] = pd.to_datetime(df['date'],format='%Y-%m-%d').dt.strftime('%Y%m%d').astype(int)
    df['basket_id'] = df['date'].astype(str)+'_'+df['id'].astype(str)
    df['item_id'] = df['dept'].astype(str)+'_'+df['category'].astype(str)+"_"+df['brand'].astype(str)+'_'+df['company'].astype(str)
    df['user_id'] = df['id'].astype(str)

    processed_df = df[['date','basket_id','user_id','item_id']].drop_duplicates()

    #all_users = list(set(processed_df['user_id'].tolist()))
    #random_users_indices = np.random.choice(range(len(all_users)), 2500, replace=False)
    #random_users = [all_users[i] for i in range(len(all_users)) if i in random_users_indices]
    #processed_df = processed_df[processed_df['user_id'].isin(random_users)]

    print(processed_df.shape)
    print(processed_df.nunique())
    last_baskets = processed_df[['user_id','basket_id','date']].drop_duplicates() \
        .groupby('user_id').apply(lambda grp: grp.nlargest(1, 'date'))
    last_baskets.index = last_baskets.index.droplevel()
    test_baskets = pd.merge(last_baskets, processed_df, how='left')
    train_baskets = pd.concat([processed_df,test_baskets]).drop_duplicates(keep=False)

    all_users = list(set(test_baskets['user_id'].tolist()))
    valid_indices = np.random.choice(range(len(all_users)),int(0.5*len(all_users)),
                                    replace=False)
    valid_users = [all_users[i] for i in valid_indices]

    valid_baskets = test_baskets[test_baskets['user_id'].isin(valid_users)]
    test_baskets = test_baskets[~test_baskets['user_id'].isin(valid_users)]


    print(valid_baskets.shape)
    print(test_baskets.shape)
    print(train_baskets.shape)

    print(valid_baskets.nunique())
    print(test_baskets.nunique())
    print(train_baskets.nunique())

    train_baskets.to_csv(train_baskets_file_path,index=False)
    test_baskets.to_csv(test_baskets_file_path,index=False)
    valid_baskets.to_csv(valid_baskets_file_path,index=False)