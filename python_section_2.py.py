#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime


# In[2]:


df=pd.read_csv("C:\\Users\\DELL\\Downloads\\dataset-2.csv")


# In[3]:


df.info()


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.isnull().sum()


# In[7]:


#Answer 9

def calculate_distance_matrix(file_path):
    df = pd.read_csv(file_path)
    locations = sorted(set(df['id_start']).union(set(df['id_end'])))
    dist_matrix = pd.DataFrame(np.inf, index=locations, columns=locations)
    np.fill_diagonal(dist_matrix.values, 0)
    
    
    for _, row in df.iterrows():
        start, end, distance = row['id_start'], row['id_end'], row['distance']
        dist_matrix.loc[start, end] = distance
        dist_matrix.loc[end, start] = distance  
    
    for k in locations:
        for i in locations:
            for j in locations:
                dist_matrix.loc[i, j] = min(dist_matrix.loc[i, j], dist_matrix.loc[i, k] + dist_matrix.loc[k, j])
    
    return dist_matrix

file_path = "C:\\Users\\DELL\\Downloads\\dataset-2.csv"
distance_matrix = calculate_distance_matrix(file_path)


print(distance_matrix)


# In[8]:


#Question 10
def unroll_distance_matrix(distance_matrix):
    unrolled_data = []

    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            if id_start != id_end:
                distance = distance_matrix.loc[id_start, id_end]
                unrolled_data.append([id_start, id_end, distance])
    unrolled_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])
    
    return unrolled_df
unrolled_df = unroll_distance_matrix(distance_matrix)

print(unrolled_df)


# In[9]:


#QUESTION 11

def find_ids_within_ten_percentage_threshold(unrolled_df, reference_id):
    reference_distances = unrolled_df[unrolled_df['id_start'] == reference_id]
    avg_distance = reference_distances['distance'].mean()
    
    lower_bound = avg_distance * 0.9 
    upper_bound = avg_distance * 1.1 
    
    within_threshold = unrolled_df[
        (unrolled_df['distance'] >= lower_bound) &
        (unrolled_df['distance'] <= upper_bound)
    ]
    
    result_ids = sorted(within_threshold['id_start'].unique())
    
    return result_ids

reference_id = 1001400 
ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)

print(ids_within_threshold)


# In[10]:


#QUESTION 12

def calculate_toll_rate(unrolled_df):
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    unrolled_df['moto'] = unrolled_df['distance'] * rate_coefficients['moto']
    unrolled_df['car'] = unrolled_df['distance'] * rate_coefficients['car']
    unrolled_df['rv'] = unrolled_df['distance'] * rate_coefficients['rv']
    unrolled_df['bus'] = unrolled_df['distance'] * rate_coefficients['bus']
    unrolled_df['truck'] = unrolled_df['distance'] * rate_coefficients['truck']
    
    return unrolled_df

toll_rate_df = calculate_toll_rate(unrolled_df)

print(toll_rate_df)


# In[11]:


#QUESTION 13

def calculate_time_based_toll_rates(toll_rate_df):
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    time_intervals = [
        (datetime.time(0, 0, 0), datetime.time(10, 0, 0)),   # 00:00 to 10:00
        (datetime.time(10, 0, 0), datetime.time(18, 0, 0)),  # 10:00 to 18:00
        (datetime.time(18, 0, 0), datetime.time(23, 59, 59)) # 18:00 to 23:59
    ]
    
    expanded_data = []
    
    for _, row in toll_rate_df.iterrows():
        for day in days_of_week:
            for start_time, end_time in time_intervals:
                if day in ['Saturday', 'Sunday']:
                    discount_factor = 0.7
                else:
                    if start_time == datetime.time(0, 0, 0):  
                        discount_factor = 0.8
                    elif start_time == datetime.time(10, 0, 0): 
                        discount_factor = 1.2
                    else:  
                        discount_factor = 0.8
                
                new_row = {
                    'id_start': row['id_start'],
                    'id_end': row['id_end'],
                    'distance': row['distance'],
                    'start_day': day,
                    'start_time': start_time,
                    'end_day': day,
                    'end_time': end_time,
                    'moto': row['moto'] * discount_factor,
                    'car': row['car'] * discount_factor,
                    'rv': row['rv'] * discount_factor,
                    'bus': row['bus'] * discount_factor,
                    'truck': row['truck'] * discount_factor
                }
                
                expanded_data.append(new_row)
    
    time_based_toll_df = pd.DataFrame(expanded_data)
    
    return time_based_toll_df

time_based_toll_df = calculate_time_based_toll_rates(toll_rate_df)

print(time_based_toll_df)


# In[ ]:




