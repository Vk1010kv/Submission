#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import re
from datetime import datetime, time, timedelta
from math import radians, sin, cos, sqrt, atan2
from typing import List, Tuple
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv("C:\\Users\\DELL\\Downloads\\dataset-1.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.info()


# In[6]:


df.columns


# In[7]:


df.isnull().sum()


# In[8]:


df['able2Hov2'].fillna(df['able2Hov2'].mean(), inplace=True)
df['able2Hov3'].fillna(df['able2Hov3'].mean(), inplace=True)
df['able3Hov2'].fillna(df['able3Hov2'].mean(), inplace=True)


# In[9]:


df.isnull().sum()


# In[10]:


df.describe().transpose()


# In[11]:


#ANSWER 1
columns_of_interest = [
    'able2Hov2', 'able2Hov3', 'able3Hov2', 'able3Hov3', 
    'able5Hov2', 'able5Hov3', 'able4Hov2', 'able4Hov3'
]

data = df[columns_of_interest].fillna(-1).values.flatten().tolist()

def reverse_in_groups(data, n):
    result = []
    for i in range(0, len(data), n):
        group = data[i:i + n]
        reversed_group = []
        for j in range(len(group) - 1, -1, -1):
            reversed_group.append(group[j])
        result.extend(reversed_group)
    return result


n = 3
reversed_data = reverse_in_groups(data, n)

reshaped_reversed_data = pd.DataFrame(
    [reversed_data[i:i+len(df)] for i in range(0, len(reversed_data), len(df))]
).T
df[columns_of_interest] = reshaped_reversed_data.values
df.to_csv('dataset_reversed.csv', index=False)


# In[12]:


df


# In[13]:


#ANSWER 2
def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    grouped = {}
    for word in lst:
        length = len(word)
        if length not in grouped:
            grouped[length] = []
        grouped[length].append(word)
    return dict(sorted(grouped.items()))
name_data = df['name'].fillna("").tolist()
result_names = group_by_length(name_data)
print(result_names)


# In[14]:


#ANSWER 5
text_data = df.astype(str).agg(' '.join, axis=1).str.cat(sep=' ')
def find_all_dates(text):
    date_pattern = r'\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2}'
    dates = re.findall(date_pattern, text)
    return dates
dates_found_in_dataset = find_all_dates(text_data)

print(dates_found_in_dataset[:10])


# In[15]:


#Answer 3
def flatten_dictionary(d, parent_key='', sep='.'):
    """
    Flattens a nested dictionary, handling lists and nested structures.
    """
    items = []
    
    if not d:
        return {}
        
    if isinstance(d, list):
        for i, item in enumerate(d):
            new_key = f"{parent_key}[{i}]" if parent_key else f"[{i}]"
            if isinstance(item, (dict, list)):
                items.extend(flatten_dictionary(item, new_key, sep=sep).items())
            else:
                items.append((new_key, item))
        return dict(items)
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, (dict, list)):
            items.extend(flatten_dictionary(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
            
    return dict(items)

def process_large_csv(file_path):
    """
    Process large CSV file in chunks and flatten each row.
    """

    
    # Initialize list to store flattened dictionaries
    flattened_data = []
    
    # Process each row
    for _, row in df.iterrows():
        # Create structured dictionary for each row
        row_dict = {
            "id": str(row['id']),
            "name": row['name'],
            "id_2": row['id_2'],
            "schedule": {
                "start": {
                    "day": row['startDay'],
                    "time": row['startTime']
                },
                "end": {
                    "day": row['endDay'],
                    "time": row['endTime']
                }
            },
            "abilities": {
                "able2": {
                    "hov2": row['able2Hov2'],
                    "hov3": row['able2Hov3']
                },
                "able3": {
                    "hov2": row['able3Hov2'],
                    "hov3": row['able3Hov3']
                },
                "able4": {
                    "hov2": row['able4Hov2'],
                    "hov3": row['able4Hov3']
                },
                "able5": {
                    "hov2": row['able5Hov2'],
                    "hov3": row['able5Hov3']
                }
            }
        }
        
        flattened = flatten_dictionary(row_dict)
        flattened_data.append(flattened)
    
    result_df = pd.DataFrame(flattened_data)
    
    return result_df

def main():
    try:
        print("Starting to process the CSV file...")
        result_df = process_large_csv('Book2.csv')
        result_df.to_csv('flattened_output.csv', index=False)
        print("\nProcessing completed successfully!")
        print(f"Total rows processed: {len(result_df)}")
        print(f"Total columns in flattened structure: {len(result_df.columns)}")
        print("\nSample of flattened data (first 2 rows):")
        print(result_df.head(2).to_string())
        print("\nFlattened column names:")
        for col in sorted(result_df.columns):
            print(col)
            
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()


# In[16]:


#ANSWER 4
import pandas as pd
from typing import List, Set, Tuple
from collections import Counter

def generate_unique_permutations(nums: List[int]) -> List[List[int]]:
    
    def backtrack(counter: Counter, curr_perm: List[int], result: List[List[int]]) -> None:
        if len(curr_perm) == len(nums):
            result.append(curr_perm[:])
            return
        for num in counter:
            if counter[num] > 0:
                curr_perm.append(num)
                counter[num] -= 1
                backtrack(counter, curr_perm, result)
                curr_perm.pop()
                counter[num] += 1
    
    result = []
    backtrack(Counter(nums), [], result)
    return result

def process_dataset_permutations(file_path: str) -> dict:
    permutation_sets = {}
    for idx, row in df.iterrows():
        able2_nums = [
            row['able2Hov2'], 
            row['able2Hov3']
        ]
        able2_nums = [x for x in able2_nums if pd.notna(x)]
        able3_nums = [
            row['able3Hov2'],
            row['able3Hov3']
        ]
        able3_nums = [x for x in able3_nums if pd.notna(x)]
        if able2_nums:
            key = f"Row_{idx}_able2"
            permutation_sets[key] = generate_unique_permutations(able2_nums)
            
        if able3_nums:
            key = f"Row_{idx}_able3"
            permutation_sets[key] = generate_unique_permutations(able3_nums)
    
    return permutation_sets

def main():
    try:
        print("Processing dataset and generating permutations...")
        permutation_results = process_dataset_permutations('Book2.csv')
        
        print("\nSample of generated permutations:")
        sample_count = 0
        for key, perms in permutation_results.items():
            if sample_count >= 5:  # Show first 5 samples
                break
            print(f"\n{key}:")
            print(f"Original numbers resulted in {len(perms)} unique permutations:")
            for perm in perms:
                print(perm)
            sample_count += 1
        total_perms = sum(len(perms) for perms in permutation_results.values())
        print(f"\nTotal number of sets processed: {len(permutation_results)}")
        print(f"Total number of permutations generated: {total_perms}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")

def test_simple_case():
    print("\nTesting simple case [1, 1, 2]:")
    test_nums = [1, 1, 2]
    result = generate_unique_permutations(test_nums)
    print(f"Input: {test_nums}")
    print("Output:")
    for perm in result:
        print(perm)

if __name__ == "__main__":
    main()
    
    test_simple_case()


# In[17]:


#ANSWER 6

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000  # Earth's radius in meters
    
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c

def create_coordinate_pairs(df: pd.DataFrame) -> List[Tuple[float, float]]:
    location_coords = {
        'Montgomery': (32.3792, -86.3077),
        'Black': (33.5186, -86.8104),
        'Emerald': (27.9944, -82.4453),
        'Foley': (30.4065, -87.6836),
        'Whittier': (33.9792, -118.0327),
        'SR': (40.7589, -73.9851),
        'San': (37.7749, -122.4194),
        'Lorain-Elyria': (41.3687, -82.1076),
        'N.Ridgeville-Cleveland': (41.4398, -81.9429)
    }
    
    coordinates = []
    for _, row in df.iterrows():
        if row['name'] in location_coords:
            coordinates.append(location_coords[row['name']])
    
    return coordinates

def process_dataset_coordinates(file_path: str) -> pd.DataFrame:
    
    coordinates = create_coordinate_pairs(df)
    
    if not coordinates:
        return pd.DataFrame(columns=['latitude', 'longitude', 'distance', 'location_name'])
    
    result_data = []
    for i, (lat, lon) in enumerate(coordinates):
        if i == 0:
            distance = 0
        else:
            prev_lat, prev_lon = coordinates[i-1]
            distance = haversine_distance(prev_lat, prev_lon, lat, lon)
        location_name = df.iloc[i]['name']
        
        result_data.append({
            'latitude': lat,
            'longitude': lon,
            'distance': distance,
            'location_name': location_name,
            'start_time': df.iloc[i]['startTime'],
            'end_time': df.iloc[i]['endTime'],
            'start_day': df.iloc[i]['startDay'],
            'end_day': df.iloc[i]['endDay']
        })
    
    return pd.DataFrame(result_data)

def analyze_route_statistics(df: pd.DataFrame) -> dict:
    stats = {
        'total_distance': df['distance'].sum(),
        'num_locations': len(df),
        'avg_distance_between_points': df['distance'][1:].mean() if len(df) > 1 else 0,
        'max_distance_between_points': df['distance'].max(),
        'unique_locations': df['location_name'].nunique(),
        'most_frequent_location': df['location_name'].mode().iloc[0],
        'total_unique_days': len(set(df['start_day'].unique()) | set(df['end_day'].unique()))
    }
    
    return stats

def main():
    try:
        print("Processing dataset coordinates...")
        df = process_dataset_coordinates('Book2.csv')
        stats = analyze_route_statistics(df)
        print("\nFirst few coordinates with distances:")
        print(df.head().to_string())
        
        print("\nRoute Statistics:")
        print(f"Total number of locations: {stats['num_locations']}")
        print(f"Total distance covered: {stats['total_distance']/1000:.2f} km")
        print(f"Average distance between points: {stats['avg_distance_between_points']/1000:.2f} km")
        print(f"Number of unique locations: {stats['unique_locations']}")
        print(f"Most frequent location: {stats['most_frequent_location']}")
        print(f"Total unique days in schedule: {stats['total_unique_days']}")
        
        df.to_csv('processed_coordinates.csv', index=False)
        print("\nResults saved to 'processed_coordinates.csv'")
        
        unique_locations = df['location_name'].unique()
        distance_matrix = pd.DataFrame(index=unique_locations, columns=unique_locations)
        
        for loc1 in unique_locations:
            for loc2 in unique_locations:
                if loc1 == loc2:
                    distance_matrix.loc[loc1, loc2] = 0
                else:
                    coord1 = df[df['location_name'] == loc1].iloc[0][['latitude', 'longitude']]
                    coord2 = df[df['location_name'] == loc2].iloc[0][['latitude', 'longitude']]
                    dist = haversine_distance(coord1['latitude'], coord1['longitude'],
                                           coord2['latitude'], coord2['longitude'])
                    distance_matrix.loc[loc1, loc2] = dist / 1000  # Convert to kilometers
        
        print("\nDistance Matrix between Locations (km):")
        print(distance_matrix.round(2))
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")

if __name__ == "__main__":
    main()


# In[18]:


#ANSWER 7
def transform_matrix(matrix):
    n = len(matrix)
    def rotate_90_clockwise(mat):
        for i in range(n):
            for j in range(i, n):
                mat[i][j], mat[j][i] = mat[j][i], mat[i][j]
        for i in range(n):
            mat[i].reverse()  
        return mat
    rotated = [row[:] for row in matrix]
    rotated = rotate_90_clockwise(rotated)
    result = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated[i]) - rotated[i][j]
            col_sum = sum(rotated[k][j] for k in range(n)) - rotated[i][j]
            result[i][j] = row_sum + col_sum
    
    return result

def test_transform():
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    result = transform_matrix(matrix)
    print("Original matrix:")
    for row in matrix:
        print(row)
    print("\nTransformed matrix:")
    for row in result:
        print(row)
    
    return result


# In[19]:



#ANSWER 8
def check_time_coverage(file_path):
    df = pd.read_csv(file_path, usecols=['id', 'id_2', 'startDay', 'startTime', 'endDay', 'endTime'])
    days_map = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }
    
    def convert_to_minutes(time_str):
        """Convert time string to minutes since midnight"""
        if time_str == '23:59:59':
            return 24 * 60  # End of day
        hours, minutes, _ = map(int, time_str.split(':'))
        return hours * 60 + minutes
    
    def check_pair_coverage(group):
        start_days = group['startDay'].map(days_map).values
        end_days = group['endDay'].map(days_map).values
        start_times = group['startTime'].apply(convert_to_minutes).values
        end_times = group['endTime'].apply(convert_to_minutes).values
        coverage = np.zeros((7, 24), dtype=bool)
        
        for i in range(len(group)):
            start_day = start_days[i]
            end_day = end_days[i]
            start_time = start_times[i]
            end_time = end_times[i]
            if end_day < start_day:
                end_day += 7
            start_hour = start_time // 60
            end_hour = (end_time + 59) // 60  # Include partial hours
            current_day = start_day
            while current_day <= end_day:
                day_idx = current_day % 7
                
                if current_day == start_day:
                    day_start_hour = start_hour
                else:
                    day_start_hour = 0
                    
                if current_day == end_day:
                    day_end_hour = end_hour
                else:
                    day_end_hour = 24
                
                coverage[day_idx, day_start_hour:day_end_hour] = True
                current_day += 1

        return not coverage.all()

    chunk_size = 10000
    result_chunks = []
    
    for chunk_df in pd.read_csv(file_path, usecols=['id', 'id_2', 'startDay', 'startTime', 'endDay', 'endTime'], 
                               chunksize=chunk_size):
        chunk_result = chunk_df.groupby(['id', 'id_2']).apply(check_pair_coverage)
        result_chunks.append(chunk_result)

    result = pd.concat(result_chunks)
    result = result[~result.index.duplicated(keep='first')]
    
    return result

if __name__ == "__main__":
    file_path = "C:\\Users\\DELL\\Downloads\\dataset-1.csv"
    
    print("Starting analysis...")
    result = check_time_coverage(file_path)
    
    print("\nResults summary:")
    print(f"Total number of (id, id_2) pairs: {len(result)}")
    print(f"Number of pairs with incomplete coverage: {result.sum()}")
    print(f"Number of pairs with complete coverage: {(~result).sum()}")
    
    incomplete_pairs = result[result].head()
    if len(incomplete_pairs) > 0:
        print("\nExample pairs with incomplete coverage:")
        print(incomplete_pairs)


# In[ ]:




