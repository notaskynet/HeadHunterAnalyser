from typing import Dict, List
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re


nltk.download('stopwords')
nltk.download('punkt_tab')
russian_stopwords = stopwords.words('russian')

EXP_LEVELS = {'Нет опыта': 0, 'От 1 года до 3 лет': 1, 'От 3 до 6 лет': 2, 'Более 6 лет': 3}
SELECTED_COLUMNS = [
    'premium', 'name', 'has_test', 'response_letter_required','area_name',
    'published_at', 'created_at', 'snippet_requirement', 'snippet_responsibility','schedule_name',
    'professional_roles_name', 'experience_name', 'employment_name',
    'salary_from', 'salary_to', 'salary_currency', 'address_lat', 'address_lng'
]

def flatten(entry: Dict, prefix: str = None, ignore_keys: list = []) -> Dict:
    flatten_entry = {}
    
    for key, value in entry.items():
        new_prefix = f"{prefix}_{key}" if prefix else key
        
        if isinstance(value, dict):
            flatten_entry.update(flatten(value, prefix=new_prefix, ignore_keys=ignore_keys))
        
        elif isinstance(value, list):
            #print(value)
            for idx, subvalue in enumerate(value):
                list_prefix = f"{new_prefix}"
                if isinstance(subvalue, dict):
                    if key in ignore_keys:
                        flatten_entry.update(flatten(subvalue, prefix="", ignore_keys=ignore_keys))
                    else:
                        flatten_entry.update(flatten(subvalue, prefix=list_prefix, ignore_keys=ignore_keys))
                else:
                    flatten_entry[list_prefix] = subvalue
        else:
            flatten_entry[new_prefix] = value

    return flatten_entry


def merge_entries(entries: List[Dict]) -> Dict:
    merged_entries = {}
    for idx, entry in enumerate(entries):
        for key, value in entry.items():
            if key not in merged_entries:
                merged_entries[key] = [pd.NA] * idx
            merged_entries[key].append(value)
        for key in [key for key in merged_entries.keys() if key not in entry.keys()]:
            merged_entries[key].append(pd.NA)
    return merged_entries


def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    df['salary_from'] = pd.to_numeric(df['salary_from'], errors='coerce')
    df['salary_to'] = pd.to_numeric(df['salary_to'], errors='coerce')

    df['salary_to'] = df['salary_to'].fillna(df['salary_from'])
    df['salary_from'] = df['salary_from'].fillna(df['salary_to'])
    df['salary_currency'] = df['salary_currency'].fillna('RUR')
    
    df['salary_from'] = df['salary_from'].fillna(0).astype(int)
    
    df['salary'] = df['salary_from'] * df['salary_currency'].apply(lambda x: 1 if x == 'RUR' else 100)
    
    df = df.drop(['salary_from', 'salary_to', 'salary_currency'], axis=1)
    df["address_lat"] = pd.to_numeric(df["address_lat"], errors="coerce")
    df["address_lng"] = pd.to_numeric(df["address_lng"], errors="coerce")

    df = df.dropna(subset=["address_lat", "address_lng"])
    return df

def process_dataframe(df: pd.DataFrame, selected_columns: List[str] = SELECTED_COLUMNS) -> pd.DataFrame:
    df = df[selected_columns]
    df = fill_gaps(df.copy())
    df['has_test'] = df['has_test'].apply(lambda x: 1 if x == 'True' else 0)
    df['premium'] = df['premium'].apply(lambda x: 1 if x == 'True' else 0)
    df['response_letter_required'] = df['response_letter_required'].apply(lambda x: 1 if x == 'True' else 0)
    df['is_salary_set'] = df['salary'].apply(lambda x: "Не указана" if x == 0 else "Указана")
    df['exp_sort'] = df['experience_name'].apply(lambda x: EXP_LEVELS[str(x)])
    df = df.sort_values(by='exp_sort', ascending=True)
    df = df.drop('exp_sort', axis=1)
    return df
