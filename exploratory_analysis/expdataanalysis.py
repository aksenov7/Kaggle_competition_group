import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Функция read_file используется для чтения файла в зависимости от его формата
# path - путь к файлу
# typefile - можно указать заранее тип файла (например, csv, json), чтобы не ориентироваться на расширение файла
# optimization - позволяет по умолчанию выполнить оптимизацию файла в памяти
# delimeter - делитель для csv файлов, по умолчанию "," (запятая)
# При успешном определении типа файла либо при заданном типе файла функция возвращает результат
# исполнения функции _read_file
def read_file(path, type_file="", optimization=True, delimeter=','):
    if (type_file!=""):
        return _read_file(path, type_file, delimeter=delimeter, optimization=optimization)
    if (path[-4:]==".csv"):
        return _read_file(path, type_file="csv", delimeter=delimeter, optimization=optimization)
    if (path[-5:]==".json"):
        return _read_file(path, type_file="json", delimeter=delimeter, optimization=optimization) 
    if (type_file==""):
        raise NameError("Please, use typefile parameter for reading this file")

# Функция _read_file производит непосредственное чтение файлов с определенным ранее типом
# path - путь к файлу
# typefile - определенный ранее тип файла (например, csv, json)
# optimization - выполнение или невыполнение оптимизации
# delimeter - делитель для csv файлов
# Функция возвращает считанный датафрейм
def _read_file(path, type_file, optimization, delimeter):
    if (type_file == "csv"):
        print("reading csv file...")
        df = pd.read_csv(path, delimeter)
    if (type_file == "json"):
        print("reading json file...")
        df = pd.read_json(path)
    if (optimization==True):
        df = optimize_mem_usage(df, True)
    get_mem_usage(df)
    return df

# Функция get_mem_usage предоставляет информацию о размере датафрейма
# df - датафрейм, для которого требуется узнать информацию о размере
# print_inf - параметр, в зависимости от которого происходит или не происходит вывод через Print
# Функция возвращает размер датафрейма
def get_mem_usage(df, print_inf = True):
    df_mem = df.memory_usage().sum() / 1024**2
    if (print_inf == True):
        print("Memory usage of dataframe is {:.2f} MB".format(df_mem))
    return df_mem


from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

# Функция optimize_mem_usage используется для оптимизации используемой датафреймом памяти
# df - оптимизируемый датафрейм
# use_float16 - позволяет использовать или не использовать float16, по умолчанию не используется
# print_inf - параметр, в зависимости от которого происходит или не происходит вывод через Print
# Функция возвращает оптимизированный датафрейм
def optimize_mem_usage(df, use_float16=False, print_inf = False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        
    """
    
    start_mem = get_mem_usage(df, False)
    
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = get_mem_usage(df, False)
    
    if (print_inf == True):
        print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
        print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
        print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    
    return df

# Выводит количество пропусков в данных по каждому слобцу
# df - датафрейм
def missing_data(df):
    cellAll = df.shape[0]
    for i in range(df.shape[1]):
        name = df.columns[i]
        cellNull = sum(df[name].isnull())
        proc = int(cellNull / cellAll * 100)
        print("{}: {} ({}%)".format(name, cellNull, proc))

# Выводит количество данных каждого типа
# df - датафрейм		
def count_types(df):
    d = dict()
    cellAll = df.shape[0]
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            cellType = type(df.iloc[i, j]).__name__
            if not cellType in d:
                d[cellType] = 0
            d[cellType] += 1
    print(d)
	


def get_corr_matrix(df, corr_method='pearson'):
    return df.corr(method=corr_method)

def plot_corr_matrix(df, figsize=(10,5), corr_method='pearson'):
    f,ax = plt.subplots(figsize=figsize)
    sns.heatmap(df.corr(method=corr_method),annot=True, linewidths=.1, fmt='.1f', ax=ax)
    plt.show()
