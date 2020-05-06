import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from pandas.plotting import scatter_matrix
from IPython.display import display, HTML

# Функция read_file используется для чтения файла в зависимости от его формата
# path - путь к файлу
# typefile - можно указать заранее тип файла (например, csv, json), чтобы не ориентироваться на расширение файла
# optimization - позволяет по умолчанию выполнить оптимизацию файла в памяти
# **kwargs - передача специальных аргументов для функции чтения файла (например, delimeter, index_col)
# При успешном определении типа файла либо при заданном типе файла функция возвращает результат
# исполнения функции _read_file
def read_file(path, type_file="", optimization=False, **kwargs):
    if (type_file!=""):
        return _read_file(path, type_file, optimization=optimization, **kwargs)
    if (path[-4:]==".csv"):
        return _read_file(path, type_file="csv", optimization=optimization, **kwargs)
    if (path[-5:]==".json"):
        return _read_file(path, type_file="json", optimization=optimization, **kwargs) 
    if (type_file==""):
        raise NameError("Please, use typefile parameter for reading this file")

# Функция _read_file производит непосредственное чтение файлов с определенным ранее типом
# path - путь к файлу
# typefile - определенный ранее тип файла (например, csv, json)
# optimization - выполнение или невыполнение оптимизации
# **kwargs - передача специальных аргументов для функции чтения файла (например, delimeter, index_col)
# Функция возвращает считанный датафрейм
def _read_file(path, type_file, optimization, **kwargs):
    if (type_file == "csv"):
        print("reading csv file...")
        df = pd.read_csv(path, **kwargs)
    if (type_file == "json"):
        print("reading json file...")
        df = pd.read_json(path, **kwargs)
    if (optimization==True):
        df = optimize_mem_usage(df, True)
    get_mem_usage(df)
    return df

# Функция get_mem_usage предоставляет информацию о размере датафрейма
# df - датафрейм, для которого требуется узнать информацию о размере
# print_inf - параметр, в зависимости от которого происходит или не происходит вывод через Print
# Функция возвращает размер датафрейма
def get_mem_usage(df, print_inf = False):
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
        print("{}: {}/{} ({}%)".format(name, cellNull, cellAll, proc))

# Выводит количество данных каждого типа
# df - датафрейм		
def count_types(df):
    print('Every column:')
    print(df.dtypes)
    print('-----')
    print('Summary:')
    print(df.dtypes.value_counts())
    print('-----')
    print('All data:')
    d = dict()
    cellAll = df.shape[0]
    for name in df.columns:
        cellType = str(df[name].dtype)
        
        if not cellType in d:
            d[cellType] = 0
        d[cellType] += cellAll
    print(d)
	

# Функция get_corr_matrix вычисляет корреляцию между полями данных
# df - исследуемый датафрейм
# corr_method - метод корреляции, по умолчанию значение 'pearson'
# Функция возвращает матрицу корреляции для вывода или дальнейшей обработки
def get_corr_matrix(df, corr_method='pearson'):
    return df.corr(method=corr_method)

# Функция plot_corr_matrix строит тепловую карту корреляционной матрицы
# df - исследуемый датафрейм
# figsize - размер выводимого графика, по умолчанию (10,5)
# corr_method - метод корреляции, по умолчанию значение 'pearson'
# Функция выводит тепловую карту
def plot_corr_matrix(df, figsize=(10,5), corr_method='pearson'):
    f,ax = plt.subplots(figsize=figsize)
    sns.heatmap(df.corr(method=corr_method),annot=True, linewidths=.1, fmt='.1f', ax=ax)
    plt.show()

# Функция plot_dependency_chart строит графики зависимости одного поля (y) от других полей датафрейма
# df - датафрейм, содержащий x-столбцы
# y - текст - название столбца
# figsize - размер выводимого графика, по умолчанию (10,10)
# chart_in_str - количество графиков в каждой строке, по умолчанию 3
# Функция выводит полученные графики
def plot_dependency_chart(df, y, figsize=(10,10), chart_in_str=3):
    plt.figure(figsize=figsize)
    df = df.select_dtypes(exclude='O')
    row = math.ceil(len(df.columns.drop([y])) / chart_in_str)
    for iterator, column in enumerate(df.columns.drop([y]), start=1):
        plt.subplot(row,chart_in_str,iterator)
        plt.plot(df.loc[:,column], df[y], '.')
        plt.xlabel(column)
        plt.ylabel(y)
    plt.show()

    #for i in range(0,len(df.columns), chart_in_str):
    #    sns.pairplot(data=df,
    #                 x_vars=df.columns[i:i+chart_in_str],
    #                 y_vars=[y])

# Функция plot_scatter_matrix строит матрицу графиков взаимной зависимости всех полей датафрейма ко всем полям
# df - исследуемый датафрейм
# Функция выводит полученные графики
def plot_scatter_matrix(df, **kwargs):
    sns.pairplot(df, **kwargs)
    plt.show()

# Функция plot_scatter строит график разброса для каждого числового столбца датафрейма
# df - исследуемый датафрейм
# y - название поля, для которого строится боксплот
# unique_filter - позволяет убрать нагромажденные графики (по умолчанию = 10; если значение отрицательное, то фильтр не используется)
# Функция выводит построенные графики
def plot_scatter(df, y, unique_filter=10):
    df.boxplot()
    plt.show()
    #df = df.select_dtypes(exclude='O')
    for iterator, column in enumerate(df.columns.drop([y]), start=1):
        if df[column].nunique()>unique_filter and unique_filter>0:
            continue
        sns.boxplot(data=df, x=column, y=y)
        plt.show()
                                  
# Функция get_description позволяет получить базовые стохастические характеристики (наибольшее, наименьшее, среднее, мода, медиана, ско)
# df - исследуемый датафрейм
# Функция возвращает датафрейм с полученными характеристиками 
def get_description(df):
    result = df.describe()
    mode = df.mode(numeric_only=True).iloc[0].rename('mode')
    med = df.median(skipna=True).rename('median')
    result = result.append(mode).append(med)
    return result
    
# Функция get_groip_description позволяет получить базовые стохастические характеристики (наибольшее, наименьшее, среднее, мода, медиана, ско) после группировки данных по определенному полю
# df - исследуемый датафрейм
# gb_column - поле для группировки
# Функция выводит базовые характеристики каждого поля после группировки
def get_group_description(df, gb_column):
    for column in df.columns.drop(gb_column):
        result = df[[column,gb_column]].groupby(gb_column).describe()
        if df[column].dtype != object:
            mode = df[[column,gb_column]].groupby(gb_column)[column].apply(lambda x: x.mode()[0])
            mode = pd.DataFrame(mode)
            mode.columns = pd.MultiIndex.from_product([mode.columns, ['mode']])
            result = result.merge(mode, how='left', on=gb_column)

            med = df[[column,gb_column]].groupby(gb_column).median()
            med.columns = pd.MultiIndex.from_product([med.columns, ['median']])
            result = result.merge(med, how='left', on=gb_column)
        display(result)
        
    
