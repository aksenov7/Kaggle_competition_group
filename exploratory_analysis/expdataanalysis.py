import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from pandas.plotting import scatter_matrix
from IPython.display import display, HTML


# Функция resolve_file_type возвращает тип файла
# path - путь к файлу
def resolve_file_type(path, file_type):
    if file_type:
        return file_type
    types = [".csv", ".json"]
    for e in types:
        if path.endswith(e):
            return e[1:]
    raise NameError("Please, use file_type parameter for reading this file")


# Функция read_file используется для чтения файла в зависимости от его формата
# path - путь к файлу
# file_type - можно указать заранее тип файла (например, csv, json), чтобы не ориентироваться на расширение файла
# optimization - позволяет по умолчанию выполнить оптимизацию файла в памяти
# **kwargs - передача специальных аргументов для функции чтения файла (например, delimeter, index_col)
# При успешном определении типа файла либо при заданном типе файла функция возвращает результат
# исполнения функции _read_file
def read_file(path, file_type="", optimization=True, print_inf=False, **kwargs):
    file_type = resolve_file_type(path, file_type)

    return _read_file(path, file_type=file_type, optimization=optimization, print_inf=print_inf, **kwargs)


# Функция _read_file производит непосредственное чтение файлов с определенным ранее типом
# path - путь к файлу
# file_type - определенный ранее тип файла (например, csv, json)
# optimization - выполнение или невыполнение оптимизации
# **kwargs - передача специальных аргументов для функции чтения файла (например, delimeter, index_col)
# Функция возвращает считанный датафрейм
def _read_file(path, file_type, optimization, print_inf=False, **kwargs):
    print("Reading " + file_type + " file...")
    if file_type == "csv":
        df = pd.read_csv(path, **kwargs)
    elif file_type == "json":
        df = pd.read_json(path, **kwargs)
    else:
        raise Exception
    print("success reading")
    if optimization:
        df = optimize_mem_usage(df, True, print_inf=print_inf)
    get_mem_usage(df)
    return df


# Функция get_mem_usage предоставляет информацию о размере датафрейма
# df - датафрейм, для которого требуется узнать информацию о размере
# print_inf - параметр, в зависимости от которого происходит или не происходит вывод через Print
# Функция возвращает размер датафрейма
def get_mem_usage(df, print_inf=False):
    df_mem = df.memory_usage().sum() / 1024 ** 2
    if print_inf:
        print("Memory usage of dataframe is {:.2f} MB".format(df_mem))
    return df_mem


from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype


# Функция optimize_mem_usage используется для оптимизации используемой датафреймом памяти
# df - оптимизируемый датафрейм
# use_float16 - позволяет использовать или не использовать float16, по умолчанию не используется
# print_inf - параметр, в зависимости от которого происходит или не происходит вывод через Print
# Функция возвращает оптимизированный датафрейм
def optimize_mem_usage(df, use_float16=False, print_inf=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        
    """
    if print_inf:
        print("\nStarting memory optimization...")
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

    if print_inf:
        print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
        print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
        print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


# Выводит количество пропусков в данных по каждому слобцу
# df - датафрейм
def print_missing_data(df, print_level=1):
    all_cells = df.shape[0]
    summ = 0
    for i in range(df.shape[1]):
        name = df.columns[i]
        null_cells = sum(df[name].isnull())
        summ += null_cells
        if print_level or null_cells:
            proc = int(null_cells / all_cells * 100)
            print("{}: {}/{} ({}%)".format(name, null_cells, all_cells, proc))
    if not summ:
        print("There is no missing data found")


# Выводит количество данных каждого типа
# df - датафрейм		
def print_count_types(df):
    print('Columns:')
    print(df.dtypes)
    print('-----')
    print('Summary:')
    print(df.dtypes.value_counts())
    print('-----')
    print('All data types:')
    d = dict()
    all_cells = df.shape[0]
    for name in df.columns:
        cell_type = str(df[name].dtype)

        if not (cell_type in d):
            d[cell_type] = 0
        d[cell_type] += all_cells
    print(d)


# Функция get_corr_matrix вычисляет корреляцию между полями данных
# df - исследуемый датафрейм
# corr_method - метод корреляции, по умолчанию значение 'pearson'
# Функция возвращает матрицу корреляции для вывода или дальнейшей обработки
def get_corr_matrix(df, corr_method='pearson'):
    return df.corr(method=corr_method)

# Функция plot_corr_matrix вычисляет корреляцию между полями данных
# и строит тепловую карту корреляционной матрицы
# df - исследуемый датафрейм
# figsize - размер выводимого графика, по умолчанию (10,5)
# corr_method - метод корреляции, по умолчанию значение 'pearson'
# Функция выводит тепловую карту
def plot_corr_matrix(df, figsize=(10,5), corr_method='pearson'):
    f, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df.corr(method=corr_method), annot=True, linewidths=.1, fmt='.2f', ax=ax)
    plt.show()


# Функция plot_scatter строит графики зависимости одного поля (y) от других полей датафрейма
# df - датафрейм, содержащий x-столбцы
# y - текст - название столбца
# figsize - размер выводимого графика, по умолчанию (10,10)
# chart_in_str - количество графиков в каждой строке, по умолчанию 3
# Функция выводит полученные графики
def plot_scatter(df, y, figsize=(10, 10), chart_in_str=3):
    plt.figure(figsize=figsize)
    df = df.select_dtypes(exclude='O')
    row = math.ceil(len(df.columns.drop([y])) / chart_in_str)
    for iterator, column in enumerate(df.columns.drop([y]), start=1):
        try:
            plt.subplot(row, chart_in_str, iterator)
            plt.plot(df.loc[:, column], df[y], '.')
            plt.xlabel(column)
            plt.ylabel(y)
        except:
            pass
    plt.show()

    # for i in range(0,len(df.columns), chart_in_str):
    #    sns.pairplot(data=df,
    #                 x_vars=df.columns[i:i+chart_in_str],
    #                 y_vars=[y])


# Функция plot_scatter_matrix строит матрицу графиков взаимной зависимости всех полей датафрейма ко всем полям
# df - исследуемый датафрейм
# Функция выводит полученные графики
def plot_scatter_matrix(df, **kwargs):
    sns.pairplot(df, **kwargs)
    plt.show()


# Функция plot_boxplots строит графики выбросов для каждого числового столбца датафрейма
# df - исследуемый датафрейм
# y - название поля, для которого строится боксплот
# unique_filter - позволяет убрать нагромажденные графики
# (по умолчанию = 10; если значение отрицательное, то фильтр не используется)
# turn - меняет ось x и y местами
# **kwargs - передача специальных аргументов при необходимости
# Функция выводит построенные графики
def plot_boxplots(df, y, unique_filter=10, turn=False, cat_list=[], **kwargs):
    df.boxplot()
    plt.show()
    # df = df.select_dtypes(exclude='O')
    for iterator, column in enumerate(df.columns.drop([y]), start=1):
        if df[column].nunique() > unique_filter > 0:
            continue
        try:
            if not turn:
                sns.boxplot(data=df, x=column, y=y, **kwargs)
            else:
                sns.boxplot(data=df, x=y, y=column, **kwargs)
            plt.show()
        except ValueError:
            print(column, y, "may be not numeric")


# Функция get_description позволяет получить базовые стохастические характеристики (наибольшее, наименьшее, среднее, мода, медиана, ско)
# df - исследуемый датафрейм
# Функция возвращает датафрейм с полученными характеристиками 
def get_description(df):
    result = df.describe()
    mode = df.mode(numeric_only=True).iloc[0].rename('mode')
    med = df.median(skipna=True).rename('median')
    result = result.append(mode).append(med)
    return result


# Функция get_group_description позволяет получить базовые стохастические характеристики
# (наибольшее, наименьшее, среднее, мода, медиана, ско) после группировки данных по определенному полю
# df - исследуемый датафрейм
# gb_column - поле для группировки
# Функция выводит базовые характеристики каждого поля после группировки
def get_group_description(df, gb_column):
    for column in df.columns.drop(gb_column):
        try:
            result = df[[column, gb_column]].groupby(gb_column).describe()
            if df[column].dtype != object:
                mode = df[[column, gb_column]].groupby(gb_column)[column].apply(lambda x: x.mode()[0])
                mode = pd.DataFrame(mode)
                mode.columns = pd.MultiIndex.from_product([mode.columns, ['mode']])
                result = result.merge(mode, how='left', on=gb_column)

                med = df[[column, gb_column]].groupby(gb_column).median()
                med.columns = pd.MultiIndex.from_product([med.columns, ['median']])
                result = result.merge(med, how='left', on=gb_column)
            display(result)
        except:
            pass

# Функция subplot позволяет строить столбчатые диаграммы (выполняет проверку x и y на 0)
def subplot(x, y):
    if len(x) > 0 and len(y) > 0:
        fig, ax = plt.subplots()
        ax.bar(x, y)
        plt.show()

# Функция find_cat позволяет находить категориальные признаки
# df - исследуемый датафрейм
# n_unique - количество уникальных записей в столбце (по умолчания - 15)
def find_cat(df, n_unique=15):
    cat_list = []
    for column in df.columns:
        if (df[column].nunique()<=n_unique):
            cat_list.append(column)
    return cat_list
