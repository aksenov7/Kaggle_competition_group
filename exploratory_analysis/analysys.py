import expdataanalysis as eda
import matplotlib.pyplot as plt
import numpy as np


def explore(df=None, path="", file_type="", optimization=True,  print_level=0, cat_exp = True, n_unique=15, boxplots=True, ignore_list=None, group_list=None, **kwargs):
    """
    Производит разведовательный анализ данных
    1. Считывает данные (если указан путь к файлу)
    2. Находит пропуски данных
    3. Выводит количество данных каждого типа
    4. Находит матрицу корелляции
    5. Определяет стандартные стохастические характеристики
    6. Находит категориальные признаки, определяет стандартные стохастические характеристики после группировки данных, строит столбчатые диаграммы
    7. Строит матрицу графиков взаимной зависимости всех полей датафрейма
    8. Строит графики выбросов для каждого числового столбца датафрейма

    @Params
    df - Исследуемый датафрейм (опционально df или path). Если указан, path игнорируется
    path - Путь к файлу (опционально df или path)
    optimization (default = True) - True - если необходима оптимизация данных
    print_level (0, 1, 2) (вроде) - Отвечает за вывод доп информации
    cat_exp (default = True) - True - если необходим анализ категориальных признаков
    n_unique (default = 15) - Граница количества уникальных элементов в столбце (для поиска категориального признака)
    boxplots (default = True) - True, если необходимы графики выбросов для каждого числового столбца датафрейма
    ignore_list - Игнор-лист столбцов. Необходим, чтобы исключить построение ненужных мусорных графиков
    group_list - Список столбцов, значения которых по-умолчанию считаются категориальными

    @return df, corr_matrix, 

    """
    print_inf = False
    if print_level:
        print_inf = True

    print("Starting explore analysis")

    if not df:
        if path:
            # read data file
            df = eda.read_file(path, optimization=optimization, file_type=file_type, print_inf=print_inf)
        else:
            raise Exception("There is no @df - dataframe or @path - file path param found")
    else:
        if optimization:
            df = optimize_mem_usage(df, print_inf=print_inf)

    print("\nLooking for some data frame missing data...")
    eda.print_missing_data(df, print_level=print_level)

    print("\nStarted type analysis:")
    eda.print_count_types(df)

    print("\nCalculating correlation matrix...")
    corr_matrix = eda.get_corr_matrix(df.drop(ignore_list, axis = 1))
    if print_level:
        print(corr_matrix)

    eda.plot_corr_matrix(df.drop(ignore_list, axis = 1))

    print("\nCalculating base stochastic characteristics")
    print(eda.get_description(df))

    cat_list = []
    if cat_exp:
        cat_list = explore_cat(df, group_list, n_unique, ignore_list=ignore_list)

    print("\nCalculating plot_scatter_matrix")
    eda.plot_scatter_matrix(df.drop(ignore_list, axis = 1))

    if boxplots:
        print("\nCalculating boxplots")
        for e in df.columns:
            if e not in ignore_list:
                eda.plot_boxplots(df, e, turn=True, cat_list=cat_list)

    return df, corr_matrix, cat_list

# Функция explore_cat
# 1. Находит категориальные признаки
# 2. Определяет базовые стохастические характеристики для каждого из них
# 3. Строит столбчатые диаграммы для каждого из них
# @params
# df - Исследуемый датафрейм
# group_list - Список параметров, которые нужно добавить в список категориальных
# n_unique - Граница количества уникальных элементов в столбце (для поиска категориального признака)
# ignore_list - Список игнорируемых столбцов
# @return
# Возвращает list категориальных признаков
def explore_cat(df, group_list, n_unique, ignore_list=[]):
    print("\nLooking for some categorial features")
    cat_list = eda.find_cat(df, n_unique=n_unique)
    print(len(cat_list), "categorial features is(are) found")
    [print(e) for e in cat_list]


    for e in group_list:
        if e not in cat_list:
            cat_list.append(e)

    cat_list = [e for e in cat_list if e not in ignore_list]
    for e in cat_list:
        print("\nCalculating base stochastic characteristics grouped by", e)
        eda.get_group_description(df, e)



        print("\nCalculating subplots for", e)
        x = df[e].unique()
        final_x = []
        y = []
        for x_value in x:
            curr_y = df[e].tolist().count(x_value)
            if str(x_value) != 'nan':
                final_x.append(x_value)
                y.append(curr_y)

        eda.subplot(final_x, y)

    return cat_list




