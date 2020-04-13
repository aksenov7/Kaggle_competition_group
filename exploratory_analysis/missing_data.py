def func(df):
    cellAll = df.shape[0]
    for i in range(df.shape[1]):
        name = df.columns[i]
        cellNull = sum(df[name].isnull())
        proc = int(cellNull / cellAll * 100)
        print("{}: {} ({}%)".format(name, cellNull, proc))
