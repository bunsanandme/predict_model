import pandas as pd
from sklearn.neighbors import LocalOutlierFactor


def proccess_df(df):
    # Выведение нового признака на основе кол-ва комнат
    # Если это студия, то ставится ноль в значении
    for i in df["type_"]:
        if i[0] == "С":
            df = df.replace(i, 1)
        else:
            df = df.replace(i, int(i[0]))

    # Удаление строк с пустыми значениями
    df = df.dropna(axis=0, subset=["area"])

    # Удаление столбца с адресом, метро и датой, т.к это неважный признак
    # Это важная строка, т.к мы здесь контролируем кол-во переменных от которых зависит результат, в данном случае
    # 4 переменных: площадь, этажность, этаж, кол-во комнат
    df = df.drop(
        columns=["address", "metro_line", "metro_station", "date", "update_date"])
    # Убираем выбросы, число в функции показывает, из скольки значений будет выбираться выброс
    model = LocalOutlierFactor(n_neighbors=20)
    y = model.fit_predict(df)
    df = df.drop(labels=df.index[y == -1])
    return df

    # Удаление строк с пустыми значениями
    # df = df.dropna(axis=0, subset=["area"])
    # df = df.dropna(axis=0, subset=["floor"])
    # df = df.dropna(axis=0, subset=["full_floor"])



path_to_train = "data/train.csv"
path_to_test = "data/test.csv"

train_df = pd.read_csv(path_to_train)
test_df = pd.read_csv(path_to_test)

proccesed_train_df = proccess_df(train_df)
proccesed_test_df = proccess_df(test_df)

print(proccesed_train_df.info())
