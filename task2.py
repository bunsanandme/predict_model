import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import LocalOutlierFactor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV


def save_predicted_price(model):
    pred = model.predict(X_test)
    pd.Series(pred, name="price").to_csv("predicted.csv", index=False)


def proccess_df(df):
    # Функция аналогична из задания 1
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
    df = df.drop(
        columns=["address", "metro_line", "metro_station", "date", "update_date"])
    model = LocalOutlierFactor(n_neighbors=20)
    y = model.fit_predict(df)
    df = df.drop(labels=df.index[y == -1])
    return df


path_to_train = "data/train.csv"
path_to_test = "data/test.csv"

train_df = pd.read_csv(path_to_train)
test_df = pd.read_csv(path_to_test)

proccesed_train_df = proccess_df(train_df)
proccesed_test_df = proccess_df(test_df)

X_train, y_train = proccesed_train_df.drop(columns=["price"]), proccesed_train_df["price"]
X_test, y_test = proccesed_test_df.drop(columns=["price"]), proccesed_test_df["price"]

# Линейная регрессия
# Обучаем и выводим RMSE по train и test
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
print("Train RMSE (Linear Regression): ", mean_squared_error(lr_model.predict(X_train), y_train, squared=False))
print("Test RMSE (Linear Regression): ", mean_squared_error(lr_model.predict(X_test), y_test, squared=False), "\n")

# Дерево решений
# Эта модель имеет много гиперпараметров, поэтому мы будем искать оптимальные
dtr_model = DecisionTreeRegressor(
    random_state=2,
    max_depth=2,
    min_samples_leaf=2,
    min_samples_split=10)
dtr_model.fit(X_train, y_train)
print("Train RMSE (Decision Tree Regressor): ", mean_squared_error(dtr_model.predict(X_train), y_train, squared=False))
print("Test RMSE (Decision Tree Regressor): ", mean_squared_error(dtr_model.predict(X_test), y_test, squared=False),
      "\n")

# Вообще, для предсказывания всегда хорошо работает случайный лес.
# Мы сейчас обучим лес и найдем оптимальные гиперпараметры и сохраним

# Все параметры, которые будем использовать
param_grid = {'bootstrap': [False], 'n_estimators': [3, 10, 30, 70], 'max_features': [1, 2, 3, 4]}

forest_reg = RandomForestRegressor()
# Начинаем "перебирать" гиперпараметры и обучать на них
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_root_mean_squared_error')

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)
print("Train RMSE (Grid): ", abs(grid_search.best_score_))
print("Test RMSE (Grid): ", mean_squared_error(grid_search.predict(X_test), y_test, squared=False),
      "\n")

# Сохраним результаты
save_predicted_price(grid_search)
