import  pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from Tools.demo.sortvisu import steps
from sklearn.compose import ColumnTransformer
from  sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from lazypredict.Supervised import LazyRegressor

data = pd.read_csv("StudentScore.xls")
# print(data.info())
# print(data[["math score", "reading score", "writing score"]].corr())

def convert_level(level):
    if level == "some high school":
        level = "high school"
    return level

# nhin diem phan bo hoi kho  => visualization
# sn.histplot(data["math score"]) #muon visualation cai cot nao, cot nay phai co ten giong trong file
# plt.title("Math score distribution")
# plt.savefig("math_score.jpg")

# sn.histplot(data["reading score"])
# plt.title("Reading score distribution")
# plt.savefig("reading score.png")

#split data
target = "math score"
x = data.drop([target], axis = 1)  #axis = 1 => drop theo cot, khong de gi thi default theo hang
y = data[target]
x["parental level of education"] = x["parental level of education"]. apply(convert_level)
# print(x["parental level of education"].unique())
# split data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, random_state= 42)

# Kiem tra cot gender xem chi co 2 gioi tinh la male or female hay con gioi tinh nao nua khong
# print(x["gender"].unique())

# gia su o du lieu bi khuyet => xu li bang lenh sau
# Tom lai: Su dung layer roi cho du lieu di qua.
# imputer = SimpleImputer(strategy= "mean") #=> dung gia tri trung binh cua cac o con lai de dien vao o nay
# x["reading score"] = imputer.fit_transform(x[["reading score"]])
# scaler = StandardScaler()
# x["reading score"] = scaler.fit_transform(x[["reading score"]])
# print(x["reading score"])

# Neu nhu co 10 layer, ta phai dung fit_transform 10 lan => bat tien
# Trong sklearn, dung ham pipeline de gom cac layer thanh 1
num_transformer = Pipeline(steps= [
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# result = num_transformer.fit_transform(x_train[["reading score"]])
# print(result)
# for i, j in zip(x_train["reading score"], result):
#     print("Before {} After {}". format(i, j))
# print(x["parental level of education"].unique())
# phai dinh nghia vi oridinal co thu bac. Neu k dinh nghia, no mac dinh theo thu tu abc
education_values = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree", "master's degree" ]
gender_values = ["male", "female"]
lunch_values = x_train["lunch"].unique()
test_values = x_train["test preparation course"].unique()
ord_transformer = Pipeline(steps = [
    ("imputer", SimpleImputer(strategy= "most_frequent")),
     # ("imputer", SimpleImputer(strategy="constant", fill_value="unknown"),
    ("encoder", OrdinalEncoder(categories=[education_values, gender_values, lunch_values, test_values]))
])

nom_transformer = Pipeline(steps = [
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("encoder", OneHotEncoder(sparse_output= False))  # Viet kieu day du, mac dinh la false
])

# result = nom_transformer.fit_transform(x_train[["race/ethnicity"]])
# for i, j in zip(x_train["race/ethnicity"], result):
#     print("Before {} After {}". format(i,j))

#     Chung ta da co 3 loai du lieu, bay gio chung ta muon biet tranformer nao duoc dung cho loai du lieu nao
preprocessor = ColumnTransformer(transformers = [
    ("num_features", num_transformer, ["reading score", "writing score"]),
    ("ordinal_features", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nominal_features", nom_transformer, ["race/ethnicity"]),
])

#tien xu li du lieu va fit vao mo hinh => Dung pipeline tiep
reg = Pipeline(steps = [
    ("preprocessor", preprocessor),
    # ("regressor", RandomForestRegressor())
])
x_train = reg.fit_transform(x_train)
x_test = reg.transform(x_test)

lazy_reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = lazy_reg.fit(x_train, x_test, y_train, y_test)
print(predictions)
# Sau khi di qua xu li du lieu va mo hinh, dong sau day dung de in du lieu ra de xem thu
# result = reg.fit_transform(x_train)

# reg.fit(x_train, y_train)
# y_predict = reg.predict(x_test)
# print("MAE {}". format(mean_absolute_error(y_test, y_predict))) #Trung binh gia tri thuc te voi gia tri du doan
# print("MSE {}". format(mean_squared_error(y_test, y_predict))) # binh phuong cua cai o tren
# # R2 score, coefficient of determination => cang gan 1 cang tot, cang gan khong cang toi
# print("R2 {}". format(r2_score(y_test, y_predict)))
# for i, j in zip(y_test, y_predict):
#     print("Actual {} Predict {}". format(i, j))

# boi vi chugn ta dang xai pipeline , he thong dang khong biet parameters cua processor hay regressor
# Vi vay, chung ta phai them cai regressor__ de he thong no hieu la chung ta dang dung cho cai nao
# Tuy nhien o cai model nay (regressor), ten parameters se giong nhau, nhung gia tri trong parameter se thay doi
# parameters = {
#     "regressor__n_estimators": [50, 100, 200, 500],
#     "regressor__criterion": ["squared_error", "absolute_error", "poisson"],
#     "regressor__max_depth": [None, 5, 10, 20],
#     "regressor__max_features": ["sqrt", "log2"],
#     "preprocessor__num_features__imputer__strategy": ["mean","median" ]

# }

# Day la mohinh regression, score accuracy chi danh cho classification thoi
# model = GridSearchCV(reg, param_grid= parameters, scoring= "r2", cv = 6, n_jobs = 8, ve)
# model.fit(x_train, y_train)
# print(model.best_score_)
# print(model.best_params_)

# Neu dung GridSearch => So luong line chay ra se la 4x3x4x2x6 => rat nhieu. chung ta co the dung RandomizedSearch de gioi han no lai.
# model = RandomizedSearchCV(reg, param_distributions= parameters, scoring= "r2", cv = 6, n_jobs = 8, verbose=2, n_iter=20)
# model.fit(x_train, y_train)
# print(model.best_score_)
# print(model.best_params_)
