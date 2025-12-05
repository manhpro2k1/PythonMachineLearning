import pandas as pd
import matplotlib.pyplot as plt  # dung de ve do thi
import seaborn as sn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from math import sqrt

# Tom tat lai cac buoc
# 1. doc du lieu vao
# 2. phan chia du lieu theo chieu doc(feature sang 1 ben, target sang 1 ben)
# 3. phan chia du lieu theo chieu ngang (xtrain, ytrain, xtest, ytest)
# 4. Tien xu li du lieu => dung scalar ...day la du lieu dung de di du doan mo hinh.
# 5. khoi tao mo hinh (cls, reg )  ==> ham fit dung de train mo hinh.
# 6. Sau khi train mo hinh xong ==> lay mo hinh di du doan
data = pd.read_csv("diabetes.csv")

# print(type(data))     # kieu du lieu
# print(data.head(7))   # so cot dau tien
# print(data.info())    # in ra thuoc tinh cua nhung cot khac nhau
# print(data.describe())  # in ra mean , max, gia tri o 25%, 50%, 70%
# result = data.corr()    #"Cooretion matrix", ma tran tuog quan
# print(data["Outcome"].value_counts())   # dem so nguoi benh va khong benh o ket qua

#Ve do thi
#b1. Dinh nghia cai figure
# plt.figure(figsize=(7,7))
# sn.displot(data["Outcome"])
# plt.title("Diabetes Distribution")
# plt.savefig("Diabetes.jpg")


#set feature and target  => tach theo chieu doc
target = 'Outcome'
x = data.drop(target, axis=1)  # tach target va feature, bo target ==> bo outcome
y = data[target]   #=> chi giu lai outcome, bo feature

#plit  => phan chia thanh bo train voi bo test. => tach theo chieu ngang
x_train, x_test, y_train,  y_test = train_test_split(x, y, train_size=0.8, random_state =42)
#=> uu tien train size
#(x_train + y_train = tong phan tu)  => uu tien train lon hon, test nho hon
# Neu khong dung random state => moi lan run program thi ket qua train test duoc chon ngau nhien, khac nhau giua cac lan hay
# neu dung random_state => Moi lan chay ra duoc ket giong nhau, de de so sanh, nguoi ta thuong chon so 42
# print(len(x_train), len(y_train))
# print(len(x_test), len(y_test))

#Preprocessing
#Dung scaler de rang buoc cac feature o cung 1 range, vi du prenancies tu 1-10 tu glucozo cung tu 1-10, neu khac nhau mo hinh kho kiem soat
# (cai scaler nay di phong van se bi hoi)

scaler = StandardScaler()
#ki vong hoac goi la gia tri trung binh
# print(scaler.mean_)
# phuong sai (var)
# print(sqrt(scaler.var_))
# Test thu cho 1 cot thoi
# x_train = scaler.fit_transform(x_train[["Pregnancies"]])
# for i, j in zip(x_train["Pregnancies"].values, x_train):
#      print("before {} after {}".format(i,j))

# test cho tat ca bo train.
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# cau hoi phong van: phan biet  => vi du ve tho may
# fit_transform
# fit
# transform
# Note: Dung fit_transform cho bo train
# Chi dung transform cho bo test thoi

# Buoc tiep theo: Dua sau lieu sau preprocessing qua mo hinh
# Dau tien phai khoi tao svc cai da

parameters= {
    "n_estimators": [50, 100, 200],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 5, 10],
    "max_features": ["sqrt", "log2"]
}

cls = GridSearchCV(RandomForestClassifier(), param_grid= parameters, scoring= "accuracy", cv = 6, verbose= 2, n_jobs= 8)
cls.fit(x_train, y_train)

print(cls.best_score_)
print(cls.best_params_)
# Sau do dem di du doan
y_predict = cls.predict(x_test)
# in ra xem thu
# for i, j in zip(y_test, y_predict):
#      print("Actual {} predict{}".format(i,j))
print(classification_report(y_test, y_predict)) #classification_report: day la cai dung de in accuracy


# tao ra report nhin cho dep hon
# cm = np.array(confusion_matrix(y_test, y_predict, labels= [0, 1]))
# confusion= pd.DataFrame(cm, index=["Not Diabetic", "Diabetic"], columns=["Not Diabetic", "Diabetic"])
# sn.heatmap(confusion, annot= True, fmt = 'g')
# plt.savefig("Diebetes_prediction.jpg")
