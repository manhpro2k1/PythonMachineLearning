import pandas as pd
import matplotlib.pyplot as plt

# Datataset cua chung ta co hai cot la Time va Co2,
# Vi trong time series, sau moi iteration thi se lay gia tri phia sau lam feature va du doan cho gia tri tiep theo.
# vi vay, nen nen y tuong la: tao ra 4 cot nua giong nhu cot lCo2 hien tai => tao ra mot ham cho de quan li
def create_recursive_data(data, window_size = 5):
    return data


data = pd.read_csv("co2.csv")
# Trong file co2.csv co 2 cot, time va co2. Trong do cot time co kieu du lieu la "Objet", tuc la khong phia kieu du liue dang so
# thi dung object de bieu thi chung
# de xu li, ta can convert kieu du lieu nay sang kieu thoi gian
# print(data.info())
data["time"] = pd.to_datetime(data["time"])
# Fill missing value by interpolation, no use "median or mean" because this is time data => uncontinuous
data["co2"] = data["co2"].interpolate()
# print(data.info())
fig, ax = plt.subplots() #hàm dùng để tạo Figure + Axes, giúp bạn có chỗ để vẽ biểu đồ.
ax.plot(data["time"], data["co2"]) #Bieu thi su tuong quan giua truc hoanh va truc tung
ax.set_xlabel("Time") # dat ten cho cot do
ax.set_ylabel("CO2")  # dat ten cho cot do
plt.show()    # khi run chuong trinh thi no ra ne

# target = "co2"
# x = data.drop(target, axis = 1)
# y= data[target]

data = create_recursive_data(data, 5)

