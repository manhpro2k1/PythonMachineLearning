import pandas as pd
from mlflow.protos.databricks_uc_registry_messages_pb2 import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTEN
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
import  re

# Ham dung dinh nghia, lay 2 ki tu cuoi trong cot location
def my_function(loc):
    result = re.findall("\ [A-Z]{2}$", loc)
    if len(result) == 1: # = 1, boi vi ham findall luon tra ve list cos 1 phan tu ==> [" CA"]
        return result[0][1:]
    else:
        return loc

data = pd.read_excel("final_project.ods", engine="odf", dtype = str)
# Trong cot location, TX, Houton => chi can lay Houton thoi nen dung ham apply
data["location"] = data["location"].apply(my_function)
data["description"] = data["description"].fillna("")  # Xu li NAN
imputer = SimpleImputer(strategy="constant", fill_value="unknown")  # cach tuong tu nhu fillna, nhung dung imputer
# imputer.fit_transform(...) trả về một numpy array 2 chiều (shape = (n_rows, 1)), nên khi gán trực
# tiếp cho data["description"] (một Series 1 chiều) sẽ vẫn chạy được, nhưng nếu bạn muốn chuẩn hơn, nên chuyển về 1 chiều: dung ravel()
data["description"] = imputer.fit_transform(data[["description"]]).ravel()
# print(len(data["location"].unique()))
# dtype = str: dinh nghia tat ca du lieu cua toi la so. Neu co la chu thi cung convert sang so.
# print(data)
# print(data.columns)
# print(data.isna().sum())
# print(data["career_level"].value_counts())

target = "career_level"
x= data.drop(target, axis = 1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 42, stratify=y)
# Tratify co nghia la: chia tach du lieu van giu nguyen ti le cua cac class
# print(y_train.value_counts())
# print("------------")
# print(y_test.value_counts())

# Over sampling (tang performance cua model) => Kiem tra xem truoc va sau no se nhu the nao
# SMOTEN chi dung cho du lieu caterogical
ros = SMOTEN(random_state=42, k_neighbors = 2, sampling_strategy= { "director_business_unit_leader": 500, "specialist": 500,
                                                                    "managing_director_small_medium_company": 500})
# print(y_train.value_counts())
# print("------------------")
# x_train, y_train = ros.fit_resample(x_train, y_train)
# print(y_train.value_counts())



# vectorizer = TfidfVectorizer(stop_words=["english"], ngram_range=(1,2), min_df=0.01, max_df=0.95)
# # stop words: chi ra cai minh dung la tieng anh hay list cua cac tu ma minh muon loai bo ra
# # result = vectorizer.fit_transform(x_train["title"])
# result = vectorizer.fit_transform(x_train["description"])
# print(result.shape)
# # shape: (so hang, so vocabulary)
# print(vectorizer.vocabulary_)
# # vocabulary: in chi so tuong ung cua tu do.

preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(stop_words=["english"], ngram_range=(1,1)),
     "title"),

    ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),

    ("description", TfidfVectorizer(stop_words=["english"], ngram_range=(1,1), min_df=0.01, max_df=0.95), "description"),

    ("function", OneHotEncoder(handle_unknown="ignore"), ["function"]),

    ("industry", TfidfVectorizer(stop_words=["english"], ngram_range=(1,1)),
     "industry")
])

# Trong xu li min_df, max_df, TFIDT, muc dich la tap trong vao nhung cho quan trong, loai bo cai it quan trong di. Vi du
# cong ty TNHH nguyen quoc manh, thi nguyen quoc manh xuat hien rat it lan. nen no khong quan trong lam.
# Chung ta co the loai bo no.

#Trong qua trinh chay, se co nhung cot nam o bo train, chu khong nam o bo test => bao loi => su dung handle_unknown
# OneHotEncoder(handle_unknown="ignore": handle_unknown: y nghia la mac ke no.

cls = Pipeline(steps= [
     ("preprocessor", preprocessor),
     ("feature_selector", SelectKBest(chi2, k=200)),   #SelectKBest => Dung de chon ra nhung class tot nhat, k la so luong feature ma minh muon giu
     # ("feature_selector", SelectPercentile(chi2, percentile=10)), #percentile = 10, so luong feature muon giu la 10%
     # ("regressor", RandomForestClassifier(random_state=42)) # random_state: dam bao moi lan chay la nhu nhau, giong nhu luc cho phan chia du lieu a
 ])

result = cls.fit_transform(x_train, y_train)
print(result.shape)

# cls.fit(x_train, y_train)
# y_predict = cls.predict(x_test)
# print(classification_report(y_test, y_predict, zero_division=0))
