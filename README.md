# ML-Project
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from xgboost import XGBClassifier



Data = pd.read_csv("healthcare_dataset.csv")
# Data.shape
# Data.head()
# Data.info()
# Data.info()
# Data.describe()
# Data.isnull().sum()
# Data.duplicated()
# Data.select_dtypes("category")
# Data.select_dtypes("number")
# Data["Test Results"].value_counts()

# Data.drop_duplicates()
# Data.fillna(Data.select_dtypes("number").median())
# Data.fillna(Data.select_dtypes("category").mode())
# Data["Date of Admission"]=pd.to_datetime(Data["Date of Admission"])
# Data["Discharge Date"]=pd.to_datetime(Data["Discharge Date"])
# Data["Total_Stay_Days"]=(Data["Discharge Date"]-Data["Date of Admission"]).dt.days
# Data["Daily_Billing"]=(Data["Total_Stay_Days"]/Data["Billing Amount"])
# unnecessary = pd.DataFrame({"Missing Values": Data.isnull().sum(), "Unique Values": Data.nunique()})
# unnecessary.drop(columns=unnecessary,errors="ignore")


# plt.figure(figsize=(8,5))
# plt.plot(Data["Age"].head(10),color="g")
# plt.xlabel("Age", fontsize=12)
# plt.ylabel("Count", fontsize=12)
# plt.show()
# plt.figure(figsize=(8,5))
# plt.plot(Data["Billing Amount"].head(10),color="b")
# plt.xlabel("Bills", fontsize=12)
# plt.ylabel("Count", fontsize=12)
# plt.show()
# sns.countplot(Data["Test Results"])
# plt.xlabel("Tests", fontsize=12)
# plt.ylabel("Count", fontsize=12)
# plt.show()
# plt.figure(figsize=(8,5))
# sns.countplot(data=Data,x="Gender",hue="Test Results",palette="Set2")
# plt.title("Gender vs Test Results Distribution", fontsize=14, fontweight="bold")
# plt.xlabel("Gender", fontsize=12)
# plt.ylabel("Count", fontsize=12)
# plt.show()
# plt.figure(figsize=(8,5))
# sns.countplot(data=Data,x="Medical Condition",hue="Test Results",palette="Set2")
# plt.title("Medical Condition vs Test Results")
# plt.xlabel("Gender", fontsize=12)
# plt.ylabel("Count", fontsize=12)
# plt.show()
# plt.figure(figsize=(8,5))
# sns.countplot(data=Data,x="Admission Type",hue="Test Results",palette="Set2")
# plt.title("Admission Type vs Test Results")
# plt.xlabel("Admission Type", fontsize=12)
# plt.ylabel("Count", fontsize=12)
# plt.show()
# sns.countplot(data=Data,x="Billing Amount",hue="Test Results",palette="Set2")
# plt.title("Billing Amount vs Test Results")
# plt.xlabel("Billing Amount", fontsize=12)
# plt.ylabel("Count", fontsize=12)
# plt.show()
# sns.heatmap(Data.select_dtypes("number").corr(),annot=True,cmap='coolwarm',fmt='.2f',linewidths=0.5)
# plt.show()


X = Data[["Age",
"Billing Amount",
"Room Number",
"Gender",
"Blood Type",
"Medical Condition",
"Insurance Provider"]]
y = Data["Test Results"]
X = X.fillna(X.mode().iloc[0])
y = y.fillna(y.mode().iloc[0])
X = pd.get_dummies(X,columns=["Gender","Blood Type","Medical Condition","Insurance Provider"],drop_first=True)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_train)
y_trans = label_encoder.transform(y_test)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
st = StandardScaler()
scaled = st.fit_transform(X)

# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", StandardScaler()),
#         ("cum",OneHotEncoder(handle_unknown='ignore'))
#     ]
# )
# pipeline = Pipeline(
#     steps=[
#         ("preproccessor",preprocessor),
#         ("model",LogisticRegression())
#     ]
# )


model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

model2 = KNeighborsClassifier(n_neighbors=2)
model2.fit(X_train,y_train)
y_pred2 = model2.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(classification_report(y_test,y_pred2))

model3 = DecisionTreeClassifier(max_depth=5,random_state=42)
model3.fit(X_train,y_train)
y_pred3 = model3.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(classification_report(y_test,y_pred3))

model4 = RandomForestClassifier(max_depth=5,random_state=42)
model4.fit(X_train,y_train)
y_pred4 = model4.predict(X_test)
print(accuracy_score(y_test,y_pred4))
print(confusion_matrix(y_test,y_pred4))
print(classification_report(y_test,y_pred4))

model5 = GaussianNB()
model5.fit(X_train,y_train)
y_pred5 = model5.predict(X_test)
print(accuracy_score(y_test,y_pred5))
print(confusion_matrix(y_test,y_pred5))
print(classification_report(y_test,y_pred5))

model6 = SVC()
model6.fit(X_train,y_train)
y_pred6 = model6.predict(X_test)
print(accuracy_score(y_test,y_pred6))
print(confusion_matrix(y_test,y_pred6))
print(classification_report(y_test,y_pred6))

model7 = GradientBoostingClassifier()
model7.fit(X_train,y_train)
y_pred7 = model7.predict(X_test)
print(accuracy_score(y_test,y_pred7))
print(confusion_matrix(y_test,y_pred7))
print(classification_report(y_test,y_pred7))

model8 = AdaBoostClassifier()
model8.fit(X_train,y_train)
y_pred8 = model8.predict(X_test)
print(accuracy_score(y_test,y_pred8))
print(confusion_matrix(y_test,y_pred8))
print(classification_report(y_test,y_pred8))

model9 = AdaBoostClassifier()
model9.fit(X_train,y_train)
y_pred9 = model9.predict(X_test)
print(accuracy_score(y_test,y_pred9))
print(confusion_matrix(y_test,y_pred9))
print(classification_report(y_test,y_pred9))


# model10 = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
# model10.fit(X_train,y_trans)
# y_pred10 = model10.predict(X_test)
# print(accuracy_score(y_test,y_pred10))
# print(confusion_matrix(y_test,y_pred10))
# print(classification_report(y_test,y_pred10))

