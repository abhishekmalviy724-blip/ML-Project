# ML-Project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


Data = pd.read_csv("healthcare_dataset.csv")


print(Data.shape)
print(Data.head())
print(Data.info())
print(Data.describe())
print(Data.isnull().sum())
print(Data.duplicated().sum())

print(Data.select_dtypes(include="number").columns)
print(Data.select_dtypes(include="object").columns)

print(Data["Test Results"].value_counts())


Data = Data.drop_duplicates()


Data["Date of Admission"] = pd.to_datetime(Data["Date of Admission"])
Data["Discharge Date"] = pd.to_datetime(Data["Discharge Date"])

Data["Total_Stay_Days"] = (
    Data["Discharge Date"] - Data["Date of Admission"]
).dt.days

Data["Daily_Billing"] = (
    Data["Billing Amount"] / Data["Total_Stay_Days"].replace(0, 1)
)


plt.figure(figsize=(8, 5))
sns.histplot(Data["Age"], kde=True)
plt.title("Age Distribution")
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(Data["Billing Amount"], kde=True)
plt.title("Billing Amount Distribution")
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(data=Data, x="Test Results")
plt.title("Test Results Distribution")
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(data=Data, x="Gender", hue="Test Results")
plt.title("Gender vs Test Results")
plt.show()


plt.figure(figsize=(10, 5))
sns.countplot(data=Data, x="Medical Condition", hue="Test Results")
plt.title("Medical Condition vs Test Results")
plt.xticks(rotation=30)
plt.show()


plt.figure(figsize=(8, 5))
sns.countplot(data=Data, x="Admission Type", hue="Test Results")
plt.title("Admission Type vs Test Results")
plt.show()


plt.figure(figsize=(8, 5))
sns.boxplot(data=Data, x="Test Results", y="Billing Amount")
plt.title("Billing Amount vs Test Results")
plt.show()


plt.figure(figsize=(10, 7))
sns.heatmap(
    Data.select_dtypes(include="number").corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.show()


target = "Test Results"

features = [
    "Age",
    "Billing Amount",
    "Room Number",
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Insurance Provider",
    "Admission Type",
    "Total_Stay_Days",
    "Daily_Billing"
]

X = Data[features]
y = Data[target]


numeric_features = [
    "Age",
    "Billing Amount",
    "Room Number",
    "Total_Stay_Days",
    "Daily_Billing"
]

categorical_features = [
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Insurance Provider",
    "Admission Type"
]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    ),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        eval_metric="mlogloss"
    )
}


results = []


for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy
        }
    )

    print("\n", "=" * 50)
    print(name)
    print("=" * 50)

    print("Accuracy:", accuracy)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))


results_df = pd.DataFrame(results)

print(results_df.sort_values(
    "Accuracy",
    ascending=False
))


plt.figure(figsize=(10, 6))

sns.barplot(
    data=results_df.sort_values(
        "Accuracy",
        ascending=False
    ),
    x="Accuracy",
    y="Model"
)

plt.title("Model Comparison")
plt.xlim(0, 1)
plt.show()


best_model_name = results_df.loc[
    results_df["Accuracy"].idxmax(),
    "Model"
]

print("Best Model:", best_model_name)


best_model = models[best_model_name]

best_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", best_model)
    ]
)


cv_scores = cross_val_score(
    best_pipeline,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("CV Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())


grid_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                random_state=42
            )
        )
    ]
)


param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 5, 8, 10],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}


grid_search = GridSearchCV(
    grid_pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:")
print(grid_search.best_params_)

print("Best CV Score:")
print(grid_search.best_score_)


random_search = RandomizedSearchCV(
    grid_pipeline,
    param_grid,
    n_iter=10,
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("Random Search Best Parameters:")
print(random_search.best_params_)

print("Random Search Best Score:")
print(random_search.best_score_)


final_model = grid_search.best_estimator_

final_pred = final_model.predict(X_test)

print("Final Accuracy:")
print(accuracy_score(y_test, final_pred))

print("Precision:")
print(
    precision_score(
        y_test,
        final_pred,
        average="weighted"
    )
)

print("Recall:")
print(
    recall_score(
        y_test,
        final_pred,
        average="weighted"
    )
)

print("F1 Score:")
print(
    f1_score(
        y_test,
        final_pred,
        average="weighted"
    )
)

print(classification_report(y_test, final_pred))


cm = confusion_matrix(y_test, final_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Final Model Confusion Matrix")
plt.show()


X_processed = preprocessor.fit_transform(X_train)

smote = SMOTE(random_state=42)

X_smote, y_smote = smote.fit_resample(
    X_processed,
    y_train
)

print("Before SMOTE:")
print(y_train.value_counts())

print("After SMOTE:")
print(pd.Series(y_smote).value_counts())


smote_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

smote_model.fit(X_smote, y_smote)

X_test_processed = preprocessor.transform(X_test)

smote_pred = smote_model.predict(X_test_processed)

print("SMOTE Accuracy:")
print(accuracy_score(y_test, smote_pred))

print(classification_report(y_test, smote_pred))


feature_names = (
    numeric_features +
    list(
        preprocessor.named_transformers_[
            "cat"
        ].get_feature_names_out(categorical_features)
    )
)


rf_model = final_model.named_steps["model"]

if hasattr(rf_model, "feature_importances_"):

    importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": rf_model.feature_importances_
        }
    )

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    print(importance.head(10))


    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=importance.head(10),
        x="Importance",
        y="Feature"
    )

    plt.title("Top 10 Feature Importance")
    plt.show()


joblib.dump(
    final_model,
    "healthcare_test_result_model.pkl"
)


loaded_model = joblib.load(
    "healthcare_test_result_model.pkl"
)


new_patient = pd.DataFrame(
    {
        "Age": [45],
        "Billing Amount": [25000],
        "Room Number": [205],
        "Gender": ["Male"],
        "Blood Type": ["A+"],
        "Medical Condition": ["Diabetes"],
        "Insurance Provider": ["Medicare"],
        "Admission Type": ["Emergency"],
        "Total_Stay_Days": [5],
        "Daily_Billing": [5000]
    }
)


prediction = loaded_model.predict(new_patient)

print("Predicted Test Result:", prediction[0])
_____________________________________________________________________________________________________________________________________________________________________________________________________________________________
