from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import pandas as pd

train_df=pd.read_csv("train.csv")
test_df=pd.read_csv("test.csv")

y=train_df["Survived"]

num_cols=["Pclass","Age","SibSp","Parch","Fare"]
X_train_num=train_df[num_cols]
X_test_num=test_df[num_cols]

my_imputer=SimpleImputer(strategy="median")
imputed_X_train=pd.DataFrame(my_imputer.fit_transform(X_train_num))
imputed_X_test=pd.DataFrame(my_imputer.transform(X_test_num))

model=XGBClassifier(random_state=42)
model.fit(imputed_X_train, y)
predictions=model.predict(imputed_X_test)

print(predictions)
