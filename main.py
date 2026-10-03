import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

df.loc[df['Age'] > df['Age'].quantile(0.9), 'Age'] = df['Age']= np.nan
df_test.loc[df_test['Age'] > df['Age'].quantile(0.9), "Age"] = np.nan

category_list = ['SibSp','Parch','Fare', 'Cabin', 'Embarked','Ticket','Name']

df.drop(category_list, axis=1, inplace=True)
df_test.drop(category_list, axis=1, inplace=True)

df["boy_pclass2"] = ((df['Age'] <= 10)) & (df['Sex'] == "male") & (df["Pclass"] == 2).astype(int)
df['Sex_Pclass'] = df['Sex'] + '_' + df['Pclass'].astype(str)
df = pd.get_dummies(df, columns=['Sex_Pclass'], drop_first=True)
df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))

label_encoder = OrdinalEncoder()
df[['Sex']] = label_encoder.fit_transform(df[['Sex']].values)
df_test[['Sex']] = label_encoder.transform(df_test[['Sex']].values)

X = df.iloc[0:, 2:].values
y = df.iloc[:, 1].values

X_test = df_test.iloc[:, 1:].values


# データを 5 等分（5分割）する設定を作る
kfold = KFold(n_splits=5)

# 5パターンの組み合わせで学習とテストを自動で行い、5回分のスコアを計算する
scores = cross_val_score(model, X, y, cv=kfold)

# 5回それぞれのテストスコアを表示する（例: [0.78, 0.81, 0.75, 0.80, 0.79]）
print('Cross-Validation scores: {}'.format(scores))

# 5回の平均スコアを出す（これがこのモデルの「真の平均実力」！）
import numpy as np
print('Average score: {}'.format(np.mean(scores)))# 交差検証 これで提出したら点数出ますかね