import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder

# 1. データの読み込み
train_path = 'train.csv'
test_path = 'test.csv'

df = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df.set_index('PassengerId', inplace=True)

df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
title_mapping = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'}
df['Title'] = df['Title'].replace(title_mapping)



for data in [df, df_test]:
    for title, median_age in title_age_medians.items():
        age_mask = (data['Age'].isnull()) & (data['Title'] == title)
        data.loc[age_mask, 'Age'] = median_age


df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df_test['Fare'] = df_test['Fare'].fillna(df['Fare'].median())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df_test['Embarked'] = df_test['Embarked'].fillna(df['Embarked'].mode()[0])

# 3. 外れ値処理（Age）
age_cap = df['Age'].quantile(0.99)
df['Age'] = np.where(df['Age'] > age_cap, age_cap, df['Age'])
df_test['Age'] = np.where(df_test['Age'] > age_cap, age_cap, df_test['Age'])

# 4. 特徴量エンジニアリング（2等の男の子フラグ）
df["boy_pclass2"] = ((df['Age'] <= 10) & (df['Sex'] == "male") & (df["Pclass"] == 2)).astype(int)
df_test["boy_pclass2"] = ((df_test['Age'] <= 10) & (df_test['Sex'] == "male") & (df_test["Pclass"] == 2)).astype(int)

# 掛け合わせ特徴量
df['Age*Class'] = df['Age'] * df['Pclass']
df['Age*Fare'] = df['Age'] * df['Fare']

# 家族関連
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# ビニング（カテゴリ化）
df['AgeBand'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, np.inf], labels=[0, 1, 2, 3, 4]).astype(int)
df['FareBand'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3]).astype(int)

# 対数変換（右に裾が長い分布の歪み対策）
df['Fare_log'] = np.log1p(df['Fare'])

# 5. 性別×客室の結合とワンホットエンコーディング
df['Sex_Pclass'] = df['Sex'].astype(str) + '_' + df['Pclass'].astype(str)
df_test['Sex_Pclass'] = df_test['Sex'].astype(str) + '_' + df_test['Pclass'].astype(str)

df_sex = pd.get_dummies(df['Sex'], prefix='sex', drop_first=True, dtype=int)
df_Pclass = pd.get_dummies(df['Pclass'], prefix='class', drop_first=True, dtype=int)
df_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True, dtype=int)
df_Title = pd.get_dummies(df['Title'], prefix='Title', drop_first=False, dtype=int)

# 結合
df = pd.concat([df, df_sex, df_Pclass, df_Embarked, df_Title], axis=1)

# 元データ・不要列の削除
drop_cols = ['Sex', 'Pclass', 'Name', 'Ticket', 'Embarked', 'Cabin', 'Title', 'Fare', 'SibSp', 'Parch']
df = df.drop(drop_cols, axis=1)

df = combined[combined['is_test'] == 0].drop('is_test', axis=1)
df_test = combined[combined['is_test'] == 1].drop('is_test', axis=1)

# 6. 不要な列の削除と Sex の数値化
category_list = ['SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'Ticket', 'Name']
df = df.drop(columns=[col for col in category_list if col in df.columns])
df_test = df_test.drop(columns=[col for col in category_list if col in df_test.columns])

label_encoder = OrdinalEncoder()
df[['Sex']] = label_encoder.fit_transform(df[['Sex']])
df_test[['Sex']] = label_encoder.transform(df_test[['Sex']])

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

mew = df[numeric_columns].mean(axis=0)
std = df[numeric_columns].std(axis=0)

df[numeric_columns] = (df[numeric_columns] - mew) / std

# 7. 特徴量（X）とターゲット（y）の分離
X = df.drop(['PassengerId', 'Survived'], axis=1)
y = df['Survived'].astype(int)
X_test = df_test.drop(['PassengerId', 'Survived'], axis=1, errors='ignore')

# 8. モデルの構築と学習
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 9. 予測と提出用ファイルの作成
predictions = model.predict(X_test)

sub = pd.DataFrame({
    'PassengerId': df_test['PassengerId'].astype(int),
    'Survived': predictions.astype(int)
})

sub.to_csv('submission.csv', index=False)
print("SUCCESS: submission.csv が正常に作成されました！")
print("\n--- 提出ファイルの確認 ---")
print(sub.head())