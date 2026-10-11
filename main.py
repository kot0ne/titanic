import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold, cross_val_score

# 1. データの読み込み
train_path = 'train.csv'
test_path = 'test.csv'

df = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df.set_index('PassengerId', inplace=True)

# 1. タイトル（敬称）の抽出（df と df_test の両方に行う）
for data in [df, df_test]:
    data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

title_mapping = {'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'}
df['Title'] = df['Title'].replace(title_mapping)
df_test['Title'] = df_test['Title'].replace(title_mapping)

for data in [df, df_test]:
    title_mask = ~data['Title'].isin(['Mr', 'Miss', 'Mrs', 'Master'])
    data.loc[title_mask, 'Title'] = data.loc[title_mask, 'Sex'].map({'male': 'Mr', 'female': 'Mrs'})

title_age_medians = {
    'Mr': 32.32, 
    'Miss': 21.68, 
    'Mrs': 35.86, 
    'Master': 4.57
}

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
for data in [df, df_test]:
    data["boy_pclass2"] = ((data['Age'] <= 10) & (data['Sex'] == "male") & (data["Pclass"] == 2)).astype(int)
    data['Age*Class'] = data['Age'] * data['Pclass']
    data['Age*Fare'] = data['Age'] * data['Fare']
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

    data['AgeBand'] = pd.cut(data['Age'], bins=[0, 12, 20, 40, 60, np.inf], labels=[0, 1, 2, 3, 4]).astype(int)
    data['FareBand'] = pd.qcut(data['Fare'], q=4, labels=[0, 1, 2, 3], duplicates='drop').astype(int)
    data['Fare_log'] = np.log1p(data['Fare'])
    data['Sex_Pclass'] = data['Sex'].astype(str) + '_' + data['Pclass'].astype(str)


df_sex = pd.get_dummies(df['Sex'], prefix='sex', drop_first=True, dtype=int)
df_Pclass = pd.get_dummies(df['Pclass'], prefix='class', drop_first=True, dtype=int)
df_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True, dtype=int)
df_Title = pd.get_dummies(df['Title'], prefix='Title', drop_first=False, dtype=int)

# 結合
df = pd.concat([df, df_sex, df_Pclass, df_Embarked, df_Title], axis=1)

df_test_sex = pd.get_dummies(df_test['Sex'], prefix='sex', drop_first=True, dtype=int)
df_test_Pclass = pd.get_dummies(df_test['Pclass'], prefix='class', drop_first=True, dtype=int)
df_test_Embarked = pd.get_dummies(df_test['Embarked'], prefix='Embarked', drop_first=True, dtype=int)
df_test_Title = pd.get_dummies(df_test['Title'], prefix='Title', drop_first=False, dtype=int)
df_test = pd.concat([df_test, df_test_sex, df_test_Pclass, df_test_Embarked, df_test_Title], axis=1)

drop_cols = ['Name', 'Ticket', 'Cabin', 'Embarked', 'Title', 'Sex', 'Pclass', 'Sex_Pclass', 'Survived']

X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df['Survived'].astype(int)

X_test = df_test.drop(columns=[col for col in drop_cols if col in df_test.columns and col != 'Survived'], errors='ignore')

# 訓練データとテストデータの列の並び・種類を完全に一致させる（Kaggleの鉄則！）
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)


# 7. 数値データの標準化（Zスコア化）
numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns
mew = X[numeric_columns].mean(axis=0)
std = X[numeric_columns].std(axis=0).replace(0, 1) # ゼロ割防止

X[numeric_columns] = (X[numeric_columns] - mew) / std
X_test[numeric_columns] = (X_test[numeric_columns] - mew) / std


# 8. 特徴量（X）から PassengerId を除外してモデル学習
X = X.drop(['PassengerId'], axis=1, errors='ignore')
X_test = X_test.drop(['PassengerId'], axis=1, errors='ignore')

model = XGBClassifier(
    random_state=42,
    max_depth=3,          # 木の深さを浅くして、複雑に考えすぎないようにする
    learning_rate=0.05,   # 学習のスピードをゆっくりにして慎重に覚えさせる
    n_estimators=100,     # 木の数
    subsample=0.8,        # データの8割だけを使ってランダムに学習する
    colsample_bytree=0.8  # 特徴量の8割だけを使ってランダムに学習する
)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# 5パターンの組み合わせで学習とテストを自動で行い、5回分のスコアを計算する
scores = cross_val_score(model, X, y, cv=kfold)

# 5回それぞれのテストスコアを表示する
print('Cross-Validation scores: {}'.format(scores))

# 5回の平均スコアを出す（これがこのモデルの「真の平均実力」！）
print('Average score: {}'.format(np.mean(scores)))
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