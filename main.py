import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder

# 1. データの読み込み
train_path = 'train.csv'
test_path = 'test.csv'

df = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# 2. 欠損値の補完（Age, Fare, Embarked）
df['Age'] = df['Age'].fillna(df['Age'].median())
df_test['Age'] = df_test['Age'].fillna(df['Age'].median())

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

# 5. 性別×客室の結合とワンホットエンコーディング
df['Sex_Pclass'] = df['Sex'].astype(str) + '_' + df['Pclass'].astype(str)
df_test['Sex_Pclass'] = df_test['Sex'].astype(str) + '_' + df_test['Pclass'].astype(str)

df['is_test'] = 0
df_test['is_test'] = 1
combined = pd.concat([df, df_test], axis=0, ignore_ignore_index=True) if hasattr(pd.concat, 'ignore_ignore_index') else pd.concat([df, df_test], axis=0, ignore_index=True)
combined = pd.get_dummies(combined, columns=['Sex_Pclass'], drop_first=True)

df = combined[combined['is_test'] == 0].drop('is_test', axis=1)
df_test = combined[combined['is_test'] == 1].drop('is_test', axis=1)

# 6. 不要な列の削除と Sex の数値化
category_list = ['SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'Ticket', 'Name']
df = df.drop(columns=[col for col in category_list if col in df.columns])
df_test = df_test.drop(columns=[col for col in category_list if col in df_test.columns])

label_encoder = OrdinalEncoder()
df[['Sex']] = label_encoder.fit_transform(df[['Sex']])
df_test[['Sex']] = label_encoder.transform(df_test[['Sex']])

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