import os, sys
import numpy as np
import pandas as pd

# 自動インストール（ライブラリ不足エラー対策）
os.system(f'"{sys.executable}" -m pip install matplotlib seaborn scikit-learn')

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# このファイルと同じフォルダを作業場所にセット
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("現在のフォルダ:", os.getcwd())
print("ここにあるファイル:", os.listdir())

# 1. データの読み込み
df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# 2. 年齢の外れ値処理（上位10%をNaNにする）
age_threshold = df['Age'].quantile(0.9)
df.loc[df['Age'] > age_threshold, 'Age'] = np.nan
df_test.loc[df_test['Age'] > age_threshold, 'Age'] = np.nan

# 3. 年齢の欠損値補完（性別×客室ごとの中央値）
df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
df_test['Age'] = df_test.groupby(['Sex', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))

# 4. 新しい特徴量の作成（2等の男の子フラグ）
df["boy_pclass2"] = ((df['Age'] <= 10) & (df['Sex'] == "male") & (df["Pclass"] == 2)).astype(int)
df_test["boy_pclass2"] = ((df_test['Age'] <= 10) & (df_test['Sex'] == "male") & (df_test["Pclass"] == 2)).astype(int)

# 5. 性別×客室の結合
df['Sex_Pclass'] = df['Sex'] + '_' + df['Pclass'].astype(str)
df_test['Sex_Pclass'] = df_test['Sex'] + '_' + df_test['Pclass'].astype(str)

# ★エラーの原因（列数のズレ）を防ぐために1つにまとめてから変換
df['is_test'] = 0
df_test['is_test'] = 1
combined = pd.concat([df, df_test], axis=0, ignore_index=True)
combined = pd.get_dummies(combined, columns=['Sex_Pclass'], drop_first=True)

# 元に戻す
df = combined[combined['is_test'] == 0].drop('is_test', axis=1)
df_test = combined[combined['is_test'] == 1].drop('is_test', axis=1)

# 6. 不要な列の削除と Sex の数値化
category_list = ['SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'Ticket', 'Name']
df = df.drop(columns=[col for col in category_list if col in df.columns])
df_test = df_test.drop(columns=[col for col in category_list if col in df_test.columns])

label_encoder = OrdinalEncoder()
df[['Sex']] = label_encoder.fit_transform(df[['Sex']])
df_test[['Sex']] = label_encoder.transform(df_test[['Sex']])

# 7. 特徴量（X）とターゲット（y）の作成
X = df.drop(['PassengerId', 'Survived'], axis=1)
y = df['Survived']
X_test = df_test.drop(['PassengerId', 'Survived'], axis=1, errors='ignore')

# 8. ★ここが前エラーになった場所：先にモデルを定義してからスコア計算★
model = RandomForestClassifier(random_state=42)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold)

print('----------------------------------------')
print('Cross-Validation scores:', scores)
print('Average score: {:.4f}'.format(np.mean(scores)))
print('----------------------------------------')

# 9. 学習と提出ファイルの作成
model.fit(X, y)
predictions = model.predict(X_test)

sub = pd.DataFrame({
    'PassengerId': df_test['PassengerId'].astype(int),
    'Survived': predictions.astype(int)
})

sub.to_csv('submission.csv', index=False)
print("SUCCESS: submission.csv が正常に作成されました！")