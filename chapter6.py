'''第6章 アルゴリズムチェーンとパイプライン'''
# ------------------------------------
import os
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/17_Introduction_to_ML')

# いつもの設定のインポート
from preamble import *
import seaborn as sns

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
# ------------------------------------


#%% データをロードして分割する
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    random_state=0)


#%% ------------------------------------
'''6.0 普通の実装'''
# 訓練データの最小値と最大値を計算(標準化)
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train) # 訓練データをスケール変換
X_test_scaled = scaler.transform(X_test) # テストデータをスケール変換

# SVCで学習と予測
svm = SVC()
svm.fit(X_train_scaled, y_train) # 学習

print('Test score:  {:.2f}'.format(svm.score(X_test_scaled, y_test)))


#%% ------------------------------------
'''6.2 パイプラインを用いた実装'''
pipe = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC())]) # パイプの作成
pipe.fit(X_train, y_train) # 学習

print(f'Test score: {pipe.score(X_test, y_test):.2f}') # テストスコア


#%% ------------------------------------
'''6.3 パイプラインを用いたグリッドサーチ'''
# グリッドサーチに用いるパラメータ
param_grid = {'svm__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

# いつものようにグリッドサーチを行えば良い
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
print("Test set score: {:.2f}".format(grid.score(X_test, y_test)))
print("Best parameters: {}".format(grid.best_params_))


#%% ------------------------------------
'''情報リークの影響'''
# X, yともにランダムに取ってきた(互いに独立)
rnd = np.random.RandomState(seed=0)
X = rnd.normal(size=(100, 10000))
y = rnd.normal(size=(100,))

# 特徴量を10個抽出し、交差検証を用いてRidge回帰
select = SelectPercentile(score_func=f_regression, percentile=5).fit(X, y)
X_selected = select.transform(X)

print("X_selected.shape: {}".format(X_selected.shape))
print("Cross-validation accuracy (cv only on ridge): {:.2f}".format(
    np.mean(cross_val_score(Ridge(), X_selected, y, cv=5)))) # 明らかに正しくない


#%% パイプラインを使った正しい交差検証
pipe = Pipeline([('select', SelectPercentile(score_func=f_regression,percentile=5)),
                ('ridge', Ridge())])

print("Cross-validation accuracy (pipeline): {:.2f}".format(
      np.mean(cross_val_score(pipe, X, y, cv=5))))


#%% ------------------------------------
'''6.4 汎用パイプラインインターフェイス'''
# make_pipelineを用いたパイプライン作成
# いつもの作り方
pipe_long = pipeline([('scaler', MinMaxScaler()), ('svm', SVC(C=100))])
# make_pipelineを用いた作り方
pipe_short = make_pipeline(MinMaxScaler(), SVC(C=100))
pipe_short.steps # 各ステップの名前の表示

#%% ステップ属性へのアクセス
pipe = make_pipeline(StandardScaler(), PCA(n_components=2), StandardScaler())
pipe.fit(cancer.data) # パイプラインを訓練
# pcaの2つの主成分を取り出す
components = pipe.named_steps['pca'].components_
components.shape

#%% グリッドサーチ内のパラメータにアクセス
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}

# データの分割とグリッドサーチ
X_train, X_test, y_train, y_test = train_test_split(cancer.data,
                                                    cancer.target,
                                                    random_state=4)
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

# 最良モデルの情報
grid.best_estimator_
# LogisticRegressionへのアクセス
grid.best_estimator_.named_steps['logisticregression']
# 個々の入力特徴量に対応する係数にアクセスできる
grid.best_estimator_.named_steps['logisticregression'].coef_
grid.best_estimator_.named_steps['logisticregression'].intercept_


#%% ------------------------------------
'''6.5 前処理ステップとモデルパラメータに対するグリッドサーチ'''
# ボストンデータを使ってパラメータチューニング
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(boston.data,
                                                    boston.target,
                                                    random_state=0)
pipe = make_pipeline(StandardScaler(),
                    PolynomialFeatures(),
                    Ridge())

# グリッドサーチするパラメータの設定
param_grid = {'polynomialfeatures__degree': [1, 2, 3],
            'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
grid.cv_results_ # scoreの結果
grid.best_params_ # 最良のパラメータ
grid.score(X_test, y_test) # 最良のモデルでのスコア
# grid.best_estimator_.score(X_test, y_test)

#%% 多項式特徴量を用いない場合
pipe = make_pipeline(StandardScaler(), Ridge())
param_grid = {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
grid.score(X_test, y_test)

#%% 交差検証の結果を折れ線で可視化
mglearn.tools.heatmap(grid.cv_results_['mean_test_score'].reshape(3, -1),
                      xlabel="ridge__alpha", ylabel="polynomialfeatures__degree",
                      xticklabels=param_grid['ridge__alpha'],
                      yticklabels=param_grid['polynomialfeatures__degree'], vmin=0)
plt.show()


#%% ------------------------------------
'''6.6 グリッドサーチによるモデルの選択'''
# SVCかRandomForestか
pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC())])
# 辞書ごとに独立でサーチする
param_grid = [
    {'classifier': [SVC()], 'preprocessing': [StandardScaler(), None],
     'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
     'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]},

    {'classifier': [RandomForestClassifier(n_estimators=100)],
     'preprocessing': [None], 'classifier__max_features': [1, 2, 3]}]

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

# モデルの探索と学習、スコア
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
grid.best_params_
grid.best_score_
grid.score(X_test, y_test)
