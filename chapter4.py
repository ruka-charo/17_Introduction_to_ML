'''4章 データの表現と特徴量エンジニアリング'''
# ------------------------------------
import os
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/17_Introduction_to_ML')

# いつもの設定のインポート
from preamble import *

from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.feature_selection import SelectPercentile, SelectFromModel, RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures, MinMaxScaler

#%% ------------------------------------
'''4.2 ビニング、離散化、線形モデル、決定木'''
# データの読み込み
X, y = mglearn.datasets.make_wave(n_samples=120)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

reg = DecisionTreeRegressor(min_samples_leaf=3).fit(X, y)
plt.plot(line, reg.predict(line), label='decision tree')

reg = LinearRegression().fit(X, y)
plt.plot(line, reg.predict(line), label="linear regression")

plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")
plt.show()

#%% ------------------------------------
# 特徴量のビニング(離散化)
kb = KBinsDiscretizer(n_bins=10, strategy='uniform', encode='onehot-dense')
kb.fit(X) # 入力レンジを分割する
print('bin edges:\n', kb.bin_edges_)
X_binned = kb.transform(X) # 10個のbinのどこに入っているかをonehotで表現
line_binned = kb.transform(line)

#%% 訓練と結果の可視化
reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='linear regression binned')

reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='decision tree binned', alpha=0.5)
plt.plot(X[:, 0], y, 'o', c='k')
plt.vlines(kb.bin_edges_[0], -3, 3, linewidth=1, alpha=.2)
plt.legend(loc="best")
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.show()

#%% ------------------------------------
'''4.3 相互作用と多項式'''
# ビニングデータに対し傾きも学習する(元の特徴量を加えなおせば良い)
X_combined = np.hstack([X, X_binned])
print(X_combined.shape)

# 学習と可視化
reg = LinearRegression().fit(X_combined, y)

line_combined = np.hstack([line, line_binned])
plt.plot(line, reg.predict(line_combined), label='linear regression combined')

plt.vlines(kb.bin_edges_[0], -3, 3, linewidth=1, alpha=.2)
plt.legend(loc="best")
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.plot(X[:, 0], y, 'o', c='k')
plt.show()


#%% 各binごとに傾きを学習させる
# (どのbinに入っているか) × (x軸のどこにあるか) を加えるとできる
X_product = np.hstack([X_binned, X * X_binned])
print(X_product.shape)

# 学習と可視化
reg = LinearRegression().fit(X_product, y)

line_product = np.hstack([line_binned, line * line_binned])
plt.plot(line, reg.predict(line_product), label='linear regression product')

plt.vlines(kb.bin_edges_[0], -3, 3, linewidth=1, alpha=.2)
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")
plt.show()


#%% 多項式回帰
# x**10までの多項式を加える
# デフォルトの'include_bias=True'だと、常に1になる特徴量を加える
poly = PolynomialFeatures(degree=10, include_bias=False)
poly.fit(X)
X_poly = poly.transform(X)

print("X_poly.shape: {}".format(X_poly.shape))
print("Entries of X:\n{}".format(X[:5]))
print("Entries of X_poly:\n{}".format(X_poly[:5]))

poly.get_feature_names() # 個々の特徴量の意味

#%% 多項式回帰の学習と可視化
reg = LinearRegression().fit(X_poly, y)

line_poly = poly.transform(line)
plt.plot(line, reg.predict(line_poly), label='polynomial linear regression')
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")
plt.show()

#%% カーネルSVMに変換をしていないデータを適用した時との比較
for gamma in [1, 10]:
    svr = SVR(gamma=gamma).fit(X, y)
    plt.plot(line, svr.predict(line), label='SVR gamma={}'.format(gamma))

plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")
plt.show()


#%% boston_housingデータに適用してみる
boston = load_boston()
X_train, X_test, y_train, y_test = train_test_split(
    boston.data, boston.target, random_state=0)

# 正規化
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2次までの多項式特徴量と相互作用を抽出
poly = PolynomialFeatures(degree=2).fit(X_train_scaled)
X_train_poly = poly.transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
print("X_train.shape: {}".format(X_train.shape))
print("X_train_poly.shape: {}".format(X_train_poly.shape))
poly.get_feature_names()

#%% Ridge回帰で相互作用特徴量を入れた時と入れてない時の比較を行う
# 入れてない時
ridge = Ridge().fit(X_train_scaled, y_train)
ridge.score(X_test_scaled, y_test)

# 入れた時
ridge = Ridge().fit(X_train_poly, y_train)
ridge.score(X_test_poly, y_test)

#%% ランダムフォレストだと精度が少し下がる
# 入れてない時
rf = RandomForestRegressor(n_estimators=100).fit(X_train_scaled, y_train)
rf.score(X_test_scaled, y_test)

# 入れた時
rf = RandomForestRegressor(n_estimators=100).fit(X_train_poly, y_train)
rf.score(X_test_poly, y_test)


#%% ------------------------------------
'''4.4 単変量非線形変換'''
rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000, 3))
w = rnd.normal(size=3)

X = rnd.poisson(10 * np.exp(X_org))
y = X_org @ w

#%% 個々の値が出現する回数の可視化
bins = np.bincount(X[:, 0])
plt.bar(range(len(bins)), bins, color='grey')
plt.ylabel("Number of appearances")
plt.xlabel("Value")
plt.show()


#%% リッジ回帰で学習してみる
# 線形モデルなのでうまく捉えることができない
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
score = Ridge().fit(X_train, y_train).score(X_test, y_test)
print("Test score: {:.3f}".format(score))

#%% ただし、対数変換を行うと…
X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)
plt.hist(X_train_log[:, 0], bins=25, color='gray')
plt.ylabel("Number of appearances")
plt.xlabel("Value")
plt.show()

score = Ridge().fit(X_train_log, y_train).score(X_test_log, y_test)
print("Test score: {:.3f}".format(score))


#%% ------------------------------------
'''自動特徴量選択'''
# 単変量統計
# cancerデータセットのクラス分類で特徴量抽出を適用してみる
cancer = load_breast_cancer()

# シードを指定して乱数を決定
rng = np.random.RandomState(42)
noise = rng.normal(size=(len(cancer.data), 50))
# ノイズ特徴量をデータに入れる
# 初めの30個は元のデータ、残り50個はノイズ
X_w_noise = np.hstack([cancer.data, noise])

X_train, X_test, y_train, y_test = train_test_split(
    X_w_noise, cancer.target, random_state=0, test_size=0.5)

# f_classif(デフォルト)とselectpercentileを使って50%の特徴量を選択
select = SelectPercentile(percentile=50)
select.fit(X_train, y_train)
X_train_selected = select.transform(X_train)

print("X_train.shape: {}".format(X_train.shape))
print("X_train_selected.shape: {}".format(X_train_selected.shape))

#%% どの特徴量が使われているか表示
mask = select.get_support()
print(mask)

# maskを可視化する。黒が「真」、白が「偽」
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")
plt.yticks(())
plt.show()


#%% ロジスティック回帰で全選択の時と抽出後の時で比較する
# テストデータの変換
X_test_selected = select.transform(X_test)
lr = LogisticRegression(max_iter=1000)

# 全選択
lr.fit(X_train, y_train)
print("Score with all features: {:.3f}".format(lr.score(X_test, y_test)))

# 抽出後
lr.fit(X_train_selected, y_train)
print("Score with only selected features: {:.3f}".format(
    lr.score(X_test_selected, y_test)))


#%% モデルベース特徴量選択
select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42),
                        threshold='0.5*median') # 'mean'も使える
select.fit(X_train, y_train)
X_train_l1 = select.transform(X_train) # 抽出
X_train_l1.shape
# maskを可視化する
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")
plt.yticks(())

#%% 抽出後のデータで性能評価
X_test_l1 = select.transform(X_test)
score = LogisticRegression(max_iter=1000).fit(X_train_l1, y_train).score(X_test_l1, y_test)
score


#%% 反復特徴量選択
select = RFE(RandomForestClassifier(n_estimators=100, random_state=42),
            n_features_to_select=40)
select.fit(X_train, y_train)

# maskを可視化する
mask = select.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")
plt.yticks(())

#%% 抽出後のデータで性能評価
X_train_rfe = select.transform(X_train) # 抽出
X_test_rfe = select.transform(X_test)

score = LogisticRegression(max_iter=1000).fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
print("Test score: {:.3f}".format(score))
