'''5章 モデルの評価と改良'''
# ------------------------------------
import os
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/17_Introduction_to_ML')

# いつもの設定のインポート
from preamble import *

from sklearn.datasets import load_iris, make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import \
train_test_split, LeaveOneOut, cross_val_score, ShuffleSplit, GroupKFold

#%% ------------------------------------
'''5.1 交差検証'''
# cross_val_score
# デフォルトでクラス分類の場合は層化k分割交差検証、回帰の場合はk分割交差検証となる

#%% 1つ抜き交差検証
iris = load_iris()
logreg = LogisticRegression(max_iter=1000)

loo = LeaveOneOut()
scores = cross_val_score(logreg, iris.data, iris.target, cv=loo)
len(scores)
scores.mean()


#%% シャッフル分割交差検証
shuffle_split = ShuffleSplit(test_size=0.5, train_size=0.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
scores
# 層化バージョン: StratifiedShuffleSplit


#%% グループ付き層化検証
# 合成データセットを作成
X, y = make_blobs(n_samples=12, random_state=0)
# 最初の3つが同じグループ、次の4つが同じグループ…といったように
groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
scores = cross_val_score(logreg, X, y, groups=groups, cv=GroupKFold(n_splits=3))
scores
