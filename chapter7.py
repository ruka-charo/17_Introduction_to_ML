'''第７章 テキストデータの処理'''
# ------------------------------------
import os
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/17_Introduction_to_ML')
import re
import spacy
import nltk

# いつもの設定のインポート
from preamble import *

from sklearn.datasets import load_files
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, \
ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
# ----------------------------------

#%% 訓練データの読み込み
reviews_train = load_files("aclImdb/train/")
# load_files は一連の訓練テキストと訓練ラベルを返す
text_train, y_train = reviews_train.data, reviews_train.target
print("type of text_train: {}".format(type(text_train)))
print("length of text_train: {}".format(len(text_train)))
print("text_train[6]:\n{}".format(text_train[6]))

#%% テストデータの読み込み
reviews_test = load_files("aclImdb/test/")
text_test, y_test = reviews_test.data, reviews_test.target
print("Number of documents in test data: {}".format(len(text_test)))
print("Samples per class (test): {}".format(np.bincount(y_test)))

#%% タグの削除
text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
text_test = [doc.replace(b"<br />", b" ") for doc in text_test]


#%% ----------------------------------
'''7.3 Bag of Wordsによるテキスト表現'''
# 例題
bards_words =["The fool doth think he is wise,",
              "but the wise man knows himself to be a fool"]
vect = CountVectorizer()
vect.fit(bards_words) # トークン分割とボキャブラリ構築

print("Vocabulary size: {}".format(len(vect.vocabulary_)))
print("Vocabulary content:\n {}".format(vect.vocabulary_))

bag_of_words = vect.transform(bards_words) # BoW表現の構築
print("bag_of_words: {}".format(repr(bag_of_words))) # スパースマトリックス

# BoW表現を可視化する(memory errorに注意)
print("Dense representation of bag_of_words:\n{}".format(
    bag_of_words.toarray()))


#%% 映画レビューのBoW
vect = CountVectorizer().fit(text_train)
X_train = vect.transform(text_train)
print("X_train:\n{}".format(repr(X_train)))

# ボキャブラリーへアクセス
feature_names = vect.get_feature_names()
print("Number of features: {}".format(len(feature_names)))
print("First 20 features:\n{}".format(feature_names[:20]))
print("Features 20010 to 20030:\n{}".format(feature_names[20010:20030]))
print("Every 2000th feature:\n{}".format(feature_names[::2000]))

#%% 交差検証(LogisticRegression)
scores = cross_val_score(LogisticRegression(),
                        X_train, y_train, cv=5)

print("Mean cross-validation accuracy: {:.2f}".format(np.mean(scores)))

#%% 正則化パラメータのチューニング
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}

grid = GridSearchCV(LogisticRegression(),
                    param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)

#%% テストデータでのスコア
X_test = vect.transform(text_test) # グリッドサーチでの最適なモデルが反映されている
print("Test score: {:.2f}".format(grid.score(X_test, y_test)))


#%% トークンとして採用されるべき単語数を設定
vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train)
print("X_train with min_df: {}".format(repr(X_train)))

#%% ボキャブラリーの確認
feature_names = vect.get_feature_names()

print("First 50 features:\n{}".format(feature_names[:50]))
print("Features 20010 to 20030:\n{}".format(feature_names[20010:20030]))
print("Every 700th feature:\n{}".format(feature_names[::700]))

#%% グリッドサーチで性能を評価
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))


#%% ----------------------------------
'''7.4 ストップワード'''
print("Number of stop words: {}".format(len(ENGLISH_STOP_WORDS)))
print("Every 10th stopword:\n{}".format(list(ENGLISH_STOP_WORDS)[::10]))

#%% CountVectorizerに適用する
vect = CountVectorizer(min_df=5, stop_words='english').fit(text_train)
X_train = vect.transform(text_train)
print("X_train with stop words:\n{}".format(repr(X_train)))

#%% グリッドサーチで性能を評価
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))


#%% ----------------------------------
'''7.5 tf_idfを用いたデータのスケール変換'''
pipe = make_pipeline(TfidfVectorizer(min_df=5, norm=None),
                    LogisticRegression())
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10]}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train) # 入力はテキストデータ
print("Best cross-validation score: {:.2f}".format(grid.best_score_))

#%% 重要だと判断された単語の確認
vectorizer = grid.best_estimator_.named_steps['tfidfvectorizer']
X_train = vectorizer.transform(text_train) # 訓練データセットの変換
# それぞれの特徴量のデータセット中での最大値を見つける
max_value = X_train.max(axis=0).toarray().ravel()
sorted_by_tfidf = max_value.argsort()
# 特徴量名を取得
feature_names = np.array(vectorizer.get_feature_names())

print("Features with lowest tfidf:\n{}".format(
      feature_names[sorted_by_tfidf[:20]]))

print("Features with highest tfidf: \n{}".format(
      feature_names[sorted_by_tfidf[-20:]]))


#%% 重要ではないと判断された単語の確認
sorted_by_idf = np.argsort(vectorizer.idf_)
print("Features with lowest idf:\n{}".format(
       feature_names[sorted_by_idf[:100]]))


#%% ----------------------------------
'''7.6 モデル係数の調査'''
# LogisticRegressionの最も大きい係数とそれに対応する単語を確認
mglearn.tools.visualize_coefficients(
    grid.best_estimator_.named_steps["logisticregression"].coef_,
    feature_names, n_top_features=40)


#%% ----------------------------------
'''7.7 1単語よりも大きい単位のBag-of-Words(n-gram)'''
print("bards_words:\n{}".format(bards_words))

# バイグラムに設定
cv = CountVectorizer(ngram_range=(2, 2)).fit(bards_words)
print("Vocabulary size: {}".format(len(cv.vocabulary_)))
print("Vocabulary:\n{}".format(cv.get_feature_names()))
print("Transformed data (dense):\n{}".format(cv.transform(bards_words).toarray()))

#%% ユニグラム、バイグラム、トリグラムを適用する
cv = CountVectorizer(ngram_range=(1, 3)).fit(bards_words)
print("Vocabulary size: {}".format(len(cv.vocabulary_)))
print("Vocabulary:\n{}".format(cv.get_feature_names()))


#%% グリッドサーチを使ってn-gramの最良値を探索する
pipe = make_pipeline(TfidfVectorizer(min_df=5),
                    LogisticRegression())
param_grid = {'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100],
              "tfidfvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)]}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters:\n{}".format(grid.best_params_))

#%% グリッドサーチ結果の可視化
scores = grid.cv_results_['mean_test_score'].reshape(-1, 3).T
heatmap = mglearn.tools.heatmap(
    scores, xlabel="C", ylabel="ngram_range", cmap="viridis", fmt="%.3f",
    xticklabels=param_grid['logisticregression__C'],
    yticklabels=param_grid['tfidfvectorizer__ngram_range'])
plt.colorbar(heatmap)


#%% ----------------------------------
'''7.8 より進んだトークン分割、語幹処理、見出し語化'''
# 語幹処理(Porter stemmer)
en_nlp = spacy.load('en') # spacyの英語モデルをロード
stemmer = nltk.stem.PorterStemmer() # インスタンスの作成

# spacyによる見出し語化とnltkによる語幹処理を比較する関数を定義
def compare_normalization(doc):
    # spacyで文書をトークン分割
    doc_spacy = en_nlp(doc)
    # spacyで見つけた見出し語を表示
    print("Lemmatization:")
    print([token.lemma_ for token in doc_spacy])
    print("Stemming:")
    print([stemmer.stem(token.norm_.lower()) for token in doc_spacy])

compare_normalization(u"Our meeting today was worse than yesterday, "
                       "I'm scared of meeting the clients tomorrow.")


#%% CountVectorizerで、見出し語化だけにspacyを用いるのが良い
# トークン分割用の正規表現
regexp = re.compile('(?u)\\b\\w\\w+\\b')

# spacyの言語モデルを読み込み、トークン分割器を取り出す
en_nlp = spacy.load('en', disable=['parser', 'ner'])
old_tokenizer = en_nlp.tokenizer

# トークン分割器を先ほどの正規表現で置き換える
en_nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(
    regexp.findall(string))

# spacyの文書処理パイプラインを用いてカスタムトークン分割器を作る
def custom_tokenizer(document):
    doc_spacy = en_nlp(document)
    return [token.lemma_ for token in doc_spacy]

# CountVectorizerをカスタムトークン分割器を使って定義する
lemma_vect = CountVectorizer(tokenizer=custom_tokenizer, min_df=5)


#%% 見出し語化を行うCountVectorizerでtext_trainを変換
X_train_lemma = lemma_vect.fit_transform(text_train)
print("X_train_lemma.shape: {}".format(X_train_lemma.shape))

# 比較のために標準のCountVectorizerでも変換
vect = CountVectorizer(min_df=5).fit(text_train)
X_train = vect.transform(text_train)
print("X_train.shape: {}".format(X_train.shape))


#%% ----------------------------------
'''7.9 トピックモデリングと文書クラスタリング'''
vect = CountVectorizer(max_features=10000, max_df=0.15)
X = vect.fit_transform(text_train)

# 10トピックで作成してみる
lda = LatentDirichletAllocation(n_components=10, learning_method='batch',
                                max_iter=25, random_state=0)
document_topics = lda.fit_transform(X)

#%% それぞれの単語のトピックに対する重要度
lda.components_.shape

# それぞれのトピックについて最も重要な単語の確認
sorting = np.argsort(lda.components_, axis=1)[:, ::-1]
# 特徴量名を取得
feature_names = np.array(vect.get_feature_names())

# 最初の10トピックを表示
mglearn.tools.print_topics(topics=range(10), feature_names=feature_names,
                           sorting=sorting, topics_per_chunk=5, n_words=10)

#%% 各トピックについて解析するには、実際の文書を見る必要がある
# 6番目が子供向けの映画と予想できるので確認する
child = np.argsort(document_topics[:, 6])[::-1]
# このトピックを最も重要としている5つの文書を表示
for i in child[:10]:
    print(b".".join(text_train[i].split(b".")[:2]) + b".\n")

#%% それぞれのトピックが全文書に対して得た重みを確認する
# それぞれのトピックに最も一般的な2つの単語を表示
fig, ax = plt.subplots(1, 2, figsize=(10, 10))
topic_names = ["{:>2} ".format(i) + " ".join(words)
               for i, words in enumerate(feature_names[sorting[:, :2]])]

for col in [0, 1]:
    start = col * 5
    end = (col + 1) * 5
    ax[col].barh(np.arange(5), np.sum(document_topics, axis=0)[start:end])
    ax[col].set_yticks(np.arange(5))
    ax[col].set_yticklabels(topic_names[start:end], ha="left", va="top")
    ax[col].invert_yaxis()
    ax[col].set_xlim(0, 2000)
    yax = ax[col].get_yaxis()
    yax.set_tick_params(pad=130)
plt.tight_layout()
