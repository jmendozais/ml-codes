from sklearn.naive_bayes import BernoulliNB
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import RandomizedPCA
from sklearn.preprocessing import StandardScaler
import string
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
def eval(clf, xtr, ytr, xte, yte):
    clf.fit(xtr, ytr)
    pred = clf.predict(xte)
    return np.mean(pred == yte)

class CustomTokenizer(object):
    def __init__(self, remove_footer=False, remove_from=True, remove_subject=False):
        self.remove_footer  = remove_footer
        self.remove_from = remove_from
        self.remove_subject = remove_subject
        self.stemmer = PorterStemmer()
    def __call__(self, doc):
        ans = []
        lines = doc.splitlines()
        doc = ''
        if(self.remove_footer):
            lines = self._remove_footer(lines)
        if(not self.remove_from):
            doc = lines[0] + '\n'
        if(not self.remove_subject):
            doc += lines[1] + '\n'
        for i in range(2, len(lines)):
            doc += lines[i] + '\n'
        doc = str(doc).lower()
        doc = doc.translate(None, string.punctuation)
        for t in word_tokenize(doc):
            if(any(char.isdigit() for char in t)==False):
                ans.append(self.stemmer.stem(t))
        return ans
    def _remove_footer(self, lines):
        footer_begin = len(lines)
        for i in reversed(range(len(lines)/2, len(lines)-1)):
            length = len(lines[i])
            if(length <= 2):
                footer_begin = i
                break
        if((len(lines) - footer_begin)> 6 | (not self.remove_footer)):
            footer_begin = len(lines)
        return lines[:footer_begin-1]

# FETCH DATA
categories = ["rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey", "sci.crypt", "sci.electronics",
              "sci.med", "sci.space"]
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

# REPRESENTATIONS
bin_vectorizer = CountVectorizer(stop_words='english', tokenizer=CustomTokenizer(), strip_accents='ascii', binary=True)
tf_vectorizer = TfidfVectorizer(stop_words='english', tokenizer=CustomTokenizer(), strip_accents='ascii', use_idf=False)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', tokenizer=CustomTokenizer(), strip_accents='ascii', use_idf=True)
bin_xtr = bin_vectorizer.fit_transform(twenty_train.data)
bin_xte = bin_vectorizer.transform(twenty_test.data)
tf_xtr = tf_vectorizer.fit_transform(twenty_train.data)
tf_xte = tf_vectorizer.transform(twenty_test.data)
tfidf_xtr = tfidf_vectorizer.fit_transform(twenty_train.data)
tfidf_xte = tfidf_vectorizer.transform(twenty_test.data)
dtfidf_xtr = tfidf_xtr.todense()
dtfidf_xte = tfidf_xte.todense()
dtf_xtr = tf_xtr.todense()
dtf_xte = tf_xte.todense()

# 1. BERNOULLI NAIVE BAYES
for alpha in [10**(v-5) for v in range(0,11)]:
    acc = eval(BernoulliNB(alpha=alpha), bin_xtr, twenty_train.target, bin_xte, twenty_test.target)
    print("Acc for alpha {} : {} ".format(alpha, acc))

# 2. SVM RBF
# Scaling
std_scaler = StandardScaler()
scl_tfidf_xtr = std_scaler.fit_transform(dtfidf_xtr)
scl_tfidf_xte = std_scaler.transform(dtfidf_xte)

gamma = [10**(v) for v in range(0,9)]
C = gamma
# OVO ( SVC works by default using One-vs-One scheme for multi-class classification )
for g in gamma:
    for c in C:
        svc = SVC(C=c, gamma=g, kernel='rbf');
        acc = eval(svc, scl_tfidf_xtr, twenty_train.target, scl_tfidf_xte, twenty_test.target)
        print("Acc with gamma {}, c {} and # sv {} -> {}".format(g, c, acc, svc.n_support_))

# OVA
for g in gamma:
    for c in C:
        acc = eval(OneVsRestClassifier(SVC(C=c, gamma=g, kernel='rbf', probability=True)), tfidf_xtr, twenty_train.target, tfidf_xte, twenty_test.target)
        print("Acc with gamma {} and c {} -> {}".format(g, c, acc))

# 3. KNN
#pca = PCA(n_components=1000)
#for ncomp in [2**x for x in range(7, 14)]:
pca = RandomizedPCA(n_components=2000)
pca.fit(dtfidf_xtr)
dtfidf_xtr_pca = pca.transform(dtfidf_xtr)
dtfidf_xte_pca = pca.transform(dtfidf_xte)

pca = RandomizedPCA(n_components=2000)
pca.fit(dtf_xtr)
dtf_xtr_pca = pca.transform(dtf_xtr)
dtf_xte_pca = pca.transform(dtf_xte)

# TFIDF
for k in [1,3,5,11,21,31]:
    acc = eval(KNeighborsClassifier(k, algorithm='brute', metric='cosine'), dtfidf_xtr_pca, twenty_train.target, dtfidf_xte_pca, twenty_test.target)
    print("tfidf k: " + str(k) + " acc: " + str(acc))

# TF
for k in [1,3,5,11,21,31]:
    acc = eval(KNeighborsClassifier(k, algorithm='brute', metric='cosine'), dtf_xtr_pca, twenty_train.target, dtf_xte_pca, twenty_test.target)
    print("tf k: " + str(k) + " acc: " + str(acc))

ys2 = []
for ncomp in [2**x for x in range(0, 15)]:
    pca = RandomizedPCA(n_components=ncomp)
    pca.fit(dtfidf_xtr)
    tr = pca.transform(dtfidf_xtr)
    te = pca.transform(dtfidf_xte)
    acc = eval(KNeighborsClassifier(21, algorithm='brute', metric='cosine'), tr, twenty_train.target, te, twenty_test.target)
    print("tfidf ncomp: " + str(ncomp) + " acc: " + str(acc))
    ys2.append(acc)

ys3 = []
for ncomp in [2**x for x in range(0, 16)]:
    pca = RandomizedPCA(n_components=ncomp)
    pca.fit(dtf_xtr)
    tr = pca.transform(dtf_xtr)
    te = pca.transform(dtf_xte)
    acc = eval(KNeighborsClassifier(21, algorithm='brute', metric='cosine'), tr, twenty_train.target, te, twenty_test.target)
    print("tfidf ncomp: " + str(ncomp) + " acc: " + str(acc))
    ys3.append(acc)

import numpy as np
import matplotlib.pyplot as plt
def barplot(labels, x, xlabel='x', ylabel='y', title=' ', color='b'):
    ind = np.arange(len(x))  # the x locations for the groups
    width = 0.6      # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind + width/2, x, width, color)
    # add some
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks(ind+width)
    ax.set_xticklabels(labels)
    plt.show()

def plot(x=None, y=None, xlabel='x', ylabel='y', title=' ', xx=None):
    plt.hold(True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if(x is None):
        plt.plot(y)
    else:
        plt.plot(x, y)
    if(xx is not None):
        plt.xticks(x, xx)
    plt.show()
    plt.hold(False)

#MULTINOMIAL NB
#0.918 removing from
#0.92205743136636165 with 0/1 stem + custom remove
#0.92 with 0/1 stem
#0.918 with 0/1
#0.91 with Counts
#0.90754181129693912]
#BERNOULLY NB
#0.82360366046071321
#0.80971915430735253 without from
#0.77690123067213634 without footer














