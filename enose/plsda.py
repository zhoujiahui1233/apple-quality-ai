# -*- coding: utf-8 -*-
"""PLS-DA（偏最小二乘判别）分类器。
用 PLSRegression 对 one-hot 类别做回归，预测取得分最高类。
独立模块：保证 joblib 保存的模型可以在其他进程中加载（app/predict）。
"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression


class PLSDA:
    def __init__(self, n_components=10, scale=True):
        self.n_components = n_components
        self.scale = scale
        self.classes_ = None
        self.pls_ = None

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        ncomp = min(self.n_components, np.asarray(X).shape[1],
                    len(self.classes_) - 1, max(2, len(self.classes_)))
        self.pls_ = PLSRegression(n_components=max(1, ncomp), scale=self.scale)
        Y = (y[:, None] == self.classes_[None, :]).astype(float)
        self.pls_.fit(X, Y)
        return self

    def predict(self, X):
        Y = self.pls_.predict(X)
        return self.classes_[np.argmax(Y, axis=1)]

    def predict_proba(self, X):
        Y = self.pls_.predict(X)
        Y = np.maximum(Y, 0)
        s = Y.sum(axis=1, keepdims=True)
        return Y / np.maximum(s, 1e-9)

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def get_params(self, deep=True):
        return {"n_components": self.n_components, "scale": self.scale}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self
