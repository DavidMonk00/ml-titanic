import sys
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt
import keras
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import joblib

import util
import algos


class Model:
    def __init__(self):
        pass

    def loadEvents(self, path):
        self.events = util.loadData(filename=path)
        self.all_tracks = alltracks = pandas.concat(self.events, ignore_index=True)

    def listTrackFeatures(self):
        return self.all_tracks.columns

    def setFeatures(self, features):
        self.features = features

    def setLabel(self, label):
        self.label = label

    def setModel(self, model):
        self.model = model

    def fit(self):
        X = self.all_tracks[self.features]
        y = self.all_tracks[self.label]
        self.X_train, self.X_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.1, random_state=42)
        self.model.fit(self.X_train, np.ravel(self.y_train))

    def evaluate(self, plot=True):
        prob_genuine = self.model.predict_proba(self.X_test)[:,1]
        effis, puris = [], [] # efficiency, purity
        for cut in np.arange(0., 1., 0.02):
            e, p = util.matchPerf(self.y_test, self.y_test[prob_genuine > cut], self.label)
            effis.append(e)
            puris.append(p)
        if plot:
            plt.plot(effis, puris, '.')
            plt.xlabel("Efficiency")
            plt.ylabel("Matching Purity")
            plt.title("Efficiency vs Purity ROC for association BDT")
            plt.show()
        return effis, puris

    def getAUC(self):
        A = 0
        x,y = self.evaluate(plot=False)
        for i in range(len(x) - 1):
            dx = abs(x[i + 1] - x[i]) # HACK:horrible hack but works until a better option can be found
            dy = y[i + 1] - y[i]
            A += dx * (y[i] + dy/2)
        return A


def main():
    m = Model()
    print("Loading events...")
    m.loadEvents("data/tmtt-kf4param-ntuple-ttbar-pu200-938relval.root")
    print(m.listTrackFeatures())
    m.setFeatures(['pt', 'z0', 'eta', 'pt2', 'chi2', 'tanL', 'inv2R', 'dz'])
    m.setLabel(['fromPV'])
    m.setModel(GradientBoostingClassifier(verbose=True))
    print("Fitting...")
    m.fit()
    print("Evaluating...")
    m.evaluate()
    m.getAUC()



if __name__ == '__main__':
    main()
