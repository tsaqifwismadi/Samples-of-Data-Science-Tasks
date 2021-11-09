from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
import joblib
from sklearn.svm import SVR
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
from sklearn.metrics import r2_score
from scipy.spatial.distance import pdist


def scale(X):
    cols = []
    descale = []
    for feature in X.T:
        minimum = feature.min(axis=0)
        maximum = feature.max(axis=0)
        col_std = np.divide((feature - minimum), (maximum - minimum))
        cols.append(col_std)
        descale.append((minimum, maximum))
    X_std = np.array(cols)
    return X_std.T, descale


def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]


def RandomForestClassification(X_train,Y_train,X_test, Y_test, Ntree, Njobs, target_names):

    cl_Final = RandomForestClassifier(n_estimators=Ntree, n_jobs=Njobs)
    cl_Final = cl_Final.fit(X_train, Y_train)
    Ypred = cl_Final.predict(X_test)

    OA = accuracy_score(Y_test,Ypred)

    kappa = cohen_kappa_score(Y_test,Ypred)

    CM = confusion_matrix(Y_test,Ypred)

    CR = classification_report(Y_test,Ypred, target_names=target_names)

    return cl_Final, OA, kappa, CM, CR, Ypred


def SVMclassification(X_train, Y_train, X_test, Y_test, target_names,vfolds):
    # CROSS-VALIDATION:
    C_s = np.logspace(-1, 3, 10)
    Gamma = 1/(2*(np.linspace(0.5,30,10) * np.mean(pdist(X_train,'euclidean')))**2)
    classifiers = []
    cl = np.unique(Y_train)[0]
    for Cp in C_s:
        for G in Gamma:
            scores = []
            for t in range(0,vfolds):
                tr_index = ind_VfoldCross(Y_train, np.round(int(len(np.where(Y_train == cl)[0])/vfolds)))
                val_index = diff(range(len(Y_train)), tr_index)
                x_t = X_train[tr_index, :]
                y_t = Y_train[tr_index]
                x_val = X_train[val_index, :]
                y_val = Y_train[val_index]
                clf = SVC(kernel='rbf', gamma=G, C=Cp)
                clf.fit(x_t, y_t)
                ypred = clf.predict(x_val)
                scores.append(accuracy_score(y_val,ypred))
            classifiers.append([Cp, G, np.mean(scores)])
    classifiers = np.array(classifiers)
    print('CV done!')
    inx = np.where(classifiers == np.amax(classifiers,axis=0)[2])[0]
    BestC = classifiers[inx, 0]
    BestG = classifiers[inx, 1]
    print('Training!')
    cl_Final = SVC(kernel='rbf', gamma=BestG[0], C=BestC[0])
    cl_Final.fit(X_train, Y_train)
    print('Predicting!')
    Y_pred = cl_Final.predict(X_test)
    OA = accuracy_score(Y_test,Y_pred)

    kappa = cohen_kappa_score(Y_test,Y_pred)

    CM = confusion_matrix(Y_test,Y_pred)

    CR = classification_report(Y_test,Y_pred, target_names=target_names)

    return cl_Final, OA, kappa, CM, CR, Y_pred


def SVMregression(X_train, Y_train, X_test, Y_test, vfolds):
    # CROSS-VALIDATION:
    C_s = np.logspace(-1, 3, 10)
    Gamma = 1/(2*(np.linspace(0.5,30,10) * np.mean(pdist(X_train,'euclidean')))**2)
    epsi = np.linspace(0.1,0.5, 5)
    
    model = []
    # kf = cross_validation.KFold(len(X_train), n_folds=vfolds)
    for Cp in C_s:
        for G in Gamma:
            for eps in epsi:
                scores = []
                for t in range(0,vfolds):
                    tr_index = random.sample(range(len(Y_train)), np.round(int(len(Y_train)/vfolds)))
                    val_index = diff(range(len(Y_train)), tr_index)
                    x_t = X_train[tr_index, :]
                    y_t = Y_train[tr_index]
                    x_val = X_train[val_index, :]
                    y_val = Y_train[val_index]
                    clf = SVR(kernel='rbf', gamma=G, C=Cp, epsilon=eps)
                    clf.fit(x_t, y_t)
                    ypred = clf.predict(x_val)
                    scores.append(np.sqrt(sum((y_val-ypred) ** 2)/len(y_val)))
                model.append([Cp, G, eps, np.mean(scores)])
    model = np.array(model)
    print('CV done!')
    inx = np.where(model == np.amin(model,axis=0)[3])[0]
    BestC = model[inx, 0]
    BestG = model[inx, 1]
    Besteps = model[inx, 2]
    print('Training!')
    model_Final = SVR(kernel='rbf', gamma=BestG[0], C=BestC[0], epsilon=Besteps)
    model_Final .fit(X_train, Y_train)
    print('Predicting!')
    Y_pred = model_Final .predict(X_test)
    RMSE = np.sqrt(sum((Y_test-Y_pred) ** 2)/len(Y_test))
    R = r2_score(Y_test, Y_pred)

    return model_Final, RMSE, R, Y_pred


def resultsClassification(classificationRes, Rootoutput, typeSelection, typefile, target_names, NumFeat, Iter, part):

    cl_Final = classificationRes[0]
    OA = classificationRes[1]
    kappa = classificationRes[2]
    CM = classificationRes[3]
    CR = classificationRes[4]

    if Iter >= 0 and part >= 0:
        output_name = typefile + '_' + str(Iter) + '_' + str(part)

    elif Iter >= 0 and part < 0:
        output_name = typefile + '_' + str(Iter)

    elif Iter < 0 and part < 0:
        output_name = typefile

    elif Iter < 0 and part >= 0:
        output_name = typefile + '_' + str(part)

    joblib.dump(cl_Final, Rootoutput + os.path.sep + typeSelection + os.path.sep + 'Classifier_model_' + output_name +
                '.pkl', compress=1)
    print('saved cl_Final')

    # save to disk as csv file
    f = open(Rootoutput + os.path.sep + typeSelection + os.path.sep + 'StatisticalsClassifier_' + typefile + '.txt',
             'a')
    if part == False:
        headerfile = ['Iteration: ', str(Iter)]
    else:
        headerfile = ['Partition: ', str(part), 'Iteration: ', str(Iter)]

    g = csv.writer(f, dialect='unix')
    g.writerow(headerfile)

    for i in range(CM.shape[0]):
        g.writerow(CM[i,:])
    g.writerow('')
    g.writerow(['Dimensions:', NumFeat])
    g.writerow('')
    g.writerow(['OA:', OA])
    g.writerow('')
    g.writerow(['Kappa:',kappa])
    # CR1 = CR.split('\n')
    # g.writerow('')
    # for i in range(8):
    #     if i == 0:
    #         CR1[i] = '_' + CR1[i]
    #     g.writerow(CR1[i].split())
    f.close()


def ind_VfoldCross(data,selec):

    cls = np.unique(data)

    arr_train = []

    for i in cls:
        #get the indexes for each
        ind = np.where(data == i)

        if len(ind[0])<=selec:
            arr_train.extend(ind[0])
        else:
            sel = random.sample(range(len(ind[0])), selec)
            arr_train.extend(ind[0][sel])

    return arr_train


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontsize=12)
    plt.yticks(tick_marks, classes,fontsize=12)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize=13)
    plt.xlabel('Predicted label',fontsize=13)
