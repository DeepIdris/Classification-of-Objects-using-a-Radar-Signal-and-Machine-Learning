import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import sys


# First, lets load our training and test data
X = pd.read_csv('binary/X.csv', header= None)
y = pd.read_csv('binary/y.csv', header= None)
X_to_classify = pd.read_csv('binary/XToClassify.csv', header= None)

# Next, we create our training and test set from the data
X_train1, X_test = train_test_split(X, test_size=0.2, random_state=42)

y_train, y_test = train_test_split(y, test_size=0.2, random_state=42)

y_train = y_train.values.ravel()

# Normalize the data to avoid numerical stability issues
sc = StandardScaler()
X_train = sc.fit_transform(X_train1)
X_test = sc.transform(X_test)
X_to_classify = sc.transform(X_to_classify)


# Description of data
def data_description():
    print("Data shape = ", X_train.shape)
    print("\nData description", pd.DataFrame.from_records(X_train).describe())
    print("\n\nCorrelation", pd.DataFrame.from_records(X_train).corr())


# Data Analysis and Visualization

# We try to explore more, some statistical properties of the data, we plot a 2-D scatter plot of the data
def visualization():
    pca = PCA(n_components=3)
    X2D = pca.fit_transform(X_train)
    plt.scatter(X2D[:,0], X2D[:,1], c = y_train)
    plt.title("Dimensionality Reduction to Two Dimensions")
    plt.xlabel('Z0')
    plt.ylabel('Z1')
    plt.savefig('2D-PCA')
    plt.show()


# In choosing a suitable subset of features, we use PCA to reduce the number of features
def suitable_subsets():
    pca = PCA(n_components= 20)
    pca.fit_transform(X_train)
    print("PCA explained variance ratio: ", pca.explained_variance_ratio_)
    var = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3) * 100)
    plt.title('PCA Analysis Cumulative')
    plt.grid(True)
    plt.xlabel('Dimensions')
    plt.ylabel('Cumulative Explained Variance')
    plt.plot(var)
    plt.savefig('PCA Analysis 1')
    plt.show()

    plt.plot((pca.explained_variance_) / 10)
    plt.title('PCA Analysis Proportion')
    plt.grid(True)
    plt.xlabel('Dimensions')
    plt.ylabel('Explained Variance')
    plt.savefig('PCA Analysis 2')
    plt.show()

# Now, we can transform our training set using a pca of 5 components
pca = PCA(n_components= 5)
X_train1 = pca.fit_transform(X_train)
X_test1 = pca.transform(X_test)
X_to_classify1 = pca.transform(X_to_classify)


# Function for displaying cross validaion scores
def display_scores(scores):
    print("Cross Validation Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# Function for plotting precision vs recall graph (Aurelien Geron's book)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, title):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.title(title)
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1.5])
    plt.xlim([7, 9])


# Function for plotting the roc curve (Aurelien Geron's book)
def plot_roc_curve(fpr, tpr, title, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


# Function to visualize the classifiers
def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()

    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.set_ylabel('Z1')
    ax.set_xlabel('Z0')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, zorder=1)

    ax.set(xlim=xlim, ylim=ylim)


# Now, let's create our binary classifier to recognize books (y = 0) and 1 (plastic case)

# Let's develop a Logistic Regression Classifier to handle this initially
def logistic_regression():
    log_reg = LogisticRegression(solver="liblinear", random_state=42)

    # Plot the decision boundary for the binary classifier
    visualize_classifier(log_reg.fit(X_train1[:, 0:2], y_train), X_train1[:, 0:2], y_train)
    plt.title('Decision surface of Logit Regression')
    plt.xlabel('Z0'), plt.ylabel('Z1')
    plt.savefig("Decision surface Logit")
    plt.show()

    # Now, fit the decision boundary to the new features from dimensionality reduction
    log_reg.fit(X_train1, y_train)

    # Now let's evaluate the performance of our model using cross validation of 5 folds
    cv_score = cross_val_score(log_reg, X_train1, y_train, cv=5, scoring="accuracy")
    print("-----------------------------------------")
    print ("Logistic Regression Classifier")
    print("-----------------------------------------")
    # print("Prediction Probabilities:", log_reg.predict_proba(X_train1))
    display_scores(cv_score)

    # To better evaluate our model, we use a confusion matrix with predictions using
    # 5 folds cross validation and actual targets
    y_train_pred = cross_val_predict(log_reg, X_train1, y_train, cv=5)
    conf_matrix = confusion_matrix(y_train, y_train_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # We can also further our evaluation by calculating the precision, recall and f1-score
    print("\nPrecision:", precision_score(y_train, y_train_pred))
    print("Recall:", recall_score(y_train, y_train_pred))
    print("F1 Score:", f1_score(y_train, y_train_pred))

    # Further more, we can explore more evaluation using the Precision and recall versus the decision threshold graph
    y_scores = cross_val_predict(log_reg, X_train1, y_train, cv=5, method="decision_function")
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds, "Logit Precision & Recall Versus Threshold")
    plt.savefig("Logit Precision & Recall Versus Threshold")
    plt.show()

    # Consequently, we can plot the graph of Precision against Recall to evaluate the model more
    plt.plot(recalls, precisions, "b", label="Precision")
    plt.title("Logit Precision versus Recall")
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.savefig("Logit Precision versus Recall")
    plt.show()

    # Finally, we can plot the ROC curve which plots the FPR against the TPR
    fpr, tpr, thresholds = roc_curve(y_train, y_scores)
    plot_roc_curve(fpr, tpr,"Logit ROC Curve", "Logit")
    plt.legend(loc="lower right", fontsize=16)
    plt.savefig("Logit ROC Curve")
    plt.show()

    print("Area under curve: ",roc_auc_score(y_train, y_scores))

    # Now, we can proceed to testing on our test data measuring the score
    print("\nScore on Test set:", log_reg.score(X_test1, y_test))

    # Finally, we can now test on the final test data and print to a file
    predictions = log_reg.predict(X_to_classify1)
    print("XTOClassify Predictions: ",predictions)


# Now, let's proceed and develop a Linear Support Vector Classifier to handle this also.
def support_vector_classifier():
    svm_clf = LinearSVC(C=1, loss="hinge", random_state=42)

    # Plot the decision boundary for the binary classifier
    visualize_classifier(svm_clf.fit(X_train1[:, 0:2], y_train), X_train1[:, 0:2], y_train)
    plt.title('Decision surface of LinearSVC')
    plt.xlabel('Z0'), plt.ylabel('Z1')
    plt.savefig("Decision surface LinearSVC")
    plt.show()

    # Now, fit the decision boundary to the new features from dimensionality reduction
    svm_clf.fit(X_train1, y_train)

    # Now let's evaluate the performance of our model using cross validation of 5 folds
    cv_score = cross_val_score(svm_clf, X_train1, y_train, cv=5, scoring="accuracy")
    print("-----------------------------------------")
    print ("Linear Support Vector Classifier")
    print("-----------------------------------------")
    display_scores(cv_score)

    # To better evaluate our model, we use a confusion matrix with predictions using
    # 5 folds cross validation and actual targets
    y_train_pred = cross_val_predict(svm_clf, X_train1, y_train, cv=5)
    conf_matrix = confusion_matrix(y_train, y_train_pred)
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # We can also further our evaluation by calculating the precision, recall and f1-score
    print("\nPrecision:", precision_score(y_train, y_train_pred))
    print("Recall:", recall_score(y_train, y_train_pred))
    print("F1 Score:", f1_score(y_train, y_train_pred))

    # Further more, we can explore more evaluation using the Precision and recall versus the decision threshold graph
    y_scores = cross_val_predict(svm_clf, X_train1, y_train, cv=5, method="decision_function")
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)

    # Consequently, we can plot the graph of Precision against Recall to evaluate the model more
    plt.plot(recalls, precisions, "b", label="Precision")
    plt.title("Linear SVC Precision versus Recall")
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.savefig("Linear SVC Precision versus Recall")
    plt.show()

    # Finally, we can plot the ROC curve which plots the FPR against the TPR
    fpr, tpr, thresholds = roc_curve(y_train, y_scores)
    plot_roc_curve(fpr, tpr,"LinearSVC ROC Curve", "LinearSVC")
    plt.legend(loc="lower right", fontsize=16)
    plt.savefig("LinearSVC ROC Curve")
    plt.show()

    print("Area under curve:", roc_auc_score(y_train, y_scores))

    # Now, we can proceed to testing on our test data measuring the score
    print("\nScore on Test set:", svm_clf.score(X_test1, y_test))

    # Finally, we can now test on the final test data and print to a file
    predictions = svm_clf.predict(X_to_classify1)
    print("XTOClassify Predictions: ",predictions)
    model_file = open('PredictedClasses.csv', 'w')
    sys.stdout = model_file
    for val in list(predictions):
        print(val)


# Let's also train a random forest classifier for this data
def random_classifier():
    rand_clf = RandomForestClassifier(n_estimators=10, n_jobs= -1, random_state=42)

    # Plot the decision boundary for the binary classifier
    visualize_classifier(rand_clf.fit(X_train1[:, 0:2], y_train), X_train1[:, 0:2], y_train)
    plt.title('Decision surface of RF')
    plt.xlabel('Z0'), plt.ylabel('Z1')
    plt.savefig("Decision surface RF")
    plt.show()

    # Now, fit the decision boundary to the new features from dimensionality reduction
    rand_clf.fit(X_train1, y_train)

    cv_score = cross_val_score(rand_clf, X_train1, y_train, cv=5, scoring="accuracy")
    print("-----------------------------------------")
    print("Random Forest Classifier")
    print("-----------------------------------------")
    display_scores(cv_score)

    y_train_pred = cross_val_predict(rand_clf, X_train1, y_train, cv=5)
    conf_matrix = confusion_matrix(y_train, y_train_pred)
    print("\nConfusion Matrix")
    print(conf_matrix)

    # We can also further our evaluation by calculating the precision, recall and f1-score
    print("\nPrecision:", precision_score(y_train, y_train_pred))
    print("Recall:", recall_score(y_train, y_train_pred))
    print("F1 Score:", f1_score(y_train, y_train_pred))

    # Further more, we can explore more evaluation using the Precision and recall versus the decision threshold graph
    y_scores = cross_val_predict(rand_clf, X_train1, y_train, cv=5, method="predict_proba")

    # Finally, we can plot the ROC curve which plots the FPR against the TPR
    y_scores_forest = y_scores[:, 1]   # score = proba of positive class
    fpr_f, tpr_f, thresholds = roc_curve(y_train, y_scores_forest)
    plot_roc_curve(fpr_f, tpr_f,"RF ROC Curve", "Random Forest")
    plt.legend(loc="lower right", fontsize=16)
    plt.savefig("RF ROC Curve")
    plt.show()

    print("Area under curve: ", roc_auc_score(y_train, y_scores_forest))

    # Finally, let's test on our the test set we created measuring the score
    print("\nScore on Test set:", rand_clf.score(X_test1, y_test))

    # Now, lets predict on our XTOCalassify dataset and print its predictions
    predictions = rand_clf.predict(X_to_classify1)
    print("XTOClassify Predictions: ",predictions)


data_description()
visualization()
suitable_subsets()
logistic_regression()
random_classifier()
support_vector_classifier()

