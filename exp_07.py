import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------
# LOAD DATA
# ---------------------------
data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ---------------------------
# BAGGING
# ---------------------------
bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=50,
    max_samples=0.8
)

bag_scores = cross_val_score(bag, X, y, cv=5)
print("Bagging CV Accuracy:", bag_scores.mean())

bag.fit(X_train, y_train)
bag_pred = bag.predict(X_test)

# ---------------------------
# BOOSTING (AdaBoost)
# ---------------------------
boost = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=0.5
)

boost_scores = cross_val_score(boost, X, y, cv=5)
print("Boosting CV Accuracy:", boost_scores.mean())

boost.fit(X_train, y_train)
boost_pred = boost.predict(X_test)

# ---------------------------
# STACKING
# ---------------------------
estimators = [
    ('dt', DecisionTreeClassifier()),
    ('nb', GaussianNB()),
    ('svm', SVC(probability=True))
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)

stack_scores = cross_val_score(stack, X, y, cv=5)
print("Stacking CV Accuracy:", stack_scores.mean())

stack.fit(X_train, y_train)
stack_pred = stack.predict(X_test)

# ---------------------------
# FINAL ACCURACY
# ---------------------------
print("\nFinal Test Accuracy:")
print("Bagging:", accuracy_score(y_test, bag_pred))
print("Boosting:", accuracy_score(y_test, boost_pred))
print("Stacking:", accuracy_score(y_test, stack_pred))

print("\nStacking Classification Report:\n")
print(classification_report(y_test, stack_pred))
