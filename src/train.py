from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import joblib


def load_data():
    iris = load_iris()
    return iris.data, iris.target, iris.target_names


def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train, random_state=42):
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_test, y_test, target_names):
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Decision Tree")
    plt.show()


def main():
    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    clf = train_model(X_train, y_train)
    evaluate_model(clf, X_test, y_test, target_names)
    joblib.dump(clf, "outputs/decision_tree_model.joblib")


if __name__ == "__main__":
    main()