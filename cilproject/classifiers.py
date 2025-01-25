"""Stores the classification heads and sklearn training code."""
import sklearn.linear_model as linear_model
import sklearn.svm as svm
import sklearn.ensemble as ensemble
import sklearn.naive_bayes as naive_bayes
import sklearn.neighbors as neighbors
import sklearn.neural_network as neural_network


def get_classifier(classifier_name: str):
    """Returns a classifier."""
    if classifier_name == "linear":
        return linear_model.LogisticRegression(multi_class="multinomial", max_iter=1500)
    if classifier_name == "svm":
        return svm.SVC(probability=True)
    if classifier_name == "rfc":
        return ensemble.RandomForestClassifier()
    if classifier_name == "knn":
        return neighbors.KNeighborsClassifier(n_neighbors=1, weights="distance")
    if classifier_name == "naive_bayes":
        return naive_bayes.GaussianNB()
    if classifier_name == "mlp":
        return neural_network.MLPClassifier()
    if classifier_name == "vote":
        return ensemble.VotingClassifier(
            estimators=[
                ("lr", linear_model.LogisticRegression(multi_class="multinomial")),
                ("mlp", neural_network.MLPClassifier()),
                ("knn", neighbors.KNeighborsClassifier()),
            ],
            voting="soft",
        )
    else:
        raise ValueError(f"Unknown classifier {classifier_name}.")


def train_classifier(classifier_name, history):
    """Trains a classifier."""
    classifier = get_classifier(classifier_name)
    X = []
    y = []
    for label, embeddings in history.items():
        X.extend(embeddings)
        y.extend([label] * len(embeddings))
    classifier.fit(X, y)
    return classifier
