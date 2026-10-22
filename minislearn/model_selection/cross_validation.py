import numpy as np 


def cross_val_score(model, X, y, cv=5):
    X= np.asarray(X)
    y= np.asarray(y)

    indices= np.arange(len(X))
    folds= np.array_split(indices, cv)

    scores= []

    for i in range(cv):
        test_indices= folds[i]
        train_indices= np.concatenate([
            folds[j] for j in range(cv) if j != i 
        ])

        x_train= X[train_indices]
        x_test= X[test_indices]

        y_train= y[train_indices]
        y_test= y[test_indices]

        model.fit(x_train, y_train)

        score= np.mean(model.predict(x_test)== y_test)
        scores.append(score)

    return np.array(scores)

