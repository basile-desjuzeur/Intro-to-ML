import numpy as np

def generate_tree_dataset(n=50, slope=3, intercept=5, add_constant=False):
    """
    Dataset synthétique longueur 
    de l'arbre vs circonférence
    """
    # génération de données synthétiques
    np.random.seed(42)
    X = np.linspace(1, 10, n).reshape(-1, 1)

    # on ajoute un bruit gaussien aux données
    # ce bruit correspond à des variations aléatoires (arbres d'essences différentes,
    # erreurs de mesure, etc.  )
    noise = np.random.normal(0, 3, size=n)
    
    y = slope * X.flatten() + intercept + noise

    if add_constant:
        X = np.hstack([np.ones([n, 1]), X])

    return X, y

def mse(y_hat, y):
    return np.mean((y_hat - y) ** 2)


def generate_mse_grid(X, y, intercept_grid, slope_grid):

    # meshgrid pour avoir tous les couples (slope, intercept)
    S, I = np.meshgrid(slope_grid, intercept_grid)

    # Calcul vectorié du MSE pour chaque couple (slope, intercept) ---
    MSE_vals = np.mean(
                        (
                            y.reshape(1, 1, -1)
                            - (S[:, :, np.newaxis] * X.flatten() + I[:, :, np.newaxis])
                        )
                        ** 2,
                        axis=2,
                    )
    
    return MSE_vals

def gradient_mse(x, y, a, b):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = len(x)
    y_pred = a * x + b
    error = y_pred - y
    d_mse_d_a = (2.0/n) * np.sum(x * error)
    d_mse_d_b = (2.0/n) * np.sum(error)
    return float(d_mse_d_a), float(d_mse_d_b)

def gradient_descent(derivative, x, y, initial_condition=(0,0), lr=0.01, n=100):
    """
    Applique l'algorithme de descente de gradient à la fonction de coût MSE.

    Paramètres :
        - derivative (function): gradient de la fonction de coût
        - x (ndarray): données d'entrée
        - y (ndarray): valeurs cibles
        - initial_condition (tuple): point de départ (a, b)
        - lr (float): learning rate
        - n (int): nombre d'itérations

    Retourne:
        - iteration_history (list): liste des itérés successifs (a, b)
        - derivation_history (list): valeurs du gradient à chaque itération
    """
    a, b = initial_condition
    iteration_history = [(a, b)]
    derivation_history = []

    for _ in range(n):
        da, db = derivative(x, y, a, b)
        derivation_history.append((da, db))

        # Mise à jour des paramètres
        a -= lr * da
        b -= lr * db

        iteration_history.append((a, b))

    return iteration_history, derivation_history
