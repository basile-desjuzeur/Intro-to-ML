import numpy as np

def generate_tree_dataset(n=50, slope=3, intercept=5, add_constant=False, noise_std=False):
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
    if noise_std == False:
        noise = np.random.normal(0, 3, size=n)
    else:
        noise = np.random.normal(0, scale=noise_std, size=n)
    
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

def get_parameters_from_gd(iteration_history):

    return iteration_history[-1][0], iteration_history[-1][1] 

import numpy as np

def generate_regression_data(n_samples=100, n_features=3, 
                             mean=0, std=1, corr=0.5, 
                             beta=None, noise_std=1.0, random_state=None):
    """
    Génère des données pour une régression linéaire multiple.
    
    Paramètres :
    -----------
    n_samples : int
        Nombre d'observations
    n_features : int
        Nombre de variables explicatives
    mean : float ou liste
        Moyenne des variables X
    std : float ou liste
        Écart-type des variables X
    corr : float
        Corrélation entre les variables X (-1 à 1)
    beta : array_like ou None
        Coefficients réels de la régression (taille n_features). Si None, générés aléatoirement
    noise_std : float
        Écart-type du bruit sur y
    random_state : int ou None
        Pour reproductibilité
    
    Retour :
    -------
    X : numpy.ndarray, shape (n_samples, n_features)
    y : numpy.ndarray, shape (n_samples,)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # --- Génération de X ---
    cov = np.full((n_features, n_features), corr)
    np.fill_diagonal(cov, 1.0)
    
    std_array = np.array(std if isinstance(std, (list, np.ndarray)) else [std]*n_features)
    cov = cov * np.outer(std_array, std_array)
    
    mean_array = np.array(mean if isinstance(mean, (list, np.ndarray)) else [mean]*n_features)
    
    X = np.random.multivariate_normal(mean_array, cov, size=n_samples)
    
    # --- Génération de beta ---
    if beta is None:
        beta = np.random.randn(n_features)
    beta = np.array(beta)
    
    # --- Génération de y ---
    epsilon = np.random.normal(0, noise_std, size=n_samples)
    y = X @ beta + epsilon
    
    return X, y, beta
