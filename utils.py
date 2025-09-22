import numpy as np

def generate_tree_dataset(n=50, add_constant=False):
    """
    Dataset synthétique longueur 
    de l'arbre vs circonférence
    """
    # génération de données synthétiques
    np.random.seed(42)
    X = np.linspace(1, 10, n).reshape(-1, 1)

    true_slope, true_intercept = 3, 5
    # on ajoute un bruit gaussien aux données
    # ce bruit correspond à des variations aléatoires (arbres d'essences différentes,
    # erreurs de mesure, etc.  )
    noise = np.random.normal(0, 3, size=n)
    
    y = true_slope * X.flatten() + true_intercept + noise

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