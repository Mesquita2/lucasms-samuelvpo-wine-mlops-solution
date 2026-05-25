"""Carregamento e split do dataset Wine (sklearn)."""
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def load_split():
    """Carrega o dataset Wine e devolve splits 60/20/20 estratificados por classe.

    Retorna:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    Tamanhos esperados: 106 / 36 / 36.
    """
    data = load_wine(as_frame=True)
    X, y = data.data, data.target

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    (Xt, yt), (Xv, yv), (Xs, ys) = load_split()
    print(f"train={len(Xt)} val={len(Xv)} test={len(Xs)}")
