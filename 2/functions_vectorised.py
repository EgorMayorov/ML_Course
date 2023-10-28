import numpy as np


def sum_non_neg_diag(X: np.ndarray) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """
    # return np.sum(np.diag(X)[np.where(np.diag(X) > 0)])
    a = np.diag(X)[np.where(np.diag(X) >= 0)]
    return np.sum(a) if a.size > 0 else -1


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    return np.array_equal(np.sort(x), np.sort(y))
    # pass


def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    '''print(x.size)
    print(x[0] % 3 == 0)
    if x.size == 1 and x[0] % 3 == 0:
        return x[0]'''
    a = (np.roll(x, 1)*x)[1:]
    a = a[np.where(a % 3 == 0)]
    return np.max(a) if a.size else -1
    # pass


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    return image.dot(weights)
    # pass


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    def recoded(a):
        return np.repeat(a[:, 0], a[:, 1], axis=0)
    x_recoded = recoded(x)
    y_recoded = recoded(y)
    return -1 if len(x_recoded) != len(y_recoded) else np.sum(x_recoded * y_recoded)
    # pass


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y.
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        distance = np.divide(X @ Y.T, (
            np.linalg.norm(X, axis=1)[:, None] *
            np.linalg.norm(Y, axis=1)[None, :]
        ))
    distance[np.where(np.isnan(distance))] = 1
    return distance
