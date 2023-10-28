from typing import List


def sum_non_neg_diag(X: List[List[int]]) -> int:
    """
    Вернуть  сумму неотрицательных элементов на диагонали прямоугольной матрицы X. 
    Если неотрицательных элементов на диагонали нет, то вернуть -1
    """ 
    s = 0
    ch = 0
    for i in range(len(X)):
        for j in range(len(X[0])):
            if (i == j) and (X[i][j] >= 0):
                s += X[i][j]
                ch = 1
    return s if ch else -1
    # pass


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    """
    Проверить, задают ли два вектора одно и то же мультимножество.
    """
    return sorted(x) == sorted(y)
    # pass


def max_prod_mod_3(x: List[int]) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    maximum = -1
    for i in range(len(x) - 1):
        if (((x[i]*x[i+1]) % 3 == 0) and (x[i]*x[i+1] > maximum)):
            maximum = x[i]*x[i+1]
    return maximum
    # pass


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    """
    Сложить каналы изображения с указанными весами.
    """
    res = []
    ind = 0
    for i in range(len(image)):
        x0 = []
        for j in range(len(image[i])):
            elem = 0
            index = 0
            for k in range(0, len(image[i][j])):
                elem += image[i][j][k]*weights[k]
            x0.insert(index, elem)
            index += 1
        x0.reverse()
        res.insert(ind, x0)
        ind += 1
    return res
    # pass


def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    x_n = []
    y_n = []
    ind = 0
    s = 0
    for i in range(len(x)):
        for j in range(x[i][1]):
            x_n.insert(ind, x[i][0])
    ind = 0
    for i in range(len(y)):
        for j in range(y[i][1]):
            y_n.insert(ind, y[i][0])
    if len(x_n) != len(y_n):
        return -1
    for i in range(len(x_n)):
        s += x_n[i]*y_n[i]
    return s
    # pass


def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    lst = []
    ind = 0
    for i in range(len(X)):
        x0 = []
        index = 0
        for j in range(len(Y)):
            scl = 0
            lnx = 0
            lny = 0
            for k in range(len(Y[j])):
                scl += X[i][k]*Y[j][k]
            for k in range(len(Y[j])):
                lnx += X[i][k]**2
                lny += Y[j][k]**2
            if lnx*lny == 0:
                x0.insert(index, 1)
            else:
                x0.insert(index, scl/((lnx*lny)**0.5))
            index += 1
        lst.insert(ind, x0)
        ind += 1
    return lst
    # pass
