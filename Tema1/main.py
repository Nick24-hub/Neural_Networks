def parse_file(file_name):
    file1 = open(file_name, 'r')
    Lines = file1.readlines()
    numbers = "0123456789"
    for i in range(3):
        auxx = ['', Lines[i]]
        if 'x' in Lines[i]:
            auxx = Lines[i].split('x')
            sgn = 1
            coef = 0
            for c in auxx[0]:
                if c == '-':
                    sgn = -1
                elif c in numbers:
                    coef = 1
                    A[i][0] = A[i][0] * 10 + int(c)
            if coef == 0:
                A[i][0] = 1
            A[i][0] = A[i][0] * sgn
        auxy = ['', auxx[1]]
        if 'y' in auxx[1]:
            auxy = auxx[1].split('y')
            sgn = 1
            coef = 0
            for c in auxy[0]:
                if c == '-':
                    sgn = -1
                elif c in numbers:
                    coef = 1
                    A[i][1] = A[i][1] * 10 + int(c)
            if coef == 0:
                A[i][1] = 1
            A[i][1] = A[i][1] * sgn
        auxz = ['', auxy[1]]
        if 'z' in auxy[1]:
            auxz = auxy[1].split('z')
            sgn = 1
            coef = 0
            for c in auxz[0]:
                if c == '-':
                    sgn = -1
                elif c in numbers:
                    coef = 1
                    A[i][2] = A[i][2] * 10 + int(c)
            if coef == 0:
                A[i][2] = 1
            A[i][2] = A[i][2] * sgn
        auxr = auxz[1].split('=')
        sgn = 1
        for c in auxr[1]:
            if c == '-':
                sgn = -1
            elif c in numbers:
                R[i][0] = R[i][0] * 10 + int(c)
        R[i][0] = R[i][0] * sgn


def determinant(M):
    if len(M) == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    elif len(M) == 3:
        return M[0][0] * M[1][1] * M[2][2] + M[1][0] * M[2][1] * M[0][2] + M[2][0] * M[0][1] * M[1][2] - M[0][2] * M[1][
            1] * M[2][0] - M[1][2] * M[2][1] * M[0][0] - M[2][2] * M[0][1] * M[1][0]


def get_transpouse(M):
    n = len(M)
    T = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            T[i][j] = M[j][i]
    return T


def get_minor(x, y, M):
    n = len(M)
    minor = [[0 for i in range(n - 1)] for j in range(n - 1)]
    for i in range(n):
        for j in range(n):
            if i < x and j < y:
                minor[i][j] = M[i][j]
            elif i > x and j < y:
                minor[i - 1][j] = M[i][j]
            elif i < x and j > y:
                minor[i][j - 1] = M[i][j]
            elif i > x and j > y:
                minor[i - 1][j - 1] = M[i][j]
    return minor


def get_Adjunct(M):
    Transpouse = get_transpouse(M)
    n = len(M)
    Adjunct = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            Adjunct[i][j] = determinant(get_minor(i, j, Transpouse))
            Adjunct[i][j] = Adjunct[i][j] * (-1) ** (i + j)
    return Adjunct


def get_inverse(M):
    n = len(M)
    Adjunct = get_Adjunct(M)
    x = determinant(M)
    Inverse = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            Inverse[i][j] = Adjunct[i][j] / x
    return Inverse


def simple_resolve():
    if determinant(A) != 0:
        Inverse = get_inverse(A)
        for i in range(3):
            for j in range(3):
                X[i][0] = X[i][0] + Inverse[i][j] * R[j][0]
        print("Solution vector:\n", X)
    else:
        print("Determinant of matrix A is 0.")


def numpy_resolve():
    if determinant(A) != 0:
        import numpy as np
        print("Solution vector using numpy:\n", np.linalg.solve(A, R))
    else:
        print("Determinant of matrix A is 0.")


A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
X = [[0], [0], [0]]
R = [[0], [0], [0]]

parse_file("input2.txt")
print("Linear system matrix:\n", A)
print("Results vector:\n", R)
simple_resolve()
numpy_resolve()
