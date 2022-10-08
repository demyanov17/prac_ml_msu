import numpy as np
import time
import scipy
from oracles import BinaryLogistic
from scipy.special import expit


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        step_alpha - float, параметр выбора шага из текста задания
        step_beta- float, параметр выбора шага из текста задания
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход
        max_iter - максимальное число итераций
        **kwargs - аргументы, необходимые для инициализации
        """
        self.loss_function = loss_function
        self.alpha, self.beta = step_alpha, step_beta
        self.tolerance, self.max_iter = tolerance, max_iter
        if loss_function == 'binary_logistic':
            self.oracle = BinaryLogistic(**kwargs)

    def fit(self, X, y, X_test=None, y_test=None, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        trace - переменная типа bool
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)
        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        history = {'time': [0], 'func': [], 'acc': []}
        if w_0 is None:
            self.w = np.zeros(X.shape[1])
        else:
            self.w = w_0
        history['func'].append(self.get_objective(X, y))
        if X_test is not None:
            history['acc'].append((self.predict(X_test) == y_test).sum() / len(y_test))
        prev_time = time.time()
        loss = history['func'][0]
        for k in range(self.max_iter):
            self.w = self.w - self.alpha / ((k+1) ** self.beta) * self.get_gradient(X, y)
            new_loss = self.get_objective(X, y)

            if trace:
                curr_time = time.time()
                history['time'].append(curr_time - prev_time)
                history['func'].append(self.get_objective(X, y))
                if X_test is not None:
                    history['acc'].append((self.predict(X_test) == y_test).sum() / len(y_test))
                prev_time = curr_time

            if abs(abs(new_loss) - abs(loss)) < self.tolerance:
                break
            loss = new_loss

        if trace:
            return history

    def predict(self, X):
        """
        Получение меток ответов на выборке X
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: одномерный numpy array с предсказаниями
        """
        pred = np.ones(X.shape[0])
        pred[self.predict_proba(X)[..., 0] > 0.5] = -1
        return pred

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        proba = np.zeros((X.shape[0], 2))
        proba[..., 1] = expit(X.dot(self.w))
        proba[:, 0] = 1 - proba[:, 1]
        return proba

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: float
        """
        return self.oracle.func(X, y, self.w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: numpy array, размерность зависит от задачи
        """
        return self.oracle.grad(X, y, self.w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function, batch_size, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        batch_size - размер подвыборки, по которой считается градиент
        step_alpha - float, параметр выбора шага из текста задания
        step_beta- float, параметр выбора шага из текста задания
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход
        max_iter - максимальное число итераций (эпох)
        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.
        **kwargs - аргументы, необходимые для инициализации
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.alpha, self.beta = step_alpha, step_beta
        self.tolerance, self.max_iter = tolerance, max_iter
        np.random.seed(random_seed)
        if loss_function == 'binary_logistic':
            self.oracle = BinaryLogistic(**kwargs)

    def fit(self, X, y, X_test=None, y_test=None, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}
        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.
        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        history = {'epoch_num': [0], 'time': [0], 'func': [], 'weights_diff': [0], 'acc': []}

        if w_0 is None:
            self.w = np.zeros(X.shape[1])
        else:
            self.w = w_0

        history['func'].append(self.get_objective(X, y))
        if X_test is not None:
            history['acc'].append((self.predict(X_test) == y_test).sum() / len(y_test))
        loss = history['func'][0]
        sgd_processed, num_changer = 0, 0
        prev_time = time.time()

        i = 0
        while i < self.max_iter:
            r = np.random.permutation(X.shape[0])
            X_r, y_r = X[r], y[r]
            prev_w = self.w
            epoch_num = sgd_processed / y.shape[0]
            epoch_iter = 1

            while epoch_num - num_changer <= log_freq:
                X_ep = X_r[(epoch_iter-1)*self.batch_size: self.batch_size*epoch_iter]
                y_ep = y_r[(epoch_iter-1)*self.batch_size: self.batch_size*epoch_iter]
                self.w = self.w - self.alpha / ((i+1) ** self.beta) * self.get_gradient(X_ep, y_ep)
                sgd_processed += self.batch_size
                epoch_num = sgd_processed / y.shape[0]
                epoch_iter += 1
                i += 1
                if i >= self.max_iter:
                    if trace:
                        return history
                    else:
                        return
            new_loss = self.get_objective(X, y)

            if epoch_num - num_changer > log_freq:
                num_changer = epoch_num
                curr_time = time.time()
                if trace:
                    history['epoch_num'].append(epoch_num)
                    history['time'].append(curr_time - prev_time)
                    history['func'].append(new_loss)
                    history['weights_diff'].append(np.linalg.norm(self.w - prev_w) ** 2)
                    if X_test is not None:
                        history['acc'].append((self.predict(X_test) == y_test).sum() / len(y_test))
                prev_time = curr_time

            if abs(new_loss - loss) < self.tolerance:
                break
            loss = new_loss

        if trace:
            return history
