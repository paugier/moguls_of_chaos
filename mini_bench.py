from numpy import pi, cos, sin
import numpy as np

from transonic import jit


def runge_kutta_step(f, x0, dt, t=None):
    """
        runge_kutta_step(f, x0, dt, t=None)

    Computes a step using the Runge-Kutta 4th order method.

    f is a RHS function, x0 is an array of variables to solve,
    dt is the timestep, and t correspond to an extra paramameter of
    of the RHS.
    """

    k1 = f(t, x0) * dt
    k2 = f(t, x0 + k1 / 2) * dt
    k3 = f(t, x0 + k2 / 2) * dt
    k4 = f(t, x0 + k3) * dt
    x_new = x0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return x_new


def board(t, X_0):
    """
        board(t, X_0)

    Right-hand-side of the equations for a board going down a slope with moguls.

    X_0 is the the set of initial conditions containing [x, y, u, v], in that oder.
    t optional parameter.
    """

    x0 = X_0[0]
    y0 = X_0[1]
    u0 = X_0[2]
    v0 = X_0[3]

    g = 9.81
    c = 0.5
    a = 0.25
    b = 0.5
    p = (2 * pi) / 10.0
    q = (2 * pi) / 4.0

    H_x = -a + b * p * sin(p * x0) * cos(q * y0)
    H_xx = b * p ** 2 * cos(p * x0) * cos(q * y0)
    H_y = b * q * cos(p * x0) * sin(q * y0)
    H_yy = b * q ** 2 * cos(p * x0) * cos(q * y0)
    H_xy = -b * q * p * sin(p * x0) * sin(q * y0)

    F = (g + H_xx * u0 ** 2 + 2 * H_xy * u0 * v0 + H_yy * v0 ** 2) / (
        1 + H_x ** 2 + H_y ** 2
    )

    dU = -F * H_x - c * u0
    dV = -F * H_y - c * v0

    return np.array([u0, v0, dU, dV])


def solver(f, x0, y0, v0, u0, dt, N_t, N, b=0.5):
    """
        solver(f, x0, y0, v0, u0, dt, N_t, N, b = 0.5)

    Function iterate the solution using runge_kutta_step and a RHS function f, for several points or initial
    conditions given by N.

    f is a RHS function
    x0, y0, v0, u0 are arrays containing the initial condition x,y,v,u; with N dimensions.
    dt is the timestep,
    N_t number of time steps.
    N number of initial conditions to iterate.
    b height of the moguls, parameter fixed.
    """

    solutions = np.zeros((N, N_t + 1, 4))
    solutions[:, 0, 0] = x0
    solutions[:, 0, 1] = y0
    solutions[:, 0, 2] = u0
    solutions[:, 0, 3] = v0

    for k in range(N):
        values_one_step = solutions[k, 0, :]
        for i in range(1, N_t + 1):
            values_one_step = solutions[k, i, :] = runge_kutta_step(
                f, values_one_step, dt, b
            )

    return solutions


def bench():
    n_sleds = 10
    n_time = 1000
    x_init = np.zeros(n_sleds)
    y_init = np.random.rand(n_sleds)
    v_init = np.zeros(n_sleds)
    u_init = np.zeros(n_sleds) + 3.5

    solver(board, x_init, y_init, v_init, u_init, 0.01, n_time, n_sleds)


bench_pythran = jit(bench)
bench_numba = jit(backend="numba")(bench)


if __name__ == "__main__":

    from transonic.util import timeit_verbose as timeit

    g = locals()
    norm = timeit("bench()", globals=g)
    timeit("bench_pythran()", globals=g, norm=norm)
    timeit("bench_numba()", globals=g, norm=norm)

"""
bench                            : 1.000 * norm
norm = 8.35e-01 s
bench_pythran                    : 0.007 * norm
bench_numba                      : 0.009 * norm

(~140 speedup!)

"""
