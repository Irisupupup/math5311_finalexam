import numpy as np
import matplotlib.pyplot as plt


def exact_solution(x, y, t):
    return np.sin(x) * np.sin(y) * np.cos(np.sqrt(2) * t)


def initial_u(x, y):
    return exact_solution(x, y, 0)


def initial_ut(x, y):
    return 0



def stability_condition(dx, n):
    return dx ** n



def central_difference_scheme(X, Y, N, T, dt):
    # dx = 2 * np.pi / N
    # dy = 2 * np.pi / N

    # X = np.zeros((N + 1, N + 1))
    # Y = np.zeros((N + 1, N + 1))
    # for i in range(N + 1):
    #     for j in range(N + 1):
    #         X[i, j] = i * dx
    #         Y[i, j] = j * dy

    u = np.zeros((N + 1, N + 1))
    u_prev = np.zeros((N + 1, N + 1))
    u_next = np.zeros((N + 1, N + 1))

    Nt = int(T / dt)
    ddt = T / Nt

    # initialize
    for i in range(N + 1):
        for j in range(N + 1):
            u[i, j] = initial_u(X[i, j], Y[i, j]) + ddt * initial_ut(X[i, j], Y[i, j])
            u_prev[i, j] = initial_u(X[i, j], Y[i, j])


    # time advancing
    for n in range(Nt - 1):
        for i in range(1, N):
            for j in range(1, N):
                u_next[i, j] = 2 * u[i, j] - u_prev[i, j] + (ddt ** 2 / dx ** 2) * (
                            u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) + (ddt ** 2 / dy ** 2) * (
                                            u[i, j + 1] - 2 * u[i, j] + u[i, j - 1])
        u_prev = u.copy()
        u = u_next.copy()

    
    return u


# get error
N_values = [20, 40, 80, 160]
errors = []
T = 1
for N in N_values:
    dt = stability_condition(2 * np.pi / N,2)
    # dt = 0.01
    dx = 2 * np.pi / N
    dy = 2 * np.pi / N
    X = np.zeros((N + 1, N + 1))
    Y = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            X[i, j] = i * dx
            Y[i, j] = j * dy

    u_numerical = central_difference_scheme(X, Y, N, T, dt)
    u_exact = exact_solution(X, Y, T)
    r = u_numerical - u_exact
    
    res = np.linalg.norm(r/N, 2)    
    errors.append(res)
    
    # u_exact = exact_solution(X, Y, T)
    error = np.max(np.abs(u_exact - u_numerical))
    print(f"For N = {N}, the max error is {error}")

    # plt.scatter(X, Y, c=np.abs(u_exact - u_numerical), cmap=plt.cm.rainbow, vmin=0.0, vmax=0.08)
    plt.contourf(X, Y, np.abs(u_exact - u_numerical))
    plt.colorbar()
    plt.title(f"Error plot for N = {N}")
    plt.show()

plt.plot(N_values, errors)
plt.yscale("log")
plt.title("Error vs N")
plt.show()