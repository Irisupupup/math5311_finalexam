import numpy as np
import matplotlib.pyplot as plt


def exact_solution(x, y, t):
    return np.sin(x) * np.sin(y) * np.cos(np.sqrt(2) * t)


def initial_u(x, y):
    return exact_solution(x, y, 0)


def initial_ut(x, y):
    return 0



def stability_condition(dx, dy):
    return 0.5 * dx


def central_difference_scheme(N, T, dt):
    dx = 2 * np.pi / N
    dy = 2 * np.pi / N
    x = np.linspace(0, 2 * np.pi, N+1)
    y = np.linspace(0, 2 * np.pi, N+1)
    X, Y = np.meshgrid(x, y)
    u = np.zeros((N+1, N+1))
    u_prev = np.zeros((N+1, N+1))
    u_next = np.zeros((N+1, N+1))

    # initialize
    for i in range(N+1):
        for j in range(N+1):
            u[i, j] = initial_u(X[i, j], Y[i, j]) + dt * initial_ut(X[i, j], Y[i, j])
            u_prev[i, j] = initial_u(X[i, j], Y[i, j])

    Nt = int(T / dt)

    # time advancing
    tt = dt
    for n in range(Nt-1):
        for i in range(1, N):
            for j in range(1, N):
                u_next[i, j] = 2 * u[i, j] - u_prev[i, j] + (dt ** 2 / dx ** 2) * (
                                u[i + 1, j] - 2 * u[i, j] + u[i - 1, j]) + (dt ** 2 / dy ** 2) * (
                                u[i, j + 1] - 2 * u[i, j] + u[i, j - 1])

    tt += dt
    # print(tt)
    u_prev = u.copy()
    u = u_next.copy()

    return u


# get error
N_values = [20, 40, 80, 160]
T = 1.0
for N in N_values:
    dt = 0.0001
    u_numerical = central_difference_scheme(N, T, dt)
    x = np.linspace(0, 2 * np.pi, N+1)
    y = np.linspace(0, 2 * np.pi, N+1)
    X, Y = np.meshgrid(x, y)
    u_exact = exact_solution(X, Y, T)
    error = np.max(np.abs(u_exact - u_numerical))
    print(f"For N = {N}, the max error is {error}")

    # plt.scatter(X, Y, c=np.abs(u_exact - u_numerical), cmap=plt.cm.rainbow, vmin=0.0, vmax=0.08)
    # plt.contourf(X, Y, u_numerical)
    # plt.colorbar()
    # plt.title(f"Error plot for N = {N}")
    # plt.show()



N = np.array([20,40,80,160])
dx = 2*np.pi/N
err = np.array([0.0058, 0.0015, 0.0004288, 0.00015959])
order2 = dx**2

plt.plot(dx, err)
plt.plot(dx, order2, "--")
plt.yscale("log")
plt.xscale("log")
plt.xlabel(r"$\Delta x$")
plt.ylabel(r"Err")
plt.legend([r"$|u-u_h|$", r"$\Delta x^2$"])
plt.show()