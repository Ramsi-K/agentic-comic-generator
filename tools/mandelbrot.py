import matplotlib.pyplot as plt
import numpy as np


def mandelbrot(h, w, max_iter=1000):
    y, x = np.ogrid[-1.5 : 1.5 : h * 1j, -2 : 1 : w * 1j]
    c = x + y * 1j
    z = c
    divtime = max_iter + np.zeros(z.shape, dtype=int)

    for i in range(max_iter):
        z = z**2 + c
        diverge = z * np.conj(z) > 4
        newly_diverged = diverge & (divtime == max_iter)
        divtime[newly_diverged] = i
        z[diverge] = 2

    return divtime


plt.imshow(
    mandelbrot(400, 400), cmap="twilight_shifted", extent=(-2, 1, -1.5, 1.5)
)
plt.colorbar()
plt.title("Mandelbrot Set")
plt.show()
