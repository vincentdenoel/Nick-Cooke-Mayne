import numpy as np
from wind_psd import wind_psd
from SSDpy.dyn import EOMsolvers as solvers

def gen_sample(typ, T=6000):
    n_windows = int(T/600)
    N = 2**16 * n_windows

    dt = T / N
    df = 1 / T
    dom = 2 * np.pi * df
    f = np.arange(0, N) * df

    t = np.arange(0, N) * dt

    if typ == 0:  # Bandlimited White Noise
        S_ = np.abs(f) < 5

        fft_w = N * np.sqrt(S_ * dom) * np.exp(2j * np.pi * np.random.rand(N))
        fft_w[0] = 0
        fft_w[N // 2] = 0
        fft_w[N // 2 + 1:] = np.conj(fft_w[N // 2 - 1:0:-1])

        x = np.fft.ifft(fft_w).real
        x /= np.std(x)

    elif typ in {1, 2, 3, 4}:  # Davenport PSD
        sigu = 1
        U = 30

        S_ = wind_psd(f, [sigu, U], 2)

        fft_w = N * np.sqrt(S_ * dom) * np.exp(2j * np.pi * np.random.rand(N))
        fft_w[0] = 0
        fft_w[N // 2] = 0
        fft_w[N // 2 + 1:] = np.conj(fft_w[N // 2 - 1:0:-1])

        x = np.fft.ifft(fft_w).real
        x /= np.std(x)

    # Additional filtering for cases 3, 4, and 5
    if typ == 2:
        t, x, _, _, _ = solvers.newmark(1, 0.01, 2, x, dt, N - 1)
        x /= np.std(x)

    elif typ == 3:
        t, x_full, _, _, _ = solvers.NewmarkMDDL(
            np.diag([1, 1]),
            np.array([[400, -100], [-100, 300]]),
            np.diag([0.04, 0.01]),
            np.vstack([x, np.zeros(N)]),
            dt,
            N - 1
        )
        x = x_full[0, :] / np.std(x_full[0, :])
        x = np.append(x, x[-1])

    elif typ == 4:
        t, x, _, _, _ = solvers.newmark(1, 0.001, 2, x, dt, N - 1, 0.25, 0.5, 0.05, 0)
        x /= np.std(x)

    return x, t, dt
