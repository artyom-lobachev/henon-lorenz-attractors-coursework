import numpy as np
import matplotlib.pyplot as plt


def fixed_points(M1, M2, B):
    discriminant = ((B + M2 - 1) ** 2) / 4 + M1

    if discriminant < 0:
        return []

    sqrt_d = np.sqrt(discriminant)

    x1 = (B + M2 - 1) / 2 + sqrt_d
    x2 = (B + M2 - 1) / 2 - sqrt_d

    return [
        (x1, x1, x1),
        (x2, x2, x2)
    ]


def initial_point_near_p1(M1, M2, B, eps=1e-6):
    points = fixed_points(M1, M2, B)

    if not points:
        return None

    p1 = points[0]

    return p1[0] + eps, p1[1], p1[2]


def henon3d_step(x, y, z, M1, M2, B):
    x_next = y
    y_next = z
    z_next = M1 + B * x + M2 * y - z ** 2

    return x_next, y_next, z_next


def jacobian_matrix(x, y, z, M1, M2, B):
    return np.array([
        [0, 1, 0],
        [0, 0, 1],
        [B, M2, -2 * z]
    ])


def lyapunov_exponents(
    x0, y0, z0,
    M1, M2, B,
    n_iter=8000,
    n_skip=1500,
    escape_bound=1e6
):
    x = x0
    y = y0
    z = z0

    for _ in range(n_skip):
        if max(abs(x), abs(y), abs(z)) > escape_bound:
            return None

        x, y, z = henon3d_step(x, y, z, M1, M2, B)

        if not np.isfinite(x + y + z):
            return None

    Q = np.eye(3)
    lyap_sum = np.zeros(3)

    for _ in range(n_iter):
        if max(abs(x), abs(y), abs(z)) > escape_bound:
            return None

        J = jacobian_matrix(x, y, z, M1, M2, B)
        Z = J @ Q

        Q, R = np.linalg.qr(Z)

        stretch = np.abs(np.diag(R))
        stretch = np.maximum(stretch, 1e-300)

        lyap_sum += np.log(stretch)

        x, y, z = henon3d_step(x, y, z, M1, M2, B)

        if not np.isfinite(x + y + z):
            return None

    return lyap_sum / n_iter


def classify_regime(lyap, zero_tolerance=5e-4):
    if lyap is None:
        return 0

    lambda1, lambda2, lambda3 = lyap

    if lambda1 <= 0:
        return 1

    if abs(lambda2) < zero_tolerance:
        return 3

    if lambda2 > 0:
        return 4

    return 2


def build_lyapunov_map(
    B=0.7,
    grid_size=120,
    n_iter=8000,
    n_skip=1500,
    m1_min=-0.08,
    m1_max=0.08,
    m2_min=0.75,
    m2_max=0.95,
    eps=1e-6
):
    M1_values = np.linspace(m1_min, m1_max, grid_size)
    M2_values = np.linspace(m2_min, m2_max, grid_size)

    shape = (len(M2_values), len(M1_values))

    regime_map = np.zeros(shape)
    lambda1_map = np.full(shape, np.nan)
    lambda2_map = np.full(shape, np.nan)
    lambda3_map = np.full(shape, np.nan)

    total = len(M1_values) * len(M2_values)
    counter = 0

    for i, M2 in enumerate(M2_values):
        for j, M1 in enumerate(M1_values):
            initial_point = initial_point_near_p1(M1, M2, B, eps=eps)

            if initial_point is None:
                lyap = None
            else:
                x0, y0, z0 = initial_point

                lyap = lyapunov_exponents(
                    x0=x0,
                    y0=y0,
                    z0=z0,
                    M1=M1,
                    M2=M2,
                    B=B,
                    n_iter=n_iter,
                    n_skip=n_skip
                )

            regime_map[i, j] = classify_regime(lyap)

            if lyap is not None:
                lambda1_map[i, j] = lyap[0]
                lambda2_map[i, j] = lyap[1]
                lambda3_map[i, j] = lyap[2]

            counter += 1

            if counter % 200 == 0:
                print(f"Calculated {counter} of {total}")

    filename = f"lyapunov_map_B_{str(B).replace('.', '_')}.npz"

    np.savez(
        filename,
        M1_values=M1_values,
        M2_values=M2_values,
        regime_map=regime_map,
        lambda1_map=lambda1_map,
        lambda2_map=lambda2_map,
        lambda3_map=lambda3_map,
        B=B,
        grid_size=grid_size,
        n_iter=n_iter,
        n_skip=n_skip,
        eps=eps
    )

    print(f"Saved data to {filename}")

    return M1_values, M2_values, regime_map, lambda1_map, lambda2_map, lambda3_map


def plot_map(values, M1_values, M2_values, B, title, colorbar_label, filename, vmin=None, vmax=None):
    plt.figure(figsize=(9, 7))

    plt.imshow(
        values,
        origin="lower",
        extent=[
            M1_values[0],
            M1_values[-1],
            M2_values[0],
            M2_values[-1]
        ],
        aspect="auto",
        vmin=vmin,
        vmax=vmax
    )

    plt.colorbar(label=colorbar_label)

    plt.scatter([0], [0.85], s=60, marker="o", label="Figure 1(a)")
    plt.scatter([0], [0.815], s=60, marker="x", label="Figure 1(b)")

    plt.title(f"{title}, B = {B}")
    plt.xlabel("M1")
    plt.ylabel("M2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

    print(f"Saved figure to {filename}")


def run_experiment(B):
    M1_values, M2_values, regime_map, lambda1_map, lambda2_map, lambda3_map = build_lyapunov_map(
        B=B,
        grid_size=100,
        n_iter=6000,
        n_skip=1200,
        m1_min=-0.08,
        m1_max=0.08,
        m2_min=0.75,
        m2_max=0.95,
        eps=1e-6
    )

    b_label = str(B).replace(".", "_")

    plot_map(
        values=lambda1_map,
        M1_values=M1_values,
        M2_values=M2_values,
        B=B,
        title="Lambda_1 map for 3D Henon map",
        colorbar_label="Lambda_1",
        filename=f"lambda1_map_B_{b_label}.png"
    )

    plot_map(
        values=lambda2_map,
        M1_values=M1_values,
        M2_values=M2_values,
        B=B,
        title="Lambda_2 map for 3D Henon map",
        colorbar_label="Lambda_2",
        filename=f"lambda2_map_B_{b_label}.png",
        vmin=-0.005,
        vmax=0.005
    )

    plot_map(
        values=regime_map,
        M1_values=M1_values,
        M2_values=M2_values,
        B=B,
        title="Regime map for 3D Henon map",
        colorbar_label="Regime code",
        filename=f"regime_map_B_{b_label}.png"
    )


if __name__ == "__main__":
    for B_value in [0.7, 0.6, 0.5]:
        run_experiment(B_value)