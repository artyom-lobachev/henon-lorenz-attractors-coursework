import numpy as np


def henon3d_step(x, y, z, M1, M2, B):
    x_next = y
    y_next = z
    z_next = M1 + B * x + M2 * y - z ** 2
    return x_next, y_next, z_next


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


def jacobian_matrix(x, y, z, M1, M2, B):
    return np.array([
        [0, 1, 0],
        [0, 0, 1],
        [B, M2, -2 * z]
    ])


def lyapunov_exponents(
    x0, y0, z0,
    M1, M2, B,
    n_iter=100000,
    n_skip=1000
):
    x = x0
    y = y0
    z = z0

    for _ in range(n_skip):
        x, y, z = henon3d_step(x, y, z, M1, M2, B)

    Q = np.eye(3)
    lyap_sum = np.zeros(3)

    for _ in range(n_iter):
        J = jacobian_matrix(x, y, z, M1, M2, B)
        Z = J @ Q

        Q, R = np.linalg.qr(Z)

        stretch = np.abs(np.diag(R))
        stretch = np.maximum(stretch, 1e-300)

        lyap_sum += np.log(stretch)

        x, y, z = henon3d_step(x, y, z, M1, M2, B)

    return lyap_sum / n_iter


def print_fixed_points_and_multipliers(M1, M2, B):
    points = fixed_points(M1, M2, B)

    if not points:
        print("Неподвижных точек нет")
        return

    for i, point in enumerate(points, start=1):
        x, y, z = point

        J = jacobian_matrix(x, y, z, M1, M2, B)
        eigenvalues = np.linalg.eigvals(J)

        print(f"P{i} = ({x:.6f}, {y:.6f}, {z:.6f})")
        print("Матрица Якоби:")
        print(J)
        print("Мультипликаторы:")
        print(eigenvalues)
        print("Модули мультипликаторов:")
        print(np.abs(eigenvalues))
        print()


def analyze_case(M1, M2, B, x0=0.1, y0=0.1, z0=0.1):
    print("=" * 70)
    print(f"Параметры: M1 = {M1}, M2 = {M2}, B = {B}")
    print("=" * 70)

    print("\nНеподвижные точки и мультипликаторы:")
    print_fixed_points_and_multipliers(M1, M2, B)

    lyap = lyapunov_exponents(
        x0=x0,
        y0=y0,
        z0=z0,
        M1=M1,
        M2=M2,
        B=B
    )

    print("Показатели Ляпунова:")
    print(f"Lambda_1 = {lyap[0]:.10f}")
    print(f"Lambda_2 = {lyap[1]:.10f}")
    print(f"Lambda_3 = {lyap[2]:.10f}")

    print("\nПроверка суммы:")
    print(f"Lambda_1 + Lambda_2 + Lambda_3 = {np.sum(lyap):.10f}")
    print(f"ln(|B|)                       = {np.log(abs(B)):.10f}")
    print()


if __name__ == "__main__":
    analyze_case(M1=0, M2=0.85, B=0.7)
    analyze_case(M1=0, M2=0.815, B=0.7)
