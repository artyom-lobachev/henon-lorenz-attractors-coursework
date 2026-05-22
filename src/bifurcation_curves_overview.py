import numpy as np
import matplotlib.pyplot as plt


def L_plus(M2, B):
    return -0.25 * (M2 + B - 1) ** 2


def L1_minus(M2, B):
    return 0.25 * (1 - M2 + B) * (3 - 3 * M2 - B)


def L_phi(M2, B):
    if abs(B) < 1e-14:
        return np.array([]), np.array([])

    mask = np.abs(M2 + 1) < 2 * abs(B)
    M2_valid = M2[mask]

    if len(M2_valid) == 0:
        return np.array([]), np.array([])

    M1_valid = (
        (1 - B) ** 2 * (1 + (M2_valid + 1) / (2 * B)) ** 2
        - 0.25 * (B + M2_valid - 1) ** 2
    )

    return M1_valid, M2_valid


def clip_curve(M1_curve, M2_curve, M1_min, M1_max, M2_min, M2_max):
    mask = (
        (M1_curve >= M1_min) & (M1_curve <= M1_max)
        & (M2_curve >= M2_min) & (M2_curve <= M2_max)
    )

    return M1_curve[mask], M2_curve[mask]


def plot_overview(B=0.7):
    M1_min, M1_max = -2.0, 3.0
    M2_min, M2_max = -2.5, 1.2

    M2_values = np.linspace(M2_min, M2_max, 5000)

    M1_Lplus = L_plus(M2_values, B)
    M1_Lplus, M2_Lplus = clip_curve(
        M1_Lplus,
        M2_values,
        M1_min,
        M1_max,
        M2_min,
        M2_max
    )

    M2_L1 = M2_values[M2_values <= 1]
    M1_L1 = L1_minus(M2_L1, B)
    M1_L1, M2_L1 = clip_curve(
        M1_L1,
        M2_L1,
        M1_min,
        M1_max,
        M2_min,
        M2_max
    )

    M1_Lphi, M2_Lphi = L_phi(M2_values, B)

    if len(M1_Lphi) > 0:
        M1_Lphi, M2_Lphi = clip_curve(
            M1_Lphi,
            M2_Lphi,
            M1_min,
            M1_max,
            M2_min,
            M2_max
        )

    plt.figure(figsize=(10, 7))

    if len(M1_Lplus) > 0:
        plt.plot(M1_Lplus, M2_Lplus, linewidth=2.5, label="L+")

    if len(M1_L1) > 0:
        plt.plot(M1_L1, M2_L1, linewidth=2.5, label="L1-")

    if len(M1_Lphi) > 0:
        plt.plot(M1_Lphi, M2_Lphi, linewidth=2.5, label="Lphi")

    detail_M1_min, detail_M1_max = -0.08, 0.08
    detail_M2_min, detail_M2_max = 0.75, 0.95

    rect_x = [
        detail_M1_min,
        detail_M1_max,
        detail_M1_max,
        detail_M1_min,
        detail_M1_min
    ]

    rect_y = [
        detail_M2_min,
        detail_M2_min,
        detail_M2_max,
        detail_M2_max,
        detail_M2_min
    ]

    plt.plot(rect_x, rect_y, linewidth=2, linestyle=":", label="detailed map window")

    plt.scatter([0], [0.85], s=70, marker="o", label="Figure 1(a)")
    plt.scatter([0], [0.815], s=70, marker="x", label="Figure 1(b)")

    plt.title(f"Bifurcation curves overview, B = {B}")
    plt.xlabel("M1")
    plt.ylabel("M2")

    plt.xlim(M1_min, M1_max)
    plt.ylim(M2_min, M2_max)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    b_label = str(B).replace(".", "_")
    output_filename = f"bifurcation_curves_overview_B_{b_label}.png"

    plt.savefig(output_filename, dpi=300)
    plt.show()

    print(f"Saved figure to {output_filename}")


if __name__ == "__main__":
    plot_overview(B=0.7)