import numpy as np
import matplotlib.pyplot as plt


def load_map_data(filename):
    data = np.load(filename)

    return {
        "M1_values": data["M1_values"],
        "M2_values": data["M2_values"],
        "lambda1_map": data["lambda1_map"],
        "lambda2_map": data["lambda2_map"],
        "regime_map": data["regime_map"],
        "B": float(data["B"])
    }


def L_plus(M2, B):
    return -0.25 * (M2 + B - 1) ** 2


def L_minus(M2, B):
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


def prepare_curves(B, M1_values, M2_values):
    M1_min, M1_max = M1_values[0], M1_values[-1]
    M2_min, M2_max = M2_values[0], M2_values[-1]

    M2_dense = np.linspace(M2_min, M2_max, 3000)

    M1_Lplus = L_plus(M2_dense, B)
    M1_Lplus, M2_Lplus = clip_curve(
        M1_Lplus,
        M2_dense,
        M1_min,
        M1_max,
        M2_min,
        M2_max
    )

    M2_L1 = M2_dense[M2_dense <= 1]
    M1_L1 = L_minus(M2_L1, B)
    M1_L1, M2_L1 = clip_curve(
        M1_L1,
        M2_L1,
        M1_min,
        M1_max,
        M2_min,
        M2_max
    )

    M1_Lphi, M2_Lphi = L_phi(M2_dense, B)

    if len(M1_Lphi) > 0:
        M1_Lphi, M2_Lphi = clip_curve(
            M1_Lphi,
            M2_Lphi,
            M1_min,
            M1_max,
            M2_min,
            M2_max
        )

    return {
        "L_plus": (M1_Lplus, M2_Lplus),
        "L1_minus": (M1_L1, M2_L1),
        "L_phi": (M1_Lphi, M2_Lphi)
    }


def add_curves(curves):
    M1_Lplus, M2_Lplus = curves["L_plus"]
    M1_L1, M2_L1 = curves["L1_minus"]
    M1_Lphi, M2_Lphi = curves["L_phi"]

    if len(M1_Lplus) > 0:
        plt.plot(M1_Lplus, M2_Lplus, linewidth=2, label="L+")

    if len(M1_L1) > 0:
        plt.plot(M1_L1, M2_L1, linewidth=2, label="L1-")

    if len(M1_Lphi) > 0:
        plt.plot(M1_Lphi, M2_Lphi, linewidth=2, label="Lphi")


def plot_lambda1_with_curves(data, curves, output_filename):
    M1_values = data["M1_values"]
    M2_values = data["M2_values"]
    lambda1_map = data["lambda1_map"]
    B = data["B"]

    plt.figure(figsize=(10, 8))

    plt.imshow(
        lambda1_map,
        origin="lower",
        extent=[
            M1_values[0],
            M1_values[-1],
            M2_values[0],
            M2_values[-1]
        ],
        aspect="auto"
    )

    plt.colorbar(label="Lambda_1")

    add_curves(curves)

    plt.scatter([0], [0.85], s=70, marker="o", label="Figure 1(a)")
    plt.scatter([0], [0.815], s=70, marker="x", label="Figure 1(b)")

    plt.title(f"Lambda_1 map with bifurcation curves, B = {B}")
    plt.xlabel("M1")
    plt.ylabel("M2")

    plt.xlim(M1_values[0], M1_values[-1])
    plt.ylim(M2_values[0], M2_values[-1])

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.show()

    print(f"Saved figure to {output_filename}")


def run_for_B(B):
    b_label = str(B).replace(".", "_")

    input_filename = f"lyapunov_map_B_{b_label}.npz"
    output_filename = f"lambda1_map_B_{b_label}_with_bifurcation_curves.png"

    data = load_map_data(input_filename)

    curves = prepare_curves(
        B=data["B"],
        M1_values=data["M1_values"],
        M2_values=data["M2_values"]
    )

    plot_lambda1_with_curves(data, curves, output_filename)


if __name__ == "__main__":
    for B_value in [0.7, 0.6, 0.5]:
        run_for_B(B_value)