# SmartProjectionPoseCalFactor: Mathematical Derivation

## Overview

This document explains the mathematics behind `SmartProjectionPoseCalFactor`, a GTSAM smart factor that optimizes camera poses and calibrations while marginalizing out 3D landmark points via the Schur complement.

## Problem Setup

We observe a single 3D landmark $\mathbf{p} \in \mathbb{R}^3$ from $m$ cameras. Each camera $i$ has:
- A pose $T_i \in SE(3)$ (6 DOF)
- A calibration $K_i$ (e.g., `Cal3_S2` has 5 DOF: $f_x, f_y, s, u_0, v_0$)
- A 2D measurement $\mathbf{z}_i \in \mathbb{R}^2$

The projection function is:
$$h_i(\mathbf{p}, T_i, K_i) = \pi(K_i, T_i^{-1} \mathbf{p})$$

where $\pi$ projects a 3D point in camera coordinates to 2D pixel coordinates using calibration $K_i$.

## Error Model

The reprojection error for measurement $i$ is:
$$\mathbf{e}_i = h_i(\mathbf{p}, T_i, K_i) - \mathbf{z}_i \in \mathbb{R}^2$$

The total error vector (before whitening) is:
$$\mathbf{e} = \begin{bmatrix} \mathbf{e}_1 \\ \mathbf{e}_2 \\ \vdots \\ \mathbf{e}_m \end{bmatrix} \in \mathbb{R}^{2m}$$

After whitening by the noise model $\Sigma^{-1/2}$:
$$\mathbf{b} = -\Sigma^{-1/2} \mathbf{e}$$

**Note:** The negative sign is because GTSAM uses the convention $\mathbf{b} = \mathbf{z} - h(\mathbf{x})$ for the RHS of the linearized system.

---

## Jacobian Definitions

### Variable in Code: `Fs` (Pose Jacobians)

$F_i$ is the Jacobian of the reprojection error w.r.t. the pose $T_i$:

$$F_i = \frac{\partial \mathbf{e}_i}{\partial T_i} \in \mathbb{R}^{2 \times 6}$$

In code, `Fs` is a vector of $m$ matrices, one per measurement:
```cpp
std::vector<Eigen::Matrix<double, 2, 6>> Fs;  // Fs[i] = F_i
```

After whitening: $F_i \leftarrow \Sigma^{-1/2} F_i$

### Variable in Code: `Gs` (Calibration Jacobians)

$G_i$ is the Jacobian of the reprojection error w.r.t. the calibration $K_i$:

$$G_i = \frac{\partial \mathbf{e}_i}{\partial K_i} \in \mathbb{R}^{2 \times d_K}$$

where $d_K$ is the calibration dimension (e.g., 5 for `Cal3_S2`).

In code, `Gs` is a vector of $m$ matrices:
```cpp
std::vector<Eigen::Matrix<double, 2, DimK>> Gs;  // Gs[i] = G_i
```

After whitening: $G_i \leftarrow \Sigma^{-1/2} G_i$

### Variable in Code: `E` (Point Jacobian)

$E$ is the stacked Jacobian of all reprojection errors w.r.t. the 3D point $\mathbf{p}$:

$$E = \begin{bmatrix} E_1 \\ E_2 \\ \vdots \\ E_m \end{bmatrix} \in \mathbb{R}^{2m \times 3}$$

where each block is:
$$E_i = \frac{\partial \mathbf{e}_i}{\partial \mathbf{p}} \in \mathbb{R}^{2 \times 3}$$

In code:
```cpp
Matrix E;  // (2m × 3)
E.block<2, 3>(2*i, 0);  // = E_i
```

After whitening: $E \leftarrow \Sigma^{-1/2} E$ (applied row-wise)

### Variable in Code: `b` (Whitened Error Vector)

$$\mathbf{b} = -\Sigma^{-1/2} \mathbf{e} \in \mathbb{R}^{2m}$$

This is the negative whitened reprojection error. The sign convention matches GTSAM's linearization where the RHS is $\mathbf{z} - h(\mathbf{x})$.

In code:
```cpp
Vector b;  // (2m × 1)
b.segment<2>(2*i);  // = b_i for measurement i
```

### Variable in Code: `P` (Point Covariance for Schur Complement)

$P$ is the inverse of the point's information matrix, used in the Schur complement:

$$P = (E^T E + \lambda I)^{-1} \in \mathbb{R}^{3 \times 3}$$

where $\lambda$ is an optional damping factor (Levenberg-Marquardt).

In code:
```cpp
Matrix3 P = Cameras::PointCov(E, lambda, diagonalDamping);
```

---

## The Schur Complement

### Why Marginalize the Point?

Smart factors marginalize out the 3D point $\mathbf{p}$ to create a factor only on poses and calibrations. This is more efficient because:
1. Each landmark only appears in one factor
2. The resulting Hessian is denser but smaller

### Full System Before Marginalization

The linearized least-squares problem is:
$$\min_{\delta T, \delta K, \delta \mathbf{p}} \| J \begin{bmatrix} \delta T \\ \delta K \\ \delta \mathbf{p} \end{bmatrix} - \mathbf{b} \|^2$$

where the full Jacobian is:
$$J = \begin{bmatrix} F_1 & G_1 & E_1 \\ F_2 & G_2 & E_2 \\ \vdots & \vdots & \vdots \\ F_m & G_m & E_m \end{bmatrix}$$

The normal equations give:
$$\begin{bmatrix} H_{TT} & H_{TK} & H_{Tp} \\ H_{KT} & H_{KK} & H_{Kp} \\ H_{pT} & H_{pK} & H_{pp} \end{bmatrix} \begin{bmatrix} \delta T \\ \delta K \\ \delta \mathbf{p} \end{bmatrix} = \begin{bmatrix} \mathbf{g}_T \\ \mathbf{g}_K \\ \mathbf{g}_p \end{bmatrix}$$

where:
- $H_{pp} = E^T E$
- $H_{Tp} = F^T E$ (stacked $F_i^T E_i$)
- etc.

### Schur Complement Formula

Eliminating $\delta \mathbf{p}$ via the Schur complement:

$$\tilde{H} = H_{cameras} - H_{cp} \cdot H_{pp}^{-1} \cdot H_{pc}$$
$$\tilde{\mathbf{g}} = \mathbf{g}_{cameras} - H_{cp} \cdot H_{pp}^{-1} \cdot \mathbf{g}_p$$

where "cameras" refers to all pose and calibration variables.

Using $P = H_{pp}^{-1} = (E^T E)^{-1}$:

$$\tilde{H} = J_{cam}^T J_{cam} - J_{cam}^T E \cdot P \cdot E^T J_{cam}$$

---

## Block-by-Block Computation

For each measurement $i$, we compute contributions to the Hessian blocks.

### Notation
- $F_i$: pose Jacobian for measurement $i$ (2×6)
- $G_i$: calibration Jacobian for measurement $i$ (2×$d_K$)
- $E_i$: point Jacobian for measurement $i$ (2×3)
- $\mathbf{b}_i$: whitened error for measurement $i$ (2×1)
- $P$: point covariance (3×3)

### Diagonal Blocks (Same Variable)

**Pose-Pose block** (for pose $j$, accumulated over all measurements from pose $j$):
$$H_{T_j T_j} = \sum_{i: \text{pose}_i = j} \left( F_i^T F_i - F_i^T E_i P E_i^T F_i \right)$$

**Calibration-Calibration block** (for calibration $k$, accumulated over all measurements using calibration $k$):
$$H_{K_k K_k} = \sum_{i: \text{cal}_i = k} \left( G_i^T G_i - G_i^T E_i P E_i^T G_i \right)$$

### Off-Diagonal Blocks (Different Variables)

**Pose-Calibration block** (for pose $j$ and calibration $k$, from measurement $i$ that uses both):
$$H_{T_j K_k} = \sum_{i: \text{pose}_i=j, \text{cal}_i=k} \left( F_i^T G_i - F_i^T E_i P E_i^T G_i \right)$$

**Cross terms between different measurements** (measurements $i$ and $j$ observing same point):

For poses $T_a$ (from measurement $i$) and $T_b$ (from measurement $j$):
$$H_{T_a T_b} \mathrel{+}= -F_i^T E_i P E_j^T F_j$$

Similarly for calibration-calibration and pose-calibration cross terms.

### Information Vector

**Pose information** (for pose $j$):
$$\mathbf{g}_{T_j} = \sum_{i: \text{pose}_i = j} \left( F_i^T \mathbf{b}_i - F_i^T E_i P (E^T \mathbf{b}) \right)$$

**Calibration information** (for calibration $k$):
$$\mathbf{g}_{K_k} = \sum_{i: \text{cal}_i = k} \left( G_i^T \mathbf{b}_i - G_i^T E_i P (E^T \mathbf{b}) \right)$$

Note: $E^T \mathbf{b}$ is precomputed once as `E_T_b` in the code.

---

## Code Variable Mapping

| Math Symbol | Code Variable | Dimensions | Description |
|-------------|---------------|------------|-------------|
| $F_i$ | `Fs[i]` | 2 × 6 | Pose Jacobian for measurement $i$ |
| $G_i$ | `Gs[i]` | 2 × $d_K$ | Calibration Jacobian for measurement $i$ |
| $E$ | `E` | 2m × 3 | Stacked point Jacobians |
| $E_i$ | `E.block<2,3>(2*i, 0)` | 2 × 3 | Point Jacobian for measurement $i$ |
| $\mathbf{b}$ | `b` | 2m × 1 | Whitened error vector |
| $\mathbf{b}_i$ | `b.segment<2>(2*i)` | 2 × 1 | Whitened error for measurement $i$ |
| $P$ | `P` | 3 × 3 | Point covariance $(E^T E)^{-1}$ |
| $E^T \mathbf{b}$ | `E_T_b` | 3 × 1 | Precomputed for efficiency |
| $E_i P$ | `Ei_P` | 2 × 3 | Intermediate: $E_i \cdot P$ |
| $E_i P E_i^T$ | `Ei_P_EiT` | 2 × 2 | Intermediate: $E_i P E_i^T$ |

---

## Key Indices in Code

| Code Variable | Description |
|---------------|-------------|
| `nonUniqueKeys_[i]` | Pose key for measurement $i$ (may repeat) |
| `calibrationKeys_[i]` | Calibration key for measurement $i$ (may repeat) |
| `poseIdx` | Index into Hessian for a pose variable |
| `calIdx` | Index into Hessian for a calibration variable |
| `keyToIndex[key]` | Maps a GTSAM Key to its Hessian block index |

---

## Hessian Structure

The final Hessian has blocks arranged as:
```
[  H_T1T1   H_T1T2  ...  H_T1K1   H_T1K2  ... | g_T1  ]
[  H_T2T1   H_T2T2  ...  H_T2K1   H_T2K2  ... | g_T2  ]
[   ...      ...    ...   ...      ...    ... |  ...  ]
[  H_K1T1   H_K1T2  ...  H_K1K1   H_K1K2  ... | g_K1  ]
[  H_K2T1   H_K2T2  ...  H_K2K1   H_K2K2  ... | g_K2  ]
[   ...      ...    ...   ...      ...    ... |  ...  ]
[  g_T1^T   g_T2^T  ...  g_K1^T   g_K2^T  ... | c     ]
```

where $c = \mathbf{b}^T \mathbf{b}$ is the constant term (sum of squared errors).

---

## Example: 2 Poses, 1 Shared Calibration

Setup:
- Pose $T_1$ with key `x0`
- Pose $T_2$ with key `x1`
- Shared calibration $K$ with key `K0`
- 2 measurements: $(z_1, x0, K0)$ and $(z_2, x1, K0)$

Hessian structure (3 variables: $T_1$, $T_2$, $K$):
```
[  H_T1T1    H_T1T2    H_T1K  | g_T1 ]   (6×6)  (6×6)  (6×5)
[  H_T2T1    H_T2T2    H_T2K  | g_T2 ]   (6×6)  (6×6)  (6×5)
[  H_KT1     H_KT2     H_KK   | g_K  ]   (5×6)  (5×6)  (5×5)
[  g_T1^T    g_T2^T    g_K^T  | c    ]
```

Block computations:
- $H_{T_1 T_1} = F_1^T F_1 - F_1^T E_1 P E_1^T F_1$
- $H_{T_2 T_2} = F_2^T F_2 - F_2^T E_2 P E_2^T F_2$
- $H_{T_1 T_2} = -F_1^T E_1 P E_2^T F_2$ (cross-measurement term)
- $H_{K K} = G_1^T G_1 + G_2^T G_2 - G_1^T E_1 P E_1^T G_1 - G_2^T E_2 P E_2^T G_2 - 2 G_1^T E_1 P E_2^T G_2$
- $H_{T_1 K} = F_1^T G_1 - F_1^T E_1 P E_1^T G_1 - F_1^T E_1 P E_2^T G_2$
- etc.

---

## References

1. Carlone, L., Kira, Z., Beall, C., Indelman, V., Dellaert, F. (2014). "Eliminating conditionally independent sets in factor graphs: a unifying perspective based on smart factors." ICRA.

2. GTSAM documentation on smart factors and Schur complement.
