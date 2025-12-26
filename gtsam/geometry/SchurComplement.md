# Schur Complement in GTSAM's CameraSet

This document explains the `SchurComplement` function in `CameraSet.h` and how it implements the mathematical marginalization of a 3D point from a camera projection factor.

---

## Background: Nonlinear Least Squares and Linearization

### The Nonlinear Problem

We want to minimize a sum of squared residuals:
$$\min_{x} \sum_i \| r_i(x) \|^2 = \min_x \| r(x) \|^2$$

where $r(x)$ is the stacked residual vector and $x$ are the variables (camera poses, calibrations, 3D points, etc.).

### Why Linearize?

This is a nonlinear optimization problem — we can't solve it directly. Instead, we use iterative methods (Gauss-Newton, Levenberg-Marquardt) that:

1. **Linearize** around the current estimate $x_0$
2. **Solve** for an update $\delta x$
3. **Apply** the update: $x \leftarrow x_0 + \delta x$ (or $x \leftarrow x_0 \oplus \delta x$ for manifolds like $SE(3)$)
4. **Repeat** until convergence

### First-Order Taylor Expansion

The residual at a nearby point $x_0 + \delta x$ is approximately:
$$r(x_0 + \delta x) \approx r(x_0) + \underbrace{\frac{\partial r}{\partial x}\bigg|_{x_0}}_{J} \delta x$$

where $J$ is the **Jacobian** of the residual evaluated at $x_0$.

### The Linearized Least-Squares Problem

Substituting the linearization into the cost function:
$$\| r(x_0 + \delta x) \|^2 \approx \| r(x_0) + J \delta x \|^2$$

Let $\mathbf{e}_0 = r(x_0)$ be the current residual. We want to minimize:
$$\min_{\delta x} \| J \delta x + \mathbf{e}_0 \|^2$$

Or equivalently, defining $\mathbf{b} = -\mathbf{e}_0$ (GTSAM's sign convention):
$$\min_{\delta x} \| J \delta x - \mathbf{b} \|^2$$

---

## Deriving the Normal Equations

### Expanding the Squared Norm

$$\| J \delta x - \mathbf{b} \|^2 = (J \delta x - \mathbf{b})^T (J \delta x - \mathbf{b})$$

Expanding:
$$= \delta x^T J^T J \delta x - 2 \mathbf{b}^T J \delta x + \mathbf{b}^T \mathbf{b}$$

This is a **quadratic function** in $\delta x$:
$$f(\delta x) = \frac{1}{2} \delta x^T H \delta x - g^T \delta x + c$$

where:
- $H = J^T J$ is the **Hessian** (or Gauss-Newton approximation to it)
- $g = J^T \mathbf{b}$ is the **gradient** (also called information vector)
- $c = \frac{1}{2}\mathbf{b}^T \mathbf{b}$ is a **constant** (doesn't affect the optimal $\delta x$)

### Finding the Minimum

To minimize, take the derivative and set to zero:
$$\frac{\partial f}{\partial \delta x} = H \delta x - g = 0$$

This gives the **normal equations**:
$$H \delta x = g$$
$$J^T J \delta x = J^T \mathbf{b}$$

### Why "Normal" Equations?

The name comes from the geometric interpretation: at the optimum, the residual $r = J\delta x - \mathbf{b}$ is orthogonal (normal) to the column space of $J$:
$$J^T (J \delta x - \mathbf{b}) = 0$$

---

## The Camera-Point Problem Setup

We have $m$ cameras observing a single 3D point $\mathbf{p}$. The variables are:
- Camera parameters $c$ (poses, calibrations, etc.)
- 3D point $\mathbf{p}$

The linearized least-squares problem is:
$$\min_{\delta c, \delta \mathbf{p}} \left\| F \delta c + E \delta \mathbf{p} - \mathbf{b} \right\|^2$$

where:
- $\delta c$ = stacked camera variable updates
- $\delta \mathbf{p}$ = 3D point update
- $F = \frac{\partial r}{\partial c}$ = stacked Jacobians w.r.t. cameras
- $E = \frac{\partial r}{\partial \mathbf{p}}$ = stacked Jacobians w.r.t. point
- $\mathbf{b}$ = negative whitened error at linearization point

The combined Jacobian is $J = [F \mid E]$ and the combined update is $\delta x = [\delta c; \delta \mathbf{p}]$.

## The Full Normal Equations

The normal equations for this least-squares problem are:

$$\begin{bmatrix} F^T F & F^T E \\ E^T F & E^T E \end{bmatrix} \begin{bmatrix} \delta c \\ \delta \mathbf{p} \end{bmatrix} = \begin{bmatrix} F^T \mathbf{b} \\ E^T \mathbf{b} \end{bmatrix}$$

Or in block notation:
$$\begin{bmatrix} H_{cc} & H_{cp} \\ H_{pc} & H_{pp} \end{bmatrix} \begin{bmatrix} \delta c \\ \delta \mathbf{p} \end{bmatrix} = \begin{bmatrix} \mathbf{g}_c \\ \mathbf{g}_p \end{bmatrix}$$

## The Schur Complement

To eliminate $\delta \mathbf{p}$, we use the Schur complement. From the second block row:

$$\delta \mathbf{p} = H_{pp}^{-1} (\mathbf{g}_p - H_{pc} \delta c)$$

Substituting into the first block row:

$$H_{cc} \delta c + H_{cp} H_{pp}^{-1} (\mathbf{g}_p - H_{pc} \delta c) = \mathbf{g}_c$$

Rearranging:

$$(H_{cc} - H_{cp} H_{pp}^{-1} H_{pc}) \delta c = \mathbf{g}_c - H_{cp} H_{pp}^{-1} \mathbf{g}_p$$

The **reduced Hessian** (Schur complement) is:
$$\tilde{H} = H_{cc} - H_{cp} H_{pp}^{-1} H_{pc} = F^T F - F^T E (E^T E)^{-1} E^T F$$

The **reduced information vector** is:
$$\tilde{\mathbf{g}} = \mathbf{g}_c - H_{cp} H_{pp}^{-1} \mathbf{g}_p = F^T \mathbf{b} - F^T E (E^T E)^{-1} E^T \mathbf{b}$$

## The Code: Key Variables

```cpp
template <int N, int ND>
static SymmetricBlockMatrix SchurComplement(
    const std::vector<Eigen::Matrix<double, ZDim, ND>, ...>& Fs,  // F blocks
    const Matrix& E,                    // Point Jacobian (2m × 3)
    const Eigen::Matrix<double, N, N>& P,  // (E^T E)^{-1}
    const Vector& b)                    // Whitened error (2m × 1)
```

| Parameter | Math | Dimensions | Description |
|-----------|------|------------|-------------|
| `Fs` | $F_i$ | Vector of (2 × ND) | Per-camera Jacobians w.r.t. camera variables |
| `E` | $E$ | (2m × N) | Stacked point Jacobians, N=3 typically |
| `P` | $(E^T E)^{-1}$ | (N × N) | Precomputed point covariance |
| `b` | $\mathbf{b}$ | (2m × 1) | Whitened error vector |
| `ND` | — | — | Camera variable dimension (e.g., 6 for pose) |
| `ZDim` | — | 2 | Measurement dimension |
| `N` | — | 3 | Point dimension |

## Block Structure

The Jacobian $F$ is block-diagonal (each camera only affects its own measurement):

$$F = \begin{bmatrix} F_1 & 0 & \cdots & 0 \\ 0 & F_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & F_m \end{bmatrix}$$

where each $F_i \in \mathbb{R}^{2 \times ND}$.

This block-diagonal structure makes $F^T F$ block-diagonal, and allows efficient computation of the Schur complement block-by-block.

## Line-by-Line Code Explanation

### Setup
```cpp
size_t m = Fs.size();  // Number of cameras/measurements

// Create augmented Hessian: (ND*m + 1) × (ND*m + 1)
// Block structure: [H_11, H_12, ..., H_1m, g_1]
//                  [H_21, H_22, ..., H_2m, g_2]
//                  [  ...                     ]
//                  [g_1^T, g_2^T, ..., g_m^T, c]
size_t M1 = ND * m + 1;
std::vector<DenseIndex> dims(m + 1);
std::fill(dims.begin(), dims.end() - 1, ND);  // m blocks of size ND
dims.back() = 1;  // Final block for constant term
SymmetricBlockMatrix augmentedHessian(dims, Matrix::Zero(M1, M1));
```

### Main Loop: Diagonal and Off-Diagonal Blocks

```cpp
for (size_t i = 0; i < m; i++) {
    const Eigen::Matrix<double, ZDim, ND>& Fi = Fs[i];
    const auto FiT = Fi.transpose();  // (ND × 2)

    // Ei_P = E_i * P = E_i * (E^T E)^{-1}
    // This is (2 × 3) * (3 × 3) = (2 × 3)
    const Eigen::Matrix<double, ZDim, N> Ei_P =
        E.block(ZDim * i, 0, ZDim, N) * P;
```

**Computing $E_i P$:** This intermediate result $E_i (E^T E)^{-1}$ is reused multiple times.

### Information Vector Block (g_i)

```cpp
    // g_i = F_i^T * b_i - F_i^T * E_i * P * E^T * b
    augmentedHessian.setOffDiagonalBlock(
        i, m,                           // Block position (i, last column)
        FiT * b.segment<ZDim>(ZDim * i)  // F_i^T * b_i
            - FiT * (Ei_P * (E.transpose() * b)));  // - F_i^T * E_i * P * (E^T * b)
```

**Math:** The reduced information vector for camera $i$ is:
$$\tilde{\mathbf{g}}_i = F_i^T \mathbf{b}_i - F_i^T E_i P (E^T \mathbf{b})$$

Note: $E^T \mathbf{b}$ is the full $(3 \times 1)$ vector, not just the contribution from measurement $i$.

### Diagonal Hessian Block (H_ii)

```cpp
    // H_ii = F_i^T * F_i - F_i^T * E_i * P * E_i^T * F_i
    augmentedHessian.setDiagonalBlock(
        i,
        FiT * (Fi - Ei_P * E.block(ZDim * i, 0, ZDim, N).transpose() * Fi));
```

**Math:** The diagonal block of the reduced Hessian is:
$$\tilde{H}_{ii} = F_i^T F_i - F_i^T E_i P E_i^T F_i$$

Rewritten using $E_i P$:
$$\tilde{H}_{ii} = F_i^T (F_i - E_i P E_i^T F_i) = F_i^T (I - E_i P E_i^T) F_i$$

### Off-Diagonal Hessian Blocks (H_ij, i < j)

```cpp
    for (size_t j = i + 1; j < m; j++) {
        const Eigen::Matrix<double, ZDim, ND>& Fj = Fs[j];

        // H_ij = -F_i^T * E_i * P * E_j^T * F_j
        augmentedHessian.setOffDiagonalBlock(
            i, j,
            -FiT * (Ei_P * E.block(ZDim * j, 0, ZDim, N).transpose() * Fj));
    }
```

**Math:** The off-diagonal blocks arise from the Schur complement term:
$$\tilde{H}_{ij} = 0 - F_i^T E_i P E_j^T F_j = -F_i^T E_i P E_j^T F_j$$

The first term is zero because $F^T F$ is block-diagonal (camera $i$ doesn't directly affect camera $j$'s measurement). The second term couples all cameras through the shared 3D point.

### Constant Term

```cpp
augmentedHessian.diagonalBlock(m)(0, 0) += b.squaredNorm();
```

**Math:** The constant term in the quadratic cost is:
$$c = \mathbf{b}^T \mathbf{b} = \|\mathbf{b}\|^2$$

## Derivation: Why This Formula?

Starting from the Schur complement:
$$\tilde{H} = F^T F - F^T E (E^T E)^{-1} E^T F$$

Since $F$ is block-diagonal, $F^T F$ is also block-diagonal with blocks $F_i^T F_i$.

For the second term, let's compute the $(i, j)$ block:
$$(F^T E P E^T F)_{ij} = F_i^T E_i P E_j^T F_j$$

where we use the block structure:
- $(F^T)_i = \begin{bmatrix} 0 & \cdots & F_i^T & \cdots & 0 \end{bmatrix}$
- $(E)_i = E_i$ (the $i$-th 2×3 block of $E$)

Therefore:
$$\tilde{H}_{ij} = \underbrace{(F^T F)_{ij}}_{\text{= } F_i^T F_i \text{ if } i=j, \text{ else } 0} - F_i^T E_i P E_j^T F_j$$

## Geometric Interpretation

The Schur complement can be interpreted as:

1. **$F_i^T F_i$**: Direct information about camera $i$ from its measurement
2. **$-F_i^T E_i P E_j^T F_j$**: Information "lost" because the point is uncertain

The term $E P E^T = E (E^T E)^{-1} E^T$ is a projection matrix onto the column space of $E$. The Schur complement removes the component of $F$ that lies in this subspace, leaving only the information that constrains cameras independently of point position.

## Example: 2 Cameras

With 2 cameras observing 1 point, the augmented Hessian is:

```
[  H_11    H_12  | g_1  ]
[  H_21    H_22  | g_2  ]
[---------+------+------]
[  g_1^T   g_2^T |  c   ]
```

where:
- $H_{11} = F_1^T F_1 - F_1^T E_1 P E_1^T F_1$ (ND × ND)
- $H_{22} = F_2^T F_2 - F_2^T E_2 P E_2^T F_2$ (ND × ND)
- $H_{12} = -F_1^T E_1 P E_2^T F_2$ (ND × ND)
- $\mathbf{g}_1 = F_1^T \mathbf{b}_1 - F_1^T E_1 P (E^T \mathbf{b})$ (ND × 1)
- $\mathbf{g}_2 = F_2^T \mathbf{b}_2 - F_2^T E_2 P (E^T \mathbf{b})$ (ND × 1)
- $c = \|\mathbf{b}\|^2$ (scalar)

---

## SchurComplementAndRearrangeBlocks

### The Problem: Repeated Keys

In some scenarios, multiple measurements may share the same variable. For example:
- In a **camera rig**, the same body pose is observed by multiple cameras
- In `SmartProjectionRigFactor`, a single landmark might be seen by 3 cameras, but only 2 unique body poses

The basic `SchurComplement` function creates one Hessian block per measurement. But if measurements 1 and 3 both come from pose $T_1$, we need to **combine** their contributions into a single block.

### Example: 3 Measurements, 2 Unique Poses

Suppose we have:
- Measurement 1 from pose $T_1$
- Measurement 2 from pose $T_2$
- Measurement 3 from pose $T_1$ (same as measurement 1)

The `jacobianKeys` (one per measurement) would be: `[T1, T2, T1]`
The `hessianKeys` (unique keys) would be: `[T1, T2]`

After basic Schur complement, we'd have a 3×3 block Hessian:
```
[  H_11    H_12    H_13  | g_1  ]     (measurement 1)
[  H_21    H_22    H_23  | g_2  ]     (measurement 2)
[  H_31    H_32    H_33  | g_3  ]     (measurement 3)
[  g_1^T   g_2^T   g_3^T |  c   ]
```

But we want a 2×2 block Hessian (one block per unique pose):
```
[  H_T1T1    H_T1T2  | g_T1  ]
[  H_T2T1    H_T2T2  | g_T2  ]
[  g_T1^T    g_T2^T  |  c    ]
```

### The Rearrangement

The function `SchurComplementAndRearrangeBlocks` does this by:

1. First computing the standard Schur complement with `nrNonuniqueKeys` blocks
2. Building a map from each key to its slot in the output Hessian
3. Iterating over the non-unique Hessian and **accumulating** contributions to the unique Hessian

### The Key Insight: Addition

When two measurements share the same key, their contributions **add**:

$$H_{T_1 T_1} = H_{11} + H_{33} + H_{13} + H_{31}$$

The diagonal contributions ($H_{11}$, $H_{33}$) add directly. The cross-terms ($H_{13}$, $H_{31}$) also contribute to the diagonal because both indices map to $T_1$.

Similarly for the information vector:
$$\mathbf{g}_{T_1} = \mathbf{g}_1 + \mathbf{g}_3$$

### Template Parameters

```cpp
template <int N, int ND, int NDD>
static SymmetricBlockMatrix SchurComplementAndRearrangeBlocks(
    const std::vector<Eigen::Matrix<double, ZDim, ND>, ...>& Fs,
    const Matrix& E,
    const Eigen::Matrix<double, N, N>& P,
    const Vector& b,
    const KeyVector& jacobianKeys,   // Keys per measurement (may repeat)
    const KeyVector& hessianKeys)    // Unique keys for output
```

| Parameter | Description |
|-----------|-------------|
| `N` | Point dimension (typically 3) |
| `ND` | Jacobian block dimension (columns of each $F_i$) |
| `NDD` | Hessian block dimension (size of output blocks) |
| `jacobianKeys` | One key per measurement, may have duplicates |
| `hessianKeys` | Unique keys, defines output Hessian structure |

**Note:** In `SmartProjectionRigFactor`, `ND = NDD = 6` because pose Jacobians are 2×6 and pose Hessian blocks are 6×6.

### Code Walkthrough

```cpp
// Step 1: Compute standard Schur complement (one block per measurement)
SymmetricBlockMatrix augmentedHessian = SchurComplement<N, ND>(Fs, E, P, b);
```

If all keys are unique, we're done — just repackage with the right block dimensions:
```cpp
if (nrUniqueKeys == nrNonuniqueKeys) {
    augmentedHessianUniqueKeys = SymmetricBlockMatrix(
        dims, Matrix(augmentedHessian.selfadjointView()));
}
```

Otherwise, we need to rearrange:
```cpp
else {
    // Build map: key -> slot in output Hessian
    std::map<Key, size_t> keyToSlotMap;
    for (size_t k = 0; k < nrUniqueKeys; k++) {
        keyToSlotMap[hessianKeys[k]] = k;
    }

    // Initialize output to zero
    augmentedHessianUniqueKeys = SymmetricBlockMatrix(
        dims, Matrix::Zero(NDD * nrUniqueKeys + 1, NDD * nrUniqueKeys + 1));

    // Accumulate contributions
    for (size_t i = 0; i < nrNonuniqueKeys; i++) {
        Key key_i = jacobianKeys.at(i);

        // Accumulate information vector
        augmentedHessianUniqueKeys.updateOffDiagonalBlock(
            keyToSlotMap[key_i], nrUniqueKeys,
            augmentedHessian.aboveDiagonalBlock(i, nrNonuniqueKeys));

        for (size_t j = i; j < nrNonuniqueKeys; j++) {
            Key key_j = jacobianKeys.at(j);

            if (i == j) {
                // Diagonal: always accumulate to the mapped slot
                augmentedHessianUniqueKeys.updateDiagonalBlock(
                    keyToSlotMap[key_i], augmentedHessian.diagonalBlock(i));
            } else {
                if (keyToSlotMap[key_i] != keyToSlotMap[key_j]) {
                    // Off-diagonal in output: accumulate as off-diagonal
                    augmentedHessianUniqueKeys.updateOffDiagonalBlock(
                        keyToSlotMap[key_i], keyToSlotMap[key_j],
                        augmentedHessian.aboveDiagonalBlock(i, j));
                } else {
                    // Both map to same key: add to diagonal (with transpose)
                    augmentedHessianUniqueKeys.updateDiagonalBlock(
                        keyToSlotMap[key_i],
                        augmentedHessian.aboveDiagonalBlock(i, j) +
                            augmentedHessian.aboveDiagonalBlock(i, j).transpose());
                }
            }
        }
    }

    // Copy constant term
    augmentedHessianUniqueKeys.updateDiagonalBlock(
        nrUniqueKeys, augmentedHessian.diagonalBlock(nrNonuniqueKeys));
}
```

### The Transpose Trick

When an off-diagonal block $H_{ij}$ (with $i < j$) maps to a diagonal position (because $key_i = key_j$), we need to add both $H_{ij}$ and $H_{ij}^T$ to get the full symmetric contribution:

$$H_{T_1 T_1} \mathrel{+}= H_{13} + H_{13}^T$$

This is because the symmetric matrix stores only the upper triangle, but when $i \neq j$ map to the same output slot, both contributions matter.

### Limitation: Uniform Block Sizes

A key limitation of `SchurComplementAndRearrangeBlocks` is that it assumes **all output blocks have the same dimension** `NDD`. This works for:
- Poses only (all 6×6)
- Calibrations only (all DimK × DimK)

But it **doesn't directly support** mixed variable types where poses are 6×6 and calibrations are 5×5, which is why `SmartProjectionPoseCalFactor` needs a custom implementation.

---

## Connection to SmartProjectionPoseCalFactor

In `SmartProjectionPoseCalFactor`, we extend this by having:
- $F_i$ = Jacobian w.r.t. pose (2 × 6)
- $G_i$ = Jacobian w.r.t. calibration (2 × DimK)

Conceptually, we could form an augmented Jacobian $[F_i | G_i]$ of size (2 × (6 + DimK)) and use the standard Schur complement. However, since poses and calibrations may be shared differently across measurements, we need to:
1. Compute the Schur complement with the full Jacobians
2. Rearrange/accumulate blocks based on which variables are shared

This is analogous to what `SchurComplementAndRearrangeBlocks` does for repeated pose keys, but extended to handle mixed variable types (poses and calibrations) with different dimensions.
