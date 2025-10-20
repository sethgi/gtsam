#include <gtsam/slam/EssentialMatrixTernaryConstraint.h>

#include <ostream>


namespace gtsam {

void EssentialMatrixTernaryConstraint::print(const std::string& s,
    const KeyFormatter& keyFormatter) const {
  std::cout << s << "EssentialMatrixTernaryConstraint(";
  Base::print("", keyFormatter);
}

bool EssentialMatrixTernaryConstraint::equals(const NonlinearFactor& expected, double tol) const {
  const This* e = dynamic_cast<const This*>(&expected);
  if (e == nullptr) return false;
  return Base::equals(*e, tol);
}

Vector EssentialMatrixTernaryConstraint::evaluateError(const Pose3& p_left,
                                            const Pose3& p_right,
                                            const EssentialMatrix& ess_mat,
                                            OptionalMatrixType Hp_left,
                                            OptionalMatrixType Hp_right,
                                            OptionalMatrixType Hess_mat) const
{
  // 1. Relative pose between left and right
  Pose3 P_left_right = p_left.between(p_right, Hp_left, Hp_right);

  // 2. Convert relative pose to predicted essential matrix
  Matrix D_E_Plr;  // 5x6 Jacobian of EssentialMatrix w.r.t. relative pose
  EssentialMatrix E_pred;
  const bool computeJacobians = (Hp_left || Hp_right || Hess_mat);
  if (computeJacobians) {
    E_pred = EssentialMatrix::FromPose3(P_left_right, D_E_Plr);
  } else {
    E_pred = EssentialMatrix::FromPose3(P_left_right, OptionalNone);
  }

  // 3. Propagate derivatives back to poses
  if (Hp_left) {
    const Matrix& D_Plr_pleft = *Hp_left;    // 6x6
    *Hp_left = D_E_Plr * D_Plr_pleft;        // (5x6)*(6x6) = 5x6
  }
  if (Hp_right) {
    const Matrix& D_Plr_pright = *Hp_right;  // 6x6
    *Hp_right = D_E_Plr * D_Plr_pright;      // 5x6
  }

  // 4. Compute 5D residual on manifold between estimated and predicted E
  // Note: following convention in e.g. EssentialMatrixConstraint and BetweenFactor, 
  // we assume dlog(err)/derr = I
  Vector5 err = ess_mat.localCoordinates(E_pred);
  if (Hess_mat) {
    // derivative wrt EssentialMatrix variable itself: -I (approximation)
    *Hess_mat = -Matrix::Identity(5,5);
  }

  return err;
}


}  // namespace gtsam