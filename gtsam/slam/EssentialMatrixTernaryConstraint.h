#pragma once

#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/geometry/EssentialMatrix.h>

namespace gtsam {


class GTSAM_EXPORT EssentialMatrixTernaryConstraint : public NoiseModelFactorN<Pose3, Pose3, EssentialMatrix> {

private:
  typedef EssentialMatrixTernaryConstraint This;
  typedef NoiseModelFactorN<Pose3, Pose3, EssentialMatrix> Base;


public:
  using Base::evaluateError;
  typedef std::shared_ptr<EssentialMatrixTernaryConstraint> shared_ptr;

  EssentialMatrixTernaryConstraint(){}

  EssentialMatrixTernaryConstraint(Key leftPoseKey, Key rightPoseKey, Key essMatKey,
                        const SharedNoiseModel& model) :
    Base(model, leftPoseKey, rightPoseKey, essMatKey) {}

  ~EssentialMatrixTernaryConstraint() override {}

  /// @return a deep copy of this factor
  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return std::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  /** print */
  void print(const std::string& s = "",
      const KeyFormatter& keyFormatter = DefaultKeyFormatter) const override;

  /** equals */
  bool equals(const NonlinearFactor& expected, double tol = 1e-9) const override;

  /** implement functions needed to derive from Factor */

  /** vector of errors */
  Vector evaluateError(const Pose3& p1, const Pose3& p2, const EssentialMatrix& E,
      OptionalMatrixType Hp_left, OptionalMatrixType Hp_right, OptionalMatrixType Hess_mat) const override;

};


} // namespace gtsam