
#include <gtsam/slam/EssentialMatrixTernaryConstraint.h>
#include <gtsam/nonlinear/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/EssentialMatrix.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/base/TestableAssertions.h>

#include <CppUnitLite/TestHarness.h>

using namespace std;
using namespace gtsam;
using namespace std::placeholders;

/* ************************************************************************* */
TEST(EssentialMatrixTernaryConstraint, test) {
  // Create a factor
  Key poseKey1(1);
  Key poseKey2(2);
  Key essKey(3);

  Rot3 trueRotation = Rot3::RzRyRx(0.15, 0.15, -0.20);
  Point3 trueTranslation(+0.5, -1.0, +1.0);
  Unit3 trueDirection(trueTranslation);
  EssentialMatrix trueE(trueRotation, trueDirection);

  SharedNoiseModel model = noiseModel::Isotropic::Sigma(5, 0.25);
  EssentialMatrixTernaryConstraint factor(poseKey1, poseKey2, essKey, model);

  // Create a linearization point at the zero-error point
  Pose3 pose1(Rot3::RzRyRx(0.00, -0.15, 0.30), Point3(-4.0, 7.0, -10.0));
  Pose3 pose2(
      Rot3::RzRyRx(0.179693265735950, 0.002945368776519, 0.102274823253840),
      Point3(-3.37493895, 6.14660244, -8.93650986));
  EssentialMatrix ess_est = EssentialMatrix::FromPose3(pose1.between(pose2));

  // Expect zero error at consistent configuration
  Vector expected = Z_5x1;
  Vector actual = factor.evaluateError(pose1, pose2, ess_est);
  CHECK(assert_equal(expected, actual, 1e-8));

  // --- Numerical derivatives for validation ---

  // d(error)/d(pose1)
  Matrix expectedH1 = numericalDerivative11<Vector5, Pose3>(
      [&](const Pose3& p1) { return factor.evaluateError(p1, pose2, ess_est); },
      pose1);

  // d(error)/d(pose2)
  Matrix expectedH2 = numericalDerivative11<Vector5, Pose3>(
      [&](const Pose3& p2) { return factor.evaluateError(pose1, p2, ess_est); },
      pose2);

  // d(error)/d(EssentialMatrix)
  Matrix expectedH3 = numericalDerivative11<Vector5, EssentialMatrix>(
      [&](const EssentialMatrix& E) { return factor.evaluateError(pose1, pose2, E); },
      ess_est);

  // Use the factor to calculate the analytic Jacobians
  Matrix actualH1, actualH2, actualH3;
  factor.evaluateError(pose1, pose2, ess_est, actualH1, actualH2, actualH3);

  // --- Debug prints ---
  cout << std::endl << "=== EssentialMatrixTernaryConstraint Debug Output ===" << std::endl;
  cout << "Error vector:\n" << actual << std::endl;

  cout << "\nExpected H1 (numerical dErr/dPose1):\n" << expectedH1 << std::endl;
  cout << "Actual   H1 (analytic dErr/dPose1):\n" << actualH1 << std::endl;
  cout << "Diff H1:\n" << (actualH1 - expectedH1) << std::endl;

  cout << "\nExpected H2 (numerical dErr/dPose2):\n" << expectedH2 << std::endl;
  cout << "Actual   H2 (analytic dErr/dPose2):\n" << actualH2 << std::endl;
  cout << "Diff H2:\n" << (actualH2 - expectedH2) << std::endl;

  cout << "\nExpected H3 (numerical dErr/dEssentialMatrix):\n" << expectedH3 << std::endl;
  cout << "Actual   H3 (analytic dErr/dEssentialMatrix):\n" << actualH3 << std::endl;
  cout << "Diff H3:\n" << (actualH3 - expectedH3) << std::endl;
  cout << "======================================================" << std::endl;

  // Verify Jacobians numerically
  CHECK(assert_equal(expectedH1, actualH1, 1e-5));
  CHECK(assert_equal(expectedH2, actualH2, 1e-5));
  CHECK(assert_equal(expectedH3, actualH3, 1e-5));
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
