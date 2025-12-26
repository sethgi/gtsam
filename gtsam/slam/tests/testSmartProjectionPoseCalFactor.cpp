/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file   testSmartProjectionPoseCalFactor.cpp
 * @brief  Unit tests for SmartProjectionPoseCalFactor
 * @author Seth Isaacson
 * @date   December 2025
 */

#include <gtsam/slam/SmartProjectionPoseCalFactor.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/PriorFactor.h>
#include <CppUnitLite/TestHarness.h>

using namespace std;
using namespace gtsam;

// Make the typename shorter
typedef SmartProjectionPoseCalFactor<Cal3_S2> SmartFactor;

// Create a noise model for measurements
static SharedNoiseModel measurementNoise =
    noiseModel::Isotropic::Sigma(2, 1.0);

/* ************************************************************************* */
TEST(SmartProjectionPoseCalFactor, Constructor) {
  SmartProjectionParams params;
  params.setLinearizationMode(HESSIAN);
  params.setDegeneracyMode(ZERO_ON_DEGENERACY);

  SmartFactor factor(measurementNoise, params);

  // New factor should have no measurements or keys
  EXPECT_LONGS_EQUAL(0, factor.measured().size());
  EXPECT_LONGS_EQUAL(0, factor.keys().size());
}

/* ************************************************************************* */
TEST(SmartProjectionPoseCalFactor, Add) {
  SmartProjectionParams params;
  params.setLinearizationMode(HESSIAN);
  params.setDegeneracyMode(ZERO_ON_DEGENERACY);

  SmartFactor factor(measurementNoise, params);

  // Add measurements from two poses with shared calibration
  factor.add(Point2(0, 0), Symbol('x', 0), Symbol('K', 0));
  factor.add(Point2(0, 0), Symbol('x', 1), Symbol('K', 0));

  // Check that we have 3 keys: 2 poses + 1 calibration
  EXPECT_LONGS_EQUAL(3, factor.keys().size());

  // Check that calibration key is present
  const KeyVector& keys = factor.keys();
  EXPECT(std::find(keys.begin(), keys.end(), Symbol('K', 0)) != keys.end());
  EXPECT(std::find(keys.begin(), keys.end(), Symbol('x', 0)) != keys.end());
  EXPECT(std::find(keys.begin(), keys.end(), Symbol('x', 1)) != keys.end());
}

/* ************************************************************************* */
TEST(SmartProjectionPoseCalFactor, AddWithDifferentCalibrations) {
  SmartProjectionParams params;
  params.setLinearizationMode(HESSIAN);
  params.setDegeneracyMode(ZERO_ON_DEGENERACY);

  SmartFactor factor(measurementNoise, params);

  // Add measurements with different calibration keys
  factor.add(Point2(0, 0), Symbol('x', 0), Symbol('K', 0));
  factor.add(Point2(0, 0), Symbol('x', 1), Symbol('K', 1));

  // Check that we have 4 keys: 2 poses + 2 calibrations
  EXPECT_LONGS_EQUAL(4, factor.keys().size());

  // Check that both calibration keys are present
  const KeyVector& keys = factor.keys();
  EXPECT(std::find(keys.begin(), keys.end(), Symbol('K', 0)) != keys.end());
  EXPECT(std::find(keys.begin(), keys.end(), Symbol('K', 1)) != keys.end());
}

/* ************************************************************************* */
TEST(SmartProjectionPoseCalFactor, RepeatedPoses) {
  SmartProjectionParams params;
  params.setLinearizationMode(HESSIAN);
  params.setDegeneracyMode(ZERO_ON_DEGENERACY);

  SmartFactor factor(measurementNoise, params);

  // Add two measurements from the same pose (but different calibrations)
  factor.add(Point2(0, 0), Symbol('x', 0), Symbol('K', 0));
  factor.add(Point2(1, 0), Symbol('x', 0), Symbol('K', 1));

  // Keys should contain: x0, K0, K1 (not x0 twice)
  EXPECT_LONGS_EQUAL(3, factor.keys().size());

  // But poseKeys should contain x0 twice
  EXPECT_LONGS_EQUAL(2, factor.poseKeys().size());
  EXPECT(factor.poseKeys()[0] == Symbol('x', 0));
  EXPECT(factor.poseKeys()[1] == Symbol('x', 0));
}

/* ************************************************************************* */
TEST(SmartProjectionPoseCalFactor, Cameras) {
  SmartProjectionParams params;
  params.setLinearizationMode(HESSIAN);
  params.setDegeneracyMode(ZERO_ON_DEGENERACY);

  SmartFactor factor(measurementNoise, params);
  factor.add(Point2(0, 0), Symbol('x', 0), Symbol('K', 0));
  factor.add(Point2(0, 0), Symbol('x', 1), Symbol('K', 0));

  // Create values
  Values values;
  Pose3 pose1 = Pose3(Rot3(), Point3(0, 0, 0));
  Pose3 pose2 = Pose3(Rot3(), Point3(1, 0, 0));
  Cal3_S2 K(500, 500, 0, 320, 240);

  values.insert(Symbol('x', 0), pose1);
  values.insert(Symbol('x', 1), pose2);
  values.insert(Symbol('K', 0), K);

  // Get cameras
  auto cameras = factor.cameras(values);

  // Should have 2 cameras (one per measurement)
  EXPECT_LONGS_EQUAL(2, cameras.size());
}

/* ************************************************************************* */
TEST(SmartProjectionPoseCalFactor, ErrorNoiseless) {
  SmartProjectionParams params;
  params.setLinearizationMode(HESSIAN);
  params.setDegeneracyMode(ZERO_ON_DEGENERACY);
  params.setRankTolerance(1e-9);

  SmartFactor factor(measurementNoise, params);

  // Create ground truth scenario
  Point3 landmark(0, 0, 5);  // 5 meters in front
  Cal3_S2 trueK(500, 500, 0, 320, 240);

  // Two poses looking at the landmark
  Pose3 pose1 = Pose3(Rot3(), Point3(0, 0, 0));
  Pose3 pose2 = Pose3(Rot3(), Point3(1, 0, 0));

  // Create cameras and project
  PinholeCamera<Cal3_S2> cam1(pose1, trueK);
  PinholeCamera<Cal3_S2> cam2(pose2, trueK);

  Point2 measurement1 = cam1.project(landmark);
  Point2 measurement2 = cam2.project(landmark);

  // Add measurements with shared calibration
  factor.add(measurement1, Symbol('x', 0), Symbol('K', 0));
  factor.add(measurement2, Symbol('x', 1), Symbol('K', 0));

  // Create values with ground truth
  Values values;
  values.insert(Symbol('x', 0), pose1);
  values.insert(Symbol('x', 1), pose2);
  values.insert(Symbol('K', 0), trueK);

  // Error should be zero (within tolerance)
  double error = factor.error(values);
  EXPECT_DOUBLES_EQUAL(0.0, error, 1e-9);
}

/* ************************************************************************* */
TEST(SmartProjectionPoseCalFactor, Linearization) {
  SmartProjectionParams params;
  params.setLinearizationMode(HESSIAN);
  params.setDegeneracyMode(ZERO_ON_DEGENERACY);

  SmartFactor factor(measurementNoise, params);

  // Create scenario
  Point3 landmark(0, 0, 5);
  Cal3_S2 K(500, 500, 0, 320, 240);
  Pose3 pose1 = Pose3(Rot3(), Point3(0, 0, 0));
  Pose3 pose2 = Pose3(Rot3(), Point3(1, 0, 0));

  // Project
  PinholeCamera<Cal3_S2> cam1(pose1, K);
  PinholeCamera<Cal3_S2> cam2(pose2, K);
  Point2 measurement1 = cam1.project(landmark);
  Point2 measurement2 = cam2.project(landmark);

  factor.add(measurement1, Symbol('x', 0), Symbol('K', 0));
  factor.add(measurement2, Symbol('x', 1), Symbol('K', 0));

  // Create values
  Values values;
  values.insert(Symbol('x', 0), pose1);
  values.insert(Symbol('x', 1), pose2);
  values.insert(Symbol('K', 0), K);

  // Linearize
  auto linearFactor = factor.linearize(values);

  // Should return a valid factor
  EXPECT(linearFactor != nullptr);

  // Check that it's a HessianFactor
  auto hessianFactor = std::dynamic_pointer_cast<HessianFactor>(linearFactor);
  EXPECT(hessianFactor != nullptr);

  // Check that keys match
  EXPECT_LONGS_EQUAL(factor.keys().size(), hessianFactor->keys().size());
}

/* ************************************************************************* */
TEST(SmartProjectionPoseCalFactor, LinearizeDamped) {
  SmartProjectionParams params;
  params.setLinearizationMode(HESSIAN);
  params.setDegeneracyMode(ZERO_ON_DEGENERACY);

  SmartFactor factor(measurementNoise, params);

  // Create scenario
  Point3 landmark(0, 0, 5);
  Cal3_S2 K(500, 500, 0, 320, 240);
  Pose3 pose1 = Pose3(Rot3(), Point3(0, 0, 0));
  Pose3 pose2 = Pose3(Rot3(), Point3(1, 0, 0));

  // Project
  PinholeCamera<Cal3_S2> cam1(pose1, K);
  PinholeCamera<Cal3_S2> cam2(pose2, K);
  Point2 measurement1 = cam1.project(landmark);
  Point2 measurement2 = cam2.project(landmark);

  factor.add(measurement1, Symbol('x', 0), Symbol('K', 0));
  factor.add(measurement2, Symbol('x', 1), Symbol('K', 0));

  // Create values
  Values values;
  values.insert(Symbol('x', 0), pose1);
  values.insert(Symbol('x', 1), pose2);
  values.insert(Symbol('K', 0), K);

  // Linearize with damping
  double lambda = 1.0;
  auto linearFactor = factor.linearizeDamped(values, lambda);

  // Should return a valid factor
  EXPECT(linearFactor != nullptr);

  // Check that it's a HessianFactor
  auto hessianFactor = std::dynamic_pointer_cast<HessianFactor>(linearFactor);
  EXPECT(hessianFactor != nullptr);
}

/* ************************************************************************* */
TEST(SmartProjectionPoseCalFactor, Optimization) {
  SmartProjectionParams params;
  params.setLinearizationMode(HESSIAN);
  params.setDegeneracyMode(ZERO_ON_DEGENERACY);

  // Ground truth
  Cal3_S2 trueK(500, 500, 0, 320, 240);
  Pose3 truePose1 = Pose3(Rot3(), Point3(0, 0, 0));
  Pose3 truePose2 = Pose3(Rot3(), Point3(1, 0, 0));
  Pose3 truePose3 = Pose3(Rot3(), Point3(0, 1, 0));

  // Multiple landmarks at different depths for better conditioning
  std::vector<Point3> landmarks = {
    Point3(0, 0, 5),
    Point3(1, 0, 6),
    Point3(0, 1, 4),
    Point3(0.5, 0.5, 5),
    Point3(-0.5, 0.5, 7)
  };

  // Create measurements
  PinholeCamera<Cal3_S2> cam1(truePose1, trueK);
  PinholeCamera<Cal3_S2> cam2(truePose2, trueK);
  PinholeCamera<Cal3_S2> cam3(truePose3, trueK);

  // Create factor graph
  NonlinearFactorGraph graph;

  // Create a factor for each landmark
  std::cerr << "x0:" << Symbol('x', 0).key() << std::endl;
  std::cerr << "x1:" << Symbol('x', 1).key() << std::endl;
  std::cerr << "x2:" << Symbol('x', 2).key() << std::endl;
  std::cerr << "K0" <<  Symbol('K', 0).key() << std::endl;

  for (const auto& landmark : landmarks) {
    Point2 z1 = cam1.project(landmark);
    Point2 z2 = cam2.project(landmark);
    Point2 z3 = cam3.project(landmark);

    auto factor = std::make_shared<SmartFactor>(measurementNoise, params);
    factor->add(z1, Symbol('x', 0), Symbol('K', 0));
    factor->add(z2, Symbol('x', 1), Symbol('K', 0));
    factor->add(z3, Symbol('x', 2), Symbol('K', 0));
    graph.push_back(factor);
  }

  // Add prior on first pose (anchor)
  auto posePrior = noiseModel::Isotropic::Sigma(6, 0.01);
  graph.addPrior(Symbol('x', 0), truePose1, posePrior);
  graph.addPrior(Symbol('x', 1), truePose2, posePrior);
  graph.addPrior(Symbol('x', 2), truePose3, posePrior);


  // Create perturbed initial values
  Values initial;
  initial.insert(Symbol('x', 0), truePose1);  // First pose is anchored
  initial.insert(Symbol('x', 1), truePose2.compose(Pose3(Rot3(), Point3(0.1, 0, 0))));
  initial.insert(Symbol('x', 2), truePose3.compose(Pose3(Rot3(), Point3(0, 0.1, 0))));

  // Perturb calibration
  Cal3_S2 initialK(505, 495, 0.005, 323, 238);
  initial.insert(Symbol('K', 0), initialK);

  // Optimize
  LevenbergMarquardtParams lmParams;
  lmParams.setVerbosity("ERROR");
  lmParams.setAbsoluteErrorTol(1e-10);
  lmParams.setlambdaUpperBound(1e8);
  LevenbergMarquardtOptimizer optimizer(graph, initial, lmParams);
  Values result = optimizer.optimize();

  // Check convergence to ground truth
  EXPECT(assert_equal(truePose1, result.at<Pose3>(Symbol('x', 0)), 1e-4));
  EXPECT(assert_equal(truePose2, result.at<Pose3>(Symbol('x', 1)), 1e-4));
  EXPECT(assert_equal(truePose3, result.at<Pose3>(Symbol('x', 2)), 1e-4));
  EXPECT(assert_equal(trueK, result.at<Cal3_S2>(Symbol('K', 0)), 1e-2));
}

/* ************************************************************************* */
TEST(SmartProjectionPoseCalFactor, MultipleFactorsSharedCalibration) {
  SmartProjectionParams params;
  params.setLinearizationMode(HESSIAN);
  params.setDegeneracyMode(ZERO_ON_DEGENERACY);

  // Ground truth
  Cal3_S2 trueK(500, 500, 0, 320, 240);
  Pose3 truePose1 = Pose3(Rot3(), Point3(0, 0, 0));
  Pose3 truePose2 = Pose3(Rot3(), Point3(1, 0, 0));

  Point3 landmark1(0, 0, 5);
  Point3 landmark2(0.5, 0.5, 6);

  // Create measurements for landmarks
  PinholeCamera<Cal3_S2> cam1(truePose1, trueK);
  PinholeCamera<Cal3_S2> cam2(truePose2, trueK);

  Point2 z1_1 = cam1.project(landmark1);
  Point2 z1_2 = cam2.project(landmark1);
  Point2 z2_1 = cam1.project(landmark2);
  Point2 z2_2 = cam2.project(landmark2);

  // Create two factors, both using the same calibration key
  Key calKey = Symbol('K', 0);
  auto factor1 = std::make_shared<SmartFactor>(measurementNoise, params);
  factor1->add(z1_1, Symbol('x', 0), calKey);
  factor1->add(z1_2, Symbol('x', 1), calKey);

  auto factor2 = std::make_shared<SmartFactor>(measurementNoise, params);
  factor2->add(z2_1, Symbol('x', 0), calKey);
  factor2->add(z2_2, Symbol('x', 1), calKey);

  // Create factor graph
  NonlinearFactorGraph graph;
  graph.push_back(factor1);
  graph.push_back(factor2);

  // Add prior
  auto posePrior = noiseModel::Isotropic::Sigma(6, 0.01);
  graph.addPrior(Symbol('x', 0), truePose1, posePrior);

  // Perturbed initial
  Values initial;
  initial.insert(Symbol('x', 0), truePose1);
  initial.insert(Symbol('x', 1), truePose2.compose(Pose3(Rot3(), Point3(0.1, 0, 0))));
  Cal3_S2 initialK(520, 480, 0, 330, 250);
  initial.insert(calKey, initialK);

  // Optimize
  LevenbergMarquardtParams lmParams;
  LevenbergMarquardtOptimizer optimizer(graph, initial, lmParams);
  Values result = optimizer.optimize();

  // Both factors should help constrain the shared calibration
  EXPECT(assert_equal(truePose1, result.at<Pose3>(Symbol('x', 0)), 1e-4));
  EXPECT(assert_equal(truePose2, result.at<Pose3>(Symbol('x', 1)), 1e-4));
  EXPECT(assert_equal(trueK, result.at<Cal3_S2>(calKey), 1e-2));
}

/* ************************************************************************* */
TEST(SmartProjectionPoseCalFactor, DifferentCalibrationsPerCamera) {
  SmartProjectionParams params;
  params.setLinearizationMode(HESSIAN);
  params.setDegeneracyMode(ZERO_ON_DEGENERACY);

  // Ground truth - two different calibrations
  Cal3_S2 trueK0(500, 500, 0, 320, 240);
  Cal3_S2 trueK1(600, 600, 0, 320, 240);  // Different focal length
  Pose3 truePose1 = Pose3(Rot3(), Point3(0, 0, 0));
  Pose3 truePose2 = Pose3(Rot3(), Point3(1, 0, 0));

  Point3 landmark(0, 0, 5);

  // Create measurements with different cameras
  PinholeCamera<Cal3_S2> cam1(truePose1, trueK0);
  PinholeCamera<Cal3_S2> cam2(truePose2, trueK1);

  Point2 z1 = cam1.project(landmark);
  Point2 z2 = cam2.project(landmark);

  // Create factor with different calibration keys per measurement
  auto factor = std::make_shared<SmartFactor>(measurementNoise, params);
  factor->add(z1, Symbol('x', 0), Symbol('K', 0));
  factor->add(z2, Symbol('x', 1), Symbol('K', 1));

  // Verify keys
  EXPECT_LONGS_EQUAL(4, factor->keys().size());  // 2 poses + 2 calibrations
  EXPECT_LONGS_EQUAL(2, factor->calibrationKeys().size());

  // Create factor graph
  NonlinearFactorGraph graph;
  graph.push_back(factor);

  // Add priors
  auto posePrior = noiseModel::Isotropic::Sigma(6, 0.01);
  graph.addPrior(Symbol('x', 0), truePose1, posePrior);
  graph.addPrior(Symbol('x', 1), truePose2, posePrior);

  // Perturbed initial
  Values initial;
  initial.insert(Symbol('x', 0), truePose1);
  initial.insert(Symbol('x', 1), truePose2);
  initial.insert(Symbol('K', 0), Cal3_S2(510, 510, 0, 325, 245));
  initial.insert(Symbol('K', 1), Cal3_S2(610, 610, 0, 325, 245));

  // Optimize
  LevenbergMarquardtParams lmParams;
  LevenbergMarquardtOptimizer optimizer(graph, initial, lmParams);
  Values result = optimizer.optimize();

  // Check convergence
  EXPECT(assert_equal(trueK0, result.at<Cal3_S2>(Symbol('K', 0)), 1e-1));
  EXPECT(assert_equal(trueK1, result.at<Cal3_S2>(Symbol('K', 1)), 1e-1));
}

/* ************************************************************************* */
// Tests for key ordering and Hessian structure
/* ************************************************************************* */

TEST(SmartProjectionPoseCalFactor, KeyOrderingPosesBeforeCalibrations) {
  SmartProjectionParams params;
  params.setLinearizationMode(HESSIAN);
  params.setDegeneracyMode(ZERO_ON_DEGENERACY);

  SmartFactor factor(measurementNoise, params);

  // Add measurements in a way that would interleave keys if not properly ordered
  // Add: (x0, K0), (x1, K0), (x2, K1)
  factor.add(Point2(0, 0), Symbol('x', 0), Symbol('K', 0));
  factor.add(Point2(0, 0), Symbol('x', 1), Symbol('K', 0));
  factor.add(Point2(0, 0), Symbol('x', 2), Symbol('K', 1));

  // keys_ should be ordered as [x0, x1, x2, K0, K1] (poses first, then cals)
  const KeyVector& keys = factor.keys();
  EXPECT_LONGS_EQUAL(5, keys.size());

  // First 3 should be poses
  EXPECT(keys[0] == Symbol('x', 0));
  EXPECT(keys[1] == Symbol('x', 1));
  EXPECT(keys[2] == Symbol('x', 2));

  // Last 2 should be calibrations
  EXPECT(keys[3] == Symbol('K', 0));
  EXPECT(keys[4] == Symbol('K', 1));
}

/* ************************************************************************* */
TEST(SmartProjectionPoseCalFactor, KeyOrderingWithSharedCalibration) {
  SmartProjectionParams params;
  params.setLinearizationMode(HESSIAN);
  params.setDegeneracyMode(ZERO_ON_DEGENERACY);

  SmartFactor factor(measurementNoise, params);

  // All measurements share the same calibration
  factor.add(Point2(0, 0), Symbol('x', 0), Symbol('K', 0));
  factor.add(Point2(0, 0), Symbol('x', 1), Symbol('K', 0));
  factor.add(Point2(0, 0), Symbol('x', 2), Symbol('K', 0));

  // keys_ should be [x0, x1, x2, K0]
  const KeyVector& keys = factor.keys();
  EXPECT_LONGS_EQUAL(4, keys.size());

  EXPECT(keys[0] == Symbol('x', 0));
  EXPECT(keys[1] == Symbol('x', 1));
  EXPECT(keys[2] == Symbol('x', 2));
  EXPECT(keys[3] == Symbol('K', 0));
}

/* ************************************************************************* */
TEST(SmartProjectionPoseCalFactor, HessianBlockDimensions) {
  SmartProjectionParams params;
  params.setLinearizationMode(HESSIAN);
  params.setDegeneracyMode(ZERO_ON_DEGENERACY);

  SmartFactor factor(measurementNoise, params);

  // Create scenario with 2 poses sharing 1 calibration
  Point3 landmark(0, 0, 5);
  Cal3_S2 K(500, 500, 0, 320, 240);
  Pose3 pose1 = Pose3(Rot3(), Point3(0, 0, 0));
  Pose3 pose2 = Pose3(Rot3(), Point3(1, 0, 0));

  PinholeCamera<Cal3_S2> cam1(pose1, K);
  PinholeCamera<Cal3_S2> cam2(pose2, K);
  Point2 z1 = cam1.project(landmark);
  Point2 z2 = cam2.project(landmark);

  factor.add(z1, Symbol('x', 0), Symbol('K', 0));
  factor.add(z2, Symbol('x', 1), Symbol('K', 0));

  Values values;
  values.insert(Symbol('x', 0), pose1);
  values.insert(Symbol('x', 1), pose2);
  values.insert(Symbol('K', 0), K);

  // Linearize
  auto linearFactor = factor.linearize(values);
  auto hessianFactor = std::dynamic_pointer_cast<HessianFactor>(linearFactor);
  EXPECT(hessianFactor != nullptr);

  // Check keys match factor keys
  EXPECT(hessianFactor->keys() == factor.keys());

  // Expected structure: [x0(6), x1(6), K0(5)]
  // Total dimension: 6 + 6 + 5 = 17
  // The augmented info should have this structure
  Matrix info = hessianFactor->information();
  EXPECT_LONGS_EQUAL(17, info.rows());
  EXPECT_LONGS_EQUAL(17, info.cols());
}

/* ************************************************************************* */
TEST(SmartProjectionPoseCalFactor, HessianBlockDimensionsDifferentCals) {
  SmartProjectionParams params;
  params.setLinearizationMode(HESSIAN);
  params.setDegeneracyMode(ZERO_ON_DEGENERACY);

  SmartFactor factor(measurementNoise, params);

  // 2 poses with different calibrations
  Point3 landmark(0, 0, 5);
  Cal3_S2 K0(500, 500, 0, 320, 240);
  Cal3_S2 K1(600, 600, 0, 320, 240);
  Pose3 pose1 = Pose3(Rot3(), Point3(0, 0, 0));
  Pose3 pose2 = Pose3(Rot3(), Point3(1, 0, 0));

  PinholeCamera<Cal3_S2> cam1(pose1, K0);
  PinholeCamera<Cal3_S2> cam2(pose2, K1);
  Point2 z1 = cam1.project(landmark);
  Point2 z2 = cam2.project(landmark);

  factor.add(z1, Symbol('x', 0), Symbol('K', 0));
  factor.add(z2, Symbol('x', 1), Symbol('K', 1));

  Values values;
  values.insert(Symbol('x', 0), pose1);
  values.insert(Symbol('x', 1), pose2);
  values.insert(Symbol('K', 0), K0);
  values.insert(Symbol('K', 1), K1);

  auto linearFactor = factor.linearize(values);
  auto hessianFactor = std::dynamic_pointer_cast<HessianFactor>(linearFactor);
  EXPECT(hessianFactor != nullptr);

  // Expected structure: [x0(6), x1(6), K0(5), K1(5)]
  // Total dimension: 6 + 6 + 5 + 5 = 22
  Matrix info = hessianFactor->information();
  EXPECT_LONGS_EQUAL(22, info.rows());
  EXPECT_LONGS_EQUAL(22, info.cols());
}

/* ************************************************************************* */
TEST(SmartProjectionPoseCalFactor, HessianSymmetry) {
  SmartProjectionParams params;
  params.setLinearizationMode(HESSIAN);
  params.setDegeneracyMode(ZERO_ON_DEGENERACY);

  SmartFactor factor(measurementNoise, params);

  Point3 landmark(0, 0, 5);
  Cal3_S2 K(500, 500, 0, 320, 240);
  Pose3 pose1 = Pose3(Rot3(), Point3(0, 0, 0));
  Pose3 pose2 = Pose3(Rot3(), Point3(1, 0, 0));
  Pose3 pose3 = Pose3(Rot3(), Point3(0, 1, 0));

  PinholeCamera<Cal3_S2> cam1(pose1, K);
  PinholeCamera<Cal3_S2> cam2(pose2, K);
  PinholeCamera<Cal3_S2> cam3(pose3, K);

  factor.add(cam1.project(landmark), Symbol('x', 0), Symbol('K', 0));
  factor.add(cam2.project(landmark), Symbol('x', 1), Symbol('K', 0));
  factor.add(cam3.project(landmark), Symbol('x', 2), Symbol('K', 0));

  Values values;
  values.insert(Symbol('x', 0), pose1);
  values.insert(Symbol('x', 1), pose2);
  values.insert(Symbol('x', 2), pose3);
  values.insert(Symbol('K', 0), K);

  auto linearFactor = factor.linearize(values);
  auto hessianFactor = std::dynamic_pointer_cast<HessianFactor>(linearFactor);

  Matrix info = hessianFactor->information();

  // Check symmetry
  EXPECT(assert_equal(info, info.transpose(), 1e-9));
}

/* ************************************************************************* */
TEST(SmartProjectionPoseCalFactor, HessianPositiveSemiDefinite) {
  SmartProjectionParams params;
  params.setLinearizationMode(HESSIAN);
  params.setDegeneracyMode(ZERO_ON_DEGENERACY);

  SmartFactor factor(measurementNoise, params);

  Point3 landmark(0, 0, 5);
  Cal3_S2 K(500, 500, 0, 320, 240);
  Pose3 pose1 = Pose3(Rot3(), Point3(0, 0, 0));
  Pose3 pose2 = Pose3(Rot3(), Point3(1, 0, 0));

  PinholeCamera<Cal3_S2> cam1(pose1, K);
  PinholeCamera<Cal3_S2> cam2(pose2, K);

  factor.add(cam1.project(landmark), Symbol('x', 0), Symbol('K', 0));
  factor.add(cam2.project(landmark), Symbol('x', 1), Symbol('K', 0));

  Values values;
  values.insert(Symbol('x', 0), pose1);
  values.insert(Symbol('x', 1), pose2);
  values.insert(Symbol('K', 0), K);

  auto linearFactor = factor.linearize(values);
  auto hessianFactor = std::dynamic_pointer_cast<HessianFactor>(linearFactor);

  Matrix info = hessianFactor->information();

  // Check positive semi-definiteness via eigenvalues
  Eigen::SelfAdjointEigenSolver<Matrix> solver(info);
  Vector eigenvalues = solver.eigenvalues();

  // All eigenvalues should be >= 0 (within tolerance for numerical errors)
  for (int i = 0; i < eigenvalues.size(); i++) {
    EXPECT(eigenvalues(i) >= -1e-9);
  }
}

/* ************************************************************************* */
TEST(SmartProjectionPoseCalFactor, DebugOutputDifferentCals) {
  SmartProjectionParams params;
  params.setLinearizationMode(HESSIAN);
  params.setDegeneracyMode(ZERO_ON_DEGENERACY);

  SmartFactor factor(measurementNoise, params);

  Point3 landmark(0, 0, 5);
  Cal3_S2 K0(500, 500, 0, 320, 240);
  Cal3_S2 K1(600, 600, 0, 320, 240);
  Pose3 pose1 = Pose3(Rot3(), Point3(0, 0, 0));
  Pose3 pose2 = Pose3(Rot3(), Point3(1, 0, 0));

  PinholeCamera<Cal3_S2> cam1(pose1, K0);
  PinholeCamera<Cal3_S2> cam2(pose2, K1);

  factor.add(cam1.project(landmark), Symbol('x', 0), Symbol('K', 0));
  factor.add(cam2.project(landmark), Symbol('x', 1), Symbol('K', 1));

  Values values;
  values.insert(Symbol('x', 0), pose1);
  values.insert(Symbol('x', 1), pose2);
  values.insert(Symbol('K', 0), K0);
  values.insert(Symbol('K', 1), K1);

  // Enable debug and linearize
  auto linearFactor = factor.linearize(values);

  EXPECT(linearFactor != nullptr);
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
