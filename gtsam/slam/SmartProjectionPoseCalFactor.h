/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file   SmartProjectionPoseCalFactor.h
 * @brief  Smart factor on poses with optimizable calibration(s)
 * @author Seth Isaacson
 * @date   December 2025
 */

#pragma once

#include <gtsam/slam/SmartProjectionFactor.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/linear/HessianFactor.h>

#include <set>

namespace gtsam {

/**
 * @brief Smart factor that optimizes poses and calibration(s)
 *
 * This factor extends SmartProjectionFactor to allow calibrations
 * to be optimized. Each camera can reference its own calibration key,
 * enabling flexible sharing patterns:
 * - All cameras share one calibration (stereo rig with identical intrinsics)
 * - Each camera has its own calibration (independent calibrations)
 * - Subsets share calibrations (e.g., left cameras share K0, right cameras share K1)
 *
 * Unlike SmartProjectionPoseFactor where calibration is fixed, this factor
 * includes calibration(s) as variable(s) in the factor graph.
 *
 * Example usage (shared calibration):
 * @code
 * auto factor = std::make_shared<SmartProjectionPoseCalFactor<Cal3_S2>>(
 *     noiseModel, params);
 * factor->add(measurement1, Symbol('x', 0), Symbol('K', 0));  // pose 0, cal 0
 * factor->add(measurement2, Symbol('x', 1), Symbol('K', 0));  // pose 1, cal 0 (shared)
 * graph.push_back(factor);
 * @endcode
 *
 * Example usage (per-camera calibration):
 * @code
 * auto factor = std::make_shared<SmartProjectionPoseCalFactor<Cal3_S2>>(
 *     noiseModel, params);
 * factor->add(measurement1, Symbol('x', 0), Symbol('K', 0));  // pose 0, cal 0
 * factor->add(measurement2, Symbol('x', 1), Symbol('K', 1));  // pose 1, cal 1
 * graph.push_back(factor);
 * @endcode
 *
 * @tparam CALIBRATION Camera calibration type (e.g., Cal3_S2, Cal3Bundler)
 * @ingroup slam
 */
template <class CALIBRATION>
class SmartProjectionPoseCalFactor
    : public SmartProjectionFactor<PinholeCamera<CALIBRATION>> {
 private:
  typedef PinholeCamera<CALIBRATION> Camera;
  typedef SmartProjectionFactor<Camera> Base;
  typedef SmartProjectionPoseCalFactor<CALIBRATION> This;

 protected:
  /// Pose keys (one per measurement), may contain duplicates
  KeyVector poseKeys_;

  /// Calibration keys (one per measurement), may contain duplicates
  KeyVector calibrationKeys_;

  static const int DimPose = 6;  ///< Pose3 dimension
  static const int DimK = FixedDimension<CALIBRATION>::value;  ///< Calibration dimension
  static const int DimCam = DimPose + DimK;  ///< Full camera dimension
  static const int ZDim = 2;  ///< Measurement dimension

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// shorthand for a smart pointer to a factor
  typedef std::shared_ptr<This> shared_ptr;

  /// CameraSet of PinholeCamera
  typedef CameraSet<Camera> Cameras;

  /**
   * Default constructor, only for serialization
   */
  SmartProjectionPoseCalFactor() {}

  /**
   * Constructor
   * @param sharedNoiseModel isotropic noise model for the 2D feature measurements
   * @param params parameters for the smart projection factors
   */
  SmartProjectionPoseCalFactor(
      const SharedNoiseModel& sharedNoiseModel,
      const SmartProjectionParams& params = SmartProjectionParams())
      : Base(sharedNoiseModel, params) {
    // Require HESSIAN linearization mode
    if (Base::params_.linearizationMode != HESSIAN) {
      throw std::runtime_error(
          "SmartProjectionPoseCalFactor: linearizationMode must be HESSIAN");
    }
    // Require ZERO_ON_DEGENERACY mode
    if (Base::params_.degeneracyMode != ZERO_ON_DEGENERACY) {
      throw std::runtime_error(
          "SmartProjectionPoseCalFactor: degeneracyMode must be ZERO_ON_DEGENERACY");
    }
  }

  /**
   * Constructor with body_P_sensor
   * @param sharedNoiseModel isotropic noise model for the 2D feature measurements
   * @param body_P_sensor pose of the camera in the body frame
   * @param params parameters for the smart projection factors
   */
  SmartProjectionPoseCalFactor(
      const SharedNoiseModel& sharedNoiseModel,
      const std::optional<Pose3> body_P_sensor,
      const SmartProjectionParams& params = SmartProjectionParams())
      : SmartProjectionPoseCalFactor(sharedNoiseModel, params) {
    this->body_P_sensor_ = body_P_sensor;
  }

  /** Virtual destructor */
  ~SmartProjectionPoseCalFactor() override {}

  /**
   * Add a new measurement with pose and calibration keys
   * @param measured 2D measurement
   * @param poseKey key for the pose observing this measurement
   * @param calibrationKey key for the calibration used for this measurement
   */
  void add(const Point2& measured, const Key& poseKey, const Key& calibrationKey) {
    // Add measurement to base class
    this->measured_.push_back(measured);

    // Store pose and calibration keys (may have duplicates)
    poseKeys_.push_back(poseKey);
    calibrationKeys_.push_back(calibrationKey);

    // Rebuild keys_ to maintain [unique poses..., unique calibrations...] order
    // This ensures keys_ matches the ordering used in schurComplementAndRearrange
    rebuildKeys();
  }

  /**
   * Add measurements in bulk
   * @param measurements vector of 2D measurements
   * @param poseKeys vector of pose keys
   * @param calibrationKeys vector of calibration keys
   */
  void add(const Point2Vector& measurements,
           const KeyVector& poseKeys,
           const KeyVector& calibrationKeys) {
    if (measurements.size() != poseKeys.size() ||
        measurements.size() != calibrationKeys.size()) {
      throw std::runtime_error(
          "SmartProjectionPoseCalFactor::add: measurements, poseKeys, and "
          "calibrationKeys must have same size");
    }
    for (size_t i = 0; i < measurements.size(); i++) {
      this->measured_.push_back(measurements[i]);
      poseKeys_.push_back(poseKeys[i]);
      calibrationKeys_.push_back(calibrationKeys[i]);
    }
    // Rebuild keys_ once after all additions
    rebuildKeys();
  }

  /**
   * print
   * @param s optional string naming the factor
   * @param keyFormatter optional formatter useful for printing Symbols
   */
  void print(const std::string& s = "", const KeyFormatter& keyFormatter =
      DefaultKeyFormatter) const override {
    std::cout<< s << "SmartProjectionPoseCalFactor" << std::endl;
    std::cout << "  Number of measurements: " << this->measured_.size() << std::endl;
    std::cout<< "  Pose keys: ";
    for (const Key& k : poseKeys_) {
      std::cerr<< keyFormatter(k) << " ";
    }
    std::cout<< std::endl;
    std::cout<< "  Calibration keys: ";
    for (const Key& k : calibrationKeys_) {
      std::cerr<< keyFormatter(k) << " ";
    }
    std::cout<< std::endl;
    Base::print("", keyFormatter);
  }

  /// equals
  bool equals(const NonlinearFactor& p, double tol = 1e-9) const override {
    const This *e = dynamic_cast<const This*>(&p);
    return e && Base::equals(p, tol) &&
           poseKeys_ == e->poseKeys_ &&
           calibrationKeys_ == e->calibrationKeys_;
  }

  /**
   * Collect all cameras involved in this factor
   * @param values Values structure which must contain poses and calibrations
   * @return CameraSet with one camera per measurement
   */
  Cameras cameras(const Values& values) const override {
    Cameras cameras;

    // Create cameras from poses + calibrations (one per measurement)
    for (size_t i = 0; i < poseKeys_.size(); i++) {
      const Key& poseKey = poseKeys_[i];
      const Key& calKey = calibrationKeys_[i];

      const Pose3& pose = values.at<Pose3>(poseKey);
      const CALIBRATION& K = values.at<CALIBRATION>(calKey);

      const Pose3 world_P_sensor =
          this->body_P_sensor_ ? pose * *this->body_P_sensor_ : pose;
      cameras.emplace_back(world_P_sensor, K);
    }
    return cameras;
  }

  /**
   * error calculates the error of the factor
   */
  double error(const Values& values) const override {
    if (this->active(values)) {
      return this->totalReprojectionError(cameras(values));
    } else {
      return 0.0;
    }
  }

  /// Return the pose keys (one per measurement)
  const KeyVector& poseKeys() const { return poseKeys_; }

  /// Return the calibration keys (one per measurement)
  const KeyVector& calibrationKeys() const { return calibrationKeys_; }

  /**
   * Linearize to a Hessian factor (with optional Levenberg-Marquardt damping)
   * @param cameras CameraSet containing all cameras
   * @param lambda LM damping parameter (default 0.0)
   * @param diagonalDamping Whether to use diagonal damping (default false)
   * @return Hessian factor with derivatives wrt poses and calibrations
   */
  std::shared_ptr<GaussianFactor> linearizeDamped(
      const Cameras& cameras,
      const double lambda = 0.0,
      bool diagonalDamping = false) const {

    // Triangulate the point
    this->triangulateSafe(cameras);

    // Handle degeneracy: return zero factor
    if (!this->result_) {
      // keys_ is ordered as [unique poses..., unique calibrations...] by rebuildKeys()
      std::vector<Matrix> Gs;
      std::vector<Vector> gs;

      // Count unique poses and calibrations
      std::set<Key> uniquePoses(poseKeys_.begin(), poseKeys_.end());
      std::set<Key> uniqueCals(calibrationKeys_.begin(), calibrationKeys_.end());
      size_t nrUniquePoses = uniquePoses.size();
      size_t nrUniqueCals = uniqueCals.size();
      size_t nrUniqueKeys = nrUniquePoses + nrUniqueCals;

      // Build upper-triangular blocks: n*(n+1)/2 blocks for n keys
      for (size_t i = 0; i < nrUniqueKeys; i++) {
        int dim_i = (i < nrUniquePoses) ? DimPose : DimK;
        for (size_t j = i; j < nrUniqueKeys; j++) {
          int dim_j = (j < nrUniquePoses) ? DimPose : DimK;
          Gs.push_back(Matrix::Zero(dim_i, dim_j));
        }
        gs.push_back(Vector::Zero(dim_i));
      }

      return std::make_shared<HessianFactor>(this->keys_, Gs, gs, 0.0);
    }

    const Point3& point = *this->result_;
    size_t m = this->measured_.size();  // number of measurements

    // Compute full camera Jacobians (F blocks of size 2 x (6+DimK)) and point Jacobian (E)
    typename Cameras::FBlocks Fs;  // Each block is 2 x DimCam
    Matrix E;
    Vector b;

    // Project and compute Jacobians
    b = -cameras.reprojectionError(point, this->measured_, Fs, E);

    // Whiten using noise model
    this->noiseModel_->WhitenSystem(E, b);
    for (size_t i = 0; i < m; i++) {
      Fs[i] = this->noiseModel_->Whiten(Fs[i]);
    }

    // Compute point covariance P = (E^T E + lambda*I)^{-1} with optional damping
    const Matrix3 P = Cameras::PointCov(E, lambda, diagonalDamping);

    // Compute Schur complement and rearrange for shared poses/calibrations
    return schurComplementAndRearrange(Fs, E, P, b);
  }

  /**
   * Linearize to a Hessian factor (with optional LM damping), takes Values
   * @param values Values structure with poses and calibrations
   * @param lambda LM damping parameter (default 0.0)
   * @return Hessian factor with derivatives wrt poses and calibrations
   */
  std::shared_ptr<GaussianFactor> linearizeDamped(
      const Values& values,
      const double lambda = 0.0) const {
    Cameras cameras = this->cameras(values);
    return linearizeDamped(cameras, lambda);
  }

  /**
   * Linearize (no damping) - calls linearizeDamped with lambda=0
   * @param values Values structure with poses and calibrations
   * @return Hessian factor with derivatives wrt poses and calibrations
   */
  std::shared_ptr<GaussianFactor> linearize(const Values& values) const override {
    return linearizeDamped(values, 0.0);
  }

 private:
  /**
   * Rebuild keys_ to maintain [unique poses..., unique calibrations...] order.
   * This ensures keys_ matches the ordering used in schurComplementAndRearrange.
   */
  void rebuildKeys() {
    this->keys_.clear();

    // First, add unique pose keys
    for (const Key& k : poseKeys_) {
      if (std::find(this->keys_.begin(), this->keys_.end(), k) == this->keys_.end()) {
        this->keys_.push_back(k);
      }
    }

    // Then, add unique calibration keys
    for (const Key& k : calibrationKeys_) {
      if (std::find(this->keys_.begin(), this->keys_.end(), k) == this->keys_.end()) {
        this->keys_.push_back(k);
      }
    }
  }

  /**
   * Compute Schur complement and rearrange blocks for shared poses/calibrations.
   *
   * This method mirrors the pattern in CameraSet::SchurComplementAndRearrangeBlocks:
   * 1. First compute the standard Schur complement using Cameras::SchurComplement
   *    (one block per measurement, with full camera dimension DimCam = 6 + DimK)
   * 2. Then rearrange by splitting each camera block into pose/calibration parts
   *    and accumulating contributions for shared poses/calibrations
   *
   * The input Fs blocks are (2 x DimCam) where DimCam = 6 + DimK.
   * Each block contains [F_pose | F_cal] horizontally concatenated.
   */
  std::shared_ptr<GaussianFactor> schurComplementAndRearrange(
      const typename Cameras::FBlocks& Fs,
      const Matrix& E,
      const Matrix3& P,
      const Vector& b) const {

    size_t m = Fs.size();  // number of measurements

    // Step 1: Compute standard Schur complement with one block per measurement
    // This gives us an augmented Hessian with blocks of size DimCam x DimCam
    SymmetricBlockMatrix augmentedHessian =
        Cameras::template SchurComplement<3, DimCam>(Fs, E, P, b);


    // Step 2: Build key-to-slot mappings for output Hessian
    // keys_ is already ordered as [unique poses..., unique calibrations...] by rebuildKeys()
    std::map<Key, size_t> poseKeyToSlot;
    std::map<Key, size_t> calKeyToSlot;
    std::vector<DenseIndex> outputDims;

    // Build slot maps and dimensions from keys_
    // First count unique poses to know where calibrations start
    // std::cerr<< "Building Unique Poses..." << std::endl;
    size_t nrUniquePoses = 0;
    for (const Key& k : poseKeys_) {
      if (poseKeyToSlot.find(k) == poseKeyToSlot.end()) {
        poseKeyToSlot[k] = nrUniquePoses;
        outputDims.push_back(DimPose);
        // std::cerr<< "Pose: " << k << " -> Slot: " << nrUniquePoses << std::endl;
        nrUniquePoses++;
      }
    }

    // Then map calibration keys (they come after poses in keys_)
    // std::cerr<< "Building Unique Calibrations..." << std::endl;
    size_t calSlot = nrUniquePoses;
    for (const Key& k : calibrationKeys_) {
      if (calKeyToSlot.find(k) == calKeyToSlot.end()) {
        calKeyToSlot[k] = calSlot;
        outputDims.push_back(DimK);
        // std::cerr<< "Calibration: " << k << " -> Slot: " << calSlot << std::endl;
        calSlot++;
      }
    }
    size_t nrUniqueKeys = this->keys_.size();
    outputDims.push_back(1);  // For the info vector / constant term

    // Compute total dimension for output
    size_t totalOutputDim = std::accumulate(outputDims.begin(), outputDims.end(), 0);

    // Initialize output augmented Hessian to zero
    // std::cerr<< "Building Output Hessian..." << std::endl;
    SymmetricBlockMatrix outputHessian(outputDims,
        Matrix::Zero(totalOutputDim, totalOutputDim));

    // Step 3: Rearrange blocks from input to output
    // For each pair of measurements (i, j), extract the block from augmentedHessian
    // and split it into pose-pose, pose-cal, cal-pose, cal-cal sub-blocks,
    // then accumulate into the appropriate output slots.

    for (size_t i = 0; i < m; i++) {
      size_t poseSlot_i = poseKeyToSlot[poseKeys_[i]];
      size_t calSlot_i = calKeyToSlot[calibrationKeys_[i]];

      // Get info vector block for measurement i: (DimCam x 1)
      auto gi = augmentedHessian.aboveDiagonalBlock(i, m);

      // Split into pose and cal parts
      auto gi_pose = gi.template topRows<DimPose>();
      auto gi_cal = gi.template bottomRows<DimK>();

      // Accumulate info vectors
      outputHessian.updateOffDiagonalBlock(poseSlot_i, nrUniqueKeys, gi_pose);
      outputHessian.updateOffDiagonalBlock(calSlot_i, nrUniqueKeys, gi_cal);

      // Process Hessian blocks
      for (size_t j = i; j < m; j++) {
        size_t poseSlot_j = poseKeyToSlot[poseKeys_[j]];
        size_t calSlot_j = calKeyToSlot[calibrationKeys_[j]];

        // std::cerr<< "Hessian Block (" << i << ", " << j << "): " << std::endl;

        // Get the (DimCam x DimCam) block from the input Hessian
        Matrix Hij;
        if (i == j) {
          Hij = augmentedHessian.diagonalBlock(i);
        } else {
          Hij = augmentedHessian.aboveDiagonalBlock(i, j);
        }

        // Split the block into 4 sub-blocks:
        // [pose-pose  pose-cal ]   [6 x 6      6 x DimK  ]
        // [cal-pose   cal-cal  ] = [DimK x 6   DimK x DimK]
        auto Hij_pp = Hij.template topLeftCorner<DimPose, DimPose>();
        auto Hij_pc = Hij.template topRightCorner<DimPose, DimK>();
        auto Hij_cp = Hij.template bottomLeftCorner<DimK, DimPose>();
        auto Hij_cc = Hij.template bottomRightCorner<DimK, DimK>();
        // std::cerr<< "Full Block:\n" << Hij << std::endl;

        // std::cerr<< "Pose-Pose:\n" << Hij_pp << std::endl;
        // std::cerr<< "Pose-Cal:\n" << Hij_pc << std::endl;
        // std::cerr<< "Cal-Pose:\n" << Hij_cp << std::endl;
        // std::cerr<< "Cal-Cal:\n" << Hij_cc << std::endl;

        if (i == j) {
          // Diagonal measurement: accumulate to diagonal blocks
          outputHessian.updateDiagonalBlock(poseSlot_i, Hij_pp);
          outputHessian.updateDiagonalBlock(calSlot_i, Hij_cc);

          // Pose-cal cross term: poseSlot < calSlot always (poses ordered before cals)
          if (poseSlot_i != calSlot_i) {
            outputHessian.updateOffDiagonalBlock(poseSlot_i, calSlot_i, Hij_pc);
          } else {
            throw std::runtime_error("Got pose/cal keys that are identitical, but they shouldn't be.");
          }
          // std::cerr<< "Updated Hessian Pose Block:" << outputHessian.block(poseSlot_i, poseSlot_i) << std::endl;
          // std::cerr<< "Updated Hessian Calibration Block:" << outputHessian.block(calSlot_i, calSlot_i) << std::endl;
          // std::cerr<< "Updated Hessian Pose-Cal Cross Term Block:" << outputHessian.block(poseSlot_i, calSlot_i) << std::endl;
        } else {
          // Off-diagonal measurement pair (i < j)
          // Since output is ordered [poses..., cals...]:
          // - poseSlot_i, poseSlot_j are both < nrUniquePoses
          // - calSlot_i, calSlot_j are both >= nrUniquePoses
          // So: poseSlot < calSlot always, but pose-pose and cal-cal pairs
          // can have either ordering depending on which keys are shared.

          // std::cerr<< "Off-diagonal case: i=" << i << ", j=" << j << std::endl;
          // std::cerr<< "poseSlot_i=" << poseSlot_i << ", poseSlot_j=" << poseSlot_j << std::endl;
          // std::cerr<< "calSlot_i=" << calSlot_i << ", calSlot_j=" << calSlot_j << std::endl;

          // Pose_i - Pose_j: both in pose range, could be equal or either order
          if (poseSlot_i == poseSlot_j) { // both measurements share a pose key (unlikely, but not disallowed.)
            outputHessian.updateDiagonalBlock(poseSlot_i, Hij_pp + Hij_pp.transpose());
            // std::cerr<< "Pose-Pose (shared): Updated diagonal block " << poseSlot_i << std::endl;
            // std::cerr<< "Updated Hessian Pose Block:" << outputHessian.block(poseSlot_i, poseSlot_i) << std::endl;
          } else if (poseSlot_i < poseSlot_j) {
            outputHessian.updateOffDiagonalBlock(poseSlot_i, poseSlot_j, Hij_pp);
            // std::cerr<< "Pose-Pose: Updated off-diagonal block (" << poseSlot_i << ", " << poseSlot_j << ")" << std::endl;
            // std::cerr<< "Updated Hessian Pose-Pose Block:" << outputHessian.block(poseSlot_i, poseSlot_j) << std::endl;
          } else {
            outputHessian.updateOffDiagonalBlock(poseSlot_j, poseSlot_i, Hij_pp.transpose());
            // std::cerr<< "Pose-Pose (transposed): Updated off-diagonal block (" << poseSlot_j << ", " << poseSlot_i << ")" << std::endl;
            // std::cerr<< "Updated Hessian Pose-Pose Block:" << outputHessian.block(poseSlot_j, poseSlot_i) << std::endl;
          }

          // Cal_i - Cal_j: both in cal range, could be equal or either order
          if (calSlot_i == calSlot_j) { // two cameras will OFTEN share a calibration...
            outputHessian.updateDiagonalBlock(calSlot_i, Hij_cc + Hij_cc.transpose());
            // std::cerr<< "Cal-Cal (shared): Updated diagonal block " << calSlot_i << std::endl;
            // std::cerr<< "Updated Hessian Cal Block:" << outputHessian.block(calSlot_i, calSlot_i) << std::endl;
          } else if (calSlot_i < calSlot_j) {
            outputHessian.updateOffDiagonalBlock(calSlot_i, calSlot_j, Hij_cc);
            // std::cerr<< "Cal-Cal: Updated off-diagonal block (" << calSlot_i << ", " << calSlot_j << ")" << std::endl;
            // std::cerr<< "Updated Hessian Cal-Cal Block:" << outputHessian.block(calSlot_i, calSlot_j) << std::endl;
          } else {
            outputHessian.updateOffDiagonalBlock(calSlot_j, calSlot_i, Hij_cc.transpose());
            // std::cerr<< "Cal-Cal (transposed): Updated off-diagonal block (" << calSlot_j << ", " << calSlot_i << ")" << std::endl;
            // std::cerr<< "Updated Hessian Cal-Cal Block:" << outputHessian.block(calSlot_j, calSlot_i) << std::endl;
          }

          // Pose_i - Cal_j: poseSlot_i < calSlot_j always (pose range < cal range)
          outputHessian.updateOffDiagonalBlock(poseSlot_i, calSlot_j, Hij_pc);
          // std::cerr<< "Pose_i-Cal_j: Updated off-diagonal block (" << poseSlot_i << ", " << calSlot_j << ")" << std::endl;
          // std::cerr<< "Updated Hessian Pose-Cal Block:" << outputHessian.block(poseSlot_i, calSlot_j) << std::endl;

          // Cal_i - Pose_j: calSlot_i > poseSlot_j always, so transpose
          outputHessian.updateOffDiagonalBlock(poseSlot_j, calSlot_i, Hij_cp.transpose());
          // std::cerr<< "Cal_i-Pose_j (transposed): Updated off-diagonal block (" << poseSlot_j << ", " << calSlot_i << ")" << std::endl;
          // std::cerr<< "Updated Hessian Cal-Pose Block:" << outputHessian.block(poseSlot_j, calSlot_i) << std::endl;
        }
      }
    }

    // Copy constant term
    outputHessian.updateDiagonalBlock(nrUniqueKeys,
        augmentedHessian.diagonalBlock(m));

    return std::make_shared<HessianFactor>(this->keys_, outputHessian);
  }

#if GTSAM_ENABLE_BOOST_SERIALIZATION
  /// Serialization function
  friend class boost::serialization::access;
  template<class ARCHIVE>
  void serialize(ARCHIVE & ar, const unsigned int /*version*/) {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(Base);
    ar & BOOST_SERIALIZATION_NVP(poseKeys_);
    ar & BOOST_SERIALIZATION_NVP(calibrationKeys_);
  }
#endif
};

// traits
template<class CALIBRATION>
struct traits<SmartProjectionPoseCalFactor<CALIBRATION> > :
    public Testable<SmartProjectionPoseCalFactor<CALIBRATION> > {
};

}  // namespace gtsam
