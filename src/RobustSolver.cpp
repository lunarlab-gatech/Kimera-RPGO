/*
Robust solver class
author: Yun Chang, Luca Carlone
*/

#include "KimeraRPGO/RobustSolver.h"

#include <gtsam/inference/inferenceExceptions.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/GncOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/slam/dataset.h>

#include <chrono>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "KimeraRPGO/Logger.h"
#include "KimeraRPGO/outlier/Pcm.h"
#include "KimeraRPGO/utils/TypeUtils.h"

namespace KimeraRPGO {

typedef std::pair<gtsam::NonlinearFactorGraph, gtsam::Values> GraphAndValues;

RobustSolver::RobustSolver(const RobustSolverParams& params)
    : GenericSolver(params.solver, params.specialSymbols),
      gnc_weights_(),
      gnc_num_inliers_(0),
      latest_num_lc_(0),
      params_(params) {
  switch (params.outlierRemovalMethod) {
    case OutlierRemovalMethod::NONE: {
      outlier_removal_ =
          nullptr;  // only returns optimize true or optimize false
    } break;
    case OutlierRemovalMethod::PCM2D: {
      outlier_removal_ =
          KimeraRPGO::make_unique<Pcm2D>(params.pcm_params,
                                         params.multirobot_align_method,
                                         params.multirobot_align_gnc_prob,
                                         params.specialSymbols);
    } break;
    case OutlierRemovalMethod::PCM3D: {
      outlier_removal_ =
          KimeraRPGO::make_unique<Pcm3D>(params.pcm_params,
                                         params.multirobot_align_method,
                                         params.multirobot_align_gnc_prob,
                                         params.specialSymbols);
    } break;
    case OutlierRemovalMethod::PCM_Simple2D: {
      outlier_removal_ =
          KimeraRPGO::make_unique<PcmSimple2D>(params.pcm_params,
                                               params.multirobot_align_method,
                                               params.multirobot_align_gnc_prob,
                                               params.specialSymbols);
    } break;
    case OutlierRemovalMethod::PCM_Simple3D: {
      outlier_removal_ =
          KimeraRPGO::make_unique<PcmSimple3D>(params.pcm_params,
                                               params.multirobot_align_method,
                                               params.multirobot_align_gnc_prob,
                                               params.specialSymbols);
    } break;
    default: {
      log<WARNING>("Undefined outlier removal method");
      exit(EXIT_FAILURE);
    }
  }

  // toggle verbosity
  switch (params.verbosity) {
    case Verbosity::UPDATE: {
      if (outlier_removal_) outlier_removal_->setQuiet();
    } break;
    case Verbosity::QUIET: {
      if (outlier_removal_)
        outlier_removal_->setQuiet();  // set outlier removal quiet
      setQuiet();                      // set solver quiet
    } break;
    case Verbosity::VERBOSE: {
      log<INFO>("Starting RobustSolver.");
    } break;
    default: {
      log<WARNING>("Unrecognized verbosity. Automatically setting to UPDATE. ");
    }
  }

  // set log output
  if (params.log_output) {
    if (outlier_removal_) outlier_removal_->logOutput(params.log_folder);
    log_ = true;
    log_folder_ = params.log_folder;
    std::string filename = log_folder_ + "/rpgo_status.csv";
    std::ofstream outfile;
    outfile.open(filename);
    outfile << "graph-size,spin-time(mu-s),num-lc,num-inliers\n";
    outfile.close();
  }
}

void RobustSolver::getGncKnownInliers(InlierVectorType* known_inliers) {
  size_t num_odom_factors = outlier_removal_->getNumOdomFactors();
  size_t num_special_factors = outlier_removal_->getNumSpecialFactors();
  // Set odometry and special factors as known inliers
  known_inliers->resize(num_odom_factors + num_special_factors);
  std::iota(std::begin(*known_inliers),
            std::begin(*known_inliers) + num_odom_factors + num_special_factors,
            0);
}

void RobustSolver::optimize() {
  gtsam::Values result;
  gtsam::Values full_values = values_;
  gtsam::NonlinearFactorGraph full_nfg = nfg_;
  // Merge in temporary values and factors
  full_values.insert(temp_values_);
  full_nfg.add(temp_nfg_);

  if (solver_type_ == Solver::LM) {
    gtsam::LevenbergMarquardtParams lmParams;
    lmParams.diagonalDamping = params_.lm_diagonal_damping;
    if (debug_) {
      lmParams.setVerbosityLM("SUMMARY");
      log<INFO>("Running LM");
    }
    if (params_.use_gnc_ && outlier_removal_ &&
        !(params_.gnc_params.fix_prev_inliers_ &&
          outlier_removal_->getNumLC() == latest_num_lc_)) {
      gtsam::GncParams<gtsam::LevenbergMarquardtParams> gncParams(lmParams);
      InlierVectorType known_inlier_factor_indices;
      getGncKnownInliers(&known_inlier_factor_indices);
      gncParams.setKnownInliers(known_inlier_factor_indices);
      gncParams.setMaxIterations(params_.gnc_params.max_iterations_);
      gncParams.setMuStep(params_.gnc_params.mu_step_);
      gncParams.setRelativeCostTol(params_.gnc_params.relative_cost_tol_);
      gncParams.setWeightsTol(params_.gnc_params.weights_tol_);
      // Create GNC optimizer
      gtsam::GncOptimizer<gtsam::GncParams<gtsam::LevenbergMarquardtParams> >
          gnc_optimizer(full_nfg, full_values, gncParams);
      if (params_.gnc_params.bias_odom_) {
        // Set initial weights to bias odom
        gtsam::Vector init_weights = Eigen::VectorXd::Zero(full_nfg.size());
        for (const auto& ind : known_inlier_factor_indices) {
          init_weights(ind) = 1;
        }
        gnc_optimizer.setWeights(init_weights);
      }
      switch (params_.gnc_params.gnc_threshold_mode_) {
        case (GncParams::GncThresholdMode::COST):
          gnc_optimizer.setInlierCostThresholds(
              params_.gnc_params.gnc_inlier_threshold_);
          break;
        case (GncParams::GncThresholdMode::PROBABILITY):
          gnc_optimizer.setInlierCostThresholdsAtProbability(
              params_.gnc_params.gnc_inlier_threshold_);
          break;
        default:
          log<WARNING>("Unsupported GNC threshold mode. ");
      }
      // Optimize and get weights
      auto opt_start_t = std::chrono::high_resolution_clock::now();
      result = gnc_optimizer.optimize();
      gtsam::Vector gnc_all_weights = gnc_optimizer.getWeights();
      gnc_weights_ = gnc_all_weights.head(nfg_.size());
      gnc_temp_weights_ = gnc_all_weights.tail(temp_nfg_.size());
      gnc_num_inliers_ = static_cast<size_t>(gnc_weights_.sum()) -
                         known_inlier_factor_indices.size();
      auto opt_stop_t = std::chrono::high_resolution_clock::now();
      auto opt_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
          opt_stop_t - opt_start_t);
      if (debug_) {
        log<INFO>() << "GNC optimize took " << opt_duration.count()
                    << " milliseconds. " << outlier_removal_->getNumLCInliers()
                    << " loop closures with " << gnc_num_inliers_
                    << " inliers.";
      }
    } else {
      auto opt_start_t = std::chrono::high_resolution_clock::now();
      if (params_.gnc_params.fix_prev_inliers_) {
        // TODO(yun) clean up
        // remove gnc inliers from previous iterations
        size_t prev_k = gnc_weights_.size() - latest_num_lc_;
        size_t k = outlier_removal_->getNumOdomFactors() +
                   outlier_removal_->getNumSpecialFactors();
        full_nfg = gtsam::NonlinearFactorGraph(nfg_.begin(), nfg_.begin() + k);
        for (size_t i = 0; i < latest_num_lc_; i++) {
          if (gnc_weights_(prev_k + i) > 0.5) {
            full_nfg.add(nfg_.at(k + i));
          }
        }
        full_nfg.add(temp_nfg_);
      }
      result =
          gtsam::LevenbergMarquardtOptimizer(full_nfg, full_values, lmParams)
              .optimize();
      // TODO(Yun) fix: Note here there's an implicit assumption that temp_factors are inliers
      // Update GNC weights
      gtsam::Vector gnc_weights_lc = gnc_weights_.tail(latest_num_lc_);
      // All except loop closures implicitly assumed to be inliers
      gnc_weights_ = gtsam::Vector::Ones(nfg_.size());
      // Directly copy over weights for loop closures from last iteration
      gnc_weights_.tail(latest_num_lc_) = gnc_weights_lc;
      // Implicitly assume all temp factors are inliers
      gnc_temp_weights_ = gtsam::Vector::Ones(temp_nfg_.size());
      auto opt_stop_t = std::chrono::high_resolution_clock::now();
      auto opt_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
          opt_stop_t - opt_start_t);
      if (debug_) {
        log<INFO>() << "Optimize took " << opt_duration.count()
                    << " milliseconds. " << outlier_removal_->getNumLC()
                    << " loop closures with " << getNumLCInliers()
                    << " inliers.";
      }
    }
  } else if (solver_type_ == Solver::GN) {
    gtsam::GaussNewtonParams gnParams;
    if (debug_) {
      gnParams.setVerbosity("ERROR");
      log<INFO>("Running GN");
    }
    if (params_.use_gnc_ && outlier_removal_ &&
        !(params_.gnc_params.fix_prev_inliers_ &&
          outlier_removal_->getNumLC() == latest_num_lc_)) {
      gtsam::GncParams<gtsam::GaussNewtonParams> gncParams(gnParams);
      InlierVectorType known_inlier_factor_indices;
      getGncKnownInliers(&known_inlier_factor_indices);
      gncParams.setKnownInliers(known_inlier_factor_indices);
      gncParams.setMaxIterations(params_.gnc_params.max_iterations_);
      gncParams.setMuStep(params_.gnc_params.mu_step_);
      gncParams.setRelativeCostTol(params_.gnc_params.relative_cost_tol_);
      gncParams.setWeightsTol(params_.gnc_params.weights_tol_);
      // Create GNC optimizer
      gtsam::GncOptimizer<gtsam::GncParams<gtsam::GaussNewtonParams> >
          gnc_optimizer(full_nfg, full_values, gncParams);
      if (params_.gnc_params.bias_odom_) {
        // Set initial weights to bias odom
        gtsam::Vector init_weights = Eigen::VectorXd::Zero(full_nfg.size());
        for (const auto& ind : known_inlier_factor_indices) {
          init_weights(ind) = 1;
        }
        gnc_optimizer.setWeights(init_weights);
      }
      switch (params_.gnc_params.gnc_threshold_mode_) {
        case (GncParams::GncThresholdMode::COST):
          gnc_optimizer.setInlierCostThresholds(
              params_.gnc_params.gnc_inlier_threshold_);
          break;
        case (GncParams::GncThresholdMode::PROBABILITY):
          gnc_optimizer.setInlierCostThresholdsAtProbability(
              params_.gnc_params.gnc_inlier_threshold_);
          break;
        default:
          log<WARNING>("Unsupported GNC threshold mode. ");
      }
      // Optimize and get weights
      auto opt_start_t = std::chrono::high_resolution_clock::now();
      result = gnc_optimizer.optimize();
      gtsam::Vector gnc_all_weights = gnc_optimizer.getWeights();
      gnc_weights_ = gnc_all_weights.head(nfg_.size());
      gnc_temp_weights_ = gnc_all_weights.tail(temp_nfg_.size());
      gnc_num_inliers_ = static_cast<size_t>(gnc_weights_.sum()) -
                         known_inlier_factor_indices.size();
      auto opt_stop_t = std::chrono::high_resolution_clock::now();
      auto opt_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
          opt_stop_t - opt_start_t);
      if (debug_) {
        log<INFO>() << "GNC optimize took " << opt_duration.count()
                    << " milliseconds. " << outlier_removal_->getNumLCInliers()
                    << " loop closures with " << gnc_num_inliers_
                    << " inliers.";
      }
    } else {
      // TODO(Yun) remove this duplicate (from LM)
      if (params_.gnc_params.fix_prev_inliers_) {
        size_t prev_k = gnc_weights_.size() - latest_num_lc_;
        size_t k = outlier_removal_->getNumOdomFactors() +
                   outlier_removal_->getNumSpecialFactors();
        full_nfg = gtsam::NonlinearFactorGraph(nfg_.begin(), nfg_.begin() + k);
        for (size_t i = 0; i < latest_num_lc_; i++) {
          if (gnc_weights_(prev_k + i) > 0.5) {
            full_nfg.add(nfg_.at(k + i));
          }
        }
        full_nfg.add(temp_nfg_);
      }
      result = gtsam::GaussNewtonOptimizer(full_nfg, full_values, gnParams)
                   .optimize();
      gtsam::Vector gnc_weights_lc = gnc_weights_.tail(latest_num_lc_);
      gnc_weights_ = gtsam::Vector::Ones(nfg_.size());
      gnc_weights_.tail(latest_num_lc_) = gnc_weights_lc;
      gnc_temp_weights_ = gtsam::Vector::Ones(temp_nfg_.size());
    }

  } else {
    log<WARNING>("Unsupported Solver");
    exit(EXIT_FAILURE);
  }
  updateValues(result);
  if (outlier_removal_) {
    latest_num_lc_ = outlier_removal_->getNumLC();
  }
}

void RobustSolver::forceUpdate(const gtsam::NonlinearFactorGraph& nfg,
                               const gtsam::Values& values) {
  // Start timer
  auto start = std::chrono::high_resolution_clock::now();
  if (outlier_removal_) {
    outlier_removal_->removeOutliers(nfg, values, &nfg_, &values_);
  } else {
    addAndCheckIfOptimize(nfg, values);
  }
  // optimize
  optimize();

  // Stop timer and save
  auto stop = std::chrono::high_resolution_clock::now();
  auto spin_time =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  // Log status
  if (log_) {
    std::string filename = log_folder_ + "/rpgo_status.csv";
    std::ofstream outfile;
    outfile.open(filename, std::ofstream::out | std::ofstream::app);
    outfile << nfg_.size() << "," << spin_time.count() << "," << getNumLC()
            << "," << getNumLCInliers() << std::endl;
    outfile.close();
    saveData(log_folder_);
  }
}

void RobustSolver::update(const gtsam::NonlinearFactorGraph& factors,
                          const gtsam::Values& values,
                          bool optimize_graph) {
  // Start timer
  auto start = std::chrono::high_resolution_clock::now();

  bool do_optimize;
  if (outlier_removal_) {
    do_optimize =
        outlier_removal_->removeOutliers(factors, values, &nfg_, &values_);
  } else {
    do_optimize = addAndCheckIfOptimize(factors, values);
  }

  if (do_optimize && optimize_graph) optimize();  // optimize once after loading

  // Stop timer and save
  auto stop = std::chrono::high_resolution_clock::now();
  auto spin_time =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  // Log status
  if (log_ && optimize_graph) {
    std::string filename = log_folder_ + "/rpgo_status.csv";
    std::ofstream outfile;
    outfile.open(filename, std::ofstream::out | std::ofstream::app);
    outfile << nfg_.size() << "," << spin_time.count() << "," << getNumLC()
            << "," << getNumLCInliers() << std::endl;
    outfile.close();
    saveData(log_folder_);
  }
  return;
}

void RobustSolver::removePriorFactorsWithPrefix(const char& prefix,
                                                bool optimize_graph) {
  if (outlier_removal_) {
    // removing loop closure so values should not change
    outlier_removal_->removePriorFactorsWithPrefix(prefix, &nfg_);
  } else {
    removePriorsWithPrefix(prefix);
  }
  if (optimize_graph) optimize();
  return;
}

EdgePtr RobustSolver::removeLastLoopClosure(char prefix_1, char prefix_2) {
  ObservationId id(prefix_1, prefix_2);
  EdgePtr removed_edge;
  if (outlier_removal_) {
    // removing loop closure so values should not change
    removed_edge = outlier_removal_->removeLastLoopClosure(id, &nfg_);
  } else {
    removed_edge = removeLastFactor();
  }

  optimize();
  return removed_edge;
}

EdgePtr RobustSolver::removeLastLoopClosure() {
  EdgePtr removed_edge;
  if (outlier_removal_) {
    // removing loop closure so values should not change
    removed_edge = outlier_removal_->removeLastLoopClosure(&nfg_);
  } else {
    removed_edge = removeLastFactor();
  }

  optimize();
  return removed_edge;
}

void RobustSolver::ignorePrefix(char prefix) {
  if (outlier_removal_) {
    outlier_removal_->ignoreLoopClosureWithPrefix(prefix, &nfg_);
  } else {
    log<WARNING>(
        "'ignorePrefix' currently not implemented for no outlier rejection "
        "case");
  }

  optimize();
  return;
}

void RobustSolver::revivePrefix(char prefix) {
  if (outlier_removal_) {
    outlier_removal_->reviveLoopClosureWithPrefix(prefix, &nfg_);
  } else {
    log<WARNING>(
        "'revivePrefix' and 'ignorePrefix' currently not implemented for no "
        "outlier rejection case");
  }

  optimize();
  return;
}

std::vector<char> RobustSolver::getIgnoredPrefixes() {
  if (outlier_removal_) {
    return outlier_removal_->getIgnoredPrefixes();
  } else {
    log<WARNING>(
        "'revivePrefix' and 'ignorePrefix' currently not implemented for no "
        "outlier rejection case");
  }
  std::vector<char> empty;
  return empty;
}

void RobustSolver::saveData(std::string folder_path) const {
  std::string g2o_file_path = folder_path + "/result.g2o";
  KimeraRPGO::writeG2o(nfg_, values_, g2o_file_path);
  if (outlier_removal_) {
    outlier_removal_->saveData(folder_path);
  }

  std::string gnc_weights_file = folder_path + "/gnc_weights.csv";
  if (params_.use_gnc_) {
    const static Eigen::IOFormat CSVFormat(
        Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    std::ofstream gnc_file(gnc_weights_file);
    if (gnc_file.is_open()) {
      gnc_file << gnc_weights_.format(CSVFormat);
      gnc_file.close();
    }

    // Save inlier loop closures to separate g2o files per robot and robot pair.
    // Files are named by the gtsam character prefix (a, b, c, ...) which maps
    // to data_params.runs order in the calling application.

    // Collect all robot character prefixes present in the solution
    std::set<char> robot_char_set;
    for (gtsam::Key key : values_.keys()) {
      char c = gtsam::Symbol(key).chr();
      if (!isSpecialSymbol(c)) robot_char_set.insert(c);
    }
    std::vector<char> robot_chars(robot_char_set.begin(), robot_char_set.end());

    // Open one intra-robot file per robot (may remain empty)
    std::map<char, std::unique_ptr<std::ofstream>> intra_streams;
    for (char c : robot_chars) {
      intra_streams[c] = std::unique_ptr<std::ofstream>(
          new std::ofstream(folder_path + "/inlier_lc_intra_" + std::string(1, c) + ".g2o"));
    }

    // Open one inter-robot file per ordered pair (may remain empty)
    std::map<std::pair<char, char>, std::unique_ptr<std::ofstream>> inter_streams;
    for (size_t ri = 0; ri < robot_chars.size(); ri++) {
      for (size_t rj = ri + 1; rj < robot_chars.size(); rj++) {
        char lo = robot_chars[ri], hi = robot_chars[rj];
        inter_streams[{lo, hi}] = std::unique_ptr<std::ofstream>(
            new std::ofstream(folder_path + "/inlier_lc_inter_" +
                              std::string(1, lo) + "_" + std::string(1, hi) + ".g2o"));
      }
    }

    // nfg_ is ordered: [nfg_odom_ | nfg_special_ | LC consistent_factors...]
    // as established by Pcm::buildGraphToOptimize() (Pcm.h:978-993).
    // The same boundary is used at lines 190-191 and 283-284.
    // Skip all non-LC factors by index instead of using a key-adjacency
    // heuristic, so that real LCs remapped to adjacent keyframe indices
    // are not silently dropped.
    // outlier_removal_ must be non-null when use_gnc_ is true — same
    // assumption already made at line 109 which dereferences it directly.
    assert(outlier_removal_);
    size_t lc_start_idx = outlier_removal_->getNumOdomFactors() +
                          outlier_removal_->getNumSpecialFactors();

    for (size_t i = 0; i < nfg_.size() && i < static_cast<size_t>(gnc_weights_.size()); i++) {
      if (i < lc_start_idx) continue;  // skip odometry and special factors

      const auto& factor_ = nfg_[i];
      auto factor3D = factor_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(factor_);
      if (!factor3D) continue;

      gtsam::Key key1 = factor3D->key1();
      gtsam::Key key2 = factor3D->key2();
      char c1 = gtsam::Symbol(key1).chr();
      char c2 = gtsam::Symbol(key2).chr();

      // Skip factors involving special symbols (e.g., landmarks)
      if (isSpecialSymbol(c1) || isSpecialSymbol(c2)) continue;

      // Select the appropriate output stream
      std::ofstream* out_stream = nullptr;
      if (c1 == c2) {
        auto it = intra_streams.find(c1);
        if (it == intra_streams.end()) {
          throw std::runtime_error(
              std::string("No intra-robot stream for prefix: ") + c1);
        }
        out_stream = it->second.get();
      } else {
        char lo = std::min(c1, c2), hi = std::max(c1, c2);
        auto it = inter_streams.find({lo, hi});
        if (it == inter_streams.end()) {
          throw std::runtime_error(
              std::string("No inter-robot stream for pair: ") + lo + "_" + hi);
        }
        out_stream = it->second.get();
      }

      // Write inlier loop closure
      if (out_stream && out_stream->is_open() && gnc_weights_(i) > 0.5) {
        gtsam::SharedNoiseModel model = factor3D->noiseModel();
        auto gaussianModel = factor_pointer_cast<gtsam::noiseModel::Gaussian>(model);
        if (gaussianModel) {
          Eigen::Matrix<double, 6, 6> Info = gaussianModel->R().transpose() * gaussianModel->R();
          const gtsam::Pose3 pose3D = factor3D->measured();
          const gtsam::Point3 p = pose3D.translation();
          const auto q = pose3D.rotation().toQuaternion();
          *out_stream << "# LC:" << std::endl;
          *out_stream << "EDGE_SE3:QUAT " << key1 << " " << key2
                      << " " << p.x() << " " << p.y() << " " << p.z() << " " << q.x()
                      << " " << q.y() << " " << q.z() << " " << q.w();
          Eigen::Matrix<double, 6, 6> InfoG2o = Eigen::MatrixXd::Identity(6, 6);
          InfoG2o.block<3, 3>(0, 0) = Info.block<3, 3>(3, 3);
          InfoG2o.block<3, 3>(3, 3) = Info.block<3, 3>(0, 0);
          InfoG2o.block<3, 3>(0, 3) = Info.block<3, 3>(0, 3);
          InfoG2o.block<3, 3>(3, 0) = Info.block<3, 3>(3, 0);
          for (size_t ii = 0; ii < 6; ii++) {
            for (size_t jj = ii; jj < 6; jj++) {
              *out_stream << " " << InfoG2o(ii, jj);
            }
          }
          *out_stream << std::endl;
        }
      }
    }
    // Streams are closed when unique_ptrs go out of scope
  }
}

}  // namespace KimeraRPGO
