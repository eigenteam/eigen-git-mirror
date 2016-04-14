// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Rasmus Munk Larsen <rmlarsen@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_COST_MODEL_H
#define EIGEN_CXX11_TENSOR_TENSOR_COST_MODEL_H

#if !defined(EIGEN_USE_GPU)
#define EIGEN_USE_COST_MODEL
#endif

namespace Eigen {

/** \class TensorEvaluator
  * \ingroup CXX11_Tensor_Module
  *
  * \brief A cost model used to limit the number of threads used for evaluating
  * tensor expression.
  *
  */

// Class storing the cost of evaluating a tensor expression in terms of the
// estimated number of operand bytes loads, bytes stored, and compute cycles.
class TensorOpCost {
 public:
  // TODO(rmlarsen): Fix the scalar op costs in Eigen proper. Even a simple
  // model based on minimal reciprocal throughput numbers from Intel or
  // Agner Fog's tables would be better than what is there now.
  template <typename ArgType>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE static int MulCost() {
    return internal::functor_traits<
        internal::scalar_product_op<ArgType, ArgType>>::Cost;
  }
  template <typename ArgType>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE static int AddCost() {
    return internal::functor_traits<internal::scalar_sum_op<ArgType>>::Cost;
  }
  template <typename ArgType>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE static int DivCost() {
    return internal::functor_traits<
        internal::scalar_quotient_op<ArgType, ArgType>>::Cost;
  }
  template <typename ArgType>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE static int ModCost() {
    return internal::functor_traits<internal::scalar_mod_op<ArgType>>::Cost;
  }
  template <typename SrcType, typename TargetType>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE static int CastCost() {
    return internal::functor_traits<
        internal::scalar_cast_op<SrcType, TargetType>>::Cost;
  }

  TensorOpCost() : bytes_loaded_(0), bytes_stored_(0), compute_cycles_(0) {}
  TensorOpCost(double bytes_loaded, double bytes_stored, double compute_cycles)
      : bytes_loaded_(bytes_loaded),
        bytes_stored_(bytes_stored),
        compute_cycles_(compute_cycles) {}

  TensorOpCost(double bytes_loaded, double bytes_stored, double compute_cycles,
               bool vectorized, double packet_size)
      : bytes_loaded_(bytes_loaded),
        bytes_stored_(bytes_stored),
        compute_cycles_(vectorized ? compute_cycles / packet_size
                                   : compute_cycles) {
    using std::isfinite;
    eigen_assert(bytes_loaded >= 0 && (isfinite)(bytes_loaded));
    eigen_assert(bytes_stored >= 0 && (isfinite)(bytes_stored));
    eigen_assert(compute_cycles >= 0 && (isfinite)(compute_cycles));
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double bytes_loaded() const {
    return bytes_loaded_;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double bytes_stored() const {
    return bytes_stored_;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double compute_cycles() const {
    return compute_cycles_;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE double total_cost(
      double load_cost, double store_cost, double compute_cost) const {
    return load_cost * bytes_loaded_ + store_cost * bytes_stored_ +
           compute_cost * compute_cycles_;
  }

  // TODO(rmlarsen): Define min in terms of total cost, not elementwise.
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost& cwiseMin(
      const TensorOpCost& rhs) {
    bytes_loaded_ = numext::mini(bytes_loaded_, rhs.bytes_loaded());
    bytes_stored_ = numext::mini(bytes_stored_, rhs.bytes_stored());
    compute_cycles_ = numext::mini(compute_cycles_, rhs.compute_cycles());
    return *this;
  }

  // TODO(rmlarsen): Define max in terms of total cost, not elementwise.
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost& cwiseMax(
      const TensorOpCost& rhs) {
    bytes_loaded_ = numext::maxi(bytes_loaded_, rhs.bytes_loaded());
    bytes_stored_ = numext::maxi(bytes_stored_, rhs.bytes_stored());
    compute_cycles_ = numext::maxi(compute_cycles_, rhs.compute_cycles());
    return *this;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost& operator+=(
      const TensorOpCost& rhs) {
    bytes_loaded_ += rhs.bytes_loaded();
    bytes_stored_ += rhs.bytes_stored();
    compute_cycles_ += rhs.compute_cycles();
    return *this;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE TensorOpCost& operator*=(double rhs) {
    bytes_loaded_ *= rhs;
    bytes_stored_ *= rhs;
    compute_cycles_ *= rhs;
    return *this;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE friend TensorOpCost operator+(
      TensorOpCost lhs, const TensorOpCost& rhs) {
    lhs += rhs;
    return lhs;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE friend TensorOpCost operator*(
      TensorOpCost lhs, double rhs) {
    lhs *= rhs;
    return lhs;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE friend TensorOpCost operator*(
      double lhs, TensorOpCost rhs) {
    rhs *= lhs;
    return rhs;
  }

  friend std::ostream& operator<<(std::ostream& os, const TensorOpCost& tc) {
    return os << "[bytes_loaded = " << tc.bytes_loaded()
              << ", bytes_stored = " << tc.bytes_stored()
              << ", compute_cycles = " << tc.compute_cycles() << "]";
  }

 private:
  double bytes_loaded_;
  double bytes_stored_;
  double compute_cycles_;
};

// TODO(rmlarsen): Implement a policy that chooses an "optimal" number of theads
// in [1:max_threads] instead of just switching multi-threading off for small
// work units.
template <typename Device>
class TensorCostModel {
 public:
  // Costs in device cycles.
  static const int kLoadCycles = 3;
  static const int kStoreCycles = 3;
  // Scaling from Eigen compute cost to device cycles.
  static const int kDeviceCyclesPerComputeCycle = 1;

  // Implements a simple "binary" policy: Return 1 if total cost is below
  // kMinWorkToParallelize and max_threads otherwise.
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE static int numThreads(
      double output_size, const TensorOpCost& cost_per_coeff, int max_threads) {
    // Compute total cost C in device cycles.
    const double total_cost =
        output_size *
        cost_per_coeff.total_cost(kLoadCycles, kStoreCycles,
                                  kDeviceCyclesPerComputeCycle);
    // Smallest work unit to parallelize.
    const double kMinParallelCost = 1e6;
    return total_cost < kMinParallelCost ? 1 : max_threads;
  }
};

}  // namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_COST_MODEL_H
