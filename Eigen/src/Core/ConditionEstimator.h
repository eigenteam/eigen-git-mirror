// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Rasmus Munk Larsen (rmlarsen@google.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CONDITIONESTIMATOR_H
#define EIGEN_CONDITIONESTIMATOR_H

namespace Eigen {

namespace internal {
template <typename Decomposition, bool IsComplex>
struct EstimateInverseMatrixL1NormImpl {};
}  // namespace internal

template <typename Decomposition>
class ConditionEstimator {
 public:
  typedef typename Decomposition::MatrixType MatrixType;
  typedef typename internal::traits<MatrixType>::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef typename internal::plain_col_type<MatrixType>::type Vector;

  /** \class ConditionEstimator
    * \ingroup Core_Module
    *
    * \brief Condition number estimator.
    *
    * Computing a decomposition of a dense matrix takes O(n^3) operations, while
    * this method estimates the condition number quickly and reliably in O(n^2)
    * operations.
    *
    * \returns an estimate of the reciprocal condition number
    * (1 / (||matrix||_1 * ||inv(matrix)||_1)) of matrix, given the matrix and
    * its decomposition. Supports the following decompositions: FullPivLU,
    * PartialPivLU.
    *
    * \sa FullPivLU, PartialPivLU.
    */
  static RealScalar rcond(const MatrixType& matrix, const Decomposition& dec) {
    eigen_assert(matrix.rows() == dec.rows());
    eigen_assert(matrix.cols() == dec.cols());
    eigen_assert(matrix.rows() == matrix.cols());
    if (dec.rows() == 0) {
      return RealScalar(1);
    }
    return rcond(MatrixL1Norm(matrix), dec);
  }

  /** \class ConditionEstimator
    * \ingroup Core_Module
    *
    * \brief Condition number estimator.
    *
    * Computing a decomposition of a dense matrix takes O(n^3) operations, while
    * this method estimates the condition number quickly and reliably in O(n^2)
    * operations.
    *
    * \returns an estimate of the reciprocal condition number
    * (1 / (||matrix||_1 * ||inv(matrix)||_1)) of matrix, given ||matrix||_1 and
    * its decomposition. Supports the following decompositions: FullPivLU,
    * PartialPivLU.
    *
    * \sa FullPivLU, PartialPivLU.
    */
  static RealScalar rcond(RealScalar matrix_norm, const Decomposition& dec) {
    eigen_assert(dec.rows() == dec.cols());
    if (dec.rows() == 0) {
      return 1;
    }
    if (matrix_norm == 0) {
      return 0;
    }
    const RealScalar inverse_matrix_norm = EstimateInverseMatrixL1Norm(dec);
    return inverse_matrix_norm == 0 ? 0
                                    : (1 / inverse_matrix_norm) / matrix_norm;
  }

  /**
   * \returns an estimate of ||inv(matrix)||_1 given a decomposition of
   * matrix that implements .solve() and .adjoint().solve() methods.
   *
   * The method implements Algorithms 4.1 and 5.1 from
   *   http://www.maths.manchester.ac.uk/~higham/narep/narep135.pdf
   * which also forms the basis for the condition number estimators in
   * LAPACK. Since at most 10 calls to the solve method of dec are
   * performed, the total cost is O(dims^2), as opposed to O(dims^3)
   * needed to compute the inverse matrix explicitly.
   *
   * The most common usage is in estimating the condition number
   * ||matrix||_1 * ||inv(matrix)||_1. The first term ||matrix||_1 can be
   * computed directly in O(n^2) operations.
   */
  static RealScalar EstimateInverseMatrixL1Norm(const Decomposition& dec) {
    eigen_assert(dec.rows() == dec.cols());
    if (dec.rows() == 0) {
      return 0;
    }
    return internal::EstimateInverseMatrixL1NormImpl<
        Decomposition, NumTraits<Scalar>::IsComplex>::compute(dec);
  }

  /**
   * \returns the induced matrix l1-norm
   * ||matrix||_1 = sup ||matrix * v||_1 / ||v||_1, which is equal to
   * the greatest absolute column sum.
   */
  inline static Scalar MatrixL1Norm(const MatrixType& matrix) {
    return matrix.cwiseAbs().colwise().sum().maxCoeff();
  }
};

namespace internal {

// Partial specialization for real matrices.
template <typename Decomposition>
struct EstimateInverseMatrixL1NormImpl<Decomposition, 0> {
  typedef typename Decomposition::MatrixType MatrixType;
  typedef typename internal::traits<MatrixType>::Scalar Scalar;
  typedef typename internal::plain_col_type<MatrixType>::type Vector;

  // Shorthand for vector L1 norm in Eigen.
  inline static Scalar VectorL1Norm(const Vector& v) {
    return v.template lpNorm<1>();
  }

  static inline Scalar compute(const Decomposition& dec) {
    const int n = dec.rows();
    const Vector plus = Vector::Ones(n);
    Vector v = plus / n;
    v = dec.solve(v);
    Scalar lower_bound = VectorL1Norm(v);
    if (n == 1) {
      return lower_bound;
    }
    // lower_bound is a lower bound on
    //   ||inv(matrix)||_1  = sup_v ||inv(matrix) v||_1 / ||v||_1
    // and is the objective maximized by the ("super-") gradient ascent
    // algorithm.
    // Basic idea: We know that the optimum is achieved at one of the simplices
    // v = e_i, so in each iteration we follow a super-gradient to move towards
    // the optimal one.
    Scalar old_lower_bound = lower_bound;
    const Vector minus = -Vector::Ones(n);
    Vector sign_vector = (v.cwiseAbs().array() == 0).select(plus, minus);
    Vector old_sign_vector = sign_vector;
    int v_max_abs_index = -1;
    int old_v_max_abs_index = v_max_abs_index;
    for (int k = 0; k < 4; ++k) {
      // argmax |inv(matrix)^T * sign_vector|
      v = dec.adjoint().solve(sign_vector);
      v.cwiseAbs().maxCoeff(&v_max_abs_index);
      if (v_max_abs_index == old_v_max_abs_index) {
        // Break if the solution stagnated.
        break;
      }
      // Move to the new simplex e_j, where j = v_max_abs_index.
      v.setZero();
      v[v_max_abs_index] = 1;
      v = dec.solve(v);  // v = inv(matrix) * e_j.
      lower_bound = VectorL1Norm(v);
      if (lower_bound <= old_lower_bound) {
        // Break if the gradient step did not increase the lower_bound.
        break;
      }
      sign_vector = (v.array() < 0).select(plus, minus);
      if (sign_vector == old_sign_vector) {
        // Break if the solution stagnated.
        break;
      }
      old_sign_vector = sign_vector;
      old_v_max_abs_index = v_max_abs_index;
      old_lower_bound = lower_bound;
    }
    // The following calculates an independent estimate of ||matrix||_1 by
    // multiplying matrix by a vector with entries of slowly increasing
    // magnitude and alternating sign:
    //   v_i = (-1)^{i} (1 + (i / (dim-1))), i = 0,...,dim-1.
    // This improvement to Hager's algorithm above is due to Higham. It was
    // added to make the algorithm more robust in certain corner cases where
    // large elements in the matrix might otherwise escape detection due to
    // exact cancellation (especially when op and op_adjoint correspond to a
    // sequence of backsubstitutions and permutations), which could cause
    // Hager's algorithm to vastly underestimate ||matrix||_1.
    Scalar alternating_sign = 1;
    for (int i = 0; i < n; ++i) {
      v[i] = alternating_sign * static_cast<Scalar>(1) +
             (static_cast<Scalar>(i) / (static_cast<Scalar>(n - 1)));
      alternating_sign = -alternating_sign;
    }
    v = dec.solve(v);
    const Scalar alternate_lower_bound =
        (2 * VectorL1Norm(v)) / (3 * static_cast<Scalar>(n));
    return numext::maxi(lower_bound, alternate_lower_bound);
  }
};

// Partial specialization for complex matrices.
template <typename Decomposition>
struct EstimateInverseMatrixL1NormImpl<Decomposition, 1> {
  typedef typename Decomposition::MatrixType MatrixType;
  typedef typename internal::traits<MatrixType>::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef typename internal::plain_col_type<MatrixType>::type Vector;
  typedef typename internal::plain_col_type<MatrixType, RealScalar>::type
      RealVector;

  // Shorthand for vector L1 norm in Eigen.
  inline static RealScalar VectorL1Norm(const Vector& v) {
    return v.template lpNorm<1>();
  }

  static inline RealScalar compute(const Decomposition& dec) {
    const int n = dec.rows();
    const Vector ones = Vector::Ones(n);
    Vector v = ones / n;
    v = dec.solve(v);
    RealScalar lower_bound = VectorL1Norm(v);
    if (n == 1) {
      return lower_bound;
    }
    // lower_bound is a lower bound on
    //   ||inv(matrix)||_1  = sup_v ||inv(matrix) v||_1 / ||v||_1
    // and is the objective maximized by the ("super-") gradient ascent
    // algorithm.
    // Basic idea: We know that the optimum is achieved at one of the simplices
    // v = e_i, so in each iteration we follow a super-gradient to move towards
    // the optimal one.
    RealScalar old_lower_bound = lower_bound;
    int v_max_abs_index = -1;
    int old_v_max_abs_index = v_max_abs_index;
    for (int k = 0; k < 4; ++k) {
      // argmax |inv(matrix)^* * sign_vector|
      RealVector abs_v = v.cwiseAbs();
      const Vector psi =
          (abs_v.array() == 0).select(v.cwiseQuotient(abs_v), ones);
      v = dec.adjoint().solve(psi);
      const RealVector z = v.real();
      z.cwiseAbs().maxCoeff(&v_max_abs_index);
      if (v_max_abs_index == old_v_max_abs_index) {
        // Break if the solution stagnated.
        break;
      }
      // Move to the new simplex e_j, where j = v_max_abs_index.
      v.setZero();
      v[v_max_abs_index] = 1;
      v = dec.solve(v);  // v = inv(matrix) * e_j.
      lower_bound = VectorL1Norm(v);
      if (lower_bound <= old_lower_bound) {
        // Break if the gradient step did not increase the lower_bound.
        break;
      }
      old_v_max_abs_index = v_max_abs_index;
      old_lower_bound = lower_bound;
    }
    // The following calculates an independent estimate of ||matrix||_1 by
    // multiplying matrix by a vector with entries of slowly increasing
    // magnitude and alternating sign:
    //   v_i = (-1)^{i} (1 + (i / (dim-1))), i = 0,...,dim-1.
    // This improvement to Hager's algorithm above is due to Higham. It was
    // added to make the algorithm more robust in certain corner cases where
    // large elements in the matrix might otherwise escape detection due to
    // exact cancellation (especially when op and op_adjoint correspond to a
    // sequence of backsubstitutions and permutations), which could cause
    // Hager's algorithm to vastly underestimate ||matrix||_1.
    RealScalar alternating_sign = 1;
    for (int i = 0; i < n; ++i) {
      v[i] = alternating_sign * static_cast<RealScalar>(1) +
             (static_cast<RealScalar>(i) / (static_cast<RealScalar>(n - 1)));
      alternating_sign = -alternating_sign;
    }
    v = dec.solve(v);
    const RealScalar alternate_lower_bound =
        (2 * VectorL1Norm(v)) / (3 * static_cast<RealScalar>(n));
    return numext::maxi(lower_bound, alternate_lower_bound);
  }
};

}  // namespace internal
}  // namespace Eigen

#endif
