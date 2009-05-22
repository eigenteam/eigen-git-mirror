// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

#ifndef EIGEN_EVAL_OMP_H
#define EIGEN_EVAL_OMP_H

/** \class EvalOMP
  *
  * \brief Parallel evaluation of an expression using OpenMP
  *
  * The template parameter Expression is the type of the expression that we are evaluating.
  *
  * This class is the return type of MatrixBase::evalOMP() and most of the time this is the
  * only way it is used.
  *
  * Note that if OpenMP is not enabled, then this class is equivalent to Eval.
  *
  * \sa MatrixBase::evalOMP(), class Eval, MatrixBase::eval()
  */
template<typename ExpressionType>
struct ei_traits<EvalOMP<ExpressionType> >
{
  typedef typename ExpressionType::Scalar Scalar;
  enum {
    RowsAtCompileTime = ExpressionType::RowsAtCompileTime,
    ColsAtCompileTime = ExpressionType::ColsAtCompileTime,
    MaxRowsAtCompileTime = ExpressionType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = ExpressionType::MaxColsAtCompileTime,
    Flags = ExpressionType::Flags & ~LazyBit
  };
};

template<typename ExpressionType> class EvalOMP : ei_no_assignment_operator,
  public Matrix< typename ExpressionType::Scalar,
                 ExpressionType::RowsAtCompileTime,
                 ExpressionType::ColsAtCompileTime,
                 ExpressionType::Flags,
                 ExpressionType::MaxRowsAtCompileTime,
                 ExpressionType::MaxColsAtCompileTime>
{
  public:

    /** The actual matrix type to evaluate to. This type can be used independently
      * of the rest of this class to get the actual matrix type to evaluate and store
      * the value of an expression.
      */
    typedef Matrix<typename ExpressionType::Scalar,
                   ExpressionType::RowsAtCompileTime,
                   ExpressionType::ColsAtCompileTime,
                   ExpressionType::Flags,
                   ExpressionType::MaxRowsAtCompileTime,
                   ExpressionType::MaxColsAtCompileTime> MatrixType;

    _EIGEN_GENERIC_PUBLIC_INTERFACE(EvalOMP, MatrixType)

    #ifdef _OPENMP
    explicit EvalOMP(const ExpressionType& other)
      : MatrixType(other.rows(), other.cols())
    {
      #ifdef __INTEL_COMPILER
      #pragma omp parallel default(none) shared(other)
      #else
      #pragma omp parallel default(none)
      #endif
      {
        if (this->cols()>this->rows())
        {
          #pragma omp for
          for(int j = 0; j < this->cols(); j++)
            for(int i = 0; i < this->rows(); i++)
              this->coeffRef(i, j) = other.coeff(i, j);
        }
        else
        {
          #pragma omp for
          for(int i = 0; i < this->rows(); i++)
            for(int j = 0; j < this->cols(); j++)
              this->coeffRef(i, j) = other.coeff(i, j);
        }
      }
    }
    #else
    explicit EvalOMP(const ExpressionType& other) : MatrixType(other) {}
    #endif
};

/** Evaluates *this in a parallel fashion using OpenMP and returns the obtained matrix.
  *
  * Of course, it only makes sense to call this function for complex expressions, and/or
  * large matrices (>32x32), \b and if there is no outer loop which can be parallelized.
  *
  * It is the responsibility of the user manage the OpenMP parameters, for instance:
  * \code
  * #include <omp.h>
  * // ...
  * omp_set_num_threads(omp_get_num_procs());
  * \endcode
  * You also need to enable OpenMP on your compiler (e.g., -fopenmp) during both compilation and linking.
  *
  * Note that if OpenMP is not enabled, then evalOMP() is equivalent to eval().
  *
  * \sa class EvalOMP, eval()
  */
template<typename Derived>
const EvalOMP<Derived> MatrixBase<Derived>::evalOMP() const
{
  return EvalOMP<Derived>(*static_cast<const Derived*>(this));
}

#endif // EIGEN_EVAL_OMP_H
