// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_ASSOCIATIVE_FUNCTORS_H
#define EIGEN_ASSOCIATIVE_FUNCTORS_H

/** \internal
  * \brief Template functor to compute the sum of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::operator+, class PartialRedux, MatrixBase::sum()
  */
template<typename Scalar> struct ei_scalar_sum_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a, const Scalar& b) const { return a + b; }
  enum { Cost = NumTraits<Scalar>::AddCost };
};

/** \internal
  * \brief Template functor to compute the product of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::cwiseProduct(), class PartialRedux, MatrixBase::redux()
  */
template<typename Scalar> struct ei_scalar_product_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a, const Scalar& b) const { return a * b; }
  enum { Cost = NumTraits<Scalar>::MulCost };
};

/** \internal
  * \brief Template functor to compute the min of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::cwiseMin, class PartialRedux, MatrixBase::minCoeff()
  */
template<typename Scalar> struct ei_scalar_min_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a, const Scalar& b) const { return std::min(a, b); }
  enum { Cost = ConditionalJumpCost + NumTraits<Scalar>::AddCost };
};

/** \internal
  * \brief Template functor to compute the max of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::cwiseMax, class PartialRedux, MatrixBase::maxCoeff()
  */
template<typename Scalar> struct ei_scalar_max_op EIGEN_EMPTY_STRUCT {
  const Scalar operator() (const Scalar& a, const Scalar& b) const { return std::max(a, b); }
  enum { Cost = ConditionalJumpCost + NumTraits<Scalar>::AddCost };
};

// default ei_functor_traits for STL functors:

template<typename Result, typename Arg0, typename Arg1>
struct ei_functor_traits<std::binary_function<Result,Arg0,Arg1> >
{ enum { Cost = 10 }; };

template<typename Result, typename Arg0>
struct ei_functor_traits<std::unary_function<Result,Arg0> >
{ enum { Cost = 5 }; };

template<typename T>
struct ei_functor_traits<std::binder2nd<T> >
{ enum { Cost = 5 }; };

template<typename T>
struct ei_functor_traits<std::binder1st<T> >
{ enum { Cost = 5 }; };

template<typename T>
struct ei_functor_traits<std::greater<T> >
{ enum { Cost = 1 }; };

template<typename T>
struct ei_functor_traits<std::less<T> >
{ enum { Cost = 1 }; };

template<typename T>
struct ei_functor_traits<std::greater_equal<T> >
{ enum { Cost = 1 }; };

template<typename T>
struct ei_functor_traits<std::less_equal<T> >
{ enum { Cost = 1 }; };

template<typename T>
struct ei_functor_traits<std::equal_to<T> >
{ enum { Cost = 1 }; };

#endif // EIGEN_ASSOCIATIVE_FUNCTORS_H
