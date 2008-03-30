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
struct ei_scalar_sum_op EIGEN_EMPTY_STRUCT {
    template<typename Scalar> Scalar operator() (const Scalar& a, const Scalar& b) const { return a + b; }
};

/** \internal
  * \brief Template functor to compute the product of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::cwiseProduct(), class PartialRedux, MatrixBase::redux()
  */
struct ei_scalar_product_op EIGEN_EMPTY_STRUCT {
    template<typename Scalar> Scalar operator() (const Scalar& a, const Scalar& b) const { return a * b; }
};

/** \internal
  * \brief Template functor to compute the min of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::cwiseMin, class PartialRedux, MatrixBase::minCoeff()
  */
struct ei_scalar_min_op EIGEN_EMPTY_STRUCT {
    template<typename Scalar> Scalar operator() (const Scalar& a, const Scalar& b) const { return std::min(a, b); }
};

/** \internal
  * \brief Template functor to compute the max of two scalars
  *
  * \sa class CwiseBinaryOp, MatrixBase::cwiseMax, class PartialRedux, MatrixBase::maxCoeff()
  */
struct ei_scalar_max_op EIGEN_EMPTY_STRUCT {
    template<typename Scalar> Scalar operator() (const Scalar& a, const Scalar& b) const { return std::max(a, b); }
};

#endif // EIGEN_ASSOCIATIVE_FUNCTORS_H
