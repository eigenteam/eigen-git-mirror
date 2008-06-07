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

#ifndef EIGEN_CROSS_H
#define EIGEN_CROSS_H

/** \returns the cross product of \c *this and \a other */
template<typename Derived>
template<typename OtherDerived>
typename ei_eval<Derived>::type
inline MatrixBase<Derived>::cross(const MatrixBase<OtherDerived>& other) const
{
  // Note that there is no need for an expression here since the compiler
  // optimize such a small temporary very well (even within a complex expression)
  const typename ei_nested<Derived,2>::type lhs(derived());
  const typename ei_nested<OtherDerived,2>::type rhs(other.derived());
  return typename ei_eval<Derived>::type(
    lhs.coeff(1) * rhs.coeff(2) - lhs.coeff(2) * rhs.coeff(1),
    lhs.coeff(2) * rhs.coeff(0) - lhs.coeff(0) * rhs.coeff(2),
    lhs.coeff(0) * rhs.coeff(1) - lhs.coeff(1) * rhs.coeff(0)
  );
}

#endif // EIGEN_CROSS_H
