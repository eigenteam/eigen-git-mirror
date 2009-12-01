// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_EXPRESSIONMAKER_H
#define EIGEN_EXPRESSIONMAKER_H

// computes the shape of a matrix from its traits flag
template<typename XprType> struct ei_shape_of
{
  enum { ret = ei_traits<XprType>::Flags&SparseBit ? IsSparse : IsDense };
};


// Since the Sparse module is completely separated from the Core module, there is
// no way to write the type of a generic expression working for both dense and sparse
// matrix. Unless we change the overall design, here is a workaround.
// There is an example in unsuported/Eigen/src/AutoDiff/AutoDiffScalar.

template<typename Func, typename XprType, int Shape = ei_shape_of<XprType>::ret>
struct MakeCwiseUnaryOp
{
  typedef CwiseUnaryOp<Func,XprType> Type;
};

template<typename Func, typename A, typename B, int Shape = ei_shape_of<A>::ret>
struct MakeCwiseBinaryOp
{
  typedef CwiseBinaryOp<Func,A,B> Type;
};

// TODO complete the list


#endif // EIGEN_EXPRESSIONMAKER_H
