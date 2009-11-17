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
#if 0
#ifndef EIGEN_SPARSE_EXPRESSIONMAKER_H
#define EIGEN_SPARSE_EXPRESSIONMAKER_H

template<typename XprType>
struct MakeNestByValue<XprType,IsSparse>
{
  typedef SparseNestByValue<XprType> Type;
};

template<typename Func, typename XprType>
struct MakeCwiseUnaryOp<Func,XprType,IsSparse>
{
  typedef SparseCwiseUnaryOp<Func,XprType> Type;
};

template<typename Func, typename A, typename B>
struct MakeCwiseBinaryOp<Func,A,B,IsSparse>
{
  typedef SparseCwiseBinaryOp<Func,A,B> Type;
};

// TODO complete the list

#endif // EIGEN_SPARSE_EXPRESSIONMAKER_H
#endif