// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
//
// Eigen is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 2 or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along
// with Eigen; if not, write to the Free Software Foundation, Inc., 51
// Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
//
// As a special exception, if other files instantiate templates or use macros
// or functions from this file, or you compile this file and link it
// with other works to produce a work based on this file, this file does not
// by itself cause the resulting work to be covered by the GNU General Public
// License. This exception does not invalidate any other reasons why a work
// based on this file might be covered by the GNU General Public License.

#ifndef EIGEN_FORWARDDECLARATIONS_H
#define EIGEN_FORWARDDECLARATIONS_H

template<typename _Scalar, int _Rows, int _Cols, int _StorageOrder, int _MaxRows, int _MaxCols> class Matrix;
template<typename MatrixType> class MatrixRef;
template<typename NewScalar, typename MatrixType> class Cast;
template<typename MatrixType> class Row;
template<typename MatrixType> class Column;
template<typename MatrixType> class Minor;
template<typename MatrixType> class Block;
template<typename MatrixType, int BlockRows, int BlockCols> class FixedBlock;
template<typename MatrixType> class Transpose;
template<typename MatrixType> class Conjugate;
template<typename MatrixType> class Opposite;
template<typename Lhs, typename Rhs> class Sum;
template<typename Lhs, typename Rhs> class Difference;
template<typename Lhs, typename Rhs> class Product;
template<typename FactorType, typename MatrixType> class ScalarMultiple;
template<typename MatrixType> class Random;
template<typename MatrixType> class Zero;
template<typename MatrixType> class Ones;
template<typename CoeffsVectorType> class DiagonalMatrix;
template<typename MatrixType> class DiagonalCoeffs;
template<typename MatrixType> class Identity;
template<typename ExpressionType> class Eval;
template<typename MatrixType> class Map;

template<typename T> struct Reference
{
  typedef T Type;
};

template<typename _Scalar, int _Rows, int _Cols, int _StorageOrder, int _MaxRows, int _MaxCols>
struct Reference<Matrix<_Scalar, _Rows, _Cols, _StorageOrder, _MaxRows, _MaxCols> >
{
  typedef MatrixRef<Matrix<_Scalar, _Rows, _Cols, _StorageOrder, _MaxRows, _MaxCols> > Type;
};

template<typename ExpressionType>
struct Reference<Eval<ExpressionType> >
{
  typedef MatrixRef<Eval<ExpressionType> > Type;
};

#endif // EIGEN_FORWARDDECLARATIONS_H
