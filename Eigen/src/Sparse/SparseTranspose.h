// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_SPARSETRANSPOSE_H
#define EIGEN_SPARSETRANSPOSE_H

template<typename MatrixType> class TransposeImpl<MatrixType,Sparse>
  : public SparseMatrixBase<Transpose<MatrixType> >
{
  public:

    EIGEN_SPARSE_PUBLIC_INTERFACE(Transpose<MatrixType>)

    class InnerIterator;
    class ReverseInnerIterator;

    inline int nonZeros() const { return derived().nestedExpression().nonZeros(); }

    // FIXME should be keep them ?
    inline Scalar& coeffRef(int row, int col)
    { return const_cast_derived().nestedExpression().coeffRef(col, row); }

    inline const Scalar coeff(int row, int col) const
    { return derived().nestedExpression().coeff(col, row); }

    inline const Scalar coeff(int index) const
    { return derived().nestedExpression().coeff(index); }

    inline Scalar& coeffRef(int index)
    { return const_cast_derived().nestedExpression().coeffRef(index); }
};

template<typename MatrixType> class TransposeImpl<MatrixType,Sparse>::InnerIterator : public MatrixType::InnerIterator
{
    typedef typename MatrixType::InnerIterator Base;
  public:

    EIGEN_STRONG_INLINE InnerIterator(const TransposeImpl& trans, int outer)
      : Base(trans.derived().nestedExpression(), outer)
    {}
    inline int row() const { return Base::col(); }
    inline int col() const { return Base::row(); }
};

template<typename MatrixType> class TransposeImpl<MatrixType,Sparse>::ReverseInnerIterator : public MatrixType::ReverseInnerIterator
{
    typedef typename MatrixType::ReverseInnerIterator Base;
  public:

    EIGEN_STRONG_INLINE ReverseInnerIterator(const TransposeImpl& xpr, int outer)
      : Base(xpr.derived().nestedExpression(), outer)
    {}
    inline int row() const { return Base::col(); }
    inline int col() const { return Base::row(); }
};

#endif // EIGEN_SPARSETRANSPOSE_H
