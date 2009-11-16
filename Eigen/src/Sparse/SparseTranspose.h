// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
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

#ifndef EIGEN_SPARSETRANSPOSE_H
#define EIGEN_SPARSETRANSPOSE_H

// template<typename MatrixType>
// struct ei_traits<SparseTranspose<MatrixType> > : ei_traits<Transpose<MatrixType> >
// {};

template<typename MatrixType> class TransposeImpl<MatrixType,Sparse>
  : public SparseMatrixBase<Transpose<MatrixType> >
{
    const typename ei_cleantype<typename MatrixType::Nested>::type& matrix() const
    { return derived().nestedExpression(); }
    typename ei_cleantype<typename MatrixType::Nested>::type& matrix()
    { return derived().nestedExpression(); }

  public:


//     _EIGEN_SPARSE_GENERIC_PUBLIC_INTERFACE(TransposeImpl,SparseMatrixBase<Transpose<MatrixType> >)
//     EIGEN_EXPRESSION_IMPL_COMMON(SparseMatrixBase<Transpose<MatrixType> >)
    EIGEN_SPARSE_PUBLIC_INTERFACE(Transpose<MatrixType>)

    class InnerIterator;
    class ReverseInnerIterator;

//     inline SparseTranspose(const MatrixType& matrix) : m_matrix(matrix) {}

    //EIGEN_INHERIT_ASSIGNMENT_OPERATORS(SparseTranspose)

//     inline int rows() const { return m_matrix.cols(); }
//     inline int cols() const { return m_matrix.rows(); }
    inline int nonZeros() const { return matrix().nonZeros(); }

    // FIXME should be keep them ?
    inline Scalar& coeffRef(int row, int col)
    { return matrix().const_cast_derived().coeffRef(col, row); }

    inline const Scalar coeff(int row, int col) const
    { return matrix().coeff(col, row); }

    inline const Scalar coeff(int index) const
    { return matrix().coeff(index); }

    inline Scalar& coeffRef(int index)
    { return matrix().const_cast_derived().coeffRef(index); }

//   protected:
//     const typename MatrixType::Nested m_matrix;
};

template<typename MatrixType> class TransposeImpl<MatrixType,Sparse>::InnerIterator : public MatrixType::InnerIterator
{
    typedef typename MatrixType::InnerIterator Base;
  public:

    EIGEN_STRONG_INLINE InnerIterator(const TransposeImpl& trans, int outer)
      : Base(trans.matrix(), outer)
    {}
    inline int row() const { return Base::col(); }
    inline int col() const { return Base::row(); }
};

template<typename MatrixType> class TransposeImpl<MatrixType,Sparse>::ReverseInnerIterator : public MatrixType::ReverseInnerIterator
{
    typedef typename MatrixType::ReverseInnerIterator Base;
  public:

    EIGEN_STRONG_INLINE ReverseInnerIterator(const TransposeImpl& xpr, int outer)
      : Base(xpr.matrix(), outer)
    {}
    inline int row() const { return Base::col(); }
    inline int col() const { return Base::row(); }
};

#endif // EIGEN_SPARSETRANSPOSE_H
