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

#ifndef EIGEN_SPARSE_TRIANGULARVIEW_H
#define EIGEN_SPARSE_TRIANGULARVIEW_H

template<typename MatrixType, int Mode>
struct ei_traits<SparseTriangularView<MatrixType,Mode> >
: public ei_traits<MatrixType>
{};

template<typename MatrixType, int Mode> class SparseTriangularView
  : public SparseMatrixBase<SparseTriangularView<MatrixType,Mode> >
{
    enum { SkipFirst = (Mode==LowerTriangular && (!MatrixType::Flags&RowMajorBit))
                    || (Mode==UpperTriangular && ( MatrixType::Flags&RowMajorBit)) };
  public:

    class InnerIterator;

    inline int rows() { return m_matrix.rows(); }
    inline int cols() { return m_matrix.cols(); }

    typedef typename ei_traits<MatrixType>::Scalar Scalar;
    typedef typename ei_meta_if<ei_must_nest_by_value<MatrixType>::ret,
        MatrixType, const MatrixType&>::ret MatrixTypeNested;

    inline SparseTriangularView(const MatrixType& matrix) : m_matrix(matrix) {}

    /** \internal */
    inline const MatrixType& nestedExpression() const { return m_matrix; }

    template<typename OtherDerived>
    typename ei_plain_matrix_type_column_major<OtherDerived>::type
    solve(const MatrixBase<OtherDerived>& other) const;

    template<typename OtherDerived> void solveInPlace(MatrixBase<OtherDerived>& other) const;
    template<typename OtherDerived> void solveInPlace(SparseMatrixBase<OtherDerived>& other) const;

  protected:
    MatrixTypeNested m_matrix;
};

template<typename MatrixType, int Mode>
class SparseTriangularView<MatrixType,Mode>::InnerIterator : public MatrixType::InnerIterator
{
    typedef typename MatrixType::InnerIterator Base;
  public:

    EIGEN_STRONG_INLINE InnerIterator(const SparseTriangularView& view, int outer)
      : Base(view.nestedExpression(), outer)
    {
      if(SkipFirst)
        while((*this) && this->index()<outer)
          ++(*this);
    }
    inline int row() const { return Base::row(); }
    inline int col() const { return Base::col(); }

    EIGEN_STRONG_INLINE operator bool() const
    {
      return SkipFirst ? Base::operator bool() : (Base::operator bool() && this->index() < this->outer());
    }
};

template<typename Derived>
template<int Mode>
inline const SparseTriangularView<Derived, Mode>
SparseMatrixBase<Derived>::triangularView() const
{
  return derived();
}

#endif // EIGEN_SPARSE_TRIANGULARVIEW_H
