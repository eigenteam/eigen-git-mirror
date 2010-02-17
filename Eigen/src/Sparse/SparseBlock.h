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

#ifndef EIGEN_SPARSE_BLOCK_H
#define EIGEN_SPARSE_BLOCK_H

template<typename MatrixType, int Size>
struct ei_traits<SparseInnerVectorSet<MatrixType, Size> >
{
  typedef typename ei_traits<MatrixType>::Scalar Scalar;
  typedef typename ei_traits<MatrixType>::StorageType StorageType;
  enum {
    IsRowMajor = (int(MatrixType::Flags)&RowMajorBit)==RowMajorBit,
    Flags = MatrixType::Flags,
    RowsAtCompileTime = IsRowMajor ? Size : MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = IsRowMajor ? MatrixType::ColsAtCompileTime : Size,
    MaxRowsAtCompileTime = RowsAtCompileTime,
    MaxColsAtCompileTime = ColsAtCompileTime,
    CoeffReadCost = MatrixType::CoeffReadCost
  };
};

template<typename MatrixType, int Size>
class SparseInnerVectorSet : ei_no_assignment_operator,
  public SparseMatrixBase<SparseInnerVectorSet<MatrixType, Size> >
{
  public:

    enum { IsRowMajor = ei_traits<SparseInnerVectorSet>::IsRowMajor };

    EIGEN_SPARSE_GENERIC_PUBLIC_INTERFACE(SparseInnerVectorSet)
    class InnerIterator: public MatrixType::InnerIterator
    {
      public:
        inline InnerIterator(const SparseInnerVectorSet& xpr, int outer)
          : MatrixType::InnerIterator(xpr.m_matrix, xpr.m_outerStart + outer), m_outer(outer)
        {}
        inline int row() const { return IsRowMajor ? m_outer : this->index(); }
        inline int col() const { return IsRowMajor ? this->index() : m_outer; }
      protected:
        int m_outer;
    };

    inline SparseInnerVectorSet(const MatrixType& matrix, int outerStart, int outerSize)
      : m_matrix(matrix), m_outerStart(outerStart), m_outerSize(outerSize)
    {
      ei_assert( (outerStart>=0) && ((outerStart+outerSize)<=matrix.outerSize()) );
    }

    inline SparseInnerVectorSet(const MatrixType& matrix, int outer)
      : m_matrix(matrix), m_outerStart(outer), m_outerSize(Size)
    {
      ei_assert(Size!=Dynamic);
      ei_assert( (outer>=0) && (outer<matrix.outerSize()) );
    }

//     template<typename OtherDerived>
//     inline SparseInnerVectorSet& operator=(const SparseMatrixBase<OtherDerived>& other)
//     {
//       return *this;
//     }

//     template<typename Sparse>
//     inline SparseInnerVectorSet& operator=(const SparseMatrixBase<OtherDerived>& other)
//     {
//       return *this;
//     }

    EIGEN_STRONG_INLINE int rows() const { return IsRowMajor ? m_outerSize.value() : m_matrix.rows(); }
    EIGEN_STRONG_INLINE int cols() const { return IsRowMajor ? m_matrix.cols() : m_outerSize.value(); }

  protected:

    const typename MatrixType::Nested m_matrix;
    int m_outerStart;
    const ei_int_if_dynamic<Size> m_outerSize;

};

/***************************************************************************
* specialisation for DynamicSparseMatrix
***************************************************************************/

template<typename _Scalar, int _Options, int Size>
class SparseInnerVectorSet<DynamicSparseMatrix<_Scalar, _Options>, Size>
  : public SparseMatrixBase<SparseInnerVectorSet<DynamicSparseMatrix<_Scalar, _Options>, Size> >
{
    typedef DynamicSparseMatrix<_Scalar, _Options> MatrixType;
  public:

    enum { IsRowMajor = ei_traits<SparseInnerVectorSet>::IsRowMajor };

    EIGEN_SPARSE_GENERIC_PUBLIC_INTERFACE(SparseInnerVectorSet)
    class InnerIterator: public MatrixType::InnerIterator
    {
      public:
        inline InnerIterator(const SparseInnerVectorSet& xpr, int outer)
          : MatrixType::InnerIterator(xpr.m_matrix, xpr.m_outerStart + outer), m_outer(outer)
        {}
        inline int row() const { return IsRowMajor ? m_outer : this->index(); }
        inline int col() const { return IsRowMajor ? this->index() : m_outer; }
      protected:
        int m_outer;
    };

    inline SparseInnerVectorSet(const MatrixType& matrix, int outerStart, int outerSize)
      : m_matrix(matrix), m_outerStart(outerStart), m_outerSize(outerSize)
    {
      ei_assert( (outerStart>=0) && ((outerStart+outerSize)<=matrix.outerSize()) );
    }

    inline SparseInnerVectorSet(const MatrixType& matrix, int outer)
      : m_matrix(matrix), m_outerStart(outer), m_outerSize(Size)
    {
      ei_assert(Size!=Dynamic);
      ei_assert( (outer>=0) && (outer<matrix.outerSize()) );
    }

    template<typename OtherDerived>
    inline SparseInnerVectorSet& operator=(const SparseMatrixBase<OtherDerived>& other)
    {
      if (IsRowMajor != ((OtherDerived::Flags&RowMajorBit)==RowMajorBit))
      {
        // need to transpose => perform a block evaluation followed by a big swap
        DynamicSparseMatrix<Scalar,IsRowMajor?RowMajorBit:0> aux(other);
        *this = aux.markAsRValue();
      }
      else
      {
        // evaluate/copy vector per vector
        for (int j=0; j<m_outerSize.value(); ++j)
        {
          SparseVector<Scalar,IsRowMajor ? RowMajorBit : 0> aux(other.innerVector(j));
          m_matrix.const_cast_derived()._data()[m_outerStart+j].swap(aux._data());
        }
      }
      return *this;
    }

    inline SparseInnerVectorSet& operator=(const SparseInnerVectorSet& other)
    {
      return operator=<SparseInnerVectorSet>(other);
    }

    int nonZeros() const
    {
      int count = 0;
      for (int j=0; j<m_outerSize; ++j)
        count += m_matrix._data()[m_outerStart+j].size();
      return count;
    }

    const Scalar& lastCoeff() const
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(SparseInnerVectorSet);
      ei_assert(m_matrix.data()[m_outerStart].size()>0);
      return m_matrix.data()[m_outerStart].vale(m_matrix.data()[m_outerStart].size()-1);
    }

//     template<typename Sparse>
//     inline SparseInnerVectorSet& operator=(const SparseMatrixBase<OtherDerived>& other)
//     {
//       return *this;
//     }

    EIGEN_STRONG_INLINE int rows() const { return IsRowMajor ? m_outerSize.value() : m_matrix.rows(); }
    EIGEN_STRONG_INLINE int cols() const { return IsRowMajor ? m_matrix.cols() : m_outerSize.value(); }

  protected:

    const typename MatrixType::Nested m_matrix;
    int m_outerStart;
    const ei_int_if_dynamic<Size> m_outerSize;

};


/***************************************************************************
* specialisation for SparseMatrix
***************************************************************************/

template<typename _Scalar, int _Options, int Size>
class SparseInnerVectorSet<SparseMatrix<_Scalar, _Options>, Size>
  : public SparseMatrixBase<SparseInnerVectorSet<SparseMatrix<_Scalar, _Options>, Size> >
{
    typedef SparseMatrix<_Scalar, _Options> MatrixType;
  public:

    enum { IsRowMajor = ei_traits<SparseInnerVectorSet>::IsRowMajor };

    EIGEN_SPARSE_GENERIC_PUBLIC_INTERFACE(SparseInnerVectorSet)
    class InnerIterator: public MatrixType::InnerIterator
    {
      public:
        inline InnerIterator(const SparseInnerVectorSet& xpr, int outer)
          : MatrixType::InnerIterator(xpr.m_matrix, xpr.m_outerStart + outer), m_outer(outer)
        {}
        inline int row() const { return IsRowMajor ? m_outer : this->index(); }
        inline int col() const { return IsRowMajor ? this->index() : m_outer; }
      protected:
        int m_outer;
    };

    inline SparseInnerVectorSet(const MatrixType& matrix, int outerStart, int outerSize)
      : m_matrix(matrix), m_outerStart(outerStart), m_outerSize(outerSize)
    {
      ei_assert( (outerStart>=0) && ((outerStart+outerSize)<=matrix.outerSize()) );
    }

    inline SparseInnerVectorSet(const MatrixType& matrix, int outer)
      : m_matrix(matrix), m_outerStart(outer), m_outerSize(Size)
    {
      ei_assert(Size==1);
      ei_assert( (outer>=0) && (outer<matrix.outerSize()) );
    }

    template<typename OtherDerived>
    inline SparseInnerVectorSet& operator=(const SparseMatrixBase<OtherDerived>& other)
    {
      if (IsRowMajor != ((OtherDerived::Flags&RowMajorBit)==RowMajorBit))
      {
        // need to transpose => perform a block evaluation followed by a big swap
        DynamicSparseMatrix<Scalar,IsRowMajor?RowMajorBit:0> aux(other);
        *this = aux.markAsRValue();
      }
      else
      {
        // evaluate/copy vector per vector
        for (int j=0; j<m_outerSize.value(); ++j)
        {
          SparseVector<Scalar,IsRowMajor ? RowMajorBit : 0> aux(other.innerVector(j));
          m_matrix.const_cast_derived()._data()[m_outerStart+j].swap(aux._data());
        }
      }
      return *this;
    }

    inline SparseInnerVectorSet& operator=(const SparseInnerVectorSet& other)
    {
      return operator=<SparseInnerVectorSet>(other);
    }

    inline const Scalar* _valuePtr() const
    { return m_matrix._valuePtr() + m_matrix._outerIndexPtr()[m_outerStart]; }
    inline Scalar* _valuePtr()
    { return m_matrix.const_cast_derived()._valuePtr() + m_matrix._outerIndexPtr()[m_outerStart]; }

    inline const int* _innerIndexPtr() const
    { return m_matrix._innerIndexPtr() + m_matrix._outerIndexPtr()[m_outerStart]; }
    inline int* _innerIndexPtr()
    { return m_matrix.const_cast_derived()._innerIndexPtr() + m_matrix._outerIndexPtr()[m_outerStart]; }

    inline const int* _outerIndexPtr() const
    { return m_matrix._outerIndexPtr() + m_outerStart; }
    inline int* _outerIndexPtr()
    { return m_matrix.const_cast_derived()._outerIndexPtr() + m_outerStart; }

    int nonZeros() const
    {
      return  size_t(m_matrix._outerIndexPtr()[m_outerStart+m_outerSize.value()])
            - size_t(m_matrix._outerIndexPtr()[m_outerStart]); }

    const Scalar& lastCoeff() const
    {
      EIGEN_STATIC_ASSERT_VECTOR_ONLY(SparseInnerVectorSet);
      ei_assert(nonZeros()>0);
      return m_matrix._valuePtr()[m_matrix._outerIndexPtr()[m_outerStart+1]-1];
    }

//     template<typename Sparse>
//     inline SparseInnerVectorSet& operator=(const SparseMatrixBase<OtherDerived>& other)
//     {
//       return *this;
//     }

    EIGEN_STRONG_INLINE int rows() const { return IsRowMajor ? m_outerSize.value() : m_matrix.rows(); }
    EIGEN_STRONG_INLINE int cols() const { return IsRowMajor ? m_matrix.cols() : m_outerSize.value(); }

  protected:

    const typename MatrixType::Nested m_matrix;
    int m_outerStart;
    const ei_int_if_dynamic<Size> m_outerSize;

};

//----------

/** \returns the i-th row of the matrix \c *this. For row-major matrix only. */
template<typename Derived>
SparseInnerVectorSet<Derived,1> SparseMatrixBase<Derived>::row(int i)
{
  EIGEN_STATIC_ASSERT(IsRowMajor,THIS_METHOD_IS_ONLY_FOR_ROW_MAJOR_MATRICES);
  return innerVector(i);
}

/** \returns the i-th row of the matrix \c *this. For row-major matrix only.
  * (read-only version) */
template<typename Derived>
const SparseInnerVectorSet<Derived,1> SparseMatrixBase<Derived>::row(int i) const
{
  EIGEN_STATIC_ASSERT(IsRowMajor,THIS_METHOD_IS_ONLY_FOR_ROW_MAJOR_MATRICES);
  return innerVector(i);
}

/** \returns the i-th column of the matrix \c *this. For column-major matrix only. */
template<typename Derived>
SparseInnerVectorSet<Derived,1> SparseMatrixBase<Derived>::col(int i)
{
  EIGEN_STATIC_ASSERT(!IsRowMajor,THIS_METHOD_IS_ONLY_FOR_COLUMN_MAJOR_MATRICES);
  return innerVector(i);
}

/** \returns the i-th column of the matrix \c *this. For column-major matrix only.
  * (read-only version) */
template<typename Derived>
const SparseInnerVectorSet<Derived,1> SparseMatrixBase<Derived>::col(int i) const
{
  EIGEN_STATIC_ASSERT(!IsRowMajor,THIS_METHOD_IS_ONLY_FOR_COLUMN_MAJOR_MATRICES);
  return innerVector(i);
}

/** \returns the \a outer -th column (resp. row) of the matrix \c *this if \c *this
  * is col-major (resp. row-major).
  */
template<typename Derived>
SparseInnerVectorSet<Derived,1> SparseMatrixBase<Derived>::innerVector(int outer)
{ return SparseInnerVectorSet<Derived,1>(derived(), outer); }

/** \returns the \a outer -th column (resp. row) of the matrix \c *this if \c *this
  * is col-major (resp. row-major). Read-only.
  */
template<typename Derived>
const SparseInnerVectorSet<Derived,1> SparseMatrixBase<Derived>::innerVector(int outer) const
{ return SparseInnerVectorSet<Derived,1>(derived(), outer); }

//----------

/** \returns the i-th row of the matrix \c *this. For row-major matrix only. */
template<typename Derived>
SparseInnerVectorSet<Derived,Dynamic> SparseMatrixBase<Derived>::subrows(int start, int size)
{
  EIGEN_STATIC_ASSERT(IsRowMajor,THIS_METHOD_IS_ONLY_FOR_ROW_MAJOR_MATRICES);
  return innerVectors(start, size);
}

/** \returns the i-th row of the matrix \c *this. For row-major matrix only.
  * (read-only version) */
template<typename Derived>
const SparseInnerVectorSet<Derived,Dynamic> SparseMatrixBase<Derived>::subrows(int start, int size) const
{
  EIGEN_STATIC_ASSERT(IsRowMajor,THIS_METHOD_IS_ONLY_FOR_ROW_MAJOR_MATRICES);
  return innerVectors(start, size);
}

/** \returns the i-th column of the matrix \c *this. For column-major matrix only. */
template<typename Derived>
SparseInnerVectorSet<Derived,Dynamic> SparseMatrixBase<Derived>::subcols(int start, int size)
{
  EIGEN_STATIC_ASSERT(!IsRowMajor,THIS_METHOD_IS_ONLY_FOR_COLUMN_MAJOR_MATRICES);
  return innerVectors(start, size);
}

/** \returns the i-th column of the matrix \c *this. For column-major matrix only.
  * (read-only version) */
template<typename Derived>
const SparseInnerVectorSet<Derived,Dynamic> SparseMatrixBase<Derived>::subcols(int start, int size) const
{
  EIGEN_STATIC_ASSERT(!IsRowMajor,THIS_METHOD_IS_ONLY_FOR_COLUMN_MAJOR_MATRICES);
  return innerVectors(start, size);
}

/** \returns the \a outer -th column (resp. row) of the matrix \c *this if \c *this
  * is col-major (resp. row-major).
  */
template<typename Derived>
SparseInnerVectorSet<Derived,Dynamic> SparseMatrixBase<Derived>::innerVectors(int outerStart, int outerSize)
{ return SparseInnerVectorSet<Derived,Dynamic>(derived(), outerStart, outerSize); }

/** \returns the \a outer -th column (resp. row) of the matrix \c *this if \c *this
  * is col-major (resp. row-major). Read-only.
  */
template<typename Derived>
const SparseInnerVectorSet<Derived,Dynamic> SparseMatrixBase<Derived>::innerVectors(int outerStart, int outerSize) const
{ return SparseInnerVectorSet<Derived,Dynamic>(derived(), outerStart, outerSize); }

#endif // EIGEN_SPARSE_BLOCK_H
