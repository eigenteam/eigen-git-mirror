// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_PART_H
#define EIGEN_PART_H

/** \class Part
  *
  * \brief Expression of a triangular matrix extracted from a given matrix
  *
  * \param MatrixType the type of the object in which we are taking the triangular part
  * \param Mode the kind of triangular matrix expression to construct. Can be Upper, StrictlyUpper,
  *             UnitUpper, Lower, StrictlyLower, UnitLower. This is in fact a bit field; it must have either
  *             UpperTriangularBit or LowerTriangularBit, and additionnaly it may have either ZeroDiagBit or
  *             UnitDiagBit.
  *
  * This class represents an expression of the upper or lower triangular part of
  * a square matrix, possibly with a further assumption on the diagonal. It is the return type
  * of MatrixBase::part() and most of the time this is the only way it is used.
  *
  * \sa MatrixBase::part()
  */
template<typename MatrixType, unsigned int Mode>
struct ei_traits<Part<MatrixType, Mode> >
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename ei_nested<MatrixType>::type MatrixTypeNested;
  typedef typename ei_unref<MatrixTypeNested>::type _MatrixTypeNested;
  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
    Flags = (_MatrixTypeNested::Flags & (HereditaryBits) & (~(PacketAccessBit | DirectAccessBit | LinearAccessBit))) | Mode,
    CoeffReadCost = _MatrixTypeNested::CoeffReadCost
  };
};

template<typename MatrixType, unsigned int Mode> class Part
#ifndef EIGEN_PARSED_BY_DOXYGEN
  : public MatrixBase<Part<MatrixType, Mode> >
#else
  : public MatrixBase
#endif
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Part)

    inline Part(const MatrixType& matrix) : m_matrix(matrix)
    { ei_assert(ei_are_flags_consistent<Mode>::ret); }

    /** \sa MatrixBase::operator+=() */
    template<typename Other> Part&  operator+=(const Other& other);
    /** \sa MatrixBase::operator-=() */
    template<typename Other> Part&  operator-=(const Other& other);
    /** \sa MatrixBase::operator*=() */
    Part&  operator*=(const typename ei_traits<MatrixType>::Scalar& other);
    /** \sa MatrixBase::operator/=() */
    Part&  operator/=(const typename ei_traits<MatrixType>::Scalar& other);

    /** \sa operator=(), MatrixBase::lazyAssign() */
    template<typename Other> void lazyAssign(const Other& other);
    /** \sa MatrixBase::operator=() */
    template<typename Other> Part& operator=(const Other& other);

    inline int rows() const { return m_matrix.rows(); }
    inline int cols() const { return m_matrix.cols(); }
    inline int stride() const { return m_matrix.stride(); }

    inline Scalar coeff(int row, int col) const
    {
      if(Flags & LowerTriangularBit ? col>row : row>col)
        return (Flags & SelfAdjointBit) ? ei_conj(m_matrix.coeff(col, row)) : (Scalar)0;
      if(Flags & UnitDiagBit)
        return col==row ? (Scalar)1 : m_matrix.coeff(row, col);
      else if(Flags & ZeroDiagBit)
        return col==row ? (Scalar)0 : m_matrix.coeff(row, col);
      else
        return m_matrix.coeff(row, col);
    }

    inline Scalar& coeffRef(int row, int col)
    {
      EIGEN_STATIC_ASSERT(!(Flags & UnitDiagBit), writting_to_triangular_part_with_unit_diag_is_not_supported);
      EIGEN_STATIC_ASSERT(!(Flags & SelfAdjointBit), default_writting_to_selfadjoint_not_supported);
      ei_assert(   (Mode==Upper && col>=row)
                || (Mode==Lower && col<=row)
                || (Mode==StrictlyUpper && col>row)
                || (Mode==StrictlyLower && col<row));
      return m_matrix.const_cast_derived().coeffRef(row, col);
    }

    /** \internal */
    const MatrixType& _expression() const { return m_matrix; }

    /** discard any writes to a row */
    const Block<Part, 1, ColsAtCompileTime> row(int i) { return Base::row(i); }
    const Block<Part, 1, ColsAtCompileTime> row(int i) const { return Base::row(i); }
    /** discard any writes to a column */
    const Block<Part, RowsAtCompileTime, 1> col(int i) { return Base::col(i); }
    const Block<Part, RowsAtCompileTime, 1> col(int i) const { return Base::col(i); }

  protected:

    const typename MatrixType::Nested m_matrix;
};

/** \returns an expression of a triangular matrix extracted from the current matrix
  *
  * The parameter \a Mode can have the following values: \c Upper, \c StrictlyUpper, \c UnitUpper,
  * \c Lower, \c StrictlyLower, \c UnitLower.
  *
  * \addexample PartExample \label How to extract a triangular part of an arbitrary matrix
  *
  * Example: \include MatrixBase_extract.cpp
  * Output: \verbinclude MatrixBase_extract.out
  *
  * \sa class Part, part(), marked()
  */
template<typename Derived>
template<unsigned int Mode>
const Part<Derived, Mode> MatrixBase<Derived>::part() const
{
  return derived();
}

template<typename MatrixType, unsigned int Mode>
template<typename Other>
inline Part<MatrixType, Mode>& Part<MatrixType, Mode>::operator=(const Other& other)
{
  if(Other::Flags & EvalBeforeAssigningBit)
  {
    typename ei_eval<Other>::type other_evaluated(other.rows(), other.cols());
    other_evaluated.template part<Mode>().lazyAssign(other);
    lazyAssign(other_evaluated);
  }
  else
    lazyAssign(other.derived());
  return *this;
}

template<typename Derived1, typename Derived2, unsigned int Mode, int UnrollCount>
struct ei_part_assignment_impl
{
  enum {
    col = (UnrollCount-1) / Derived1::RowsAtCompileTime,
    row = (UnrollCount-1) % Derived1::RowsAtCompileTime
  };

  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    ei_part_assignment_impl<Derived1, Derived2, Mode, UnrollCount-1>::run(dst, src);

    if(Mode == SelfAdjoint)
    {
      if(row == col)
        dst.coeffRef(row, col) = ei_real(src.coeff(row, col));
      else if(row < col)
        dst.coeffRef(col, row) = ei_conj(dst.coeffRef(row, col) = src.coeff(row, col));
    }
    else
    {
      ei_assert(Mode == Upper || Mode == Lower || Mode == StrictlyUpper || Mode == StrictlyLower);
      if((Mode == Upper && row <= col)
      || (Mode == Lower && row >= col)
      || (Mode == StrictlyUpper && row < col)
      || (Mode == StrictlyLower && row > col))
        dst.coeffRef(row, col) = src.coeff(row, col);
    }
  }
};

template<typename Derived1, typename Derived2, unsigned int Mode>
struct ei_part_assignment_impl<Derived1, Derived2, Mode, 1>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    if(!(Mode & ZeroDiagBit))
      dst.coeffRef(0, 0) = src.coeff(0, 0);
  }
};

// prevent buggy user code from causing an infinite recursion
template<typename Derived1, typename Derived2, unsigned int Mode>
struct ei_part_assignment_impl<Derived1, Derived2, Mode, 0>
{
  inline static void run(Derived1 &, const Derived2 &) {}
};

template<typename Derived1, typename Derived2>
struct ei_part_assignment_impl<Derived1, Derived2, Upper, Dynamic>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    for(int j = 0; j < dst.cols(); j++)
      for(int i = 0; i <= j; i++)
        dst.coeffRef(i, j) = src.coeff(i, j);
  }
};

template<typename Derived1, typename Derived2>
struct ei_part_assignment_impl<Derived1, Derived2, Lower, Dynamic>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    for(int j = 0; j < dst.cols(); j++)
      for(int i = j; i < dst.rows(); i++)
        dst.coeffRef(i, j) = src.coeff(i, j);
  }
};

template<typename Derived1, typename Derived2>
struct ei_part_assignment_impl<Derived1, Derived2, StrictlyUpper, Dynamic>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    for(int j = 0; j < dst.cols(); j++)
      for(int i = 0; i < j; i++)
        dst.coeffRef(i, j) = src.coeff(i, j);
  }
};
template<typename Derived1, typename Derived2>
struct ei_part_assignment_impl<Derived1, Derived2, StrictlyLower, Dynamic>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    for(int j = 0; j < dst.cols(); j++)
      for(int i = j+1; i < dst.rows(); i++)
        dst.coeffRef(i, j) = src.coeff(i, j);
  }
};
template<typename Derived1, typename Derived2>
struct ei_part_assignment_impl<Derived1, Derived2, SelfAdjoint, Dynamic>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    for(int j = 0; j < dst.cols(); j++)
    {
      for(int i = 0; i < j; i++)
        dst.coeffRef(j, i) = ei_conj(dst.coeffRef(i, j) = src.coeff(i, j));
      dst.coeffRef(j, j) = ei_real(src.coeff(j, j));
    }
  }
};

template<typename MatrixType, unsigned int Mode>
template<typename Other>
void Part<MatrixType, Mode>::lazyAssign(const Other& other)
{
  const bool unroll = MatrixType::SizeAtCompileTime * Other::CoeffReadCost / 2 <= EIGEN_UNROLLING_LIMIT;
  ei_assert(m_matrix.rows() == other.rows() && m_matrix.cols() == other.cols());

  ei_part_assignment_impl
    <MatrixType, Other, Mode,
    unroll ? int(MatrixType::SizeAtCompileTime) : Dynamic
    >::run(m_matrix.const_cast_derived(), other.derived());
}

/** \returns a lvalue pseudo-expression allowing to perform special operations on \c *this.
  *
  * The \a Mode parameter can have the following values: \c Upper, \c StrictlyUpper, \c Lower,
  * \c StrictlyLower, \c SelfAdjoint.
  *
  * \addexample PartExample \label How to write to a triangular part of a matrix
  *
  * Example: \include MatrixBase_part.cpp
  * Output: \verbinclude MatrixBase_part.out
  *
  * \sa class Part, MatrixBase::extract(), MatrixBase::marked()
  */
template<typename Derived>
template<unsigned int Mode>
inline Part<Derived, Mode> MatrixBase<Derived>::part()
{
  return Part<Derived, Mode>(derived());
}

/** \returns true if *this is approximately equal to an upper triangular matrix,
  *          within the precision given by \a prec.
  *
  * \sa isLower(), extract(), part(), marked()
  */
template<typename Derived>
bool MatrixBase<Derived>::isUpper(RealScalar prec) const
{
  if(cols() != rows()) return false;
  RealScalar maxAbsOnUpperPart = static_cast<RealScalar>(-1);
  for(int j = 0; j < cols(); j++)
    for(int i = 0; i <= j; i++)
    {
      RealScalar absValue = ei_abs(coeff(i,j));
      if(absValue > maxAbsOnUpperPart) maxAbsOnUpperPart = absValue;
    }
  for(int j = 0; j < cols()-1; j++)
    for(int i = j+1; i < rows(); i++)
      if(!ei_isMuchSmallerThan(coeff(i, j), maxAbsOnUpperPart, prec)) return false;
  return true;
}

/** \returns true if *this is approximately equal to a lower triangular matrix,
  *          within the precision given by \a prec.
  *
  * \sa isUpper(), extract(), part(), marked()
  */
template<typename Derived>
bool MatrixBase<Derived>::isLower(RealScalar prec) const
{
  if(cols() != rows()) return false;
  RealScalar maxAbsOnLowerPart = static_cast<RealScalar>(-1);
  for(int j = 0; j < cols(); j++)
    for(int i = j; i < rows(); i++)
    {
      RealScalar absValue = ei_abs(coeff(i,j));
      if(absValue > maxAbsOnLowerPart) maxAbsOnLowerPart = absValue;
    }
  for(int j = 1; j < cols(); j++)
    for(int i = 0; i < j; i++)
      if(!ei_isMuchSmallerThan(coeff(i, j), maxAbsOnLowerPart, prec)) return false;
  return true;
}

template<typename MatrixType, unsigned int Mode>
template<typename Other>
inline Part<MatrixType, Mode>& Part<MatrixType, Mode>::operator+=(const Other& other)
{
  return *this = m_matrix + other;
}

template<typename MatrixType, unsigned int Mode>
template<typename Other>
inline Part<MatrixType, Mode>& Part<MatrixType, Mode>::operator-=(const Other& other)
{
  return *this = m_matrix - other;
}

template<typename MatrixType, unsigned int Mode>
inline Part<MatrixType, Mode>& Part<MatrixType, Mode>::operator*=
(const typename ei_traits<MatrixType>::Scalar& other)
{
  return *this = m_matrix * other;
}

template<typename MatrixType, unsigned int Mode>
inline Part<MatrixType, Mode>& Part<MatrixType, Mode>::operator/=
(const typename ei_traits<MatrixType>::Scalar& other)
{
  return *this = m_matrix / other;
}

#endif // EIGEN_PART_H
