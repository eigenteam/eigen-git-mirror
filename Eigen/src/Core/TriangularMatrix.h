// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008 Benoit Jacob <jacob.benoit.1@gmail.com>
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

#ifndef EIGEN_TRIANGULARMATRIX_H
#define EIGEN_TRIANGULARMATRIX_H

/** \nonstableyet
  * \class TriangularBase
  *
  * \brief Expression of a triangular matrix extracted from a given matrix
  *
  * \param MatrixType the type of the object in which we are taking the triangular part
  * \param Mode the kind of triangular matrix expression to construct. Can be UpperTriangular,
  *             LowerTriangular, UpperSelfadjoint, or LowerSelfadjoint. This is in fact a bit field;
  *             it must have either UpperBit or LowerBit, and additionnaly it may have either
  *             TraingularBit or SelfadjointBit.
  *
  * This class represents an expression of the upper or lower triangular part of
  * a square matrix, possibly with a further assumption on the diagonal. It is the return type
  * of MatrixBase::part() and most of the time this is the only way it is used.
  *
  * \sa MatrixBase::part()
  */
template<typename Derived> class TriangularBase : public MultiplierBase<Derived>
{
  public:

    enum {
      Mode = ei_traits<Derived>::Mode,
      CoeffReadCost = ei_traits<Derived>::CoeffReadCost,
      RowsAtCompileTime = ei_traits<Derived>::RowsAtCompileTime,
      ColsAtCompileTime = ei_traits<Derived>::ColsAtCompileTime,
      MaxRowsAtCompileTime = ei_traits<Derived>::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = ei_traits<Derived>::MaxColsAtCompileTime
    };
    typedef typename ei_traits<Derived>::Scalar Scalar;

    inline TriangularBase() { ei_assert(ei_are_flags_consistent<Mode>::ret); }

    inline int rows() const { return derived().rows(); }
    inline int cols() const { return derived().cols(); }
    inline int stride() const { return derived().stride(); }

    inline Scalar coeff(int row, int col) const  { return derived().coeff(row,col); }
    inline Scalar& coeffRef(int row, int col) { return derived().coeffRef(row,col); }

    /** \see MatrixBase::copyCoeff(row,col)
      */
    template<typename Other>
    EIGEN_STRONG_INLINE void copyCoeff(int row, int col, Other& other)
    {
      derived().coeffRef(row, col) = other.coeff(row, col);
    }

    inline Scalar operator()(int row, int col) const
    {
      check_coordinates(row, col);
      return coeff(row,col);
    }
    inline Scalar& operator()(int row, int col)
    {
      check_coordinates(row, col);
      return coeffRef(row,col);
    }

    #ifndef EIGEN_PARSED_BY_DOXYGEN
    inline const Derived& derived() const { return *static_cast<const Derived*>(this); }
    inline Derived& derived() { return *static_cast<Derived*>(this); }
    #endif // not EIGEN_PARSED_BY_DOXYGEN

    template<typename DenseDerived>
    void evalToDense(MatrixBase<DenseDerived> &other) const;
    template<typename DenseDerived>
    void evalToDenseLazy(MatrixBase<DenseDerived> &other) const;

  protected:

    void check_coordinates(int row, int col)
    {
      ei_assert(col>0 && col<cols() && row>0 && row<rows());
      ei_assert(   (Mode==UpperTriangular && col>=row)
                || (Mode==LowerTriangular && col<=row)
                || (Mode==StrictlyUpperTriangular && col>row)
                || (Mode==StrictlyLowerTriangular && col<row));
    }

    void check_coordinates_internal(int row, int col)
    {
      #ifdef EIGEN_INTERNAL_DEBUGGING
      check_coordinates(row, col);
      #endif
    }

};


/** \class TriangularView
  * \nonstableyet
  *
  * \brief Expression of a triangular part of a dense matrix
  *
  * \param MatrixType the type of the dense matrix storing the coefficients
  *
  * This class is an expression of a triangular part of a matrix with given dense
  * storage of the coefficients. It is the return type of MatrixBase::triangularPart()
  * and most of the time this is the only way that it is used.
  *
  * \sa class TriangularBase, MatrixBase::triangularPart(), class DiagonalWrapper
  */
template<typename MatrixType, unsigned int _Mode>
struct ei_traits<TriangularView<MatrixType, _Mode> > : ei_traits<MatrixType>
{
  typedef typename ei_nested<MatrixType>::type MatrixTypeNested;
  typedef typename ei_unref<MatrixTypeNested>::type _MatrixTypeNested;
  typedef MatrixType ExpressionType;
  enum {
    Mode = _Mode,
    Flags = (_MatrixTypeNested::Flags & (HereditaryBits) & (~(PacketAccessBit | DirectAccessBit | LinearAccessBit))) | Mode,
    CoeffReadCost = _MatrixTypeNested::CoeffReadCost
  };
};

template<typename MatrixType, unsigned int _Mode> class TriangularView
  : public TriangularBase<TriangularView<MatrixType, _Mode> >
{
  public:

    typedef TriangularBase<TriangularView> Base;
    typedef typename ei_traits<TriangularView>::Scalar Scalar;
    typedef typename MatrixType::PlainMatrixType PlainMatrixType;
    
    enum {
      Mode = _Mode,
      TransposeMode = (Mode & UpperTriangularBit ? LowerTriangularBit : 0)
                    | (Mode & LowerTriangularBit ? UpperTriangularBit : 0)
                    | (Mode & (ZeroDiagBit | UnitDiagBit))
    };

    inline TriangularView(const MatrixType& matrix) : m_matrix(matrix)
    { ei_assert(ei_are_flags_consistent<Mode>::ret); }

    inline int rows() const { return m_matrix.rows(); }
    inline int cols() const { return m_matrix.cols(); }
    inline int stride() const { return m_matrix.stride(); }

    /** \sa MatrixBase::operator+=() */
    template<typename Other> TriangularView&  operator+=(const Other& other) { return *this = m_matrix + other; }
    /** \sa MatrixBase::operator-=() */
    template<typename Other> TriangularView&  operator-=(const Other& other) { return *this = m_matrix - other; }
    /** \sa MatrixBase::operator*=() */
    TriangularView&  operator*=(const typename ei_traits<MatrixType>::Scalar& other) { return *this = m_matrix * other; }
    /** \sa MatrixBase::operator/=() */
    TriangularView&  operator/=(const typename ei_traits<MatrixType>::Scalar& other) { return *this = m_matrix / other; }

    /** \sa MatrixBase::fill() */
    void fill(const Scalar& value) { setConstant(value); }
    /** \sa MatrixBase::setConstant() */
    TriangularView& setConstant(const Scalar& value)
    { return *this = MatrixType::Constant(rows(), cols(), value); }
    /** \sa MatrixBase::setZero() */
    TriangularView& setZero() { return setConstant(Scalar(0)); }
    /** \sa MatrixBase::setOnes() */
    TriangularView& setOnes() { return setConstant(Scalar(1)); }

    /** \sa MatrixBase::coeff()
      * \warning the coordinates must fit into the referenced triangular part
      */
    inline Scalar coeff(int row, int col) const
    {
      Base::check_coordinates_internal(row, col);
      return m_matrix.coeff(row, col);
    }

    /** \sa MatrixBase::coeffRef()
      * \warning the coordinates must fit into the referenced triangular part
      */
    inline Scalar& coeffRef(int row, int col)
    {
      Base::check_coordinates_internal(row, col);
      return m_matrix.const_cast_derived().coeffRef(row, col);
    }

    /** \internal */
    const MatrixType& _expression() const { return m_matrix; }

    /** Assigns a triangular matrix to a triangular part of a dense matrix */
    template<typename OtherDerived>
    TriangularView& operator=(const TriangularBase<OtherDerived>& other);

    template<typename OtherDerived>
    TriangularView& operator=(const MatrixBase<OtherDerived>& other);

    TriangularView& operator=(const TriangularView& other)
    { return *this = other._expression(); }

    template<typename OtherDerived>
    void lazyAssign(const TriangularBase<OtherDerived>& other);

    template<typename OtherDerived>
    void lazyAssign(const MatrixBase<OtherDerived>& other);


    /** \sa MatrixBase::adjoint() */
    inline TriangularView<NestByValue<typename MatrixType::AdjointReturnType>,TransposeMode> adjoint()
    { return m_matrix.adjoint().nestByValue(); }
    /** \sa MatrixBase::adjoint() const */
    const inline TriangularView<NestByValue<typename MatrixType::AdjointReturnType>,TransposeMode> adjoint() const
    { return m_matrix.adjoint().nestByValue(); }

    /** \sa MatrixBase::transpose() */
    inline TriangularView<NestByValue<Transpose<MatrixType> >,TransposeMode> transpose()
    { return m_matrix.transpose().nestByValue(); }
    /** \sa MatrixBase::transpose() const */
    const inline TriangularView<NestByValue<Transpose<MatrixType> >,TransposeMode> transpose() const
    { return m_matrix.transpose().nestByValue(); }

    PlainMatrixType toDense() const
    {
      PlainMatrixType res(rows(), cols());
      res = *this;
      return res;
    }

    template<typename OtherDerived>
    typename ei_plain_matrix_type_column_major<OtherDerived>::type
    solve(const MatrixBase<OtherDerived>& other) const;

    template<typename OtherDerived>
    void solveInPlace(const MatrixBase<OtherDerived>& other) const;

    template<typename OtherDerived>
    void swap(const TriangularBase<OtherDerived>& other)
    {
      TriangularView<SwapWrapper<MatrixType>,Mode>(const_cast<MatrixType&>(m_matrix)).lazyAssign(other.derived());
    }

    template<typename OtherDerived>
    void swap(const MatrixBase<OtherDerived>& other)
    {
      TriangularView<SwapWrapper<MatrixType>,Mode>(const_cast<MatrixType&>(m_matrix)).lazyAssign(other.derived());
    }

  protected:

    const typename MatrixType::Nested m_matrix;
};

/***************************************************************************
* Implementation of triangular evaluation/assignment
***************************************************************************/

template<typename Derived1, typename Derived2, unsigned int Mode, int UnrollCount, bool ClearOpposite>
struct ei_part_assignment_impl
{
  enum {
    col = (UnrollCount-1) / Derived1::RowsAtCompileTime,
    row = (UnrollCount-1) % Derived1::RowsAtCompileTime
  };

  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    ei_part_assignment_impl<Derived1, Derived2, Mode, UnrollCount-1, ClearOpposite>::run(dst, src);

    if(Mode == SelfAdjoint)
    {
      if(row == col)
        dst.coeffRef(row, col) = ei_real(src.coeff(row, col));
      else if(row < col)
        dst.coeffRef(col, row) = ei_conj(dst.coeffRef(row, col) = src.coeff(row, col));
    }
    else
    {
      ei_assert( Mode == UpperTriangular || Mode == LowerTriangular
              || Mode == StrictlyUpperTriangular || Mode == StrictlyLowerTriangular
              || Mode == UnitUpperTriangular || Mode == UnitLowerTriangular);
      if((Mode == UpperTriangular && row <= col)
      || (Mode == LowerTriangular && row >= col)
      || (Mode == StrictlyUpperTriangular && row < col)
      || (Mode == StrictlyLowerTriangular && row > col)
      || (Mode == UnitUpperTriangular && row < col)
      || (Mode == UnitLowerTriangular && row > col))
        dst.copyCoeff(row, col, src);
      else if(ClearOpposite)
      {
        if (Mode&UnitDiagBit && row==col)
          dst.coeffRef(row, col) = 1;
        else
          dst.coeffRef(row, col) = 0;
      }
    }
  }
};
template<typename Derived1, typename Derived2, unsigned int Mode, bool ClearOpposite>
struct ei_part_assignment_impl<Derived1, Derived2, Mode, 1, ClearOpposite>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    if(Mode&UnitDiagBit)
    {
      if(ClearOpposite)
        dst.coeffRef(0, 0) = 1;
    }
    else if(!(Mode & ZeroDiagBit))
      dst.copyCoeff(0, 0, src);
  }
};
// prevent buggy user code from causing an infinite recursion
template<typename Derived1, typename Derived2, unsigned int Mode, bool ClearOpposite>
struct ei_part_assignment_impl<Derived1, Derived2, Mode, 0, ClearOpposite>
{
  inline static void run(Derived1 &, const Derived2 &) {}
};

template<typename Derived1, typename Derived2, bool ClearOpposite>
struct ei_part_assignment_impl<Derived1, Derived2, UpperTriangular, Dynamic, ClearOpposite>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    for(int j = 0; j < dst.cols(); ++j)
    {
      for(int i = 0; i <= j; ++i)
        dst.copyCoeff(i, j, src);
      if (ClearOpposite)
        for(int i = j+1; i < dst.rows(); ++i)
          dst.coeffRef(i, j) = 0;
    }
  }
};
template<typename Derived1, typename Derived2, bool ClearOpposite>
struct ei_part_assignment_impl<Derived1, Derived2, LowerTriangular, Dynamic, ClearOpposite>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    for(int j = 0; j < dst.cols(); ++j)
    {
      for(int i = j; i < dst.rows(); ++i)
        dst.copyCoeff(i, j, src);
      if (ClearOpposite)
        for(int i = 0; i < j; ++i)
          dst.coeffRef(i, j) = 0;
    }
  }
};

template<typename Derived1, typename Derived2, bool ClearOpposite>
struct ei_part_assignment_impl<Derived1, Derived2, StrictlyUpperTriangular, Dynamic, ClearOpposite>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    for(int j = 0; j < dst.cols(); ++j)
    {
      for(int i = 0; i < j; ++i)
        dst.copyCoeff(i, j, src);
      if (ClearOpposite)
        for(int i = j; i < dst.rows(); ++i)
          dst.coeffRef(i, j) = 0;
    }
  }
};
template<typename Derived1, typename Derived2, bool ClearOpposite>
struct ei_part_assignment_impl<Derived1, Derived2, StrictlyLowerTriangular, Dynamic, ClearOpposite>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    for(int j = 0; j < dst.cols(); ++j)
    {
      for(int i = j+1; i < dst.rows(); ++i)
        dst.copyCoeff(i, j, src);
      if (ClearOpposite)
        for(int i = 0; i <= j; ++i)
          dst.coeffRef(i, j) = 0;
    }
  }
};

template<typename Derived1, typename Derived2, bool ClearOpposite>
struct ei_part_assignment_impl<Derived1, Derived2, UnitUpperTriangular, Dynamic, ClearOpposite>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    for(int j = 0; j < dst.cols(); ++j)
    {
      for(int i = 0; i < j; ++i)
        dst.copyCoeff(i, j, src);
      if (ClearOpposite)
      {
        for(int i = j+1; i < dst.rows(); ++i)
          dst.coeffRef(i, j) = 0;
        dst.coeffRef(j, j) = 1;
      }
    }
  }
};
template<typename Derived1, typename Derived2, bool ClearOpposite>
struct ei_part_assignment_impl<Derived1, Derived2, UnitLowerTriangular, Dynamic, ClearOpposite>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    for(int j = 0; j < dst.cols(); ++j)
    {
      for(int i = j+1; i < dst.rows(); ++i)
        dst.copyCoeff(i, j, src);
      if (ClearOpposite)
      {
        for(int i = 0; i < j; ++i)
          dst.coeffRef(i, j) = 0;
        dst.coeffRef(j, j) = 1;
      }
    }
  }
};

// selfadjoint to dense matrix
template<typename Derived1, typename Derived2, bool ClearOpposite>
struct ei_part_assignment_impl<Derived1, Derived2, SelfAdjoint, Dynamic, ClearOpposite>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    for(int j = 0; j < dst.cols(); ++j)
    {
      for(int i = 0; i < j; ++i)
        dst.coeffRef(j, i) = ei_conj(dst.coeffRef(i, j) = src.coeff(i, j));
      dst.coeffRef(j, j) = ei_real(src.coeff(j, j));
    }
  }
};

// FIXME should we keep that possibility
template<typename MatrixType, unsigned int Mode>
template<typename OtherDerived>
inline TriangularView<MatrixType, Mode>&
TriangularView<MatrixType, Mode>::operator=(const MatrixBase<OtherDerived>& other)
{
  if(OtherDerived::Flags & EvalBeforeAssigningBit)
  {
    typename OtherDerived::PlainMatrixType other_evaluated(other.rows(), other.cols());
    other_evaluated.template triangularView<Mode>().lazyAssign(other.derived());
    lazyAssign(other_evaluated);
  }
  else
    lazyAssign(other.derived());
  return *this;
}

// FIXME should we keep that possibility
template<typename MatrixType, unsigned int Mode>
template<typename OtherDerived>
void TriangularView<MatrixType, Mode>::lazyAssign(const MatrixBase<OtherDerived>& other)
{
  const bool unroll =    MatrixType::SizeAtCompileTime * ei_traits<OtherDerived>::CoeffReadCost / 2
                      <= EIGEN_UNROLLING_LIMIT;
  ei_assert(m_matrix.rows() == other.rows() && m_matrix.cols() == other.cols());

  ei_part_assignment_impl
    <MatrixType, OtherDerived, int(Mode),
    unroll ? int(MatrixType::SizeAtCompileTime) : Dynamic,
    false // do not change the opposite triangular part
    >::run(m_matrix.const_cast_derived(), other.derived());
}



template<typename MatrixType, unsigned int Mode>
template<typename OtherDerived>
inline TriangularView<MatrixType, Mode>&
TriangularView<MatrixType, Mode>::operator=(const TriangularBase<OtherDerived>& other)
{
  ei_assert(Mode == OtherDerived::Mode);
  if(ei_traits<OtherDerived>::Flags & EvalBeforeAssigningBit)
  {
    typename OtherDerived::PlainMatrixType other_evaluated(other.rows(), other.cols());
    other_evaluated.template triangularView<Mode>().lazyAssign(other.derived());
    lazyAssign(other_evaluated);
  }
  else
    lazyAssign(other.derived());
  return *this;
}

template<typename MatrixType, unsigned int Mode>
template<typename OtherDerived>
void TriangularView<MatrixType, Mode>::lazyAssign(const TriangularBase<OtherDerived>& other)
{
  const bool unroll =    MatrixType::SizeAtCompileTime * ei_traits<OtherDerived>::CoeffReadCost / 2
                      <= EIGEN_UNROLLING_LIMIT;
  ei_assert(m_matrix.rows() == other.rows() && m_matrix.cols() == other.cols());

  ei_part_assignment_impl
    <MatrixType, OtherDerived, int(Mode),
    unroll ? int(MatrixType::SizeAtCompileTime) : Dynamic,
    false // preserve the opposite triangular part
    >::run(m_matrix.const_cast_derived(), other.derived()._expression());
}

/***************************************************************************
* Implementation of MatrixBase methods
***************************************************************************/

/** Assigns a triangular or selfadjoint matrix to a dense matrix.
  * If the matrix is triangular, the opposite part is set to zero. */
template<typename Derived>
template<typename DenseDerived>
void TriangularBase<Derived>::evalToDense(MatrixBase<DenseDerived> &other) const
{
  if(ei_traits<Derived>::Flags & EvalBeforeAssigningBit)
  {
    typename Derived::PlainMatrixType other_evaluated(rows(), cols());
    evalToDenseLazy(other_evaluated);
    other.derived().swap(other_evaluated);
  }
  else
    evalToDenseLazy(other.derived());
}

/** Assigns a triangular or selfadjoint matrix to a dense matrix.
  * If the matrix is triangular, the opposite part is set to zero. */
template<typename Derived>
template<typename DenseDerived>
void TriangularBase<Derived>::evalToDenseLazy(MatrixBase<DenseDerived> &other) const
{
  const bool unroll =   DenseDerived::SizeAtCompileTime * Derived::CoeffReadCost / 2
                     <= EIGEN_UNROLLING_LIMIT;
  ei_assert(this->rows() == other.rows() && this->cols() == other.cols());

  ei_part_assignment_impl
    <DenseDerived, typename ei_traits<Derived>::ExpressionType, Derived::Mode,
    unroll ? int(DenseDerived::SizeAtCompileTime) : Dynamic,
    true // clear the opposite triangular part
    >::run(other.derived(), derived()._expression());
}

/** \deprecated use MatrixBase::triangularView() */
template<typename Derived>
template<unsigned int Mode>
EIGEN_DEPRECATED const TriangularView<Derived, Mode> MatrixBase<Derived>::part() const
{
  return derived();
}

/** \deprecated use MatrixBase::triangularView() */
template<typename Derived>
template<unsigned int Mode>
EIGEN_DEPRECATED TriangularView<Derived, Mode> MatrixBase<Derived>::part()
{
  return derived();
}

/** \nonstableyet
  * \returns an expression of a triangular view extracted from the current matrix
  *
  * The parameter \a Mode can have the following values: \c UpperTriangular, \c StrictlyUpperTriangular, \c UnitUpperTriangular,
  * \c LowerTriangular, \c StrictlyLowerTriangular, \c UnitLowerTriangular.
  *
  * Example: \include MatrixBase_extract.cpp
  * Output: \verbinclude MatrixBase_extract.out
  *
  * \sa class TriangularView
  */
template<typename Derived>
template<unsigned int Mode>
TriangularView<Derived, Mode> MatrixBase<Derived>::triangularView()
{
  return derived();
}

/** This is the const version of MatrixBase::triangularView() */
template<typename Derived>
template<unsigned int Mode>
const TriangularView<Derived, Mode> MatrixBase<Derived>::triangularView() const
{
  return derived();
}

/** \returns true if *this is approximately equal to an upper triangular matrix,
  *          within the precision given by \a prec.
  *
  * \sa isLowerTriangular(), extract(), part(), marked()
  */
template<typename Derived>
bool MatrixBase<Derived>::isUpperTriangular(RealScalar prec) const
{
  if(cols() != rows()) return false;
  RealScalar maxAbsOnUpperTriangularPart = static_cast<RealScalar>(-1);
  for(int j = 0; j < cols(); ++j)
    for(int i = 0; i <= j; ++i)
    {
      RealScalar absValue = ei_abs(coeff(i,j));
      if(absValue > maxAbsOnUpperTriangularPart) maxAbsOnUpperTriangularPart = absValue;
    }
  for(int j = 0; j < cols()-1; ++j)
    for(int i = j+1; i < rows(); ++i)
      if(!ei_isMuchSmallerThan(coeff(i, j), maxAbsOnUpperTriangularPart, prec)) return false;
  return true;
}

/** \returns true if *this is approximately equal to a lower triangular matrix,
  *          within the precision given by \a prec.
  *
  * \sa isUpperTriangular(), extract(), part(), marked()
  */
template<typename Derived>
bool MatrixBase<Derived>::isLowerTriangular(RealScalar prec) const
{
  if(cols() != rows()) return false;
  RealScalar maxAbsOnLowerTriangularPart = static_cast<RealScalar>(-1);
  for(int j = 0; j < cols(); ++j)
    for(int i = j; i < rows(); ++i)
    {
      RealScalar absValue = ei_abs(coeff(i,j));
      if(absValue > maxAbsOnLowerTriangularPart) maxAbsOnLowerTriangularPart = absValue;
    }
  for(int j = 1; j < cols(); ++j)
    for(int i = 0; i < j; ++i)
      if(!ei_isMuchSmallerThan(coeff(i, j), maxAbsOnLowerTriangularPart, prec)) return false;
  return true;
}

#endif // EIGEN_TRIANGULARMATRIX_H
