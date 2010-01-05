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

/** \internal
  *
  * \class TriangularBase
  *
  * \brief Base class for triangular part in a matrix
  */
template<typename Derived> class TriangularBase : public AnyMatrixBase<Derived>
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
    void evalTo(MatrixBase<DenseDerived> &other) const;
    template<typename DenseDerived>
    void evalToLazy(MatrixBase<DenseDerived> &other) const;

  protected:

    void check_coordinates(int row, int col)
    {
      ei_assert(col>=0 && col<cols() && row>=0 && row<rows());
      ei_assert(   (Mode==UpperTriangular && col>=row)
                || (Mode==LowerTriangular && col<=row)
                || ((Mode==StrictlyUpperTriangular || Mode==UnitUpperTriangular) && col>row)
                || ((Mode==StrictlyLowerTriangular || Mode==UnitLowerTriangular) && col<row));
    }

    #ifdef EIGEN_INTERNAL_DEBUGGING
    void check_coordinates_internal(int row, int col)
    {
      check_coordinates(row, col);
    }
    #else
    void check_coordinates_internal(int , int ) {}
    #endif

};

/** \class TriangularView
  *
  * \brief Base class for triangular part in a matrix
  *
  * \param MatrixType the type of the object in which we are taking the triangular part
  * \param Mode the kind of triangular matrix expression to construct. Can be UpperTriangular,
  *             LowerTriangular, UpperSelfadjoint, or LowerSelfadjoint. This is in fact a bit field;
  *             it must have either UpperBit or LowerBit, and additionnaly it may have either
  *             TraingularBit or SelfadjointBit.
  *
  * This class represents a triangular part of a matrix, not necessarily square. Strictly speaking, for rectangular
  * matrices one should speak ok "trapezoid" parts. This class is the return type
  * of MatrixBase::triangularView() and most of the time this is the only way it is used.
  *
  * \sa MatrixBase::triangularView()
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

template<int Mode, bool LhsIsTriangular,
         typename Lhs, bool LhsIsVector,
         typename Rhs, bool RhsIsVector>
struct TriangularProduct;

template<typename _MatrixType, unsigned int _Mode> class TriangularView
  : public TriangularBase<TriangularView<_MatrixType, _Mode> >
{
  public:

    typedef TriangularBase<TriangularView> Base;
    typedef typename ei_traits<TriangularView>::Scalar Scalar;
    typedef _MatrixType MatrixType;
    typedef typename MatrixType::PlainMatrixType DenseMatrixType;
    typedef typename MatrixType::Nested MatrixTypeNested;
    typedef typename ei_cleantype<MatrixTypeNested>::type _MatrixTypeNested;

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
    inline TriangularView<typename MatrixType::AdjointReturnType,TransposeMode> adjoint()
    { return m_matrix.adjoint(); }
    /** \sa MatrixBase::adjoint() const */
    inline const TriangularView<typename MatrixType::AdjointReturnType,TransposeMode> adjoint() const
    { return m_matrix.adjoint(); }

    /** \sa MatrixBase::transpose() */
    inline TriangularView<Transpose<MatrixType>,TransposeMode> transpose()
    { return m_matrix.transpose(); }
    /** \sa MatrixBase::transpose() const */
    inline const TriangularView<Transpose<MatrixType>,TransposeMode> transpose() const
    { return m_matrix.transpose(); }

    DenseMatrixType toDenseMatrix() const
    {
      DenseMatrixType res(rows(), cols());
      evalToLazy(res);
      return res;
    }

    /** Efficient triangular matrix times vector/matrix product */
    template<typename OtherDerived>
    TriangularProduct<Mode,true,MatrixType,false,OtherDerived,OtherDerived::IsVectorAtCompileTime>
    operator*(const MatrixBase<OtherDerived>& rhs) const
    {
      return TriangularProduct
              <Mode,true,MatrixType,false,OtherDerived,OtherDerived::IsVectorAtCompileTime>
              (m_matrix, rhs.derived());
    }

    /** Efficient vector/matrix times triangular matrix product */
    template<typename OtherDerived> friend
    TriangularProduct<Mode,false,OtherDerived,OtherDerived::IsVectorAtCompileTime,MatrixType,false>
    operator*(const MatrixBase<OtherDerived>& lhs, const TriangularView& rhs)
    {
      return TriangularProduct
              <Mode,false,OtherDerived,OtherDerived::IsVectorAtCompileTime,MatrixType,false>
              (lhs.derived(),rhs.m_matrix);
    }


    template<int Side, typename OtherDerived>
    typename ei_plain_matrix_type_column_major<OtherDerived>::type
    solve(const MatrixBase<OtherDerived>& other) const;

    template<int Side, typename OtherDerived>
    void solveInPlace(const MatrixBase<OtherDerived>& other) const;

    template<typename OtherDerived>
    typename ei_plain_matrix_type_column_major<OtherDerived>::type
    solve(const MatrixBase<OtherDerived>& other) const
    { return solve<OnTheLeft>(other); }

    template<typename OtherDerived>
    void solveInPlace(const MatrixBase<OtherDerived>& other) const
    { return solveInPlace<OnTheLeft>(other); }

    const SelfAdjointView<_MatrixTypeNested,Mode> selfadjointView() const
    {
      EIGEN_STATIC_ASSERT((Mode&UnitDiagBit)==0,PROGRAMMING_ERROR);
      return SelfAdjointView<_MatrixTypeNested,Mode>(m_matrix);
    }
    SelfAdjointView<_MatrixTypeNested,Mode> selfadjointView()
    {
      EIGEN_STATIC_ASSERT((Mode&UnitDiagBit)==0,PROGRAMMING_ERROR);
      return SelfAdjointView<_MatrixTypeNested,Mode>(m_matrix);
    }

    template<typename OtherDerived>
    void swap(TriangularBase<OtherDerived> EIGEN_REF_TO_TEMPORARY other)
    {
      TriangularView<SwapWrapper<MatrixType>,Mode>(const_cast<MatrixType&>(m_matrix)).lazyAssign(other.derived());
    }

    template<typename OtherDerived>
    void swap(MatrixBase<OtherDerived> EIGEN_REF_TO_TEMPORARY other)
    {
      TriangularView<SwapWrapper<MatrixType>,Mode>(const_cast<MatrixType&>(m_matrix)).lazyAssign(other.derived());
    }

  protected:

    const MatrixTypeNested m_matrix;
};

/***************************************************************************
* Implementation of triangular evaluation/assignment
***************************************************************************/

template<typename Derived1, typename Derived2, unsigned int Mode, int UnrollCount, bool ClearOpposite>
struct ei_triangular_assignment_selector
{
  enum {
    col = (UnrollCount-1) / Derived1::RowsAtCompileTime,
    row = (UnrollCount-1) % Derived1::RowsAtCompileTime
  };

  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    ei_triangular_assignment_selector<Derived1, Derived2, Mode, UnrollCount-1, ClearOpposite>::run(dst, src);

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
};

// prevent buggy user code from causing an infinite recursion
template<typename Derived1, typename Derived2, unsigned int Mode, bool ClearOpposite>
struct ei_triangular_assignment_selector<Derived1, Derived2, Mode, 0, ClearOpposite>
{
  inline static void run(Derived1 &, const Derived2 &) {}
};

template<typename Derived1, typename Derived2, bool ClearOpposite>
struct ei_triangular_assignment_selector<Derived1, Derived2, UpperTriangular, Dynamic, ClearOpposite>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    for(int j = 0; j < dst.cols(); ++j)
    {
      int maxi = std::min(j, dst.rows()-1);
      for(int i = 0; i <= maxi; ++i)
        dst.copyCoeff(i, j, src);
      if (ClearOpposite)
        for(int i = maxi+1; i < dst.rows(); ++i)
          dst.coeffRef(i, j) = 0;
    }
  }
};

template<typename Derived1, typename Derived2, bool ClearOpposite>
struct ei_triangular_assignment_selector<Derived1, Derived2, LowerTriangular, Dynamic, ClearOpposite>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    for(int j = 0; j < dst.cols(); ++j)
    {
      for(int i = j; i < dst.rows(); ++i)
        dst.copyCoeff(i, j, src);
      int maxi = std::min(j, dst.rows());
      if (ClearOpposite)
        for(int i = 0; i < maxi; ++i)
          dst.coeffRef(i, j) = 0;
    }
  }
};

template<typename Derived1, typename Derived2, bool ClearOpposite>
struct ei_triangular_assignment_selector<Derived1, Derived2, StrictlyUpperTriangular, Dynamic, ClearOpposite>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    for(int j = 0; j < dst.cols(); ++j)
    {
      int maxi = std::min(j, dst.rows());
      for(int i = 0; i < maxi; ++i)
        dst.copyCoeff(i, j, src);
      if (ClearOpposite)
        for(int i = maxi; i < dst.rows(); ++i)
          dst.coeffRef(i, j) = 0;
    }
  }
};

template<typename Derived1, typename Derived2, bool ClearOpposite>
struct ei_triangular_assignment_selector<Derived1, Derived2, StrictlyLowerTriangular, Dynamic, ClearOpposite>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    for(int j = 0; j < dst.cols(); ++j)
    {
      for(int i = j+1; i < dst.rows(); ++i)
        dst.copyCoeff(i, j, src);
      int maxi = std::min(j, dst.rows()-1);
      if (ClearOpposite)
        for(int i = 0; i <= maxi; ++i)
          dst.coeffRef(i, j) = 0;
    }
  }
};

template<typename Derived1, typename Derived2, bool ClearOpposite>
struct ei_triangular_assignment_selector<Derived1, Derived2, UnitUpperTriangular, Dynamic, ClearOpposite>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    for(int j = 0; j < dst.cols(); ++j)
    {
      int maxi = std::min(j, dst.rows());
      for(int i = 0; i < maxi; ++i)
        dst.copyCoeff(i, j, src);
      if (ClearOpposite)
      {
        for(int i = maxi+1; i < dst.rows(); ++i)
          dst.coeffRef(i, j) = 0;
      }
    }
    dst.diagonal().setOnes();
  }
};
template<typename Derived1, typename Derived2, bool ClearOpposite>
struct ei_triangular_assignment_selector<Derived1, Derived2, UnitLowerTriangular, Dynamic, ClearOpposite>
{
  inline static void run(Derived1 &dst, const Derived2 &src)
  {
    for(int j = 0; j < dst.cols(); ++j)
    {
      int maxi = std::min(j, dst.rows());
      for(int i = maxi+1; i < dst.rows(); ++i)
        dst.copyCoeff(i, j, src);
      if (ClearOpposite)
      {
        for(int i = 0; i < maxi; ++i)
          dst.coeffRef(i, j) = 0;
      }
    }
    dst.diagonal().setOnes();
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
    typename ei_plain_matrix_type<OtherDerived>::type other_evaluated(other.rows(), other.cols());
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

  ei_triangular_assignment_selector
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
    typename OtherDerived::DenseMatrixType other_evaluated(other.rows(), other.cols());
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

  ei_triangular_assignment_selector
    <MatrixType, OtherDerived, int(Mode),
    unroll ? int(MatrixType::SizeAtCompileTime) : Dynamic,
    false // preserve the opposite triangular part
    >::run(m_matrix.const_cast_derived(), other.derived()._expression());
}

/***************************************************************************
* Implementation of TriangularBase methods
***************************************************************************/

/** Assigns a triangular or selfadjoint matrix to a dense matrix.
  * If the matrix is triangular, the opposite part is set to zero. */
template<typename Derived>
template<typename DenseDerived>
void TriangularBase<Derived>::evalTo(MatrixBase<DenseDerived> &other) const
{
  if(ei_traits<Derived>::Flags & EvalBeforeAssigningBit)
  {
    typename ei_plain_matrix_type<Derived>::type other_evaluated(rows(), cols());
    evalToLazy(other_evaluated);
    other.derived().swap(other_evaluated);
  }
  else
    evalToLazy(other.derived());
}

/** Assigns a triangular or selfadjoint matrix to a dense matrix.
  * If the matrix is triangular, the opposite part is set to zero. */
template<typename Derived>
template<typename DenseDerived>
void TriangularBase<Derived>::evalToLazy(MatrixBase<DenseDerived> &other) const
{
  const bool unroll =   DenseDerived::SizeAtCompileTime * Derived::CoeffReadCost / 2
                     <= EIGEN_UNROLLING_LIMIT;
  ei_assert(this->rows() == other.rows() && this->cols() == other.cols());

  ei_triangular_assignment_selector
    <DenseDerived, typename ei_traits<Derived>::ExpressionType, Derived::Mode,
    unroll ? int(DenseDerived::SizeAtCompileTime) : Dynamic,
    true // clear the opposite triangular part
    >::run(other.derived(), derived()._expression());
}

/***************************************************************************
* Implementation of TriangularView methods
***************************************************************************/

/***************************************************************************
* Implementation of MatrixBase methods
***************************************************************************/

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
  RealScalar maxAbsOnUpperTriangularPart = static_cast<RealScalar>(-1);
  for(int j = 0; j < cols(); ++j)
  {
    int maxi = std::min(j, rows()-1);
    for(int i = 0; i <= maxi; ++i)
    {
      RealScalar absValue = ei_abs(coeff(i,j));
      if(absValue > maxAbsOnUpperTriangularPart) maxAbsOnUpperTriangularPart = absValue;
    }
  }
  RealScalar threshold = maxAbsOnUpperTriangularPart * prec;
  for(int j = 0; j < cols(); ++j)
    for(int i = j+1; i < rows(); ++i)
      if(ei_abs(coeff(i, j)) > threshold) return false;
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
  RealScalar maxAbsOnLowerTriangularPart = static_cast<RealScalar>(-1);
  for(int j = 0; j < cols(); ++j)
    for(int i = j; i < rows(); ++i)
    {
      RealScalar absValue = ei_abs(coeff(i,j));
      if(absValue > maxAbsOnLowerTriangularPart) maxAbsOnLowerTriangularPart = absValue;
    }
  RealScalar threshold = maxAbsOnLowerTriangularPart * prec;
  for(int j = 1; j < cols(); ++j)
  {
    int maxi = std::min(j, rows()-1);
    for(int i = 0; i < maxi; ++i)
      if(ei_abs(coeff(i, j)) > threshold) return false;
  }
  return true;
}

#endif // EIGEN_TRIANGULARMATRIX_H
