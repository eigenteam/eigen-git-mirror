// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
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

#ifndef EIGEN_TRIANGULAR_H
#define EIGEN_TRIANGULAR_H

/** \class Triangular
  *
  * \brief Expression of a triangular matrix from a squared matrix
  *
  * \param Mode or-ed bit field indicating the triangular part (Upper or Lower) we are taking,
  * and the property of the diagonal if any (UnitDiagBit or NullDiagBit).
  * \param MatrixType the type of the object in which we are taking the triangular part
  *
  * This class represents an expression of the upper or lower triangular part of
  * a squared matrix. It is the return type of MatrixBase::upper(), MatrixBase::lower(),
  * MatrixBase::upperWithUnitDiagBit(), etc., and used to optimize operations involving
  * triangular matrices. Most of the time this is the only way it is used.
  *
  * Examples of some key features:
  * \code
  * m1 = (<any expression>).upper();
  * \endcode
  * In this example, the strictly lower part of the expression is not evaluated,
  * m1 might be resized and the strict lower part of m1 == 0.
  *
  * \code
  * m1.upper() = <any expression>;
  * \endcode
  * This example diverge from the previous one in the sense that the strictly
  * lower part of m1 is left unchanged, and optimal loops are employed. Note that
  * m1 might also be resized.
  *
  * Of course, in both examples \c <any \c expression> has to be a squared matrix.
  *
  * \sa MatrixBase::upper(), MatrixBase::lower(), class TriangularProduct
  */
template<int Mode, typename MatrixType>
struct ei_traits<Triangular<Mode, MatrixType> >
{
  typedef typename MatrixType::Scalar Scalar;
  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
    Flags = MatrixType::Flags & (~(VectorizableBit | Like1DArrayBit)) | Mode,
    CoeffReadCost = MatrixType::CoeffReadCost
  };
};

template<int Mode, typename MatrixType> class Triangular
  : public MatrixBase<Triangular<Mode,MatrixType> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Triangular)

    Triangular(const MatrixType& matrix)
      : m_matrix(matrix)
    {
      assert(!( (Flags&UnitDiagBit) && (Flags&NullDiagBit)));
      assert(matrix.rows()==matrix.cols());
    }

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Triangular)

    /** Overloaded to keep a Triangular expression */
    Triangular<(Upper | Lower) xor Mode, Temporary<Transpose<MatrixType> > > transpose()
    {
      return Triangular<(Upper | Lower) xor Mode, Temporary<Transpose<MatrixType> > >((m_matrix.transpose().temporary()));
    }

    /** Overloaded to keep a Triangular expression */
    const Triangular<(Upper | Lower) xor Mode, Temporary<Transpose<MatrixType> > > transpose() const
    {
      return Triangular<(Upper | Lower) xor Mode, Temporary<Transpose<MatrixType> > >((m_matrix.transpose().temporary()));
    }

    /** \returns the product of the inverse of *this with \a other.
      *
      * This function computes the inverse-matrix matrix product inv(*this) \a other
      * This process is also as forward (resp. backward) substitution if *this is an upper (resp. lower)
      * triangular matrix.
      */
    template<typename OtherDerived>
    typename OtherDerived::Eval inverseProduct(const MatrixBase<OtherDerived>& other) const
    {
      assert(_cols() == other.rows());
      assert(!(Flags & NullDiagBit));

      typename OtherDerived::Eval res(other.rows(), other.cols());

      for (int c=0 ; c<other.cols() ; ++c)
      {
        if (Flags & Lower)
        {
          // forward substitution
          if (Flags & UnitDiagBit)
            res(0,c) = other(0,c);
          else
            res(0,c) = other(0,c)/_coeff(0, 0);
          for (int i=1 ; i<_rows() ; ++i)
          {
            Scalar tmp = other(i,c) - ((this->row(i).start(i)) * res.col(c).start(i))(0,0);
            if (Flags & UnitDiagBit)
              res(i,c) = tmp;
            else
              res(i,c) = tmp/_coeff(i,i);
          }
        }
        else
        {
          // backward substitution
          if (Flags & UnitDiagBit)
            res(_cols()-1,c) = other(_cols()-1,c);
          else
            res(_cols()-1,c) = other(_cols()-1, c)/_coeff(_rows()-1, _cols()-1);
          for (int i=_rows()-2 ; i>=0 ; --i)
          {
            Scalar tmp = other(i,c) - ((this->row(i).end(_cols()-i-1)) * res.col(c).end(_cols()-i-1))(0,0);
            if (Flags & UnitDiagBit)
              res(i,c) = tmp;
            else
              res(i,c) = tmp/_coeff(i,i);
          }
        }
      }
      return res;
    }

  private:

    int _rows() const { return m_matrix.rows(); }
    int _cols() const { return m_matrix.cols(); }

    Scalar& _coeffRef(int row, int col)
    {
      ei_assert( ((! (Flags & Lower)) && row<=col) || (Flags & Lower && col<=row));
      return m_matrix.const_cast_derived().coeffRef(row, col);
    }

    Scalar _coeff(int row, int col) const
    {
      if ((Flags & Lower) ? col>row : row>col)
        return 0;
      if (Flags & UnitDiagBit)
        return col==row ? 1 : m_matrix.coeff(row, col);
      else if (Flags & NullDiagBit)
        return col==row ? 0 : m_matrix.coeff(row, col);
      else
        return m_matrix.coeff(row, col);
    }

  protected:

    const typename MatrixType::Nested m_matrix;
};

/** \returns an expression of a upper triangular matrix
  *
  * \sa isUpper(), upperWithNullDiagBit(), upperWithNullDiagBit(), lower()
  */
template<typename Derived>
Triangular<Upper, Derived> MatrixBase<Derived>::upper(void)
{
  return Triangular<Upper,Derived>(derived());
}

/** This is the const version of upper(). */
template<typename Derived>
const Triangular<Upper, Derived> MatrixBase<Derived>::upper(void) const
{
  return Triangular<Upper,Derived>(derived());
}

/** \returns an expression of a lower triangular matrix
  *
  * \sa isLower(), lowerWithUnitDiag(), lowerWithNullDiag(), upper()
  */
template<typename Derived>
Triangular<Lower, Derived> MatrixBase<Derived>::lower(void)
{
  return Triangular<Lower,Derived>(derived());
}

/** This is the const version of lower().*/
template<typename Derived>
const Triangular<Lower, Derived> MatrixBase<Derived>::lower(void) const
{
  return Triangular<Lower,Derived>(derived());
}

/** \returns an expression of a upper triangular matrix with a unit diagonal
  *
  * \sa upper(), lowerWithUnitDiagBit()
  */
template<typename Derived>
const Triangular<Upper|UnitDiagBit, Derived> MatrixBase<Derived>::upperWithUnitDiag(void) const
{
  return Triangular<Upper|UnitDiagBit, Derived>(derived());
}

/** \returns an expression of a strictly upper triangular matrix (diagonal==zero)
  * FIXME could also be called strictlyUpper() or upperStrict()
  *
  * \sa upper(), lowerWithNullDiag()
  */
template<typename Derived>
const Triangular<Upper|NullDiagBit, Derived> MatrixBase<Derived>::upperWithNullDiag(void) const
{
  return Triangular<Upper|NullDiagBit, Derived>(derived());
}

/** \returns an expression of a lower triangular matrix with a unit diagonal
  *
  * \sa lower(), upperWithUnitDiag()
  */
template<typename Derived>
const Triangular<Lower|UnitDiagBit, Derived> MatrixBase<Derived>::lowerWithUnitDiag(void) const
{
  return Triangular<Lower|UnitDiagBit, Derived>(derived());
}

/** \returns an expression of a strictly lower triangular matrix (diagonal==zero)
  * FIXME could also be called strictlyLower() or lowerStrict()
  *
  * \sa lower(), upperWithNullDiag()
  */
template<typename Derived>
const Triangular<Lower|NullDiagBit, Derived> MatrixBase<Derived>::lowerWithNullDiag(void) const
{
  return Triangular<Lower|NullDiagBit, Derived>(derived());
}

/** \returns true if *this is approximately equal to an upper triangular matrix,
  *          within the precision given by \a prec.
  *
  * \sa isLower(), upper()
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
  * \sa isUpper(), upper()
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


#endif // EIGEN_TRIANGULAR_H
