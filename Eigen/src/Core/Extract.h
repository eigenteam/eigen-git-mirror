// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_EXTRACT_H
#define EIGEN_EXTRACT_H

/** \class Extract
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
  * of MatrixBase::extract() and most of the time this is the only way it is used.
  *
  * \sa MatrixBase::extract()
  */
template<typename MatrixType, unsigned int Mode>
struct ei_traits<Extract<MatrixType, Mode> >
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename ei_nested<MatrixType>::type MatrixTypeNested;
  typedef typename ei_unref<MatrixTypeNested>::type _MatrixTypeNested;
  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,
    Flags = (_MatrixTypeNested::Flags & ~(VectorizableBit | Like1DArrayBit | DirectAccessBit)) | Mode,
    CoeffReadCost = _MatrixTypeNested::CoeffReadCost
  };
};

template<typename MatrixType, unsigned int Mode> class Extract
  : public MatrixBase<Extract<MatrixType, Mode> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Extract)

    inline Extract(const MatrixType& matrix) : m_matrix(matrix) {}

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Extract)

  private:

    inline int _rows() const { return m_matrix.rows(); }
    inline int _cols() const { return m_matrix.cols(); }

    inline Scalar _coeff(int row, int col) const
    {
      if(Flags & LowerTriangularBit ? col>row : row>col)
        return (Scalar)0;
      if(Flags & UnitDiagBit)
        return col==row ? (Scalar)1 : m_matrix.coeff(row, col);
      else if(Flags & ZeroDiagBit)
        return col==row ? (Scalar)0 : m_matrix.coeff(row, col);
      else
        return m_matrix.coeff(row, col);
    }

  protected:

    const typename MatrixType::Nested m_matrix;
};

/** \returns an expression of a triangular matrix extracted from the current matrix
  *
  * The parameter \a Mode can have the following values: \c Upper, \c StrictlyUpper, \c UnitUpper,
  * \c Lower, \c StrictlyLower, \c UnitLower.
  *
  * Example: \include MatrixBase_extract.cpp
  * Output: \verbinclude MatrixBase_extract.out
  *
  * \sa class Extract, part(), marked()
  */
template<typename Derived>
template<unsigned int Mode>
const Extract<Derived, Mode> MatrixBase<Derived>::extract() const
{
  return derived();
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

#endif // EIGEN_EXTRACT_H
