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

#ifndef EIGEN_DIAGONALMATRIX_H
#define EIGEN_DIAGONALMATRIX_H

/** \class DiagonalMatrix
  *
  * \brief Expression of a diagonal matrix
  *
  * \param CoeffsVectorType the type of the vector of diagonal coefficients
  *
  * This class is an expression of a diagonal matrix with given vector of diagonal
  * coefficients. It is the return
  * type of MatrixBase::diagonal(const OtherDerived&) and most of the time this is
  * the only way it is used.
  *
  * \sa MatrixBase::diagonal(const OtherDerived&)
  */
template<typename CoeffsVectorType>
class DiagonalMatrix : NoOperatorEquals,
  public MatrixBase<typename CoeffsVectorType::Scalar,
                    DiagonalMatrix<CoeffsVectorType> >
{
  public:
    typedef typename CoeffsVectorType::Scalar Scalar;
    typedef typename CoeffsVectorType::Ref CoeffsVecRef;
    friend class MatrixBase<Scalar, DiagonalMatrix<CoeffsVectorType> >;
    
    DiagonalMatrix(const CoeffsVecRef& coeffs) : m_coeffs(coeffs)
    {
      assert(CoeffsVectorType::Traits::IsVectorAtCompileTime
          && coeffs.size() > 0);
    }
    
  private:
    enum {
      RowsAtCompileTime = CoeffsVectorType::Traits::SizeAtCompileTime,
      ColsAtCompileTime = CoeffsVectorType::Traits::SizeAtCompileTime,
      MaxRowsAtCompileTime = CoeffsVectorType::Traits::MaxSizeAtCompileTime,
      MaxColsAtCompileTime = CoeffsVectorType::Traits::MaxSizeAtCompileTime
    };
    
    const DiagonalMatrix& _ref() const { return *this; }
    int _rows() const { return m_coeffs.size(); }
    int _cols() const { return m_coeffs.size(); }
    
    Scalar _coeff(int row, int col) const
    {
      return row == col ? m_coeffs.coeff(row) : static_cast<Scalar>(0);
    }
    
  protected:
    const CoeffsVecRef m_coeffs;
};

/** \returns an expression of a diagonal matrix with *this as vector of diagonal coefficients
  *
  * \only_for_vectors
  *
  * Example: \include MatrixBase_asDiagonal.cpp
  * Output: \verbinclude MatrixBase_asDiagonal.out
  *
  * \sa class DiagonalMatrix, isDiagonal()
  **/
template<typename Scalar, typename Derived>
const DiagonalMatrix<Derived>
MatrixBase<Scalar, Derived>::asDiagonal() const
{
  return DiagonalMatrix<Derived>(ref());
}

/** \returns true if *this is approximately equal to a diagonal matrix,
  *          within the precision given by \a prec.
  *
  * Example: \include MatrixBase_isDiagonal.cpp
  * Output: \verbinclude MatrixBase_isDiagonal.out
  *
  * \sa asDiagonal()
  */
template<typename Scalar, typename Derived>
bool MatrixBase<Scalar, Derived>::isDiagonal
(typename NumTraits<Scalar>::Real prec) const
{
  if(cols() != rows()) return false;
  RealScalar maxAbsOnDiagonal = static_cast<RealScalar>(-1);
  for(int j = 0; j < cols(); j++)
  {
    RealScalar absOnDiagonal = abs(coeff(j,j));
    if(absOnDiagonal > maxAbsOnDiagonal) maxAbsOnDiagonal = absOnDiagonal;
  }
  for(int j = 0; j < cols(); j++)
    for(int i = 0; i < j; i++)
    {
      if(!Eigen::isMuchSmallerThan(coeff(i, j), maxAbsOnDiagonal, prec)) return false;
      if(!Eigen::isMuchSmallerThan(coeff(j, i), maxAbsOnDiagonal, prec)) return false;
    }
  return true;
}

#endif // EIGEN_DIAGONALMATRIX_H
