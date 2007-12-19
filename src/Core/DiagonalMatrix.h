// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2007 Benoit Jacob <jacob@math.jussieu.fr>
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

template<typename MatrixType, typename CoeffsVectorType>
class DiagonalMatrix : NoOperatorEquals,
  public MatrixBase<typename MatrixType::Scalar,
                    DiagonalMatrix<MatrixType, CoeffsVectorType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename CoeffsVectorType::Ref CoeffsVecRef;
    friend class MatrixBase<Scalar, DiagonalMatrix<MatrixType, CoeffsVectorType> >;
    
    DiagonalMatrix(const CoeffsVecRef& coeffs) : m_coeffs(coeffs)
    {
      assert(CoeffsVectorType::IsVector
          && _RowsAtCompileTime == _ColsAtCompileTime
          && _RowsAtCompileTime == CoeffsVectorType::SizeAtCompileTime
          && coeffs.size() > 0);
    }
    
  private:
    static const int _RowsAtCompileTime = MatrixType::RowsAtCompileTime,
                     _ColsAtCompileTime = MatrixType::ColsAtCompileTime;

    const DiagonalMatrix& _ref() const { return *this; }
    int _rows() const { return m_coeffs.size(); }
    int _cols() const { return m_coeffs.size(); }
    
    Scalar _coeff(int row, int col) const
    {
      return row == col ? m_coeffs.coeff(row) : static_cast<Scalar>(0);
    }
    
  protected:
    CoeffsVecRef m_coeffs;
};

template<typename Scalar, typename Derived>
template<typename OtherDerived>
const DiagonalMatrix<Derived, OtherDerived>
MatrixBase<Scalar, Derived>::diagonal(const OtherDerived& coeffs)
{
  return DiagonalMatrix<Derived, OtherDerived>(coeffs);
}

#endif // EIGEN_DIAGONALMATRIX_H
