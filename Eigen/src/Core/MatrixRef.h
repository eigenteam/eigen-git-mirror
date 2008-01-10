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

#ifndef EIGEN_MATRIXREF_H
#define EIGEN_MATRIXREF_H

template<typename MatrixType> class MatrixRef
 : public MatrixBase<typename MatrixType::Scalar, MatrixRef<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    friend class MatrixBase<Scalar, MatrixRef>;
    
    MatrixRef(const MatrixType& matrix) : m_matrix(matrix) {}
    ~MatrixRef() {}

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(MatrixRef)

  private:
    enum {
      RowsAtCompileTime = MatrixType::Traits::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::Traits::ColsAtCompileTime
    };

    int _rows() const { return m_matrix.rows(); }
    int _cols() const { return m_matrix.cols(); }

    const Scalar& _coeff(int row, int col) const
    {
      return m_matrix._coeff(row, col);
    }
    
    Scalar& _coeffRef(int row, int col)
    {
      return const_cast<MatrixType*>(&m_matrix)->_coeffRef(row, col);
    }

  protected:
    const MatrixType& m_matrix;
};

#endif // EIGEN_MATRIXREF_H
