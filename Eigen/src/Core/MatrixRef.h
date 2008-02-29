// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_MATRIXREF_H
#define EIGEN_MATRIXREF_H

template<typename MatrixType> class MatrixRef
 : public MatrixBase<typename MatrixType::Scalar, MatrixRef<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    friend class MatrixBase<Scalar, MatrixRef>;
    typedef MatrixBase<Scalar, MatrixRef> Base;
    
    MatrixRef(const MatrixType& matrix) : m_matrix(matrix) {}
    ~MatrixRef() {}

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(MatrixRef)

  private:
    enum {
      RowsAtCompileTime = MatrixType::Traits::RowsAtCompileTime,
      ColsAtCompileTime = MatrixType::Traits::ColsAtCompileTime,
      MaxRowsAtCompileTime = MatrixType::Traits::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = MatrixType::Traits::MaxColsAtCompileTime
    };

    MatrixRef _ref() const { return *this; }
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
