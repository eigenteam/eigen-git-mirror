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

#ifndef EI_MATRIXREF_H
#define EI_MATRIXREF_H

template<typename MatrixType> class MatrixConstRef
 : public Object<typename MatrixType::Scalar, MatrixConstRef<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    friend class Object<Scalar, MatrixConstRef>;
    
    MatrixConstRef(const MatrixType& matrix) : m_matrix(matrix) {}
    MatrixConstRef(const MatrixConstRef& other) : m_matrix(other.m_matrix) {}
    ~MatrixConstRef() {}

    EI_INHERIT_ASSIGNMENT_OPERATORS(MatrixConstRef)

  private:
    int _rows() const { return m_matrix.rows(); }
    int _cols() const { return m_matrix.cols(); }

    const Scalar& _read(int row, int col) const
    {
      return m_matrix._read(row, col);
    }
    
    const MatrixType& m_matrix;
};

template<typename MatrixType> class MatrixRef
 : public Object<typename MatrixType::Scalar, MatrixRef<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    friend class Object<Scalar, MatrixRef>;
    
    MatrixRef(MatrixType& matrix) : m_matrix(matrix) {}
    MatrixRef(const MatrixRef& other) : m_matrix(other.m_matrix) {}
    ~MatrixRef() {}

    EI_INHERIT_ASSIGNMENT_OPERATORS(MatrixRef)

  private:
    int _rows() const { return m_matrix.rows(); }
    int _cols() const { return m_matrix.cols(); }

    const Scalar& _read(int row, int col) const
    {
      return m_matrix._read(row, col);
    }
    
    Scalar& _write(int row, int col)
    {
      return m_matrix.write(row, col);
    }

  protected:
    MatrixType& m_matrix;
};

#endif // EI_MATRIXREF_H
