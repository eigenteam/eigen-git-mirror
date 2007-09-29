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

template<typename MatrixType> class EiMatrixConstRef
 : public EiObject<typename MatrixType::Scalar, EiMatrixConstRef<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    friend class EiObject<Scalar, EiMatrixConstRef>;
    
    EiMatrixConstRef(const MatrixType& matrix) : m_matrix(matrix) {}
    EiMatrixConstRef(const EiMatrixConstRef& other) : m_matrix(other.m_matrix) {}
    ~EiMatrixConstRef() {}

    EI_INHERIT_ASSIGNMENT_OPERATORS(EiMatrixConstRef)

  private:
    int _rows() const { return m_matrix.rows(); }
    int _cols() const { return m_matrix.cols(); }

    const Scalar& _read(int row, int col) const
    {
      return m_matrix._read(row, col);
    }
    
  protected:
    const MatrixType& m_matrix;
};

template<typename MatrixType> class EiMatrixRef
 : public EiObject<typename MatrixType::Scalar, EiMatrixRef<MatrixType> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    friend class EiObject<Scalar, EiMatrixRef>;
    
    EiMatrixRef(MatrixType& matrix) : m_matrix(matrix) {}
    EiMatrixRef(const EiMatrixRef& other) : m_matrix(other.m_matrix) {}
    ~EiMatrixRef() {}

    EI_INHERIT_ASSIGNMENT_OPERATORS(EiMatrixRef)

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
