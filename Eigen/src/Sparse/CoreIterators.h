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

#ifndef EIGEN_COREITERATORS_H
#define EIGEN_COREITERATORS_H

template<typename Derived>
class MatrixBase<Derived>::InnerIterator
{
    typedef typename Derived::Scalar Scalar;
  public:
    InnerIterator(const Derived& mat, int col)
      : m_matrix(mat), m_row(0), m_col(col), m_end(mat.rows())
    {}

    Scalar value() { return m_matrix.coeff(m_row, m_col); }

    InnerIterator& operator++() { m_row++; return *this; }

    int index() const { return m_row; }

    operator bool() const { return m_row < m_end && m_row>=0; }

  protected:
    const Derived& m_matrix;
    int m_row;
    const int m_col;
    const int m_end;
};

template<typename UnaryOp, typename MatrixType>
class CwiseUnaryOp<UnaryOp,MatrixType>::InnerIterator
{
    typedef typename CwiseUnaryOp::Scalar Scalar;
    typedef typename ei_traits<CwiseUnaryOp>::_MatrixTypeNested _MatrixTypeNested;
    typedef typename _MatrixTypeNested::InnerIterator MatrixTypeIterator;
  public:

    InnerIterator(const CwiseUnaryOp& unaryOp, int col)
      : m_iter(unaryOp.m_matrix,col), m_functor(unaryOp.m_functor), m_id(-1)
    {
      this->operator++();
    }

    InnerIterator& operator++()
    {
      if (m_iter)
      {
        m_id = m_iter.index();
        m_value = m_functor(m_iter.value());
        ++m_iter;
      }
      else
      {
        m_id = -1;
      }
      return *this;
    }

    Scalar value() const { return m_value; }

    int index() const { return m_id; }

    operator bool() const { return m_id>=0; }

  protected:
    MatrixTypeIterator m_iter;
    const UnaryOp& m_functor;
    Scalar m_value;
    int m_id;
};

template<typename BinaryOp, typename Lhs, typename Rhs>
class CwiseBinaryOp<BinaryOp,Lhs,Rhs>::InnerIterator
{
    typedef typename CwiseBinaryOp::Scalar Scalar;
    typedef typename ei_traits<CwiseBinaryOp>::_LhsNested _LhsNested;
    typedef typename _LhsNested::InnerIterator LhsIterator;
    typedef typename ei_traits<CwiseBinaryOp>::_RhsNested _RhsNested;
    typedef typename _RhsNested::InnerIterator RhsIterator;
  public:

    InnerIterator(const CwiseBinaryOp& binOp, int col)
      : m_lhsIter(binOp.m_lhs,col), m_rhsIter(binOp.m_rhs,col), m_functor(binOp.m_functor), m_id(-1)
    {
      this->operator++();
    }

    InnerIterator& operator++()
    {
      if (m_lhsIter && m_rhsIter && (m_lhsIter.index() == m_rhsIter.index()))
      {
        m_id = m_lhsIter.index();
        m_value = m_functor(m_lhsIter.value(), m_rhsIter.value());
        ++m_lhsIter;
        ++m_rhsIter;
      }
      else if (m_lhsIter && ((!m_rhsIter) || m_lhsIter.index() < m_rhsIter.index()))
      {
        m_id = m_lhsIter.index();
        m_value = m_functor(m_lhsIter.value(), Scalar(0));
        ++m_lhsIter;
      }
      else if (m_rhsIter && ((!m_lhsIter) || m_lhsIter.index() > m_rhsIter.index()))
      {
        m_id = m_rhsIter.index();
        m_value = m_functor(Scalar(0), m_rhsIter.value());
        ++m_rhsIter;
      }
      else
      {
        m_id = -1;
      }
      return *this;
    }

    Scalar value() const { return m_value; }

    int index() const { return m_id; }

    operator bool() const { return m_id>=0; }

  protected:
    LhsIterator m_lhsIter;
    RhsIterator m_rhsIter;
    const BinaryOp& m_functor;
    Scalar m_value;
    int m_id;
};

#endif // EIGEN_COREITERATORS_H
