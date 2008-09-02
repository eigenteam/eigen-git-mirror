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

/* This file contains the respective InnerIterator definition of the expressions defined in Eigen/Core
 */

template<typename Derived>
class MatrixBase<Derived>::InnerIterator
{
    typedef typename Derived::Scalar Scalar;
  public:
    InnerIterator(const Derived& mat, int outer)
      : m_matrix(mat), m_inner(0), m_outer(outer), m_end(mat.rows())
    {}

    Scalar value() const
    {
      return (Derived::Flags&RowMajorBit) ? m_matrix.coeff(m_outer, m_inner)
                                          : m_matrix.coeff(m_inner, m_outer);
    }

    InnerIterator& operator++() { m_inner++; return *this; }

    int index() const { return m_inner; }

    operator bool() const { return m_inner < m_end && m_inner>=0; }

  protected:
    const Derived& m_matrix;
    int m_inner;
    const int m_outer;
    const int m_end;
};

template<typename MatrixType>
class Transpose<MatrixType>::InnerIterator : public MatrixType::InnerIterator
{
  public:

    InnerIterator(const Transpose& trans, int outer)
      : MatrixType::InnerIterator(trans.m_matrix, outer)
    {}
};

template<typename MatrixType, int BlockRows, int BlockCols, int PacketAccess, int _DirectAccessStatus>
class Block<MatrixType, BlockRows, BlockCols, PacketAccess, _DirectAccessStatus>::InnerIterator
{
    typedef typename Block::Scalar Scalar;
    typedef typename ei_traits<Block>::_MatrixTypeNested _MatrixTypeNested;
    typedef typename _MatrixTypeNested::InnerIterator MatrixTypeIterator;
  public:

    InnerIterator(const Block& block, int outer)
      : m_iter(block.m_matrix,(Block::Flags&RowMajor) ? block.m_startRow.value() + outer : block.m_startCol.value() + outer),
        m_start( (Block::Flags&RowMajor) ? block.m_startCol.value() : block.m_startRow.value()),
        m_end(m_start + ((Block::Flags&RowMajor) ? block.m_blockCols.value() : block.m_blockRows.value())),
        m_offset( (Block::Flags&RowMajor) ? block.m_startCol.value() : block.m_startRow.value())
    {
      while (m_iter.index()>=0 && m_iter.index()<m_start)
        ++m_iter;
    }

    InnerIterator& operator++()
    {
      ++m_iter;
      return *this;
    }

    Scalar value() const { return m_iter.value(); }

    int index() const { return m_iter.index() - m_offset; }

    operator bool() const { return m_iter && m_iter.index()<m_end; }

  protected:
    MatrixTypeIterator m_iter;
    int m_start;
    int m_end;
    int m_offset;
}; 

template<typename MatrixType, int BlockRows, int BlockCols, int PacketAccess>
class Block<MatrixType, BlockRows, BlockCols, PacketAccess, IsSparse>::InnerIterator
{
    typedef typename Block::Scalar Scalar;
    typedef typename ei_traits<Block>::_MatrixTypeNested _MatrixTypeNested;
    typedef typename _MatrixTypeNested::InnerIterator MatrixTypeIterator;
  public:

    InnerIterator(const Block& block, int outer)
      : m_iter(block.m_matrix,(Block::Flags&RowMajor) ? block.m_startRow.value() + outer : block.m_startCol.value() + outer),
        m_start( (Block::Flags&RowMajor) ? block.m_startCol.value() : block.m_startRow.value()),
        m_end(m_start + ((Block::Flags&RowMajor) ? block.m_blockCols.value() : block.m_blockRows.value())),
        m_offset( (Block::Flags&RowMajor) ? block.m_startCol.value() : block.m_startRow.value())
    {
      while (m_iter.index()>=0 && m_iter.index()<m_start)
        ++m_iter;
    }

    InnerIterator& operator++()
    {
      ++m_iter;
      return *this;
    }

    Scalar value() const { return m_iter.value(); }

    int index() const { return m_iter.index() - m_offset; }

    operator bool() const { return m_iter && m_iter.index()<m_end; }

  protected:
    MatrixTypeIterator m_iter;
    int m_start;
    int m_end;
    int m_offset;
};

template<typename UnaryOp, typename MatrixType>
class CwiseUnaryOp<UnaryOp,MatrixType>::InnerIterator
{
    typedef typename CwiseUnaryOp::Scalar Scalar;
    typedef typename ei_traits<CwiseUnaryOp>::_MatrixTypeNested _MatrixTypeNested;
    typedef typename _MatrixTypeNested::InnerIterator MatrixTypeIterator;
  public:

    InnerIterator(const CwiseUnaryOp& unaryOp, int outer)
      : m_iter(unaryOp.m_matrix,outer), m_functor(unaryOp.m_functor), m_id(-1)
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

    InnerIterator(const CwiseBinaryOp& binOp, int outer)
      : m_lhsIter(binOp.m_lhs,outer), m_rhsIter(binOp.m_rhs,outer), m_functor(binOp.m_functor), m_id(-1)
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
      else if (m_lhsIter && (!m_rhsIter || (m_lhsIter.index() < m_rhsIter.index())))
      {
        m_id = m_lhsIter.index();
        m_value = m_functor(m_lhsIter.value(), Scalar(0));
        ++m_lhsIter;
      }
      else if (m_rhsIter && (!m_lhsIter || (m_lhsIter.index() > m_rhsIter.index())))
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
