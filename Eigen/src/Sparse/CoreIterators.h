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
    EIGEN_STRONG_INLINE InnerIterator(const Derived& mat, int outer)
      : m_matrix(mat), m_inner(0), m_outer(outer), m_end(mat.rows())
    {}

    EIGEN_STRONG_INLINE Scalar value() const
    {
      return (Derived::Flags&RowMajorBit) ? m_matrix.coeff(m_outer, m_inner)
                                          : m_matrix.coeff(m_inner, m_outer);
    }

    EIGEN_STRONG_INLINE InnerIterator& operator++() { m_inner++; return *this; }

    EIGEN_STRONG_INLINE int index() const { return m_inner; }

    EIGEN_STRONG_INLINE operator bool() const { return m_inner < m_end && m_inner>=0; }

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

    EIGEN_STRONG_INLINE InnerIterator(const Transpose& trans, int outer)
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

    EIGEN_STRONG_INLINE InnerIterator(const Block& block, int outer)
      : m_iter(block.m_matrix,(Block::Flags&RowMajor) ? block.m_startRow.value() + outer : block.m_startCol.value() + outer),
        m_start( (Block::Flags&RowMajor) ? block.m_startCol.value() : block.m_startRow.value()),
        m_end(m_start + ((Block::Flags&RowMajor) ? block.m_blockCols.value() : block.m_blockRows.value())),
        m_offset( (Block::Flags&RowMajor) ? block.m_startCol.value() : block.m_startRow.value())
    {
      while (m_iter.index()>=0 && m_iter.index()<m_start)
        ++m_iter;
    }

    EIGEN_STRONG_INLINE InnerIterator& operator++()
    {
      ++m_iter;
      return *this;
    }

    EIGEN_STRONG_INLINE Scalar value() const { return m_iter.value(); }

    EIGEN_STRONG_INLINE int index() const { return m_iter.index() - m_offset; }

    EIGEN_STRONG_INLINE operator bool() const { return m_iter && m_iter.index()<m_end; }

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

    EIGEN_STRONG_INLINE InnerIterator(const Block& block, int outer)
      : m_iter(block.m_matrix,(Block::Flags&RowMajor) ? block.m_startRow.value() + outer : block.m_startCol.value() + outer),
        m_start( (Block::Flags&RowMajor) ? block.m_startCol.value() : block.m_startRow.value()),
        m_end(m_start + ((Block::Flags&RowMajor) ? block.m_blockCols.value() : block.m_blockRows.value())),
        m_offset( (Block::Flags&RowMajor) ? block.m_startCol.value() : block.m_startRow.value())
    {
      while (m_iter.index()>=0 && m_iter.index()<m_start)
        ++m_iter;
    }

    EIGEN_STRONG_INLINE InnerIterator& operator++()
    {
      ++m_iter;
      return *this;
    }

    EIGEN_STRONG_INLINE Scalar value() const { return m_iter.value(); }

    EIGEN_STRONG_INLINE int index() const { return m_iter.index() - m_offset; }

    EIGEN_STRONG_INLINE operator bool() const { return m_iter && m_iter.index()<m_end; }

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

    EIGEN_STRONG_INLINE InnerIterator(const CwiseUnaryOp& unaryOp, int outer)
      : m_iter(unaryOp.m_matrix,outer), m_functor(unaryOp.m_functor), m_id(-1)
    {
      this->operator++();
    }

    EIGEN_STRONG_INLINE InnerIterator& operator++()
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

    EIGEN_STRONG_INLINE Scalar value() const { return m_value; }

    EIGEN_STRONG_INLINE int index() const { return m_id; }

    EIGEN_STRONG_INLINE operator bool() const { return m_id>=0; }

  protected:
    MatrixTypeIterator m_iter;
    const UnaryOp& m_functor;
    Scalar m_value;
    int m_id;
};

template<typename T> struct ei_is_scalar_product { enum { ret = false }; };
template<typename T> struct ei_is_scalar_product<ei_scalar_product_op<T> > { enum { ret = true }; };

template<typename BinaryOp, typename Lhs, typename Rhs, typename Derived>
class CwiseBinaryOpInnerIterator;

template<typename BinaryOp, typename Lhs, typename Rhs>
class CwiseBinaryOp<BinaryOp,Lhs,Rhs>::InnerIterator
  : public CwiseBinaryOpInnerIterator<BinaryOp,Lhs,Rhs, typename CwiseBinaryOp<BinaryOp,Lhs,Rhs>::InnerIterator>
{
    typedef CwiseBinaryOpInnerIterator<
      BinaryOp,Lhs,Rhs, typename CwiseBinaryOp<BinaryOp,Lhs,Rhs>::InnerIterator> Base;
  public:
    typedef typename CwiseBinaryOp::Scalar Scalar;
    typedef typename ei_traits<CwiseBinaryOp>::_LhsNested _LhsNested;
    typedef typename _LhsNested::InnerIterator LhsIterator;
    typedef typename ei_traits<CwiseBinaryOp>::_RhsNested _RhsNested;
    typedef typename _RhsNested::InnerIterator RhsIterator;
//   public:
    EIGEN_STRONG_INLINE InnerIterator(const CwiseBinaryOp& binOp, int outer)
      : Base(binOp.m_lhs,binOp.m_rhs,binOp.m_functor,outer)
    {}
};

template<typename BinaryOp, typename Lhs, typename Rhs, typename Derived>
class CwiseBinaryOpInnerIterator
{
    typedef CwiseBinaryOp<BinaryOp,Lhs,Rhs> ExpressionType;
    typedef typename ExpressionType::Scalar Scalar;
    typedef typename ei_traits<ExpressionType>::_LhsNested _LhsNested;
//     typedef typename ei_traits<ExpressionType>::LhsIterator LhsIterator;
    typedef typename ei_traits<ExpressionType>::_RhsNested _RhsNested;
//     typedef typename ei_traits<ExpressionType>::RhsIterator RhsIterator;
//     typedef typename ei_traits<CwiseBinaryOp>::_LhsNested _LhsNested;
    typedef typename _LhsNested::InnerIterator LhsIterator;
//     typedef typename ei_traits<CwiseBinaryOp>::_RhsNested _RhsNested;
    typedef typename _RhsNested::InnerIterator RhsIterator;
//     enum { IsProduct = ei_is_scalar_product<BinaryOp>::ret };
  public:

    EIGEN_STRONG_INLINE CwiseBinaryOpInnerIterator(const _LhsNested& lhs, const _RhsNested& rhs,
      const BinaryOp& functor, int outer)
      : m_lhsIter(lhs,outer), m_rhsIter(rhs,outer), m_functor(functor), m_id(-1)
    {
      this->operator++();
    }

    EIGEN_STRONG_INLINE Derived& operator++()
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
      return *static_cast<Derived*>(this);
    }

    EIGEN_STRONG_INLINE Scalar value() const { return m_value; }

    EIGEN_STRONG_INLINE int index() const { return m_id; }

    EIGEN_STRONG_INLINE operator bool() const { return m_id>=0; }

  protected:
    LhsIterator m_lhsIter;
    RhsIterator m_rhsIter;
    const BinaryOp& m_functor;
    Scalar m_value;
    int m_id;
};
/*
template<typename T, typename Lhs, typename Rhs, typename Derived>
class CwiseBinaryOpInnerIterator<ei_scalar_product_op<T>,Lhs,Rhs,Derived>
{
    typedef typename CwiseBinaryOp::Scalar Scalar;
    typedef typename ei_traits<CwiseBinaryOp>::_LhsNested _LhsNested;
    typedef typename _LhsNested::InnerIterator LhsIterator;
    typedef typename ei_traits<CwiseBinaryOp>::_RhsNested _RhsNested;
    typedef typename _RhsNested::InnerIterator RhsIterator;
  public:

    EIGEN_STRONG_INLINE CwiseBinaryOpInnerIterator(const CwiseBinaryOp& binOp, int outer)
      : m_lhsIter(binOp.m_lhs,outer), m_rhsIter(binOp.m_rhs,outer), m_functor(binOp.m_functor)//, m_id(-1)
    {
      //this->operator++();
      while (m_lhsIter && m_rhsIter && m_lhsIter.index() != m_rhsIter.index())
      {
        if (m_lhsIter.index() < m_rhsIter.index())
          ++m_lhsIter;
        else
          ++m_rhsIter;
      }
    }

    EIGEN_STRONG_INLINE Derived& operator++()
    {
//       m_id = -1;
      asm("#beginwhile");
      while (m_lhsIter && m_rhsIter)
      {
        if (m_lhsIter.index() == m_rhsIter.index())
        {
//           m_id = m_lhsIter.index();
          //m_value = m_functor(m_lhsIter.value(), m_rhsIter.value());
          ++m_lhsIter;
          ++m_rhsIter;
          break;
        }
        else if (m_lhsIter.index() < m_rhsIter.index())
          ++m_lhsIter;
        else
          ++m_rhsIter;
      }
      asm("#endwhile");
      return *static_cast<Derived*>(this);
    }

    EIGEN_STRONG_INLINE Scalar value() const { return m_functor(m_lhsIter.value(), m_rhsIter.value()); }

    EIGEN_STRONG_INLINE int index() const { return m_lhsIter.index(); }

    EIGEN_STRONG_INLINE operator bool() const { return m_lhsIter && m_rhsIter; }

  protected:
    LhsIterator m_lhsIter;
    RhsIterator m_rhsIter;
    const BinaryOp& m_functor;
//     Scalar m_value;
//     int m_id;
};*/

#endif // EIGEN_COREITERATORS_H
