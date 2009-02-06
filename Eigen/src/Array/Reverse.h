// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2009 Ricard Marxer <email@ricardmarxer.com>
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

#ifndef EIGEN_REVERSE_H
#define EIGEN_REVERSE_H

#include <iostream>
using namespace std;

/** \array_module \ingroup Array
  *
  * \class Reverse
  *
  * \brief Expression of the reverse of a vector or matrix
  *
  * \param MatrixType the type of the object of which we are taking the reverse
  *
  * This class represents an expression of the reverse of a vector.
  * It is the return type of MatrixBase::reverse() and PartialRedux::reverse()
  * and most of the time this is the only way it is used.
  *
  * \sa MatrixBase::reverse(), PartialRedux::reverse()
  */
template<typename MatrixType, int Direction>
struct ei_traits<Reverse<MatrixType, Direction> >
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename ei_nested<MatrixType>::type MatrixTypeNested;
  typedef typename ei_unref<MatrixTypeNested>::type _MatrixTypeNested;
  enum {
    RowsAtCompileTime = MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime,

    // TODO: check how to correctly set the new flags
    Flags = ((int(_MatrixTypeNested::Flags) & HereditaryBits)
          & ~(LowerTriangularBit | UpperTriangularBit))
          | (int(_MatrixTypeNested::Flags)&UpperTriangularBit ? LowerTriangularBit : 0)
          | (int(_MatrixTypeNested::Flags)&LowerTriangularBit ? UpperTriangularBit : 0),

    // TODO: should add two add costs (due to the -1) or only one, and add the cost of calling .rows() and .cols()
    CoeffReadCost = _MatrixTypeNested::CoeffReadCost
  };
};

template<typename MatrixType, int Direction> class Reverse
  : public MatrixBase<Reverse<MatrixType, Direction> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Reverse)

    inline Reverse(const MatrixType& matrix) : m_matrix(matrix) { }

    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(Reverse)

    inline int rows() const { return m_matrix.rows(); }
    inline int cols() const { return m_matrix.cols(); }

    inline Scalar& coeffRef(int row, int col)
    {
      return m_matrix.const_cast_derived().coeffRef(((Direction == Vertical) || (Direction == BothDirections)) ? m_matrix.rows() - row - 1 : row,
                                                    ((Direction == Horizontal) || (Direction == BothDirections)) ? m_matrix.cols() - col - 1 : col);
    }

    inline const Scalar coeff(int row, int col) const
    {
      return m_matrix.coeff(((Direction == Vertical) || (Direction == BothDirections)) ? m_matrix.rows() - row - 1 : row,
                            ((Direction == Horizontal) || (Direction == BothDirections)) ? m_matrix.cols() - col - 1 : col);
    }

    /* could be removed */
    /*
    inline const Scalar coeff(int index) const
    {
      switch ( Direction )
        {
        case Vertical:
          return m_matrix.coeff( index + m_matrix.rows() - 2 * (index % m_matrix.rows()) - 1 );
          break;

        case Horizontal:
          return m_matrix.coeff( (index % m_matrix.rows()) + (m_matrix.cols() - 1 - index/m_matrix.rows()) * m_matrix.rows() );
          break;

        case BothDirections:
          return m_matrix.coeff((m_matrix.rows() * m_matrix.cols()) - index - 1);
          break;
        }

    }

    inline Scalar& coeffRef(int index)
    {
      switch ( Direction )
        {
        case Vertical:
          return m_matrix.const_cast_derived().coeffRef( index + m_matrix.rows() - 2 * (index % m_matrix.rows()) - 1 );
          break;

        case Horizontal:
          return m_matrix.const_cast_derived().coeffRef( (index % m_matrix.rows()) + (m_matrix.cols() - 1 - index/m_matrix.rows()) * m_matrix.rows() );
          break;

        case BothDirections:
          return m_matrix.const_cast_derived().coeffRef( (m_matrix.rows() * m_matrix.cols()) - index - 1 );
          break;
        }
    }
    */

    /* the following is not ready yet */
    /*
    // TODO: We must reverse the packet reading and writing, which is currently not done here, I think
    template<int LoadMode>
    inline const PacketScalar packet(int row, int col) const
    {
      return m_matrix.template packet<LoadMode>(((Direction == Vertical) || (Direction == BothDirections)) ? m_matrix.rows() - row - 1 : row,
                                                ((Direction == Horizontal) || (Direction == BothDirections)) ? m_matrix.cols() - col - 1 : col);
    }

    template<int LoadMode>
    inline void writePacket(int row, int col, const PacketScalar& x)
    {
      m_matrix.const_cast_derived().template writePacket<LoadMode>(((Direction == Vertical) || (Direction == BothDirections)) ? m_matrix.rows() - row - 1 : row,
                                                                   ((Direction == Horizontal) || (Direction == BothDirections)) ? m_matrix.cols() - col - 1 : col,
                                                                   x);
    }

    template<int LoadMode>
    inline const PacketScalar packet(int index) const
    {
      switch ( Direction )
        {
        case Vertical:
          return m_matrix.template packet<LoadMode>( index + m_matrix.rows() - 2 * (index % m_matrix.rows()) - 1 );
          break;

        case Horizontal:
          return m_matrix.template packet<LoadMode>( (index % m_matrix.rows()) + (m_matrix.cols() - 1 - index/m_matrix.rows()) * m_matrix.rows() );
          break;

        case BothDirections:
          return m_matrix.template packet<LoadMode>( (m_matrix.rows() * m_matrix.cols()) - index - 1 );
          break;
        }
    }
    */

    /* could be removed */
    /*
    template<int LoadMode>
    inline void writePacket(int index, const PacketScalar& x)
    {
      switch ( Direction )
        {
        case Vertical:
          return m_matrix.const_cast_derived().template packet<LoadMode>( index + m_matrix.rows() - 2 * (index % m_matrix.rows()) - 1, x );
          break;

        case Horizontal:
          return m_matrix.const_cast_derived().template packet<LoadMode>( (index % m_matrix.rows()) + (m_matrix.cols() - 1 - index/m_matrix.rows()) * m_matrix.rows(), x );
          break;

        case BothDirections:
          return m_matrix.const_cast_derived().template packet<LoadMode>( (m_matrix.rows() * m_matrix.cols()) - index - 1, x );
          break;
        }
    }
    */

  protected:
    const typename MatrixType::Nested m_matrix;
};

/** \returns an expression of the reverse of *this.
  *
  * Example: \include MatrixBase_reverse.cpp
  * Output: \verbinclude MatrixBase_reverse.out
  *
  */
template<typename Derived>
inline Reverse<Derived, BothDirections>
MatrixBase<Derived>::reverse()
{
  return derived();
}

/** This is the const version of reverse(). */
template<typename Derived>
inline const Reverse<Derived, BothDirections>
MatrixBase<Derived>::reverse() const
{
  return derived();
}

/** This is the "in place" version of reverse: it reverses \c *this.
  *
  * In most cases it is probably better to simply use the reversed expression
  * of a matrix. However, when reversing the matrix data itself is really needed,
  * then this "in-place" version is probably the right choice because it provides
  * the following additional features:
  *  - less error prone: doing the same operation with .reverse() requires special care:
  *    \code m = m.reverse().eval(); \endcode
  *  - no temporary object is created (currently there is one created but could be avoided using swap)
  *  - it allows future optimizations (cache friendliness, etc.)
  *
  * \sa reverse() */
template<typename Derived>
inline void MatrixBase<Derived>::reverseInPlace()
{
  derived() = derived().reverse().eval();
}


#endif // EIGEN_REVERSE_H
