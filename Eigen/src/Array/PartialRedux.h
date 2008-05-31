// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_PARTIAL_REDUX_H
#define EIGEN_PARTIAL_REDUX_H

/** \class PartialRedux
  *
  * \brief Generic expression of a partially reduxed matrix
  *
  * \param Direction indicates the direction of the redux (Vertical or Horizontal)
  * \param BinaryOp type of the binary functor implementing the operator (must be associative)
  * \param MatrixType the type of the matrix we are applying the redux operation
  *
  * This class represents an expression of a partial redux operator of a matrix.
  * It is the return type of MatrixBase::verticalRedux(), MatrixBase::horizontalRedux(),
  * and most of the time this is the only way it is used.
  *
  * \sa class CwiseBinaryOp
  */
template<int Direction, typename BinaryOp, typename MatrixType>
struct ei_traits<PartialRedux<Direction, BinaryOp, MatrixType> >
{
  typedef typename ei_result_of<
                     BinaryOp(typename MatrixType::Scalar)
                   >::type Scalar;
  typedef typename ei_nested<MatrixType>::type MatrixTypeNested;
  typedef typename ei_unref<MatrixTypeNested>::type _MatrixTypeNested;
  enum {
    RowsAtCompileTime = Direction==Vertical   ? 1 : MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = Direction==Horizontal ? 1 : MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = Direction==Vertical   ? 1 : MatrixType::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = Direction==Horizontal ? 1 : MatrixType::MaxColsAtCompileTime,
    Flags = ((int(RowsAtCompileTime) == Dynamic || int(ColsAtCompileTime) == Dynamic)
          ? (unsigned int)_MatrixTypeNested::Flags
          : (unsigned int)_MatrixTypeNested::Flags & ~LargeBit) & HereditaryBits,
    TraversalSize = Direction==Vertical ? RowsAtCompileTime : ColsAtCompileTime,
    CoeffReadCost = TraversalSize * _MatrixTypeNested::CoeffReadCost
                  + (TraversalSize - 1) * ei_functor_traits<BinaryOp>::Cost
  };
};

template<int Direction, typename BinaryOp, typename MatrixType>
class PartialRedux : ei_no_assignment_operator,
  public MatrixBase<PartialRedux<Direction, BinaryOp, MatrixType> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(PartialRedux)
    typedef typename ei_traits<PartialRedux>::MatrixTypeNested MatrixTypeNested;
    typedef typename ei_traits<PartialRedux>::_MatrixTypeNested _MatrixTypeNested;

    PartialRedux(const MatrixType& mat, const BinaryOp& func = BinaryOp())
      : m_matrix(mat), m_functor(func) {}

  private:

    int _rows() const { return (Direction==Vertical   ? 1 : m_matrix.rows()); }
    int _cols() const { return (Direction==Horizontal ? 1 : m_matrix.cols()); }

    const Scalar _coeff(int i, int j) const
    {
      if (Direction==Vertical)
        return m_matrix.col(j).redux(m_functor);
      else
        return m_matrix.row(i).redux(m_functor);
    }

  protected:
    const MatrixTypeNested m_matrix;
    const BinaryOp m_functor;
};

/** \returns a row vector expression of *this vertically reduxed by \a func
  *
  * The template parameter \a BinaryOp is the type of the functor
  * of the custom redux operator. Note that func must be an associative operator.
  *
  * \sa class PartialRedux, MatrixBase::horizontalRedux()
  */
template<typename Derived>
template<typename BinaryOp>
const PartialRedux<Vertical, BinaryOp, Derived>
MatrixBase<Derived>::verticalRedux(const BinaryOp& func) const
{
  return PartialRedux<Vertical, BinaryOp, Derived>(derived(), func);
}

/** \returns a row vector expression of *this horizontally reduxed by \a func
  *
  * The template parameter \a BinaryOp is the type of the functor
  * of the custom redux operator. Note that func must be an associative operator.
  *
  * \sa class PartialRedux, MatrixBase::verticalRedux()
  */
template<typename Derived>
template<typename BinaryOp>
const PartialRedux<Horizontal, BinaryOp, Derived>
MatrixBase<Derived>::horizontalRedux(const BinaryOp& func) const
{
  return PartialRedux<Horizontal, BinaryOp, Derived>(derived(), func);
}

#endif // EIGEN_PARTIAL_REDUX_H
