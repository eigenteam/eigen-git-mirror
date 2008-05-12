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

#ifndef EIGEN_REDUX_H
#define EIGEN_REDUX_H

template<typename BinaryOp, typename Derived, int Start, int Length>
struct ei_redux_unroller
{
  enum {
    HalfLength = Length/2
  };

  typedef typename ei_result_of<BinaryOp(typename Derived::Scalar)>::type Scalar;

  static Scalar run(const Derived &mat, const BinaryOp& func)
  {
    return func(
      ei_redux_unroller<BinaryOp, Derived, Start, HalfLength>::run(mat, func),
      ei_redux_unroller<BinaryOp, Derived, Start+HalfLength, Length - HalfLength>::run(mat, func));
  }
};

template<typename BinaryOp, typename Derived, int Start>
struct ei_redux_unroller<BinaryOp, Derived, Start, 1>
{
  enum {
    col = Start / Derived::RowsAtCompileTime,
    row = Start % Derived::RowsAtCompileTime
  };

  typedef typename ei_result_of<BinaryOp(typename Derived::Scalar)>::type Scalar;

  static Scalar run(const Derived &mat, const BinaryOp &)
  {
    return mat.coeff(row, col);
  }
};

template<typename BinaryOp, typename Derived, int Start>
struct ei_redux_unroller<BinaryOp, Derived, Start, Dynamic>
{
  typedef typename ei_result_of<BinaryOp(typename Derived::Scalar)>::type Scalar;
  static Scalar run(const Derived&, const BinaryOp&) { return Scalar(); }
};


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
    Flags = ((RowsAtCompileTime == Dynamic || ColsAtCompileTime == Dynamic)
          ? (unsigned int)_MatrixTypeNested::Flags
          : (unsigned int)_MatrixTypeNested::Flags & ~LargeBit) & DefaultLostFlagMask,
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


/** \returns the result of a full redux operation on the whole matrix or vector using \a func
  *
  * The template parameter \a BinaryOp is the type of the functor \a func which must be
  * an assiociative operator. Both current STL and TR1 functor styles are handled.
  *
  * \sa MatrixBase::sum(), MatrixBase::minCoeff(), MatrixBase::maxCoeff(), MatrixBase::verticalRedux(), MatrixBase::horizontalRedux()
  */
template<typename Derived>
template<typename BinaryOp>
typename ei_result_of<BinaryOp(typename ei_traits<Derived>::Scalar)>::type
MatrixBase<Derived>::redux(const BinaryOp& func) const
{
  const bool unroll = SizeAtCompileTime * CoeffReadCost
                    + (SizeAtCompileTime-1) * ei_functor_traits<BinaryOp>::Cost
                    <= EIGEN_UNROLLING_LIMIT;
  if(unroll)
    return ei_redux_unroller<BinaryOp, Derived, 0,
                             unroll ? SizeAtCompileTime : Dynamic>
           ::run(derived(), func);
  else
  {
    Scalar res;
    res = coeff(0,0);
    for(int i = 1; i < rows(); i++)
      res = func(res, coeff(i, 0));
    for(int j = 1; j < cols(); j++)
      for(int i = 0; i < rows(); i++)
        res = func(res, coeff(i, j));
    return res;
  }
}

/** \returns the sum of all coefficients of *this
  *
  * \sa trace()
  */
template<typename Derived>
typename ei_traits<Derived>::Scalar
inline MatrixBase<Derived>::sum() const
{
  return this->redux(Eigen::ei_scalar_sum_op<Scalar>());
}

/** \returns the trace of \c *this, i.e. the sum of the coefficients on the main diagonal.
  *
  * \c *this can be any matrix, not necessarily square.
  *
  * \sa diagonal(), sum()
  */
template<typename Derived>
typename ei_traits<Derived>::Scalar
inline MatrixBase<Derived>::trace() const
{
  return diagonal().sum();
}

/** \returns the minimum of all coefficients of *this
  */
template<typename Derived>
typename ei_traits<Derived>::Scalar
inline MatrixBase<Derived>::minCoeff() const
{
  return this->redux(Eigen::ei_scalar_min_op<Scalar>());
}

/** \returns the maximum of all coefficients of *this
  */
template<typename Derived>
typename ei_traits<Derived>::Scalar
inline MatrixBase<Derived>::maxCoeff() const
{
  return this->redux(Eigen::ei_scalar_max_op<Scalar>());
}



template<typename Derived, int UnrollCount>
struct ei_all_unroller
{
  enum {
    col = (UnrollCount-1) / Derived::RowsAtCompileTime,
    row = (UnrollCount-1) % Derived::RowsAtCompileTime
  };

  inline static bool run(const Derived &mat)
  {
    return ei_all_unroller<Derived, UnrollCount-1>::run(mat) && mat.coeff(row, col);
  }
};

template<typename Derived>
struct ei_all_unroller<Derived, 1>
{
  inline static bool run(const Derived &mat) { return mat.coeff(0, 0); }
};

template<typename Derived>
struct ei_all_unroller<Derived, Dynamic>
{
  inline static bool run(const Derived &) { return false; }
};

template<typename Derived, int UnrollCount>
struct ei_any_unroller
{
  enum {
    col = (UnrollCount-1) / Derived::RowsAtCompileTime,
    row = (UnrollCount-1) % Derived::RowsAtCompileTime
  };

  inline static bool run(const Derived &mat)
  {
    return ei_any_unroller<Derived, UnrollCount-1>::run(mat) || mat.coeff(row, col);
  }
};

template<typename Derived>
struct ei_any_unroller<Derived, 1>
{
  inline static bool run(const Derived &mat) { return mat.coeff(0, 0); }
};

template<typename Derived>
struct ei_any_unroller<Derived, Dynamic>
{
  inline static bool run(const Derived &) { return false; }
};

/** \returns true if all coefficients are true
  *
  * \sa MatrixBase::any()
  */
template<typename Derived>
bool MatrixBase<Derived>::all(void) const
{
  const bool unroll = SizeAtCompileTime * (CoeffReadCost + NumTraits<Scalar>::AddCost)
                      <= EIGEN_UNROLLING_LIMIT;
  if(unroll)
    return ei_all_unroller<Derived,
                           unroll ? SizeAtCompileTime : Dynamic
     >::run(derived());
  else
  {
    for(int j = 0; j < cols(); j++)
      for(int i = 0; i < rows(); i++)
        if (!coeff(i, j)) return false;
    return true;
  }
}

/** \returns true if at least one coefficient is true
  *
  * \sa MatrixBase::any()
  */
template<typename Derived>
bool MatrixBase<Derived>::any(void) const
{
  const bool unroll = SizeAtCompileTime * (CoeffReadCost + NumTraits<Scalar>::AddCost)
                      <= EIGEN_UNROLLING_LIMIT;
  if(unroll)
    return ei_any_unroller<Derived,
                           unroll ? SizeAtCompileTime : Dynamic
           >::run(derived());
  else
  {
    for(int j = 0; j < cols(); j++)
      for(int i = 0; i < rows(); i++)
        if (coeff(i, j)) return true;
    return false;
  }
}

#endif // EIGEN_REDUX_H
