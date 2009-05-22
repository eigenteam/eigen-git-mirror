// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_REPLICATE_H
#define EIGEN_REPLICATE_H

/** \nonstableyet
  * \class Replicate
  *
  * \brief Expression of the multiple replication of a matrix or vector
  *
  * \param MatrixType the type of the object we are replicating
  *
  * This class represents an expression of the multiple replication of a matrix or vector.
  * It is the return type of MatrixBase::replicate() and most of the time
  * this is the only way it is used.
  *
  * \sa MatrixBase::replicate()
  */
template<typename MatrixType,int RowFactor,int ColFactor>
struct ei_traits<Replicate<MatrixType,RowFactor,ColFactor> >
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename ei_nested<MatrixType>::type MatrixTypeNested;
  typedef typename ei_unref<MatrixTypeNested>::type _MatrixTypeNested;
  enum {
    RowsPlusOne = (MatrixType::RowsAtCompileTime != Dynamic) ?
                  int(MatrixType::RowsAtCompileTime) + 1 : Dynamic,
    ColsPlusOne = (MatrixType::ColsAtCompileTime != Dynamic) ?
                  int(MatrixType::ColsAtCompileTime) + 1 : Dynamic,
    RowsAtCompileTime = RowFactor==Dynamic || MatrixType::RowsAtCompileTime==Dynamic
                      ? Dynamic
                      : RowFactor * MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = ColFactor==Dynamic || MatrixType::ColsAtCompileTime==Dynamic
                      ? Dynamic
                      : ColFactor * MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = RowsAtCompileTime,
    MaxColsAtCompileTime = ColsAtCompileTime,
    Flags = _MatrixTypeNested::Flags & HereditaryBits,
    CoeffReadCost = _MatrixTypeNested::CoeffReadCost
  };
};

template<typename MatrixType,int RowFactor,int ColFactor> class Replicate
  : public MatrixBase<Replicate<MatrixType,RowFactor,ColFactor> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Replicate)

    inline Replicate(const MatrixType& matrix)
      : m_matrix(matrix), m_rowFactor(RowFactor), m_colFactor(ColFactor)
    {
      ei_assert(RowFactor!=Dynamic && ColFactor!=Dynamic);
    }

    inline Replicate(const MatrixType& matrix, int rowFactor, int colFactor)
      : m_matrix(matrix), m_rowFactor(rowFactor), m_colFactor(colFactor)
    {}

    inline int rows() const { return m_matrix.rows() * m_rowFactor.value(); }
    inline int cols() const { return m_matrix.cols() * m_colFactor.value(); }

    inline Scalar coeff(int row, int col) const
    {
      return m_matrix.coeff(row%m_matrix.rows(), col%m_matrix.cols());
    }

  protected:
    const typename MatrixType::Nested m_matrix;
    const ei_int_if_dynamic<RowFactor> m_rowFactor;
    const ei_int_if_dynamic<ColFactor> m_colFactor;
};

/** \nonstableyet
  * \return an expression of the replication of \c *this
  *
  * Example: \include MatrixBase_replicate.cpp
  * Output: \verbinclude MatrixBase_replicate.out
  *
  * \sa PartialRedux::replicate(), MatrixBase::replicate(int,int), class Replicate
  */
template<typename Derived>
template<int RowFactor, int ColFactor>
inline const Replicate<Derived,RowFactor,ColFactor>
MatrixBase<Derived>::replicate() const
{
  return derived();
}

/** \nonstableyet
  * \return an expression of the replication of \c *this
  *
  * Example: \include MatrixBase_replicate_int_int.cpp
  * Output: \verbinclude MatrixBase_replicate_int_int.out
  *
  * \sa PartialRedux::replicate(), MatrixBase::replicate<int,int>(), class Replicate
  */
template<typename Derived>
inline const Replicate<Derived,Dynamic,Dynamic>
MatrixBase<Derived>::replicate(int rowFactor,int colFactor) const
{
  return Replicate<Derived,Dynamic,Dynamic>(derived(),rowFactor,colFactor);
}

/** \nonstableyet
  * \return an expression of the replication of each column (or row) of \c *this
  *
  * Example: \include DirectionWise_replicate_int.cpp
  * Output: \verbinclude DirectionWise_replicate_int.out
  *
  * \sa PartialRedux::replicate(), MatrixBase::replicate(), class Replicate
  */
template<typename ExpressionType, int Direction>
const Replicate<ExpressionType,(Direction==Vertical?Dynamic:1),(Direction==Horizontal?Dynamic:1)>
PartialRedux<ExpressionType,Direction>::replicate(int factor) const
{
  return Replicate<ExpressionType,Direction==Vertical?Dynamic:1,Direction==Horizontal?Dynamic:1>
          (_expression(),Direction==Vertical?factor:1,Direction==Horizontal?factor:1);
}

/** \nonstableyet
  * \return an expression of the replication of each column (or row) of \c *this
  *
  * Example: \include DirectionWise_replicate.cpp
  * Output: \verbinclude DirectionWise_replicate.out
  *
  * \sa PartialRedux::replicate(int), MatrixBase::replicate(), class Replicate
  */
template<typename ExpressionType, int Direction>
template<int Factor>
const Replicate<ExpressionType,(Direction==Vertical?Factor:1),(Direction==Horizontal?Factor:1)>
PartialRedux<ExpressionType,Direction>::replicate(int factor) const
{
  return Replicate<ExpressionType,Direction==Vertical?Factor:1,Direction==Horizontal?Factor:1>
          (_expression(),Direction==Vertical?factor:1,Direction==Horizontal?factor:1);
}

#endif // EIGEN_REPLICATE_H
