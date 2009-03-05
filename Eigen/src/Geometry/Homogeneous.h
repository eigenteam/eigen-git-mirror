// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
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

#ifndef EIGEN_HOMOGENEOUS_H
#define EIGEN_HOMOGENEOUS_H

/** \geometry_module \ingroup Geometry_Module
  * \nonstableyet 
  * \class Homogeneous
  *
  * \brief Expression of one (or a set of) homogeneous vector(s)
  *
  * \param MatrixType the type of the object in which we are making homogeneous
  *
  * This class represents an expression of one (or a set of) homogeneous vector(s).
  * It is the return type of MatrixBase::homogeneous() and most of the time
  * this is the only way it is used.
  *
  * \sa MatrixBase::homogeneous()
  */
template<typename MatrixType,int Direction>
struct ei_traits<Homogeneous<MatrixType,Direction> >
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename ei_nested<MatrixType>::type MatrixTypeNested;
  typedef typename ei_unref<MatrixTypeNested>::type _MatrixTypeNested;
  enum {
    RowsPlusOne = (MatrixType::RowsAtCompileTime != Dynamic) ?
                  int(MatrixType::RowsAtCompileTime) + 1 : Dynamic,
    ColsPlusOne = (MatrixType::ColsAtCompileTime != Dynamic) ?
                  int(MatrixType::ColsAtCompileTime) + 1 : Dynamic,
    RowsAtCompileTime = Direction==Vertical  ?  RowsPlusOne : MatrixType::RowsAtCompileTime,
    ColsAtCompileTime = Direction==Horizontal ? ColsPlusOne : MatrixType::ColsAtCompileTime,
    MaxRowsAtCompileTime = RowsAtCompileTime,
    MaxColsAtCompileTime = ColsAtCompileTime,
    Flags = _MatrixTypeNested::Flags & HereditaryBits,
    CoeffReadCost = _MatrixTypeNested::CoeffReadCost
  };
};

template<typename MatrixType,int Direction> class Homogeneous
  : public MatrixBase<Homogeneous<MatrixType,Direction> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Homogeneous)

    inline Homogeneous(const MatrixType& matrix)
      : m_matrix(matrix)
    {}

    inline int rows() const { return m_matrix.rows() + (Direction==Vertical   ? 1 : 0); }
    inline int cols() const { return m_matrix.cols() + (Direction==Horizontal ? 1 : 0); }

    inline Scalar coeff(int row, int col) const
    {
      if(  (Direction==Vertical   && row==m_matrix.rows())
        || (Direction==Horizontal && col==m_matrix.cols()))
        return 1;
      return m_matrix.coeff(row, col);
    }

  protected:
    const typename MatrixType::Nested m_matrix;
};

/** \geometry_module
  * \nonstableyet 
  * \return an expression of the equivalent homogeneous vector
  * 
  * \vectoronly
  *
  * Example: \include MatrixBase_homogeneous.cpp
  * Output: \verbinclude MatrixBase_homogeneous.out
  *
  * \sa class Homogeneous
  */
template<typename Derived>
inline const Homogeneous<Derived,MatrixBase<Derived>::ColsAtCompileTime==1?Vertical:Horizontal>
MatrixBase<Derived>::homogeneous() const
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  return derived();
}

/** \geometry_module
  * \nonstableyet 
  * \returns a matrix expression of homogeneous column (or row) vectors
  *
  * Example: \include PartialRedux_homogeneous.cpp
  * Output: \verbinclude PartialRedux_homogeneous.out
  *
  * \sa MatrixBase::homogeneous() */
template<typename ExpressionType, int Direction>
inline const Homogeneous<ExpressionType,Direction>
PartialRedux<ExpressionType,Direction>::homogeneous() const
{
  return _expression();
}

/** \geometry_module
  * \nonstableyet 
  * \returns an expression of the homogeneous normalized vector of \c *this
  *
  * Example: \include MatrixBase_hnormalized.cpp
  * Output: \verbinclude MatrixBase_hnormalized.out
  *
  * \sa PartialRedux::hnormalized() */
template<typename Derived>
inline const typename MatrixBase<Derived>::HNormalizedReturnType
MatrixBase<Derived>::hnormalized() const
{
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Derived);
  return StartMinusOne(derived(),0,0,
    ColsAtCompileTime==1?size()-1:1,
    ColsAtCompileTime==1?1:size()-1).nestByValue() / coeff(size()-1);
}

/** \geometry_module
  * \nonstableyet 
  * \returns an expression of the homogeneous normalized vector of \c *this
  *
  * Example: \include DirectionWise_hnormalized.cpp
  * Output: \verbinclude DirectionWise_hnormalized.out
  *
  * \sa MatrixBase::hnormalized() */
template<typename ExpressionType, int Direction>
inline const typename PartialRedux<ExpressionType,Direction>::HNormalizedReturnType
PartialRedux<ExpressionType,Direction>::hnormalized() const
{
  return HNormalized_Block(_expression(),0,0,
      Direction==Vertical   ? _expression().rows()-1 : _expression().rows(),
      Direction==Horizontal ? _expression().cols()-1 : _expression().cols()).nestByValue()
    .cwise()/
      Replicate<NestByValue<HNormalized_Factors>,
                Direction==Vertical   ? HNormalized_SizeMinusOne : 1,
                Direction==Horizontal ? HNormalized_SizeMinusOne : 1>
        (HNormalized_Factors(_expression(),
          Direction==Vertical    ? _expression().rows()-1:0,
          Direction==Horizontal  ? _expression().cols()-1:0,
          Direction==Vertical    ? 1 : _expression().rows(),
          Direction==Horizontal  ? 1 : _expression().cols()).nestByValue(),
         Direction==Vertical   ? _expression().rows()-1 : 1,
         Direction==Horizontal ? _expression().cols()-1 : 1).nestByValue();
}

#endif // EIGEN_HOMOGENEOUS_H
