// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_ARRAYWRAPPER_H
#define EIGEN_ARRAYWRAPPER_H

/** \class ArrayWrapper
  *
  * \brief Expression of a mathematical vector or matrix as an array object
  *
  * This class is the return type of MatrixBase::array(), and most of the time
  * this is the only way it is use.
  *
  * \sa MatrixBase::array(), class MatrixWrapper
  */
template<typename ExpressionType>
struct ei_traits<ArrayWrapper<ExpressionType> >
  : public ei_traits<typename ei_cleantype<typename ExpressionType::Nested>::type >
{
  typedef DenseStorageArray DenseStorageType;
};

template<typename ExpressionType>
class ArrayWrapper : public ArrayBase<ArrayWrapper<ExpressionType> >
{
  public:
    typedef ArrayBase<ArrayWrapper> Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(ArrayWrapper)
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(ArrayWrapper)

    typedef typename ei_nested<ExpressionType>::type NestedExpressionType;

    inline ArrayWrapper(const ExpressionType& matrix) : m_expression(matrix) {}

    inline int rows() const { return m_expression.rows(); }
    inline int cols() const { return m_expression.cols(); }
    inline int outerStride() const { return m_expression.outerStride(); }
    inline int innerStride() const { return m_expression.innerStride(); }

    inline const CoeffReturnType coeff(int row, int col) const
    {
      return m_expression.coeff(row, col);
    }

    inline Scalar& coeffRef(int row, int col)
    {
      return m_expression.const_cast_derived().coeffRef(row, col);
    }

    inline const CoeffReturnType coeff(int index) const
    {
      return m_expression.coeff(index);
    }

    inline Scalar& coeffRef(int index)
    {
      return m_expression.const_cast_derived().coeffRef(index);
    }

    template<int LoadMode>
    inline const PacketScalar packet(int row, int col) const
    {
      return m_expression.template packet<LoadMode>(row, col);
    }

    template<int LoadMode>
    inline void writePacket(int row, int col, const PacketScalar& x)
    {
      m_expression.const_cast_derived().template writePacket<LoadMode>(row, col, x);
    }

    template<int LoadMode>
    inline const PacketScalar packet(int index) const
    {
      return m_expression.template packet<LoadMode>(index);
    }

    template<int LoadMode>
    inline void writePacket(int index, const PacketScalar& x)
    {
      m_expression.const_cast_derived().template writePacket<LoadMode>(index, x);
    }

    template<typename Dest>
    inline void evalTo(Dest& dst) const { dst = m_expression; }

  protected:
    const NestedExpressionType m_expression;
};

/** \class MatrixWrapper
  *
  * \brief Expression of an array as a mathematical vector or matrix
  *
  * This class is the return type of ArrayBase::matrix(), and most of the time
  * this is the only way it is use.
  *
  * \sa MatrixBase::matrix(), class ArrayWrapper
  */

template<typename ExpressionType>
struct ei_traits<MatrixWrapper<ExpressionType> >
 : public ei_traits<typename ei_cleantype<typename ExpressionType::Nested>::type >
{
  typedef DenseStorageMatrix DenseStorageType;
};

template<typename ExpressionType>
class MatrixWrapper : public MatrixBase<MatrixWrapper<ExpressionType> >
{
  public:
    typedef MatrixBase<MatrixWrapper<ExpressionType> > Base;
    EIGEN_DENSE_PUBLIC_INTERFACE(MatrixWrapper)
    EIGEN_INHERIT_ASSIGNMENT_OPERATORS(MatrixWrapper);

    typedef typename ei_nested<ExpressionType>::type NestedExpressionType;

    inline MatrixWrapper(const ExpressionType& matrix) : m_expression(matrix) {}

    inline int rows() const { return m_expression.rows(); }
    inline int cols() const { return m_expression.cols(); }
    inline int outerStride() const { return m_expression.outerStride(); }
    inline int innerStride() const { return m_expression.innerStride(); }

    inline const CoeffReturnType coeff(int row, int col) const
    {
      return m_expression.coeff(row, col);
    }

    inline Scalar& coeffRef(int row, int col)
    {
      return m_expression.const_cast_derived().coeffRef(row, col);
    }

    inline const CoeffReturnType coeff(int index) const
    {
      return m_expression.coeff(index);
    }

    inline Scalar& coeffRef(int index)
    {
      return m_expression.const_cast_derived().coeffRef(index);
    }

    template<int LoadMode>
    inline const PacketScalar packet(int row, int col) const
    {
      return m_expression.template packet<LoadMode>(row, col);
    }

    template<int LoadMode>
    inline void writePacket(int row, int col, const PacketScalar& x)
    {
      m_expression.const_cast_derived().template writePacket<LoadMode>(row, col, x);
    }

    template<int LoadMode>
    inline const PacketScalar packet(int index) const
    {
      return m_expression.template packet<LoadMode>(index);
    }

    template<int LoadMode>
    inline void writePacket(int index, const PacketScalar& x)
    {
      m_expression.const_cast_derived().template writePacket<LoadMode>(index, x);
    }

  protected:
    const NestedExpressionType m_expression;
};

#endif // EIGEN_ARRAYWRAPPER_H
