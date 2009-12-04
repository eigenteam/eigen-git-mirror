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

#ifndef EIGEN_ARRAYWRAPPER_H
#define EIGEN_ARRAYWRAPPER_H

template<typename ExpressionType>
struct ei_traits<ArrayWrapper<ExpressionType> >
 : public ei_traits<ExpressionType>
{};

template<typename ExpressionType>
class ArrayWrapper : public ArrayBase<ArrayWrapper<ExpressionType> >
{
  public:
    EIGEN_GENERIC_PUBLIC_INTERFACE(ArrayWrapper)

    inline ArrayWrapper(const ExpressionType& matrix) : m_expression(matrix) {}

    inline int rows() const { return m_expression.rows(); }
    inline int cols() const { return m_expression.cols(); }
    inline int stride() const { return m_expression.stride(); }

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
    const ExpressionType& m_expression;
};

template<typename ExpressionType>
struct ei_traits<MatrixWrapper<ExpressionType> >
 : public ei_traits<ExpressionType>
{};

template<typename ExpressionType>
class MatrixWrapper : public MatrixBase<MatrixWrapper<ExpressionType> >
{
  public:
    EIGEN_GENERIC_PUBLIC_INTERFACE(MatrixWrapper)

    inline MatrixWrapper(const ExpressionType& matrix) : m_expression(matrix) {}

    inline int rows() const { return m_expression.rows(); }
    inline int cols() const { return m_expression.cols(); }
    inline int stride() const { return m_expression.stride(); }

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
    const ExpressionType& m_expression;
};

#endif // EIGEN_ARRAYWRAPPER_H
