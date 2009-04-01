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

#ifndef EIGEN_AUTODIFF_VECTOR_H
#define EIGEN_AUTODIFF_VECTOR_H

namespace Eigen {

/* \class AutoDiffScalar
  * \brief A scalar type replacement with automatic differentation capability
  *
  * \param DerType the vector type used to store/represent the derivatives (e.g. Vector3f)
  *
  * This class represents a scalar value while tracking its respective derivatives.
  *
  * It supports the following list of global math function:
  *  - std::abs, std::sqrt, std::pow, std::exp, std::log, std::sin, std::cos, 
  *  - ei_abs, ei_sqrt, ei_pow, ei_exp, ei_log, ei_sin, ei_cos,
  *  - ei_conj, ei_real, ei_imag, ei_abs2.
  *
  * AutoDiffScalar can be used as the scalar type of an Eigen::Matrix object. However,
  * in that case, the expression template mechanism only occurs at the top Matrix level,
  * while derivatives are computed right away.
  *
  */
template<typename ValueType, typename JacobianType>
class AutoDiffVector
{
  public:
    typedef typename ei_traits<ValueType>::Scalar Scalar;
    
    inline AutoDiffVector() {}
    
    inline AutoDiffVector(const ValueType& values)
      : m_values(values)
    {
      m_jacobian.setZero();
    }
    
    inline AutoDiffVector(const ValueType& values, const JacobianType& jac)
      : m_values(values), m_jacobian(jac)
    {}
    
    template<typename OtherValueType, typename OtherJacobianType>
    inline AutoDiffVector(const AutoDiffVector<OtherValueType, OtherJacobianType>& other)
      : m_values(other.values()), m_jacobian(other.jacobian())
    {}
    
    inline AutoDiffVector(const AutoDiffVector& other)
      : m_values(other.values()), m_jacobian(other.jacobian())
    {}
    
    template<typename OtherValueType, typename OtherJacobianType>
    inline AutoDiffScalar& operator=(const AutoDiffVector<OtherValueType, OtherJacobianType>& other)
    {
      m_values = other.values();
      m_jacobian = other.jacobian();
      return *this;
    }
    
    inline AutoDiffVector& operator=(const AutoDiffVector& other)
    {
      m_values = other.values();
      m_jacobian = other.jacobian();
      return *this;
    }
    
    inline const ValueType& values() const { return m_values; }
    inline ValueType& values() { return m_values; }
    
    inline const JacobianType& jacobian() const { return m_jacobian; }
    inline JacobianType& jacobian() { return m_jacobian; }
    
    template<typename OtherValueType,typename OtherJacobianType>
    inline const AutoDiffVector<
      CwiseBinaryOp<ei_scalar_sum_op<Scalar>,ValueType,OtherValueType> >
      CwiseBinaryOp<ei_scalar_sum_op<Scalar>,JacobianType,OtherJacobianType> >
    operator+(const AutoDiffScalar<OtherDerType>& other) const
    {
      return AutoDiffVector<
      CwiseBinaryOp<ei_scalar_sum_op<Scalar>,ValueType,OtherValueType> >
      CwiseBinaryOp<ei_scalar_sum_op<Scalar>,JacobianType,OtherJacobianType> >(
        m_values + other.values(),
        m_jacobian + other.jacobian());
    }
    
    template<typename OtherValueType, typename OtherJacobianType>
    inline AutoDiffVector&
    operator+=(const AutoDiffVector<OtherValueType,OtherDerType>& other)
    {
      m_values += other.values();
      m_jacobian += other.jacobian();
      return *this;
    }
    
    template<typename OtherValueType,typename OtherJacobianType>
    inline const AutoDiffVector<
      CwiseBinaryOp<ei_scalar_difference_op<Scalar>,ValueType,OtherValueType> >
      CwiseBinaryOp<ei_scalar_difference_op<Scalar>,JacobianType,OtherJacobianType> >
    operator-(const AutoDiffScalar<OtherDerType>& other) const
    {
      return AutoDiffVector<
      CwiseBinaryOp<ei_scalar_difference_op<Scalar>,ValueType,OtherValueType> >
      CwiseBinaryOp<ei_scalar_difference_op<Scalar>,JacobianType,OtherJacobianType> >(
        m_values - other.values(),
        m_jacobian - other.jacobian());
    }
    
    template<typename OtherValueType, typename OtherJacobianType>
    inline AutoDiffVector&
    operator-=(const AutoDiffVector<OtherValueType,OtherDerType>& other)
    {
      m_values -= other.values();
      m_jacobian -= other.jacobian();
      return *this;
    }
    
    inline const AutoDiffVector<
      CwiseUnaryOp<ei_scalar_opposite_op<Scalar>, ValueType>
      CwiseUnaryOp<ei_scalar_opposite_op<Scalar>, JacobianType> >
    operator-() const
    {
      return AutoDiffVector<
      CwiseUnaryOp<ei_scalar_opposite_op<Scalar>, ValueType>
      CwiseUnaryOp<ei_scalar_opposite_op<Scalar>, JacobianType> >(
        -m_values,
        -m_jacobian);
    }
    
    inline const AutoDiffVector<
      CwiseUnaryOp<ei_scalar_multiple_op<Scalar>, ValueType>
      CwiseUnaryOp<ei_scalar_multiple_op<Scalar>, JacobianType> >
    operator*(const Scalar& other) const
    {
      return AutoDiffVector<
        CwiseUnaryOp<ei_scalar_multiple_op<Scalar>, ValueType>
        CwiseUnaryOp<ei_scalar_multiple_op<Scalar>, JacobianType> >(
          m_values * other,
          (m_jacobian * other));
    }
    
    friend inline const AutoDiffVector<
      CwiseUnaryOp<ei_scalar_multiple_op<Scalar>, ValueType>
      CwiseUnaryOp<ei_scalar_multiple_op<Scalar>, JacobianType> >
    operator*(const Scalar& other, const AutoDiffVector& v)
    {
      return AutoDiffVector<
        CwiseUnaryOp<ei_scalar_multiple_op<Scalar>, ValueType>
        CwiseUnaryOp<ei_scalar_multiple_op<Scalar>, JacobianType> >(
          v.values() * other,
          v.jacobian() * other);
    }
    
//     template<typename OtherValueType,typename OtherJacobianType>
//     inline const AutoDiffVector<
//       CwiseBinaryOp<ei_scalar_multiple_op<Scalar>, ValueType, OtherValueType>
//       CwiseBinaryOp<ei_scalar_sum_op<Scalar>,
//         NestByValue<CwiseUnaryOp<ei_scalar_multiple_op<Scalar>, JacobianType> >,
//         NestByValue<CwiseUnaryOp<ei_scalar_multiple_op<Scalar>, OtherJacobianType> > > >
//     operator*(const AutoDiffVector<OtherValueType,OtherJacobianType>& other) const
//     {
//       return AutoDiffVector<
//         CwiseBinaryOp<ei_scalar_multiple_op<Scalar>, ValueType, OtherValueType>
//         CwiseBinaryOp<ei_scalar_sum_op<Scalar>,
//           NestByValue<CwiseUnaryOp<ei_scalar_multiple_op<Scalar>, JacobianType> >,
//           NestByValue<CwiseUnaryOp<ei_scalar_multiple_op<Scalar>, OtherJacobianType> > > >(
//             m_values.cwise() * other.values(),
//             (m_jacobian * other.values()).nestByValue() + (m_values * other.jacobian()).nestByValue());
//     }
    
    inline AutoDiffVector& operator*=(const Scalar& other)
    {
      m_values *= other;
      m_jacobian *= other;
      return *this;
    }
    
    template<typename OtherValueType,typename OtherJacobianType>
    inline AutoDiffVector& operator*=(const AutoDiffVector<OtherValueType,OtherJacobianType>& other)
    {
      *this = *this * other;
      return *this;
    }
    
  protected:
    ValueType m_values;
    JacobianType m_jacobian;
    
};

}

#endif // EIGEN_AUTODIFF_VECTOR_H
