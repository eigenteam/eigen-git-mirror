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

#ifndef EIGEN_AUTODIFF_SCALAR_H
#define EIGEN_AUTODIFF_SCALAR_H

namespace Eigen {

template<typename A, typename B>
struct ei_make_coherent_impl {
  static void run(A& a, B& b) {}
};

// resize a to match b is a.size()==0, and conversely.
template<typename A, typename B>
void ei_make_coherent(const A& a, const B&b)
{
  ei_make_coherent_impl<A,B>::run(a.const_cast_derived(), b.const_cast_derived());
}

/** \class AutoDiffScalar
  * \brief A scalar type replacement with automatic differentation capability
  *
  * \param _DerType the vector type used to store/represent the derivatives. The base scalar type
  *                 as well as the number of derivatives to compute are determined from this type.
  *                 Typical choices include, e.g., \c Vector4f for 4 derivatives, or \c VectorXf
  *                 if the number of derivatives is not known at compile time, and/or, the number
  *                 of derivatives is large.
  *                 Note that _DerType can also be a reference (e.g., \c VectorXf&) to wrap a
  *                 existing vector into an AutoDiffScalar.
  *                 Finally, _DerType can also be any Eigen compatible expression.
  *
  * This class represents a scalar value while tracking its respective derivatives using Eigen's expression
  * template mechanism.
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
template<typename _DerType>
class AutoDiffScalar
{
  public:
    typedef typename ei_cleantype<_DerType>::type DerType;
    typedef typename ei_traits<DerType>::Scalar Scalar;

    inline AutoDiffScalar() {}

    inline AutoDiffScalar(const Scalar& value)
      : m_value(value)
    {
      if(m_derivatives.size()>0)
        m_derivatives.setZero();
    }

    inline AutoDiffScalar(const Scalar& value, const DerType& der)
      : m_value(value), m_derivatives(der)
    {}

    template<typename OtherDerType>
    inline AutoDiffScalar(const AutoDiffScalar<OtherDerType>& other)
      : m_value(other.value()), m_derivatives(other.derivatives())
    {}

    inline AutoDiffScalar(const AutoDiffScalar& other)
      : m_value(other.value()), m_derivatives(other.derivatives())
    {}

    template<typename OtherDerType>
    inline AutoDiffScalar& operator=(const AutoDiffScalar<OtherDerType>& other)
    {
      m_value = other.value();
      m_derivatives = other.derivatives();
      return *this;
    }

    inline AutoDiffScalar& operator=(const AutoDiffScalar& other)
    {
      m_value = other.value();
      m_derivatives = other.derivatives();
      return *this;
    }

//     inline operator const Scalar& () const { return m_value; }
//     inline operator Scalar& () { return m_value; }

    inline const Scalar& value() const { return m_value; }
    inline Scalar& value() { return m_value; }

    inline const DerType& derivatives() const { return m_derivatives; }
    inline DerType& derivatives() { return m_derivatives; }

    inline const AutoDiffScalar<DerType&> operator+(const Scalar& other) const
    {
      return AutoDiffScalar<DerType>(m_value + other, m_derivatives);
    }

    friend inline const AutoDiffScalar<DerType&> operator+(const Scalar& a, const AutoDiffScalar& b)
    {
      return AutoDiffScalar<DerType>(a + b.value(), b.derivatives());
    }

    inline AutoDiffScalar& operator+=(const Scalar& other)
    {
      value() += other;
      return *this;
    }

    template<typename OtherDerType>
    inline const AutoDiffScalar<typename MakeCwiseBinaryOp<ei_scalar_sum_op<Scalar>,DerType,typename ei_cleantype<OtherDerType>::type>::Type >
    operator+(const AutoDiffScalar<OtherDerType>& other) const
    {
      ei_make_coherent(m_derivatives, other.derivatives());
      return AutoDiffScalar<typename MakeCwiseBinaryOp<ei_scalar_sum_op<Scalar>,DerType,typename ei_cleantype<OtherDerType>::type>::Type >(
        m_value + other.value(),
        m_derivatives + other.derivatives());
    }

    template<typename OtherDerType>
    inline AutoDiffScalar&
    operator+=(const AutoDiffScalar<OtherDerType>& other)
    {
      (*this) = (*this) + other;
      return *this;
    }

    template<typename OtherDerType>
    inline const AutoDiffScalar<typename MakeCwiseBinaryOp<ei_scalar_difference_op<Scalar>, DerType,typename ei_cleantype<OtherDerType>::type>::Type >
    operator-(const AutoDiffScalar<OtherDerType>& other) const
    {
      ei_make_coherent(m_derivatives, other.derivatives());
      return AutoDiffScalar<typename MakeCwiseBinaryOp<ei_scalar_difference_op<Scalar>, DerType,typename ei_cleantype<OtherDerType>::type>::Type >(
        m_value - other.value(),
        m_derivatives - other.derivatives());
    }

    template<typename OtherDerType>
    inline AutoDiffScalar&
    operator-=(const AutoDiffScalar<OtherDerType>& other)
    {
      *this = *this - other;
      return *this;
    }

    template<typename OtherDerType>
    inline const AutoDiffScalar<typename MakeCwiseUnaryOp<ei_scalar_opposite_op<Scalar>, DerType>::Type >
    operator-() const
    {
      return AutoDiffScalar<typename MakeCwiseUnaryOp<ei_scalar_opposite_op<Scalar>, DerType>::Type >(
        -m_value,
        -m_derivatives);
    }

    inline const AutoDiffScalar<typename MakeCwiseUnaryOp<ei_scalar_multiple_op<Scalar>, DerType>::Type >
    operator*(const Scalar& other) const
    {
      return AutoDiffScalar<typename MakeCwiseUnaryOp<ei_scalar_multiple_op<Scalar>, DerType>::Type >(
        m_value * other,
        (m_derivatives * other));
    }

    friend inline const AutoDiffScalar<typename MakeCwiseUnaryOp<ei_scalar_multiple_op<Scalar>, DerType>::Type >
    operator*(const Scalar& other, const AutoDiffScalar& a)
    {
      return AutoDiffScalar<typename MakeCwiseUnaryOp<ei_scalar_multiple_op<Scalar>, DerType>::Type >(
        a.value() * other,
        a.derivatives() * other);
    }

    inline const AutoDiffScalar<typename MakeCwiseUnaryOp<ei_scalar_multiple_op<Scalar>, DerType>::Type >
    operator/(const Scalar& other) const
    {
      return AutoDiffScalar<typename MakeCwiseUnaryOp<ei_scalar_multiple_op<Scalar>, DerType>::Type >(
        m_value / other,
        (m_derivatives * (Scalar(1)/other)));
    }

    friend inline const AutoDiffScalar<typename MakeCwiseUnaryOp<ei_scalar_multiple_op<Scalar>, DerType>::Type >
    operator/(const Scalar& other, const AutoDiffScalar& a)
    {
      return AutoDiffScalar<typename MakeCwiseUnaryOp<ei_scalar_multiple_op<Scalar>, DerType>::Type >(
        other / a.value(),
        a.derivatives() * (-Scalar(1)/other));
    }

    template<typename OtherDerType>
    inline const AutoDiffScalar<typename MakeCwiseUnaryOp<ei_scalar_multiple_op<Scalar>,
        typename MakeNestByValue<typename MakeCwiseBinaryOp<ei_scalar_difference_op<Scalar>,
          typename MakeNestByValue<typename MakeCwiseUnaryOp<ei_scalar_multiple_op<Scalar>, DerType>::Type>::Type,
          typename MakeNestByValue<typename MakeCwiseUnaryOp<ei_scalar_multiple_op<Scalar>, typename ei_cleantype<OtherDerType>::type>::Type>::Type >::Type >::Type >::Type >
    operator/(const AutoDiffScalar<OtherDerType>& other) const
    {
      ei_make_coherent(m_derivatives, other.derivatives());
      return AutoDiffScalar<typename MakeCwiseUnaryOp<ei_scalar_multiple_op<Scalar>,
        typename MakeNestByValue<typename MakeCwiseBinaryOp<ei_scalar_difference_op<Scalar>,
          typename MakeNestByValue<typename MakeCwiseUnaryOp<ei_scalar_multiple_op<Scalar>, DerType>::Type>::Type,
          typename MakeNestByValue<typename MakeCwiseUnaryOp<ei_scalar_multiple_op<Scalar>, typename ei_cleantype<OtherDerType>::type>::Type>::Type >::Type >::Type >::Type >(
        m_value / other.value(),
          ((m_derivatives * other.value()).nestByValue() - (m_value * other.derivatives()).nestByValue()).nestByValue()
        * (Scalar(1)/(other.value()*other.value())));
    }

    template<typename OtherDerType>
    inline const AutoDiffScalar<typename MakeCwiseBinaryOp<ei_scalar_sum_op<Scalar>,
        typename MakeNestByValue<typename MakeCwiseUnaryOp<ei_scalar_multiple_op<Scalar>, DerType>::Type>::Type,
        typename MakeNestByValue<typename MakeCwiseUnaryOp<ei_scalar_multiple_op<Scalar>, typename ei_cleantype<OtherDerType>::type>::Type>::Type >::Type >
    operator*(const AutoDiffScalar<OtherDerType>& other) const
    {
      ei_make_coherent(m_derivatives, other.derivatives());
      return AutoDiffScalar<typename MakeCwiseBinaryOp<ei_scalar_sum_op<Scalar>,
        typename MakeNestByValue<typename MakeCwiseUnaryOp<ei_scalar_multiple_op<Scalar>, DerType>::Type>::Type,
        typename MakeNestByValue<typename MakeCwiseUnaryOp<ei_scalar_multiple_op<Scalar>, typename ei_cleantype<OtherDerType>::type>::Type>::Type >::Type >(
        m_value * other.value(),
        (m_derivatives * other.value()).nestByValue() + (m_value * other.derivatives()).nestByValue());
    }

    inline AutoDiffScalar& operator*=(const Scalar& other)
    {
      *this = *this * other;
      return *this;
    }

    template<typename OtherDerType>
    inline AutoDiffScalar& operator*=(const AutoDiffScalar<OtherDerType>& other)
    {
      *this = *this * other;
      return *this;
    }

  protected:
    Scalar m_value;
    DerType m_derivatives;

};

template<typename A_Scalar, int A_Rows, int A_Cols, int A_Options, int A_MaxRows, int A_MaxCols, typename B>
struct ei_make_coherent_impl<Matrix<A_Scalar, A_Rows, A_Cols, A_Options, A_MaxRows, A_MaxCols>, B> {
  typedef Matrix<A_Scalar, A_Rows, A_Cols, A_Options, A_MaxRows, A_MaxCols> A;
  static void run(A& a, B& b) {
    if((A_Rows==Dynamic || A_Cols==Dynamic) && (a.size()==0))
    {
      a.resize(b.size());
      a.setZero();
    }
  }
};

template<typename A, typename B_Scalar, int B_Rows, int B_Cols, int B_Options, int B_MaxRows, int B_MaxCols>
struct ei_make_coherent_impl<A, Matrix<B_Scalar, B_Rows, B_Cols, B_Options, B_MaxRows, B_MaxCols> > {
  typedef Matrix<B_Scalar, B_Rows, B_Cols, B_Options, B_MaxRows, B_MaxCols> B;
  static void run(A& a, B& b) {
    if((B_Rows==Dynamic || B_Cols==Dynamic) && (b.size()==0))
    {
      b.resize(a.size());
      b.setZero();
    }
  }
};

template<typename A_Scalar, int A_Rows, int A_Cols, int A_Options, int A_MaxRows, int A_MaxCols,
         typename B_Scalar, int B_Rows, int B_Cols, int B_Options, int B_MaxRows, int B_MaxCols>
struct ei_make_coherent_impl<Matrix<A_Scalar, A_Rows, A_Cols, A_Options, A_MaxRows, A_MaxCols>,
                             Matrix<B_Scalar, B_Rows, B_Cols, B_Options, B_MaxRows, B_MaxCols> > {
  typedef Matrix<A_Scalar, A_Rows, A_Cols, A_Options, A_MaxRows, A_MaxCols> A;
  typedef Matrix<B_Scalar, B_Rows, B_Cols, B_Options, B_MaxRows, B_MaxCols> B;
  static void run(A& a, B& b) {
    if((A_Rows==Dynamic || A_Cols==Dynamic) && (a.size()==0))
    {
      a.resize(b.size());
      a.setZero();
    }
    else if((B_Rows==Dynamic || B_Cols==Dynamic) && (b.size()==0))
    {
      b.resize(a.size());
      b.setZero();
    }
  }
};

}

#define EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(FUNC,CODE) \
  template<typename DerType> \
  inline const Eigen::AutoDiffScalar<typename Eigen::MakeCwiseUnaryOp<Eigen::ei_scalar_multiple_op<typename Eigen::ei_traits<DerType>::Scalar>, DerType>::Type > \
  FUNC(const Eigen::AutoDiffScalar<DerType>& x) { \
    using namespace Eigen; \
    typedef typename ei_traits<DerType>::Scalar Scalar; \
    typedef AutoDiffScalar<typename MakeCwiseUnaryOp<ei_scalar_multiple_op<Scalar>, DerType>::Type > ReturnType; \
    CODE; \
  }

namespace std
{
  EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(abs,
    return ReturnType(std::abs(x.value()), x.derivatives() * (sign(x.value())));)

  EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(sqrt,
    Scalar sqrtx = std::sqrt(x.value());
    return ReturnType(sqrtx,x.derivatives() * (Scalar(0.5) / sqrtx));)

  EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(cos,
    return ReturnType(std::cos(x.value()), x.derivatives() * (-std::sin(x.value())));)

  EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(sin,
    return ReturnType(std::sin(x.value()),x.derivatives() * std::cos(x.value()));)

  EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(exp,
    Scalar expx = std::exp(x.value());
    return ReturnType(expx,x.derivatives() * expx);)

  EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(ei_log,
    return ReturnType(std::log(x.value),x.derivatives() * (Scalar(1).x.value()));)

  template<typename DerType>
  inline const Eigen::AutoDiffScalar<typename Eigen::MakeCwiseUnaryOp<Eigen::ei_scalar_multiple_op<typename Eigen::ei_traits<DerType>::Scalar>, DerType>::Type >
  pow(const Eigen::AutoDiffScalar<DerType>& x, typename Eigen::ei_traits<DerType>::Scalar y)
  {
    using namespace Eigen;
    typedef typename ei_traits<DerType>::Scalar Scalar;
    return AutoDiffScalar<typename MakeCwiseUnaryOp<ei_scalar_multiple_op<Scalar>, DerType>::Type >(
      std::pow(x.value(),y),
      x.derivatives() * (y * std::pow(x.value(),y-1)));
  }

}

namespace Eigen {

template<typename DerType>
inline const AutoDiffScalar<DerType>& ei_conj(const AutoDiffScalar<DerType>& x)  { return x; }
template<typename DerType>
inline const AutoDiffScalar<DerType>& ei_real(const AutoDiffScalar<DerType>& x)  { return x; }
template<typename DerType>
inline typename DerType::Scalar ei_imag(const AutoDiffScalar<DerType>&)    { return 0.; }

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(ei_abs,
  return ReturnType(ei_abs(x.value()), x.derivatives() * (sign(x.value())));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(ei_abs2,
  return ReturnType(ei_abs2(x.value()), x.derivatives() * (Scalar(2)*x.value()));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(ei_sqrt,
  Scalar sqrtx = ei_sqrt(x.value());
  return ReturnType(sqrtx,x.derivatives() * (Scalar(0.5) / sqrtx));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(ei_cos,
  return ReturnType(ei_cos(x.value()), x.derivatives() * (-ei_sin(x.value())));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(ei_sin,
  return ReturnType(ei_sin(x.value()),x.derivatives() * ei_cos(x.value()));)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(ei_exp,
  Scalar expx = ei_exp(x.value());
  return ReturnType(expx,x.derivatives() * expx);)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(ei_log,
  return ReturnType(ei_log(x.value),x.derivatives() * (Scalar(1).x.value()));)

template<typename DerType>
inline const AutoDiffScalar<typename MakeCwiseUnaryOp<ei_scalar_multiple_op<typename ei_traits<DerType>::Scalar>, DerType>::Type >
ei_pow(const AutoDiffScalar<DerType>& x, typename ei_traits<DerType>::Scalar y)
{ return std::pow(x,y);}

#undef EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY

template<typename DerType> struct NumTraits<AutoDiffScalar<DerType> >
{
  typedef typename DerType::Scalar Real;
  typedef AutoDiffScalar<DerType> FloatingPoint;
  enum {
    IsComplex = 0,
    HasFloatingPoint = 1,
    ReadCost = 1,
    AddCost = 1,
    MulCost = 1
  };
};

}

#endif // EIGEN_AUTODIFF_SCALAR_H
