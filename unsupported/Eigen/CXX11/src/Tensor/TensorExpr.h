// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_EXPR_H
#define EIGEN_CXX11_TENSOR_TENSOR_EXPR_H

namespace Eigen {

/** \class TensorExpr
  * \ingroup CXX11_Tensor_Module
  *
  * \brief Tensor expression classes.
  *
  * The TensorCwiseUnaryOp class represents an expression where a unary operator
  * (e.g. cwiseSqrt) is applied to an expression.
  *
  * The TensorCwiseBinaryOp class represents an expression where a binary operator
  * (e.g. addition) is applied to a lhs and a rhs expression.
  *
  */

namespace internal {
template<typename UnaryOp, typename XprType>
struct traits<TensorCwiseUnaryOp<UnaryOp, XprType> >
 : traits<XprType>
{
  typedef typename result_of<
                     UnaryOp(typename XprType::Scalar)
                   >::type Scalar;
  typedef typename XprType::Nested XprTypeNested;
  typedef typename remove_reference<XprTypeNested>::type _XprTypeNested;
};

template<typename UnaryOp, typename XprType>
struct eval<TensorCwiseUnaryOp<UnaryOp, XprType>, Eigen::Dense>
{
  typedef const TensorCwiseUnaryOp<UnaryOp, XprType>& type;
};

template<typename UnaryOp, typename XprType>
struct nested<TensorCwiseUnaryOp<UnaryOp, XprType>, 1, typename eval<TensorCwiseUnaryOp<UnaryOp, XprType> >::type>
{
  typedef TensorCwiseUnaryOp<UnaryOp, XprType> type;
};

}  // end namespace internal



template<typename UnaryOp, typename XprType>
class TensorCwiseUnaryOp : public TensorBase<TensorCwiseUnaryOp<UnaryOp, XprType> >
{
  public:
  typedef typename Eigen::internal::traits<TensorCwiseUnaryOp>::Scalar Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename Eigen::internal::nested<TensorCwiseUnaryOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorCwiseUnaryOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorCwiseUnaryOp>::Index Index;

   inline TensorCwiseUnaryOp(const XprType& xpr, const UnaryOp& func = UnaryOp())
      : m_xpr(xpr), m_functor(func) {}

    EIGEN_DEVICE_FUNC
    const UnaryOp& functor() const { return m_functor; }

    /** \returns the nested expression */
    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename XprType::Nested>::type&
    nestedExpression() const { return m_xpr; }

  protected:
    typename XprType::Nested m_xpr;
    const UnaryOp m_functor;
};


namespace internal {
template<typename BinaryOp, typename LhsXprType, typename RhsXprType>
struct traits<TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType> >
{
  // Type promotion to handle the case where the types of the lhs and the rhs are different.
  typedef typename result_of<
                     BinaryOp(
                       typename LhsXprType::Scalar,
                       typename RhsXprType::Scalar
                     )
                   >::type Scalar;
  typedef typename promote_storage_type<typename traits<LhsXprType>::StorageKind,
                                           typename traits<RhsXprType>::StorageKind>::ret StorageKind;
  typedef typename promote_index_type<typename traits<LhsXprType>::Index,
                                         typename traits<RhsXprType>::Index>::type Index;
  typedef typename LhsXprType::Nested LhsNested;
  typedef typename RhsXprType::Nested RhsNested;
  typedef typename remove_reference<LhsNested>::type _LhsNested;
  typedef typename remove_reference<RhsNested>::type _RhsNested;
};

template<typename BinaryOp, typename LhsXprType, typename RhsXprType>
struct eval<TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>, Eigen::Dense>
{
  typedef const TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>& type;
};

template<typename BinaryOp, typename LhsXprType, typename RhsXprType>
struct nested<TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>, 1, typename eval<TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType> >::type>
{
  typedef TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType> type;
};

}  // end namespace internal



template<typename BinaryOp, typename LhsXprType, typename RhsXprType>
class TensorCwiseBinaryOp : public TensorBase<TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType> >
{
  public:
  typedef typename Eigen::internal::traits<TensorCwiseBinaryOp>::Scalar Scalar;
  typedef typename Eigen::NumTraits<Scalar>::Real RealScalar;
  typedef typename internal::promote_storage_type<typename LhsXprType::CoeffReturnType,
                                                  typename RhsXprType::CoeffReturnType>::ret CoeffReturnType;
  typedef typename Eigen::internal::nested<TensorCwiseBinaryOp>::type Nested;
  typedef typename Eigen::internal::traits<TensorCwiseBinaryOp>::StorageKind StorageKind;
  typedef typename Eigen::internal::traits<TensorCwiseBinaryOp>::Index Index;

  inline TensorCwiseBinaryOp(const LhsXprType& lhs, const RhsXprType& rhs, const BinaryOp& func = BinaryOp())
      : m_lhs_xpr(lhs), m_rhs_xpr(rhs), m_functor(func) {}

    EIGEN_DEVICE_FUNC
    const BinaryOp& functor() const { return m_functor; }

    /** \returns the nested expressions */
    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename LhsXprType::Nested>::type&
    lhsExpression() const { return m_lhs_xpr; }

    EIGEN_DEVICE_FUNC
    const typename internal::remove_all<typename RhsXprType::Nested>::type&
    rhsExpression() const { return m_rhs_xpr; }

  protected:
    typename LhsXprType::Nested m_lhs_xpr;
    typename RhsXprType::Nested m_rhs_xpr;
    const BinaryOp m_functor;
};

} // end namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_EXPR_H
