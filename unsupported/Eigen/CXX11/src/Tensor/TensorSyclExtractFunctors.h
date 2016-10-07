// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/*****************************************************************
 * TensorSyclextractFunctors.h
 *
 * \brief:
 *  Used to extract all the functors allocated to each node of the expression
*tree.
 *
*****************************************************************/

#ifndef UNSUPPORTED_EIGEN_CXX11_SRC_TENSORYSYCL_EXTRACT_FUNCTORS_HPP
#define UNSUPPORTED_EIGEN_CXX11_SRC_TENSORYSYCL_EXTRACT_FUNCTORS_HPP

namespace Eigen {
namespace TensorSycl {
namespace internal {
/// \struct FunctorExtractor:  This struct is used to extract the functors
/// constructed on
/// the host-side, to pack them and reuse them in reconstruction of the
/// expression on the device.
/// We have to do that as in Eigen the functors are not stateless so we cannot
/// re-instantiate them on the device.
/// We have to pass whatever instantiated to the device.
template <typename Evaluator>
struct FunctorExtractor;

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// TensorMap:
template <typename PlainObjectType, int Options_, typename Dev>
struct FunctorExtractor<
    TensorEvaluator<TensorMap<PlainObjectType, Options_>, Dev>> {
  using Dimensions = typename PlainObjectType::Dimensions;
  const Dimensions m_dimensions;
  const Dimensions& dimensions() const { return m_dimensions; }
  FunctorExtractor(
      const TensorEvaluator<TensorMap<PlainObjectType, Options_>, Dev>& expr)
      : m_dimensions(expr.dimensions()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// const TensorMap
template <typename PlainObjectType, int Options_, typename Dev>
struct FunctorExtractor<
    TensorEvaluator<const TensorMap<PlainObjectType, Options_>, Dev>> {
  using Dimensions = typename PlainObjectType::Dimensions;
  const Dimensions m_dimensions;
  const Dimensions& dimensions() const { return m_dimensions; }
  FunctorExtractor(
      const TensorEvaluator<const TensorMap<PlainObjectType, Options_>, Dev>&
          expr)
      : m_dimensions(expr.dimensions()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// TensorForcedEvalOp
template <typename Expr, typename Dev>
struct FunctorExtractor<TensorEvaluator<TensorForcedEvalOp<Expr>, Dev>> {
  using Dimensions = typename Expr::Dimensions;
  const Dimensions m_dimensions;
  const Dimensions& dimensions() const { return m_dimensions; }
  FunctorExtractor(const TensorEvaluator<TensorForcedEvalOp<Expr>, Dev>& expr)
      : m_dimensions(expr.dimensions()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// const TensorForcedEvalOp
template <typename Expr, typename Dev>
struct FunctorExtractor<TensorEvaluator<const TensorForcedEvalOp<Expr>, Dev>> {
  using Dimensions =
      typename TensorEvaluator<const TensorForcedEvalOp<Expr>, Dev>::Dimensions;
  const Dimensions m_dimensions;
  const Dimensions& dimensions() const { return m_dimensions; }
  FunctorExtractor(
      const TensorEvaluator<const TensorForcedEvalOp<Expr>, Dev>& expr)
      : m_dimensions(expr.dimensions()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// TensorCwiseNullaryOp
template <typename OP, typename RHSExpr, typename Dev>
struct FunctorExtractor<
    TensorEvaluator<TensorCwiseNullaryOp<OP, RHSExpr>, Dev>> {
  FunctorExtractor<TensorEvaluator<RHSExpr, Dev>> rhsExpr;
  OP func;
  FunctorExtractor(
      TensorEvaluator<TensorCwiseNullaryOp<OP, RHSExpr>, Dev>& expr)
      : rhsExpr(expr.impl()), func(expr.functor()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// const TensorCwiseNullaryOp
template <typename OP, typename RHSExpr, typename Dev>
struct FunctorExtractor<
    TensorEvaluator<const TensorCwiseNullaryOp<OP, RHSExpr>, Dev>> {
  FunctorExtractor<TensorEvaluator<RHSExpr, Dev>> rhsExpr;
  OP func;
  FunctorExtractor(
      const TensorEvaluator<const TensorCwiseNullaryOp<OP, RHSExpr>, Dev>& expr)
      : rhsExpr(expr.impl()), func(expr.functor()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// TensorBroadcastingOp
template <typename OP, typename RHSExpr, typename Dev>
struct FunctorExtractor<
    TensorEvaluator<TensorBroadcastingOp<OP, RHSExpr>, Dev>> {
  FunctorExtractor<TensorEvaluator<RHSExpr, Dev>> rhsExpr;
  OP func;
  FunctorExtractor(
      const TensorEvaluator<TensorBroadcastingOp<OP, RHSExpr>, Dev>& expr)
      : rhsExpr(expr.impl()), func(expr.functor()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// const TensorBroadcastingOp
template <typename OP, typename RHSExpr, typename Dev>
struct FunctorExtractor<
    TensorEvaluator<const TensorBroadcastingOp<OP, RHSExpr>, Dev>> {
  FunctorExtractor<TensorEvaluator<RHSExpr, Dev>> rhsExpr;
  OP func;
  FunctorExtractor(
      const TensorEvaluator<const TensorBroadcastingOp<OP, RHSExpr>, Dev>& expr)
      : rhsExpr(expr.impl()), func(expr.functor()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// TensorCwiseUnaryOp
template <typename OP, typename RHSExpr, typename Dev>
struct FunctorExtractor<TensorEvaluator<TensorCwiseUnaryOp<OP, RHSExpr>, Dev>> {
  FunctorExtractor<TensorEvaluator<RHSExpr, Dev>> rhsExpr;
  OP func;
  FunctorExtractor(
      const TensorEvaluator<TensorCwiseUnaryOp<OP, RHSExpr>, Dev>& expr)
      : rhsExpr(expr.impl()), func(expr.functor()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// const TensorCwiseUnaryOp
template <typename OP, typename RHSExpr, typename Dev>
struct FunctorExtractor<
    TensorEvaluator<const TensorCwiseUnaryOp<OP, RHSExpr>, Dev>> {
  FunctorExtractor<TensorEvaluator<RHSExpr, Dev>> rhsExpr;
  OP func;
  FunctorExtractor(
      const TensorEvaluator<const TensorCwiseUnaryOp<OP, RHSExpr>, Dev>& expr)
      : rhsExpr(expr.impl()), func(expr.functor()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// TensorCwiseBinaryOp
template <typename OP, typename LHSExpr, typename RHSExpr, typename Dev>
struct FunctorExtractor<
    TensorEvaluator<TensorCwiseBinaryOp<OP, LHSExpr, RHSExpr>, Dev>> {
  FunctorExtractor<TensorEvaluator<LHSExpr, Dev>> lhsExpr;
  FunctorExtractor<TensorEvaluator<RHSExpr, Dev>> rhsExpr;
  OP func;
  FunctorExtractor(
      const TensorEvaluator<TensorCwiseBinaryOp<OP, LHSExpr, RHSExpr>, Dev>&
          expr)
      : lhsExpr(expr.left_impl()),
        rhsExpr(expr.right_impl()),
        func(expr.functor()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// const TensorCwiseBinaryOp
template <typename OP, typename LHSExpr, typename RHSExpr, typename Dev>
struct FunctorExtractor<
    TensorEvaluator<const TensorCwiseBinaryOp<OP, LHSExpr, RHSExpr>, Dev>> {
  FunctorExtractor<TensorEvaluator<LHSExpr, Dev>> lhsExpr;
  FunctorExtractor<TensorEvaluator<RHSExpr, Dev>> rhsExpr;
  OP func;
  FunctorExtractor(const TensorEvaluator<
                   const TensorCwiseBinaryOp<OP, LHSExpr, RHSExpr>, Dev>& expr)
      : lhsExpr(expr.left_impl()),
        rhsExpr(expr.right_impl()),
        func(expr.functor()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// const TensorCwiseTernaryOp
template <typename OP, typename Arg1Expr, typename Arg2Expr, typename Arg3Expr,
          typename Dev>
struct FunctorExtractor<TensorEvaluator<
    const TensorCwiseTernaryOp<OP, Arg1Expr, Arg2Expr, Arg3Expr>, Dev>> {
  FunctorExtractor<TensorEvaluator<Arg1Expr, Dev>> arg1Expr;
  FunctorExtractor<TensorEvaluator<Arg2Expr, Dev>> arg2Expr;
  FunctorExtractor<TensorEvaluator<Arg3Expr, Dev>> arg3Expr;
  OP func;
  FunctorExtractor(const TensorEvaluator<
                   const TensorCwiseTernaryOp<OP, Arg1Expr, Arg2Expr, Arg3Expr>,
                   Dev>& expr)
      : arg1Expr(expr.arg1Impl()),
        arg2Expr(expr.arg2Impl()),
        arg3Expr(expr.arg3Impl()),
        func(expr.functor()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// TensorCwiseTernaryOp
template <typename OP, typename Arg1Expr, typename Arg2Expr, typename Arg3Expr,
          typename Dev>
struct FunctorExtractor<TensorEvaluator<
    TensorCwiseTernaryOp<OP, Arg1Expr, Arg2Expr, Arg3Expr>, Dev>> {
  FunctorExtractor<TensorEvaluator<Arg1Expr, Dev>> arg1Expr;
  FunctorExtractor<TensorEvaluator<Arg2Expr, Dev>> arg2Expr;
  FunctorExtractor<TensorEvaluator<Arg3Expr, Dev>> arg3Expr;
  OP func;
  FunctorExtractor(
      const TensorEvaluator<
          TensorCwiseTernaryOp<OP, Arg1Expr, Arg2Expr, Arg3Expr>, Dev>& expr)
      : arg1Expr(expr.arg1Impl()),
        arg2Expr(expr.arg2Impl()),
        arg3Expr(expr.arg3Impl()),
        func(expr.functor()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// const TensorCwiseSelectOp
template <typename IfExpr, typename ThenExpr, typename ElseExpr, typename Dev>
struct FunctorExtractor<
    TensorEvaluator<const TensorSelectOp<IfExpr, ThenExpr, ElseExpr>, Dev>> {
  FunctorExtractor<TensorEvaluator<IfExpr, Dev>> ifExpr;
  FunctorExtractor<TensorEvaluator<ThenExpr, Dev>> thenExpr;
  FunctorExtractor<TensorEvaluator<ElseExpr, Dev>> elseExpr;
  FunctorExtractor(const TensorEvaluator<
                   const TensorSelectOp<IfExpr, ThenExpr, ElseExpr>, Dev>& expr)
      : ifExpr(expr.cond_impl()),
        thenExpr(expr.then_impl()),
        elseExpr(expr.else_impl()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// TensorCwiseSelectOp
template <typename IfExpr, typename ThenExpr, typename ElseExpr, typename Dev>
struct FunctorExtractor<
    TensorEvaluator<TensorSelectOp<IfExpr, ThenExpr, ElseExpr>, Dev>> {
  FunctorExtractor<IfExpr> ifExpr;
  FunctorExtractor<ThenExpr> thenExpr;
  FunctorExtractor<ElseExpr> elseExpr;
  FunctorExtractor(
      const TensorEvaluator<TensorSelectOp<IfExpr, ThenExpr, ElseExpr>, Dev>&
          expr)
      : ifExpr(expr.cond_impl()),
        thenExpr(expr.then_impl()),
        elseExpr(expr.else_impl()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// TensorAssignOp
template <typename LHSExpr, typename RHSExpr, typename Dev>
struct FunctorExtractor<
    TensorEvaluator<TensorAssignOp<LHSExpr, RHSExpr>, Dev>> {
  FunctorExtractor<TensorEvaluator<LHSExpr, Dev>> lhsExpr;
  FunctorExtractor<TensorEvaluator<RHSExpr, Dev>> rhsExpr;
  FunctorExtractor(
      const TensorEvaluator<TensorAssignOp<LHSExpr, RHSExpr>, Dev>& expr)
      : lhsExpr(expr.left_impl()), rhsExpr(expr.right_impl()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// const TensorAssignOp
template <typename LHSExpr, typename RHSExpr, typename Dev>
struct FunctorExtractor<
    TensorEvaluator<const TensorAssignOp<LHSExpr, RHSExpr>, Dev>> {
  FunctorExtractor<TensorEvaluator<LHSExpr, Dev>> lhsExpr;
  FunctorExtractor<TensorEvaluator<RHSExpr, Dev>> rhsExpr;
  FunctorExtractor(
      const TensorEvaluator<const TensorAssignOp<LHSExpr, RHSExpr>, Dev>& expr)
      : lhsExpr(expr.left_impl()), rhsExpr(expr.right_impl()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// TensorEvalToOp
template <typename RHSExpr, typename Dev>
struct FunctorExtractor<TensorEvaluator<TensorEvalToOp<RHSExpr>, Dev>> {
  FunctorExtractor<TensorEvaluator<RHSExpr, Dev>> rhsExpr;
  FunctorExtractor(const TensorEvaluator<TensorEvalToOp<RHSExpr>, Dev>& expr)
      : rhsExpr(expr.impl()) {}
};

/// specialisation of the \ref FunctorExtractor struct when the node type is
/// const TensorEvalToOp
template <typename RHSExpr, typename Dev>
struct FunctorExtractor<TensorEvaluator<const TensorEvalToOp<RHSExpr>, Dev>> {
  FunctorExtractor<TensorEvaluator<RHSExpr, Dev>> rhsExpr;
  FunctorExtractor(
      const TensorEvaluator<const TensorEvalToOp<RHSExpr>, Dev>& expr)
      : rhsExpr(expr.impl()) {}
};

/// template deduction function for FunctorExtractor
template <typename Evaluator>
auto extractFunctors(const Evaluator& evaluator)
    -> FunctorExtractor<Evaluator> {
  return FunctorExtractor<Evaluator>(evaluator);
}
}  // namespace internal
}  // namespace TensorSycl
}  // namespace Eigen

#endif  // UNSUPPORTED_EIGEN_CXX11_SRC_TENSORYSYCL_EXTRACT_FUNCTORS_HPP
