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
 * TensorSyclExprConstructor.h
 *
 * \brief:
 *  This file re-create an expression on the SYCL device in order
 *  to use the original tensor evaluator.
 *
*****************************************************************/

#ifndef UNSUPPORTED_EIGEN_CXX11_SRC_TENSORYSYCL_EXPR_CONSTRUCTOR_HPP
#define UNSUPPORTED_EIGEN_CXX11_SRC_TENSORYSYCL_EXPR_CONSTRUCTOR_HPP

namespace Eigen {
namespace TensorSycl {
namespace internal {
/// this class is used by EvalToOp in order to create an lhs expression which is
/// a pointer from an accessor on device-only buffer
template <typename PtrType, size_t N, typename... Params>
struct EvalToLHSConstructor {
  PtrType expr;
  EvalToLHSConstructor(const utility::tuple::Tuple<Params...> &t)
      : expr((&(*(utility::tuple::get<N>(t).get_pointer())))) {}
};

/// \struct ExprConstructor is used to reconstruct the expression on the device
/// and
/// recreate the expression with MakeGlobalPointer containing the device address
/// space for the TensorMap pointers used in eval function.
/// It receives the original expression type, the functor of the node, the tuple
/// of accessors, and the device expression type to re-instantiate the
/// expression tree for the device
template <typename OrigExpr, typename IndexExpr, typename... Params>
struct ExprConstructor;

/// specialisation of the \ref ExprConstructor struct when the node type is
/// const TensorMap
template <typename Scalar_, int Options_, int Options2_, int Options3_,
          int NumIndices_, typename IndexType_,
          template <class> class MakePointer_, size_t N, typename... Params>
struct ExprConstructor<
    const TensorMap<Tensor<Scalar_, NumIndices_, Options_, IndexType_>,
                    Options2_, MakeGlobalPointer>,
    const Eigen::internal::PlaceHolder<
        const TensorMap<Tensor<Scalar_, NumIndices_, Options_, IndexType_>,
                        Options3_, MakePointer_>,
        N>,
    Params...> {
  using Type =
      const TensorMap<Tensor<Scalar_, NumIndices_, Options_, IndexType_>,
                      Options2_, MakeGlobalPointer>;

  Type expr;

  template <typename FuncDetector>
  ExprConstructor(FuncDetector &fd, const utility::tuple::Tuple<Params...> &t)
      : expr(Type((&(*(utility::tuple::get<N>(t).get_pointer()))),
                  fd.dimensions())) {}
};

/// specialisation of the \ref ExprConstructor struct when the node type is
/// TensorMap
template <typename Scalar_, int Options_, int Options2_, int Options3_,
          int NumIndices_, typename IndexType_,
          template <class> class MakePointer_, size_t N, typename... Params>
struct ExprConstructor<
    TensorMap<Tensor<Scalar_, NumIndices_, Options_, IndexType_>, Options2_,
              MakeGlobalPointer>,
    Eigen::internal::PlaceHolder<
        TensorMap<Tensor<Scalar_, NumIndices_, Options_, IndexType_>, Options3_,
                  MakePointer_>,
        N>,
    Params...> {
  using Type = TensorMap<Tensor<Scalar_, NumIndices_, Options_, IndexType_>,
                         Options2_, MakeGlobalPointer>;

  Type expr;
  template <typename FuncDetector>
  ExprConstructor(FuncDetector &fd, const utility::tuple::Tuple<Params...> &t)
      : expr(Type((&(*(utility::tuple::get<N>(t).get_pointer()))),
                  fd.dimensions())) {}
};

/// specialisation of the \ref ExprConstructor struct when the node type is
/// TensorCwiseNullaryOp
template <typename OP, typename OrigRHSExpr, typename RHSExpr,
          typename... Params>
struct ExprConstructor<TensorCwiseNullaryOp<OP, OrigRHSExpr>,
                       TensorCwiseNullaryOp<OP, RHSExpr>, Params...> {
  using my_type = ExprConstructor<OrigRHSExpr, RHSExpr, Params...>;
  my_type rhsExpr;
  using Type = TensorCwiseNullaryOp<OP, typename my_type::Type>;
  Type expr;

  template <typename FuncDetector>
  ExprConstructor(FuncDetector &funcD,
                  const utility::tuple::Tuple<Params...> &t)
      : rhsExpr(funcD.rhsExpr, t), expr(rhsExpr.expr, funcD.func) {}
};

/// specialisation of the \ref ExprConstructor struct when the node type is
/// const TensorCwiseNullaryOp
template <typename OP, typename OrigRHSExpr, typename RHSExpr,
          typename... Params>
struct ExprConstructor<const TensorCwiseNullaryOp<OP, OrigRHSExpr>,
                       const TensorCwiseNullaryOp<OP, RHSExpr>, Params...> {
  using my_type = const ExprConstructor<OrigRHSExpr, RHSExpr, Params...>;
  my_type rhsExpr;
  using Type = const TensorCwiseNullaryOp<OP, typename my_type::Type>;
  Type expr;

  template <typename FuncDetector>
  ExprConstructor(FuncDetector &funcD,
                  const utility::tuple::Tuple<Params...> &t)
      : rhsExpr(funcD.rhsExpr, t), expr(rhsExpr.expr, funcD.func) {}
};

/// specialisation of the \ref ExprConstructor struct when the node type is
/// TensorBroadcastingOp
template <typename OP, typename OrigRHSExpr, typename RHSExpr,
          typename... Params>
struct ExprConstructor<TensorBroadcastingOp<OP, OrigRHSExpr>,
                       TensorBroadcastingOp<OP, RHSExpr>, Params...> {
  using my_type = ExprConstructor<OrigRHSExpr, RHSExpr, Params...>;
  my_type rhsExpr;
  using Type = TensorBroadcastingOp<OP, typename my_type::Type>;
  Type expr;

  template <typename FuncDetector>
  ExprConstructor(FuncDetector &funcD,
                  const utility::tuple::Tuple<Params...> &t)
      : rhsExpr(funcD.rhsExpr, t), expr(rhsExpr.expr, funcD.func) {}
};

/// specialisation of the \ref ExprConstructor struct when the node type is
/// const TensorBroadcastingOp
template <typename OP, typename OrigRHSExpr, typename RHSExpr,
          typename... Params>
struct ExprConstructor<const TensorBroadcastingOp<OP, OrigRHSExpr>,
                       const TensorBroadcastingOp<OP, RHSExpr>, Params...> {
  using my_type = const ExprConstructor<OrigRHSExpr, RHSExpr, Params...>;
  my_type rhsExpr;
  using Type = const TensorBroadcastingOp<OP, typename my_type::Type>;
  Type expr;

  template <typename FuncDetector>
  ExprConstructor(FuncDetector &funcD,
                  const utility::tuple::Tuple<Params...> &t)
      : rhsExpr(funcD.rhsExpr, t), expr(rhsExpr.expr, funcD.func) {}
};

/// specialisation of the \ref ExprConstructor struct when the node type is
/// TensorCwiseUnaryOp
template <typename OP, typename OrigRHSExpr, typename RHSExpr,
          typename... Params>
struct ExprConstructor<TensorCwiseUnaryOp<OP, OrigRHSExpr>,
                       TensorCwiseUnaryOp<OP, RHSExpr>, Params...> {
  using my_type = ExprConstructor<OrigRHSExpr, RHSExpr, Params...>;
  using Type = TensorCwiseUnaryOp<OP, typename my_type::Type>;
  my_type rhsExpr;
  Type expr;

  template <typename FuncDetector>
  ExprConstructor(FuncDetector &funcD, utility::tuple::Tuple<Params...> &t)
      : rhsExpr(funcD.rhsExpr, t), expr(rhsExpr.expr, funcD.func) {}
};

/// specialisation of the \ref ExprConstructor struct when the node type is
/// const TensorCwiseUnaryOp
template <typename OP, typename OrigRHSExpr, typename RHSExpr,
          typename... Params>
struct ExprConstructor<const TensorCwiseUnaryOp<OP, OrigRHSExpr>,
                       const TensorCwiseUnaryOp<OP, RHSExpr>, Params...> {
  using my_type = ExprConstructor<OrigRHSExpr, RHSExpr, Params...>;
  using Type = const TensorCwiseUnaryOp<OP, typename my_type::Type>;
  my_type rhsExpr;
  Type expr;

  template <typename FuncDetector>
  ExprConstructor(FuncDetector &funcD,
                  const utility::tuple::Tuple<Params...> &t)
      : rhsExpr(funcD.rhsExpr, t), expr(rhsExpr.expr, funcD.func) {}
};

/// specialisation of the \ref ExprConstructor struct when the node type is
/// TensorCwiseBinaryOp
template <typename OP, typename OrigLHSExpr, typename OrigRHSExpr,
          typename LHSExpr, typename RHSExpr, typename... Params>
struct ExprConstructor<TensorCwiseBinaryOp<OP, OrigLHSExpr, OrigRHSExpr>,
                       TensorCwiseBinaryOp<OP, LHSExpr, RHSExpr>, Params...> {
  using my_left_type = ExprConstructor<OrigLHSExpr, LHSExpr, Params...>;
  using my_right_type = ExprConstructor<OrigRHSExpr, RHSExpr, Params...>;
  using Type = TensorCwiseBinaryOp<OP, typename my_left_type::Type,
                                   typename my_right_type::Type>;

  my_left_type lhsExpr;
  my_right_type rhsExpr;
  Type expr;

  template <typename FuncDetector>
  ExprConstructor(FuncDetector &funcD,
                  const utility::tuple::Tuple<Params...> &t)
      : lhsExpr(funcD.lhsExpr, t),
        rhsExpr(funcD.rhsExpr, t),
        expr(lhsExpr.expr, rhsExpr.expr, funcD.func) {}
};

/// specialisation of the \ref ExprConstructor struct when the node type is
/// const TensorCwiseBinaryOp
template <typename OP, typename OrigLHSExpr, typename OrigRHSExpr,
          typename LHSExpr, typename RHSExpr, typename... Params>
struct ExprConstructor<const TensorCwiseBinaryOp<OP, OrigLHSExpr, OrigRHSExpr>,
                       const TensorCwiseBinaryOp<OP, LHSExpr, RHSExpr>,
                       Params...> {
  using my_left_type = ExprConstructor<OrigLHSExpr, LHSExpr, Params...>;
  using my_right_type = ExprConstructor<OrigRHSExpr, RHSExpr, Params...>;
  using Type = const TensorCwiseBinaryOp<OP, typename my_left_type::Type,
                                         typename my_right_type::Type>;

  my_left_type lhsExpr;
  my_right_type rhsExpr;
  Type expr;

  template <typename FuncDetector>
  ExprConstructor(FuncDetector &funcD,
                  const utility::tuple::Tuple<Params...> &t)
      : lhsExpr(funcD.lhsExpr, t),
        rhsExpr(funcD.rhsExpr, t),
        expr(lhsExpr.expr, rhsExpr.expr, funcD.func) {}
};

/// specialisation of the \ref ExprConstructor struct when the node type is
/// const TensorCwiseTernaryOp
template <typename OP, typename OrigArg1Expr, typename OrigArg2Expr,
          typename OrigArg3Expr, typename Arg1Expr, typename Arg2Expr,
          typename Arg3Expr, typename... Params>
struct ExprConstructor<
    const TensorCwiseTernaryOp<OP, OrigArg1Expr, OrigArg2Expr, OrigArg3Expr>,
    const TensorCwiseTernaryOp<OP, Arg1Expr, Arg2Expr, Arg3Expr>, Params...> {
  using my_arg1_type = ExprConstructor<OrigArg1Expr, Arg1Expr, Params...>;
  using my_arg2_type = ExprConstructor<OrigArg2Expr, Arg2Expr, Params...>;
  using my_arg3_type = ExprConstructor<OrigArg3Expr, Arg3Expr, Params...>;
  using Type = const TensorCwiseTernaryOp<OP, typename my_arg1_type::Type,
                                          typename my_arg2_type::Type,
                                          typename my_arg3_type::Type>;

  my_arg1_type arg1Expr;
  my_arg2_type arg2Expr;
  my_arg3_type arg3Expr;
  Type expr;
  template <typename FuncDetector>
  ExprConstructor(FuncDetector &funcD,
                  const utility::tuple::Tuple<Params...> &t)
      : arg1Expr(funcD.arg1Expr, t),
        arg2Expr(funcD.arg2Expr, t),
        arg3Expr(funcD.arg3Expr, t),
        expr(arg1Expr.expr, arg2Expr.expr, arg3Expr.expr, funcD.func) {}
};

/// specialisation of the \ref ExprConstructor struct when the node type is
/// TensorCwiseTernaryOp
template <typename OP, typename OrigArg1Expr, typename OrigArg2Expr,
          typename OrigArg3Expr, typename Arg1Expr, typename Arg2Expr,
          typename Arg3Expr, typename... Params>
struct ExprConstructor<
    TensorCwiseTernaryOp<OP, OrigArg1Expr, OrigArg2Expr, OrigArg3Expr>,
    TensorCwiseTernaryOp<OP, Arg1Expr, Arg2Expr, Arg3Expr>, Params...> {
  using my_arg1_type = ExprConstructor<OrigArg1Expr, Arg1Expr, Params...>;
  using my_arg2_type = ExprConstructor<OrigArg2Expr, Arg2Expr, Params...>;
  using my_arg3_type = ExprConstructor<OrigArg3Expr, Arg3Expr, Params...>;
  using Type = TensorCwiseTernaryOp<OP, typename my_arg1_type::Type,
                                    typename my_arg2_type::Type,
                                    typename my_arg3_type::Type>;

  my_arg1_type arg1Expr;
  my_arg2_type arg2Expr;
  my_arg3_type arg3Expr;
  Type expr;
  template <typename FuncDetector>
  ExprConstructor(FuncDetector &funcD,
                  const utility::tuple::Tuple<Params...> &t)
      : arg1Expr(funcD.arg1Expr, t),
        arg2Expr(funcD.arg2Expr, t),
        arg3Expr(funcD.arg3Expr, t),
        expr(arg1Expr.expr, arg2Expr.expr, arg3Expr.expr, funcD.func) {}
};

/// specialisation of the \ref ExprConstructor struct when the node type is
/// const TensorCwiseSelectOp
template <typename OrigIfExpr, typename OrigThenExpr, typename OrigElseExpr,
          typename IfExpr, typename ThenExpr, typename ElseExpr,
          typename... Params>
struct ExprConstructor<
    const TensorSelectOp<OrigIfExpr, OrigThenExpr, OrigElseExpr>,
    const TensorSelectOp<IfExpr, ThenExpr, ElseExpr>, Params...> {
  using my_if_type = ExprConstructor<OrigIfExpr, IfExpr, Params...>;
  using my_then_type = ExprConstructor<OrigThenExpr, ThenExpr, Params...>;
  using my_else_type = ExprConstructor<OrigElseExpr, ElseExpr, Params...>;
  using Type = const TensorSelectOp<typename my_if_type::Type,
                                    typename my_then_type::Type,
                                    typename my_else_type::Type>;

  my_if_type ifExpr;
  my_then_type thenExpr;
  my_else_type elseExpr;
  Type expr;
  template <typename FuncDetector>
  ExprConstructor(FuncDetector &funcD,
                  const utility::tuple::Tuple<Params...> &t)
      : ifExpr(funcD.ifExpr, t),
        thenExpr(funcD.thenExpr, t),
        elseExpr(funcD.elseExpr, t),
        expr(ifExpr.expr, thenExpr.expr, elseExpr.expr) {}
};

/// specialisation of the \ref ExprConstructor struct when the node type is
/// TensorCwiseSelectOp
template <typename OrigIfExpr, typename OrigThenExpr, typename OrigElseExpr,
          typename IfExpr, typename ThenExpr, typename ElseExpr,
          typename... Params>
struct ExprConstructor<TensorSelectOp<OrigIfExpr, OrigThenExpr, OrigElseExpr>,
                       TensorSelectOp<IfExpr, ThenExpr, ElseExpr>, Params...> {
  using my_if_type = ExprConstructor<OrigIfExpr, IfExpr, Params...>;
  using my_then_type = ExprConstructor<OrigThenExpr, ThenExpr, Params...>;
  using my_else_type = ExprConstructor<OrigElseExpr, ElseExpr, Params...>;
  using Type =
      TensorSelectOp<typename my_if_type::Type, typename my_then_type::Type,
                     typename my_else_type::Type>;

  my_if_type ifExpr;
  my_then_type thenExpr;
  my_else_type elseExpr;
  Type expr;
  template <typename FuncDetector>
  ExprConstructor(FuncDetector &funcD,
                  const utility::tuple::Tuple<Params...> &t)
      : ifExpr(funcD.ifExpr, t),
        thenExpr(funcD.thenExpr, t),
        elseExpr(funcD.elseExpr, t),
        expr(ifExpr.expr, thenExpr.expr, elseExpr.expr) {}
};

/// specialisation of the \ref ExprConstructor struct when the node type is
/// TensorAssignOp
template <typename OrigLHSExpr, typename OrigRHSExpr, typename LHSExpr,
          typename RHSExpr, typename... Params>
struct ExprConstructor<TensorAssignOp<OrigLHSExpr, OrigRHSExpr>,
                       TensorAssignOp<LHSExpr, RHSExpr>, Params...> {
  using my_left_type = ExprConstructor<OrigLHSExpr, LHSExpr, Params...>;
  using my_right_type = ExprConstructor<OrigRHSExpr, RHSExpr, Params...>;
  using Type =
      TensorAssignOp<typename my_left_type::Type, typename my_right_type::Type>;

  my_left_type lhsExpr;
  my_right_type rhsExpr;
  Type expr;

  template <typename FuncDetector>
  ExprConstructor(FuncDetector &funcD,
                  const utility::tuple::Tuple<Params...> &t)
      : lhsExpr(funcD.lhsExpr, t),
        rhsExpr(funcD.rhsExpr, t),
        expr(lhsExpr.expr, rhsExpr.expr) {}
};

/// specialisation of the \ref ExprConstructor struct when the node type is
/// const TensorAssignOp
template <typename OrigLHSExpr, typename OrigRHSExpr, typename LHSExpr,
          typename RHSExpr, typename... Params>
struct ExprConstructor<const TensorAssignOp<OrigLHSExpr, OrigRHSExpr>,
                       const TensorAssignOp<LHSExpr, RHSExpr>, Params...> {
  using my_left_type = ExprConstructor<OrigLHSExpr, LHSExpr, Params...>;
  using my_right_type = ExprConstructor<OrigRHSExpr, RHSExpr, Params...>;
  using Type = const TensorAssignOp<typename my_left_type::Type,
                                    typename my_right_type::Type>;

  my_left_type lhsExpr;
  my_right_type rhsExpr;
  Type expr;
  template <typename FuncDetector>
  ExprConstructor(FuncDetector &funcD,
                  const utility::tuple::Tuple<Params...> &t)
      : lhsExpr(funcD.lhsExpr, t),
        rhsExpr(funcD.rhsExpr, t),
        expr(lhsExpr.expr, rhsExpr.expr) {}
};

/// specialisation of the \ref ExprConstructor struct when the node type is
/// const TensorEvalToOp
template <typename OrigExpr, typename Expr, typename... Params>
struct ExprConstructor<const TensorEvalToOp<OrigExpr, MakeGlobalPointer>,
                       const TensorEvalToOp<Expr>, Params...> {
  using my_expr_type = ExprConstructor<OrigExpr, Expr, Params...>;
  using my_buffer_type =
      typename TensorEvalToOp<OrigExpr, MakeGlobalPointer>::PointerType;
  using Type =
      const TensorEvalToOp<typename my_expr_type::Type, MakeGlobalPointer>;
  my_expr_type nestedExpression;
  EvalToLHSConstructor<my_buffer_type, 0, Params...> buffer;
  Type expr;

  template <typename FuncDetector>
  ExprConstructor(FuncDetector &funcD,
                  const utility::tuple::Tuple<Params...> &t)
      : nestedExpression(funcD.rhsExpr, t),
        buffer(t),
        expr(buffer.expr, nestedExpression.expr) {}
};

/// specialisation of the \ref ExprConstructor struct when the node type is
/// TensorEvalToOp
template <typename OrigExpr, typename Expr, typename... Params>
struct ExprConstructor<TensorEvalToOp<OrigExpr, MakeGlobalPointer>,
                       TensorEvalToOp<Expr>, Params...> {
  using my_expr_type = ExprConstructor<OrigExpr, Expr, Params...>;
  using my_buffer_type =
      typename TensorEvalToOp<OrigExpr, MakeGlobalPointer>::PointerType;
  using Type = TensorEvalToOp<typename my_expr_type::Type>;
  my_expr_type nestedExpression;
  EvalToLHSConstructor<my_buffer_type, 0, Params...> buffer;
  Type expr;

  template <typename FuncDetector>
  ExprConstructor(FuncDetector &funcD,
                  const utility::tuple::Tuple<Params...> &t)
      : nestedExpression(funcD.rhsExpr, t),
        buffer(t),
        expr(buffer.expr, nestedExpression.expr) {}
};

/// specialisation of the \ref ExprConstructor struct when the node type is
/// const TensorForcedEvalOp
template <typename OrigExpr, typename DevExpr, size_t N, typename... Params>
struct ExprConstructor<
    const TensorForcedEvalOp<OrigExpr, MakeGlobalPointer>,
    const Eigen::internal::PlaceHolder<const TensorForcedEvalOp<DevExpr>, N>,
    Params...> {
  using Type = const TensorMap<
      Tensor<typename TensorForcedEvalOp<DevExpr, MakeGlobalPointer>::Scalar,
             TensorForcedEvalOp<DevExpr, MakeGlobalPointer>::NumDimensions, 0,
             typename TensorForcedEvalOp<DevExpr>::Index>,
      0, MakeGlobalPointer>;

  Type expr;

  template <typename FuncDetector>
  ExprConstructor(FuncDetector &fd, const utility::tuple::Tuple<Params...> &t)
      : expr(Type((&(*(utility::tuple::get<N>(t).get_pointer()))),
                  fd.dimensions())) {}
};

/// specialisation of the \ref ExprConstructor struct when the node type is
/// TensorForcedEvalOp
template <typename OrigExpr, typename DevExpr, size_t N, typename... Params>
struct ExprConstructor<
    const TensorForcedEvalOp<OrigExpr, MakeGlobalPointer>,
    const Eigen::internal::PlaceHolder<TensorForcedEvalOp<DevExpr>, N>,
    Params...> {
  using Type = TensorMap<
      Tensor<typename TensorForcedEvalOp<DevExpr, MakeGlobalPointer>::Scalar, 1,
             0, typename TensorForcedEvalOp<DevExpr>::Index>,
      0, MakeGlobalPointer>;

  Type expr;

  template <typename FuncDetector>
  ExprConstructor(FuncDetector &fd, const utility::tuple::Tuple<Params...> &t)
      : expr(Type((&(*(utility::tuple::get<N>(t).get_pointer()))),
                  fd.dimensions())) {}
};

/// template deduction for \ref ExprConstructor struct
template <typename OrigExpr, typename IndexExpr, typename FuncD,
          typename... Params>
auto createDeviceExpression(FuncD &funcD,
                            const utility::tuple::Tuple<Params...> &t)
    -> decltype(ExprConstructor<OrigExpr, IndexExpr, Params...>(funcD, t)) {
  return ExprConstructor<OrigExpr, IndexExpr, Params...>(funcD, t);
}
}
}
}  // namespace Eigen

#endif  // UNSUPPORTED_EIGEN_CXX11_SRC_TENSORYSYCL_EXPR_CONSTRUCTOR_HPP
