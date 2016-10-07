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
 * TensorSyclExtractAccessor.h
 *
 * \brief:
 * ExtractAccessor takes Expression placeHolder expression and the tuple of sycl
 * buffers as an input. Using pre-order tree traversal, ExtractAccessor
 * recursively calls itself for its children in the expression tree. The
 * leaf node in the PlaceHolder expression is nothing but a container preserving
 * the order of the actual data in the tuple of sycl buffer. By invoking the
 * extract accessor for the PlaceHolder<N>, an accessor is created for the Nth
 * buffer in the tuple of buffers. This accessor is then added as an Nth
 * element in the tuple of accessors. In this case we preserve the order of data
 * in the expression tree.
 *
 * This is the specialisation of extract accessor method for different operation
 * type in the PlaceHolder expression.
 *
*****************************************************************/

#ifndef UNSUPPORTED_EIGEN_CXX11_SRC_TENSORYSYCL_EXTRACT_ACCESSOR_HPP
#define UNSUPPORTED_EIGEN_CXX11_SRC_TENSORYSYCL_EXTRACT_ACCESSOR_HPP

namespace Eigen {
namespace TensorSycl {
namespace internal {
/// \struct ExtractAccessor: Extract Accessor Class is used to extract the
/// accessor from a buffer.
/// Depending on the type of the leaf node we can get a read accessor or a
/// read_write accessor
template <typename Evaluator>
struct ExtractAccessor;

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// const TensorMap
template <typename PlainObjectType, int Options_, typename Dev>
struct ExtractAccessor<
    TensorEvaluator<const TensorMap<PlainObjectType, Options_>, Dev>> {
  using actual_type = typename Eigen::internal::remove_all<
      typename Eigen::internal::traits<PlainObjectType>::Scalar>::type;
  static inline auto getTuple(
      cl::sycl::handler& cgh,
      const TensorEvaluator<const TensorMap<PlainObjectType, Options_>, Dev>
          eval)
      -> decltype(utility::tuple::make_tuple(
          (eval.device()
               .template get_sycl_accessor<cl::sycl::access::mode::read, true,
                                           actual_type>(
                   eval.dimensions().TotalSize(), cgh,
                   eval.derived().data())))) {
    return utility::tuple::make_tuple(
        (eval.device()
             .template get_sycl_accessor<cl::sycl::access::mode::read, true,
                                         actual_type>(
                 eval.dimensions().TotalSize(), cgh, eval.derived().data())));
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// TensorMap
template <typename PlainObjectType, int Options_, typename Dev>
struct ExtractAccessor<
    TensorEvaluator<TensorMap<PlainObjectType, Options_>, Dev>> {
  using actual_type = typename Eigen::internal::remove_all<
      typename Eigen::internal::traits<PlainObjectType>::Scalar>::type;

  static inline auto getTuple(
      cl::sycl::handler& cgh,
      TensorEvaluator<TensorMap<PlainObjectType, Options_>, Dev> eval)
      -> decltype(utility::tuple::make_tuple(
          (eval.device()
               .template get_sycl_accessor<cl::sycl::access::mode::read_write,
                                           true, actual_type>(
                   eval.dimensions().TotalSize(), cgh,
                   eval.derived().data())))) {
    return utility::tuple::make_tuple(
        (eval.device()
             .template get_sycl_accessor<cl::sycl::access::mode::read_write,
                                         true, actual_type>(
                 eval.dimensions().TotalSize(), cgh, eval.derived().data())));
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// const TensorCwiseNullaryOp
template <typename OP, typename RHSExpr, typename Dev>
struct ExtractAccessor<
    TensorEvaluator<const TensorCwiseNullaryOp<OP, RHSExpr>, Dev>> {
  static auto getTuple(
      cl::sycl::handler& cgh,
      const TensorEvaluator<const TensorCwiseNullaryOp<OP, RHSExpr>, Dev> eval)
      -> decltype(ExtractAccessor<TensorEvaluator<RHSExpr, Dev>>::getTuple(
          cgh, eval.impl())) {
    auto RHSTuple = ExtractAccessor<TensorEvaluator<RHSExpr, Dev>>::getTuple(
        cgh, eval.impl());
    return RHSTuple;
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// TensorCwiseNullaryOp
template <typename OP, typename RHSExpr, typename Dev>
struct ExtractAccessor<
    TensorEvaluator<TensorCwiseNullaryOp<OP, RHSExpr>, Dev>> {
  static auto getTuple(
      cl::sycl::handler& cgh,
      const TensorEvaluator<TensorCwiseNullaryOp<OP, RHSExpr>, Dev> eval)
      -> decltype(ExtractAccessor<TensorEvaluator<RHSExpr, Dev>>::getTuple(
          cgh, eval.impl())) {
    auto RHSTuple = ExtractAccessor<TensorEvaluator<RHSExpr, Dev>>::getTuple(
        cgh, eval.impl());
    return RHSTuple;
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// const TensorBroadcastingOp
template <typename OP, typename RHSExpr, typename Dev>
struct ExtractAccessor<
    TensorEvaluator<const TensorBroadcastingOp<OP, RHSExpr>, Dev>> {
  static auto getTuple(
      cl::sycl::handler& cgh,
      const TensorEvaluator<const TensorBroadcastingOp<OP, RHSExpr>, Dev> eval)
      -> decltype(ExtractAccessor<TensorEvaluator<RHSExpr, Dev>>::getTuple(
          cgh, eval.impl())) {
    auto RHSTuple = ExtractAccessor<TensorEvaluator<RHSExpr, Dev>>::getTuple(
        cgh, eval.impl());
    return RHSTuple;
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// TensorBroadcastingOp
template <typename OP, typename RHSExpr, typename Dev>
struct ExtractAccessor<
    TensorEvaluator<TensorBroadcastingOp<OP, RHSExpr>, Dev>> {
  static auto getTuple(
      cl::sycl::handler& cgh,
      const TensorEvaluator<TensorBroadcastingOp<OP, RHSExpr>, Dev> eval)
      -> decltype(ExtractAccessor<TensorEvaluator<RHSExpr, Dev>>::getTuple(
          cgh, eval.impl())) {
    auto RHSTuple = ExtractAccessor<TensorEvaluator<RHSExpr, Dev>>::getTuple(
        cgh, eval.impl());
    return RHSTuple;
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// const TenosorCwiseUnary
template <typename OP, typename RHSExpr, typename Dev>
struct ExtractAccessor<
    TensorEvaluator<const TensorCwiseUnaryOp<OP, RHSExpr>, Dev>> {
  static auto getTuple(
      cl::sycl::handler& cgh,
      const TensorEvaluator<const TensorCwiseUnaryOp<OP, RHSExpr>, Dev> eval)
      -> decltype(ExtractAccessor<TensorEvaluator<RHSExpr, Dev>>::getTuple(
          cgh, eval.impl())) {
    auto RHSTuple = ExtractAccessor<TensorEvaluator<RHSExpr, Dev>>::getTuple(
        cgh, eval.impl());
    return RHSTuple;
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// TenosorCwiseUnary
template <typename OP, typename RHSExpr, typename Dev>
struct ExtractAccessor<TensorEvaluator<TensorCwiseUnaryOp<OP, RHSExpr>, Dev>> {
  static auto getTuple(
      cl::sycl::handler& cgh,
      const TensorEvaluator<TensorCwiseUnaryOp<OP, RHSExpr>, Dev> eval)
      -> decltype(ExtractAccessor<TensorEvaluator<RHSExpr, Dev>>::getTuple(
          cgh, eval.impl())) {
    auto RHSTuple = ExtractAccessor<TensorEvaluator<RHSExpr, Dev>>::getTuple(
        cgh, eval.impl());
    return RHSTuple;
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// const TensorCwiseBinaryOp
template <typename OP, typename LHSExpr, typename RHSExpr, typename Dev>
struct ExtractAccessor<
    TensorEvaluator<const TensorCwiseBinaryOp<OP, LHSExpr, RHSExpr>, Dev>> {
  static auto getTuple(cl::sycl::handler& cgh,
                       const TensorEvaluator<
                           const TensorCwiseBinaryOp<OP, LHSExpr, RHSExpr>, Dev>
                           eval)
      -> decltype(utility::tuple::append(
          ExtractAccessor<TensorEvaluator<LHSExpr, Dev>>::getTuple(
              cgh, eval.left_impl()),
          ExtractAccessor<TensorEvaluator<RHSExpr, Dev>>::getTuple(
              cgh, eval.right_impl()))) {
    auto LHSTuple = ExtractAccessor<TensorEvaluator<LHSExpr, Dev>>::getTuple(
        cgh, eval.left_impl());
    auto RHSTuple = ExtractAccessor<TensorEvaluator<RHSExpr, Dev>>::getTuple(
        cgh, eval.right_impl());
    return utility::tuple::append(LHSTuple, RHSTuple);
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// TensorCwiseBinaryOp
template <typename OP, typename LHSExpr, typename RHSExpr, typename Dev>
struct ExtractAccessor<
    TensorEvaluator<TensorCwiseBinaryOp<OP, LHSExpr, RHSExpr>, Dev>> {
  static auto getTuple(
      cl::sycl::handler& cgh,
      const TensorEvaluator<TensorCwiseBinaryOp<OP, LHSExpr, RHSExpr>, Dev>
          eval)
      -> decltype(utility::tuple::append(
          ExtractAccessor<TensorEvaluator<LHSExpr, Dev>>::getTuple(
              cgh, eval.left_impl()),
          ExtractAccessor<TensorEvaluator<RHSExpr, Dev>>::getTuple(
              cgh, eval.right_impl()))) {
    auto LHSTuple = ExtractAccessor<TensorEvaluator<LHSExpr, Dev>>::getTuple(
        cgh, eval.left_impl());
    auto RHSTuple = ExtractAccessor<TensorEvaluator<RHSExpr, Dev>>::getTuple(
        cgh, eval.right_impl());
    return utility::tuple::append(LHSTuple, RHSTuple);
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// const TensorCwiseTernaryOp
template <typename OP, typename Arg1Expr, typename Arg2Expr, typename Arg3Expr,
          typename Dev>
struct ExtractAccessor<TensorEvaluator<
    const TensorCwiseTernaryOp<OP, Arg1Expr, Arg2Expr, Arg3Expr>, Dev>> {
  static auto getTuple(
      cl::sycl::handler& cgh,
      const TensorEvaluator<
          const TensorCwiseTernaryOp<OP, Arg1Expr, Arg2Expr, Arg3Expr>, Dev>
          eval)
      -> decltype(utility::tuple::append(
          ExtractAccessor<TensorEvaluator<Arg1Expr, Dev>>::getTuple(
              cgh, eval.arg1Impl()),
          utility::tuple::append(
              ExtractAccessor<TensorEvaluator<Arg2Expr, Dev>>::getTuple(
                  cgh, eval.arg2Impl()),
              ExtractAccessor<TensorEvaluator<Arg3Expr, Dev>>::getTuple(
                  cgh, eval.arg3Impl())))) {
    auto Arg1Tuple = ExtractAccessor<TensorEvaluator<Arg1Expr, Dev>>::getTuple(
        cgh, eval.arg1Impl());
    auto Arg2Tuple = ExtractAccessor<TensorEvaluator<Arg2Expr, Dev>>::getTuple(
        cgh, eval.arg2Impl());
    auto Arg3Tuple = ExtractAccessor<TensorEvaluator<Arg3Expr, Dev>>::getTuple(
        cgh, eval.arg3Impl());
    return utility::tuple::append(Arg1Tuple,
                                  utility::tuple::append(Arg2Tuple, Arg3Tuple));
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// TensorCwiseTernaryOp
template <typename OP, typename Arg1Expr, typename Arg2Expr, typename Arg3Expr,
          typename Dev>
struct ExtractAccessor<TensorEvaluator<
    TensorCwiseTernaryOp<OP, Arg1Expr, Arg2Expr, Arg3Expr>, Dev>> {
  static auto getTuple(
      cl::sycl::handler& cgh,
      const TensorEvaluator<
          TensorCwiseTernaryOp<OP, Arg1Expr, Arg2Expr, Arg3Expr>, Dev>
          eval)
      -> decltype(utility::tuple::append(
          ExtractAccessor<TensorEvaluator<Arg1Expr, Dev>>::getTuple(
              cgh, eval.arg1Impl()),
          utility::tuple::append(
              ExtractAccessor<TensorEvaluator<Arg2Expr, Dev>>::getTuple(
                  cgh, eval.arg2Impl()),
              ExtractAccessor<TensorEvaluator<Arg3Expr, Dev>>::getTuple(
                  cgh, eval.arg3Impl())))) {
    auto Arg1Tuple = ExtractAccessor<TensorEvaluator<Arg1Expr, Dev>>::getTuple(
        cgh, eval.arg1Impl());
    auto Arg2Tuple = ExtractAccessor<TensorEvaluator<Arg2Expr, Dev>>::getTuple(
        cgh, eval.arg2Impl());
    auto Arg3Tuple = ExtractAccessor<TensorEvaluator<Arg3Expr, Dev>>::getTuple(
        cgh, eval.arg3Impl());
    return utility::tuple::append(Arg1Tuple,
                                  utility::tuple::append(Arg2Tuple, Arg3Tuple));
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// const TensorCwiseSelectOp
template <typename IfExpr, typename ThenExpr, typename ElseExpr, typename Dev>
struct ExtractAccessor<
    TensorEvaluator<const TensorSelectOp<IfExpr, ThenExpr, ElseExpr>, Dev>> {
  static auto getTuple(
      cl::sycl::handler& cgh,
      const TensorEvaluator<const TensorSelectOp<IfExpr, ThenExpr, ElseExpr>,
                            Dev>
          eval)
      -> decltype(utility::tuple::append(
          ExtractAccessor<TensorEvaluator<IfExpr, Dev>>::getTuple(
              cgh, eval.cond_impl()),
          utility::tuple::append(
              ExtractAccessor<TensorEvaluator<ThenExpr, Dev>>::getTuple(
                  cgh, eval.then_impl()),
              ExtractAccessor<TensorEvaluator<ElseExpr, Dev>>::getTuple(
                  cgh, eval.else_impl())))) {
    auto IfTuple = ExtractAccessor<TensorEvaluator<IfExpr, Dev>>::getTuple(
        cgh, eval.cond_impl());
    auto ThenTuple = ExtractAccessor<TensorEvaluator<ThenExpr, Dev>>::getTuple(
        cgh, eval.then_impl());
    auto ElseTuple = ExtractAccessor<TensorEvaluator<ElseExpr, Dev>>::getTuple(
        cgh, eval.else_impl());
    return utility::tuple::append(IfTuple,
                                  utility::tuple::append(ThenTuple, ElseTuple));
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// TensorCwiseSelectOp
template <typename IfExpr, typename ThenExpr, typename ElseExpr, typename Dev>
struct ExtractAccessor<
    TensorEvaluator<TensorSelectOp<IfExpr, ThenExpr, ElseExpr>, Dev>> {
  static auto getTuple(
      cl::sycl::handler& cgh,
      const TensorEvaluator<TensorSelectOp<IfExpr, ThenExpr, ElseExpr>, Dev>
          eval)
      -> decltype(utility::tuple::append(
          ExtractAccessor<TensorEvaluator<IfExpr, Dev>>::getTuple(
              cgh, eval.cond_impl()),
          utility::tuple::append(
              ExtractAccessor<TensorEvaluator<ThenExpr, Dev>>::getTuple(
                  cgh, eval.then_impl()),
              ExtractAccessor<TensorEvaluator<ElseExpr, Dev>>::getTuple(
                  cgh, eval.else_impl())))) {
    auto IfTuple = ExtractAccessor<TensorEvaluator<IfExpr, Dev>>::getTuple(
        cgh, eval.cond_impl());
    auto ThenTuple = ExtractAccessor<TensorEvaluator<ThenExpr, Dev>>::getTuple(
        cgh, eval.then_impl());
    auto ElseTuple = ExtractAccessor<TensorEvaluator<ElseExpr, Dev>>::getTuple(
        cgh, eval.else_impl());
    return utility::tuple::append(IfTuple,
                                  utility::tuple::append(ThenTuple, ElseTuple));
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// const TensorAssignOp
template <typename LHSExpr, typename RHSExpr, typename Dev>
struct ExtractAccessor<
    TensorEvaluator<const TensorAssignOp<LHSExpr, RHSExpr>, Dev>> {
  static auto getTuple(
      cl::sycl::handler& cgh,
      const TensorEvaluator<const TensorAssignOp<LHSExpr, RHSExpr>, Dev> eval)
      -> decltype(utility::tuple::append(
          ExtractAccessor<TensorEvaluator<LHSExpr, Dev>>::getTuple(
              cgh, eval.left_impl()),
          ExtractAccessor<TensorEvaluator<RHSExpr, Dev>>::getTuple(
              cgh, eval.right_impl()))) {
    auto LHSTuple = ExtractAccessor<TensorEvaluator<LHSExpr, Dev>>::getTuple(
        cgh, eval.left_impl());
    auto RHSTuple = ExtractAccessor<TensorEvaluator<RHSExpr, Dev>>::getTuple(
        cgh, eval.right_impl());
    return utility::tuple::append(LHSTuple, RHSTuple);
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// TensorAssignOp
template <typename LHSExpr, typename RHSExpr, typename Dev>
struct ExtractAccessor<TensorEvaluator<TensorAssignOp<LHSExpr, RHSExpr>, Dev>> {
  static auto getTuple(
      cl::sycl::handler& cgh,
      const TensorEvaluator<TensorAssignOp<LHSExpr, RHSExpr>, Dev> eval)
      -> decltype(utility::tuple::append(
          ExtractAccessor<TensorEvaluator<LHSExpr, Dev>>::getTuple(
              eval.left_impl()),
          ExtractAccessor<TensorEvaluator<RHSExpr, Dev>>::getTuple(
              eval.right_impl()))) {
    auto LHSTuple = ExtractAccessor<TensorEvaluator<LHSExpr, Dev>>::getTuple(
        cgh, eval.left_impl());
    auto RHSTuple = ExtractAccessor<TensorEvaluator<RHSExpr, Dev>>::getTuple(
        cgh, eval.right_impl());
    return utility::tuple::append(LHSTuple, RHSTuple);
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// const TensorForcedEvalOp
template <typename Expr, typename Dev>
struct ExtractAccessor<TensorEvaluator<const TensorForcedEvalOp<Expr>, Dev>> {
  using actual_type =
      typename Eigen::internal::remove_all<typename TensorEvaluator<
          const TensorForcedEvalOp<Expr>, Dev>::CoeffReturnType>::type;
  static auto getTuple(
      cl::sycl::handler& cgh,
      const TensorEvaluator<const TensorForcedEvalOp<Expr>, Dev> eval)
      -> decltype(utility::tuple::make_tuple(
          (eval.device()
               .template get_sycl_accessor<cl::sycl::access::mode::read, false,
                                           actual_type>(
                   eval.dimensions().TotalSize(), cgh, eval.data())))) {
    return utility::tuple::make_tuple(
        (eval.device()
             .template get_sycl_accessor<cl::sycl::access::mode::read, false,
                                         actual_type>(
                 eval.dimensions().TotalSize(), cgh, eval.data())));
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// TensorForcedEvalOp
template <typename Expr, typename Dev>
struct ExtractAccessor<TensorEvaluator<TensorForcedEvalOp<Expr>, Dev>>
    : ExtractAccessor<TensorEvaluator<const TensorForcedEvalOp<Expr>, Dev>> {};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// const TensorEvalToOp
template <typename Expr, typename Dev>
struct ExtractAccessor<TensorEvaluator<const TensorEvalToOp<Expr>, Dev>> {
  using actual_type =
      typename Eigen::internal::remove_all<typename TensorEvaluator<
          const TensorEvalToOp<Expr>, Dev>::CoeffReturnType>::type;

  static auto getTuple(cl::sycl::handler& cgh,
                       TensorEvaluator<const TensorEvalToOp<Expr>, Dev> eval)
      -> decltype(utility::tuple::append(
          utility::tuple::make_tuple(
              (eval.device()
                   .template get_sycl_accessor<cl::sycl::access::mode::write,
                                               false, actual_type>(
                       eval.dimensions().TotalSize(), cgh, eval.data()))),
          ExtractAccessor<TensorEvaluator<Expr, Dev>>::getTuple(cgh,
                                                                eval.impl()))) {
    auto LHSTuple = utility::tuple::make_tuple(
        (eval.device()
             .template get_sycl_accessor<cl::sycl::access::mode::write, false,
                                         actual_type>(
                 eval.dimensions().TotalSize(), cgh, eval.data())));

    auto RHSTuple =
        ExtractAccessor<TensorEvaluator<Expr, Dev>>::getTuple(cgh, eval.impl());
    return utility::tuple::append(LHSTuple, RHSTuple);
  }
};

/// specialisation of the \ref ExtractAccessor struct when the node type is
/// TensorEvalToOp
template <typename Expr, typename Dev>
struct ExtractAccessor<TensorEvaluator<TensorEvalToOp<Expr>, Dev>>
    : ExtractAccessor<TensorEvaluator<const TensorEvalToOp<Expr>, Dev>> {};

/// template deduction for \ref ExtractAccessor
template <typename Evaluator>
auto createTupleOfAccessors(cl::sycl::handler& cgh, const Evaluator& expr)
    -> decltype(ExtractAccessor<Evaluator>::getTuple(cgh, expr)) {
  return ExtractAccessor<Evaluator>::getTuple(cgh, expr);
}
}
}
}
#endif  // UNSUPPORTED_EIGEN_CXX11_SRC_TENSORYSYCL_EXTRACT_ACCESSOR_HPP
