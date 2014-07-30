// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSEASSIGN_H
#define EIGEN_SPARSEASSIGN_H

namespace Eigen { 

#ifndef EIGEN_TEST_EVALUATORS

template<typename Derived>    
template<typename OtherDerived>
Derived& SparseMatrixBase<Derived>::operator=(const EigenBase<OtherDerived> &other)
{
  other.derived().evalTo(derived());
  return derived();
}

template<typename Derived>
template<typename OtherDerived>
Derived& SparseMatrixBase<Derived>::operator=(const ReturnByValue<OtherDerived>& other)
{
  other.evalTo(derived());
  return derived();
}

template<typename Derived>
template<typename OtherDerived>
inline Derived& SparseMatrixBase<Derived>::operator=(const SparseMatrixBase<OtherDerived>& other)
{
  return assign(other.derived());
}

template<typename Derived>
inline Derived& SparseMatrixBase<Derived>::operator=(const Derived& other)
{
//   if (other.isRValue())
//     derived().swap(other.const_cast_derived());
//   else
  return assign(other.derived());
}

template<typename Derived>
template<typename OtherDerived>
inline Derived& SparseMatrixBase<Derived>::assign(const OtherDerived& other)
{
  const bool transpose = (Flags & RowMajorBit) != (OtherDerived::Flags & RowMajorBit);
  const Index outerSize = (int(OtherDerived::Flags) & RowMajorBit) ? Index(other.rows()) : Index(other.cols());
  if ((!transpose) && other.isRValue())
  {
    // eval without temporary
    derived().resize(Index(other.rows()), Index(other.cols()));
    derived().setZero();
    derived().reserve((std::max)(this->rows(),this->cols())*2);
    for (Index j=0; j<outerSize; ++j)
    {
      derived().startVec(j);
      for (typename OtherDerived::InnerIterator it(other, typename OtherDerived::Index(j)); it; ++it)
      {
        Scalar v = it.value();
        derived().insertBackByOuterInner(j,Index(it.index())) = v;
      }
    }
    derived().finalize();
  }
  else
  {
    assignGeneric(other);
  }
  return derived();
}

template<typename Derived>
template<typename OtherDerived>
inline void SparseMatrixBase<Derived>::assignGeneric(const OtherDerived& other)
{
  //const bool transpose = (Flags & RowMajorBit) != (OtherDerived::Flags & RowMajorBit);
  eigen_assert(( ((internal::traits<Derived>::SupportedAccessPatterns&OuterRandomAccessPattern)==OuterRandomAccessPattern) ||
              (!((Flags & RowMajorBit) != (OtherDerived::Flags & RowMajorBit)))) &&
              "the transpose operation is supposed to be handled in SparseMatrix::operator=");

  enum { Flip = (Flags & RowMajorBit) != (OtherDerived::Flags & RowMajorBit) };

  const Index outerSize = Index(other.outerSize());
  //typedef typename internal::conditional<transpose, LinkedVectorMatrix<Scalar,Flags&RowMajorBit>, Derived>::type TempType;
  // thanks to shallow copies, we always eval to a tempary
  Derived temp(Index(other.rows()), Index(other.cols()));

  temp.reserve((std::max)(this->rows(),this->cols())*2);
  for (Index j=0; j<outerSize; ++j)
  {
    temp.startVec(j);
    for (typename OtherDerived::InnerIterator it(other.derived(), typename OtherDerived::Index(j)); it; ++it)
    {
      Scalar v = it.value();
      temp.insertBackByOuterInner(Flip?Index(it.index()):j,Flip?j:Index(it.index())) = v;
    }
  }
  temp.finalize();

  derived() = temp.markAsRValue();
}

// template<typename Lhs, typename Rhs>
// inline Derived& operator=(const SparseSparseProduct<Lhs,Rhs>& product);
// 
// template<typename OtherDerived>
// Derived& operator+=(const SparseMatrixBase<OtherDerived>& other);
// template<typename OtherDerived>
// Derived& operator-=(const SparseMatrixBase<OtherDerived>& other);
// 
// Derived& operator*=(const Scalar& other);
// Derived& operator/=(const Scalar& other);
// 
// template<typename OtherDerived>
// Derived& operator*=(const SparseMatrixBase<OtherDerived>& other);

#else // EIGEN_TEST_EVALUATORS

template<typename Derived>    
template<typename OtherDerived>
Derived& SparseMatrixBase<Derived>::operator=(const EigenBase<OtherDerived> &other)
{
  // TODO use the evaluator mechanism
  other.derived().evalTo(derived());
  return derived();
}

template<typename Derived>
template<typename OtherDerived>
Derived& SparseMatrixBase<Derived>::operator=(const ReturnByValue<OtherDerived>& other)
{
  // TODO use the evaluator mechanism
  other.evalTo(derived());
  return derived();
}

template<typename Derived>
template<typename OtherDerived>
inline Derived& SparseMatrixBase<Derived>::operator=(const SparseMatrixBase<OtherDerived>& other)
{
  // FIXME, by default sparse evaluation do not alias, so we should be able to bypass the generic call_assignment
  internal::call_assignment/*_no_alias*/(derived(), other.derived());
  return derived();
}

template<typename Derived>
inline Derived& SparseMatrixBase<Derived>::operator=(const Derived& other)
{
  internal::call_assignment_no_alias(derived(), other.derived());
  return derived();
}

namespace internal {

template<>
struct storage_kind_to_evaluator_kind<Sparse> {
  typedef IteratorBased Kind;
};

template<>
struct storage_kind_to_shape<Sparse> {
  typedef SparseShape Shape;
};

struct Sparse2Sparse {};
struct Sparse2Dense  {};

template<> struct AssignmentKind<SparseShape,SparseShape> { typedef Sparse2Sparse Kind; };
template<> struct AssignmentKind<SparseShape,SparseTriangularShape> { typedef Sparse2Sparse Kind; };
template<> struct AssignmentKind<DenseShape,SparseShape>  { typedef Sparse2Dense  Kind; };


template<typename DstXprType, typename SrcXprType>
void assign_sparse_to_sparse(DstXprType &dst, const SrcXprType &src)
{
  eigen_assert(dst.rows() == src.rows() && dst.cols() == src.cols());
  
  typedef typename DstXprType::Index Index;
  typedef typename DstXprType::Scalar Scalar;
  typedef typename internal::evaluator<DstXprType>::type DstEvaluatorType;
  typedef typename internal::evaluator<SrcXprType>::type SrcEvaluatorType;

  SrcEvaluatorType srcEvaluator(src);

  const bool transpose = (DstEvaluatorType::Flags & RowMajorBit) != (SrcEvaluatorType::Flags & RowMajorBit);
  const Index outerEvaluationSize = (SrcEvaluatorType::Flags&RowMajorBit) ? src.rows() : src.cols();
  if ((!transpose) && src.isRValue())
  {
    // eval without temporary
    dst.resize(src.rows(), src.cols());
    dst.setZero();
    dst.reserve((std::max)(src.rows(),src.cols())*2);
    for (Index j=0; j<outerEvaluationSize; ++j)
    {
      dst.startVec(j);
      for (typename SrcEvaluatorType::InnerIterator it(srcEvaluator, j); it; ++it)
      {
        Scalar v = it.value();
        dst.insertBackByOuterInner(j,it.index()) = v;
      }
    }
    dst.finalize();
  }
  else
  {
    // eval through a temporary
    eigen_assert(( ((internal::traits<DstXprType>::SupportedAccessPatterns & OuterRandomAccessPattern)==OuterRandomAccessPattern) ||
              (!((DstEvaluatorType::Flags & RowMajorBit) != (SrcEvaluatorType::Flags & RowMajorBit)))) &&
              "the transpose operation is supposed to be handled in SparseMatrix::operator=");

    enum { Flip = (DstEvaluatorType::Flags & RowMajorBit) != (SrcEvaluatorType::Flags & RowMajorBit) };

    
    DstXprType temp(src.rows(), src.cols());

    temp.reserve((std::max)(src.rows(),src.cols())*2);
    for (Index j=0; j<outerEvaluationSize; ++j)
    {
      temp.startVec(j);
      for (typename SrcEvaluatorType::InnerIterator it(srcEvaluator, j); it; ++it)
      {
        Scalar v = it.value();
        temp.insertBackByOuterInner(Flip?it.index():j,Flip?j:it.index()) = v;
      }
    }
    temp.finalize();

    dst = temp.markAsRValue();
  }
}

// Generic Sparse to Sparse assignment
template< typename DstXprType, typename SrcXprType, typename Functor, typename Scalar>
struct Assignment<DstXprType, SrcXprType, Functor, Sparse2Sparse, Scalar>
{
  static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<typename DstXprType::Scalar> &/*func*/)
  {
    eigen_assert(dst.rows() == src.rows() && dst.cols() == src.cols());
    
    assign_sparse_to_sparse(dst.derived(), src.derived());
  }
};

// Sparse to Dense assignment
template< typename DstXprType, typename SrcXprType, typename Functor, typename Scalar>
struct Assignment<DstXprType, SrcXprType, Functor, Sparse2Dense, Scalar>
{
  static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<typename DstXprType::Scalar> &/*func*/)
  {
    eigen_assert(dst.rows() == src.rows() && dst.cols() == src.cols());
    typedef typename SrcXprType::Index Index;
    
    dst.setZero();
    typename internal::evaluator<SrcXprType>::type srcEval(src);
    typename internal::evaluator<DstXprType>::type dstEval(dst);
    const Index outerEvaluationSize = (internal::evaluator<SrcXprType>::Flags&RowMajorBit) ? src.rows() : src.cols();
    for (Index j=0; j<outerEvaluationSize; ++j)
      for (typename internal::evaluator<SrcXprType>::InnerIterator i(srcEval,j); i; ++i)
        dstEval.coeffRef(i.row(),i.col()) = i.value();
  }
};

} // end namespace internal

#endif // EIGEN_TEST_EVALUATORS

} // end namespace Eigen

#endif // EIGEN_SPARSEASSIGN_H
