// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2008-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2011 Jitse Niesen <jitse@maths.leeds.ac.uk>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#ifndef EIGEN_PRODUCTEVALUATORS_H
#define EIGEN_PRODUCTEVALUATORS_H

namespace Eigen {
  
namespace internal {

/** \internal
  * \class product_evaluator
  * Products need their own evaluator with more template arguments allowing for
  * easier partial template specializations.
  */
template< typename T,
          int ProductTag = internal::product_type<typename T::Lhs,typename T::Rhs>::ret,
          typename LhsShape = typename evaluator_traits<typename T::Lhs>::Shape,
          typename RhsShape = typename evaluator_traits<typename T::Rhs>::Shape,
          typename LhsScalar = typename T::Lhs::Scalar,
          typename RhsScalar = typename T::Rhs::Scalar
        > struct product_evaluator;

/** \internal
  * Evaluator of a product expression.
  * Since products require special treatments to handle all possible cases,
  * we simply deffer the evaluation logic to a product_evaluator class
  * which offers more partial specialization possibilities.
  * 
  * \sa class product_evaluator
  */
template<typename Lhs, typename Rhs, int Options>
struct evaluator<Product<Lhs, Rhs, Options> > 
 : public product_evaluator<Product<Lhs, Rhs, Options> >
{
  typedef Product<Lhs, Rhs, Options> XprType;
  typedef product_evaluator<XprType> Base;
  
  typedef evaluator type;
  typedef evaluator nestedType;
  
  evaluator(const XprType& xpr) : Base(xpr) {}
};
 
// Catch scalar * ( A * B ) and transform it to (A*scalar) * B
// TODO we should apply that rule only if that's really helpful
template<typename Lhs, typename Rhs, typename Scalar>
struct evaluator<CwiseUnaryOp<internal::scalar_multiple_op<Scalar>,  const Product<Lhs, Rhs, DefaultProduct>  > > 
 : public evaluator<Product<CwiseUnaryOp<internal::scalar_multiple_op<Scalar>,const Lhs>, Rhs, DefaultProduct> >
{
  typedef CwiseUnaryOp<internal::scalar_multiple_op<Scalar>, const Product<Lhs, Rhs, DefaultProduct> > XprType;
  typedef evaluator<Product<CwiseUnaryOp<internal::scalar_multiple_op<Scalar>,const Lhs>, Rhs, DefaultProduct> > Base;
  
  typedef evaluator type;
  typedef evaluator nestedType;
  
  evaluator(const XprType& xpr)
    : Base(xpr.functor().m_other * xpr.nestedExpression().lhs() * xpr.nestedExpression().rhs())
  {}
};


template<typename Lhs, typename Rhs, int DiagIndex>
struct evaluator<Diagonal<const Product<Lhs, Rhs, DefaultProduct>, DiagIndex> > 
 : public evaluator<Diagonal<const Product<Lhs, Rhs, LazyProduct>, DiagIndex> >
{
  typedef Diagonal<const Product<Lhs, Rhs, DefaultProduct>, DiagIndex> XprType;
  typedef evaluator<Diagonal<const Product<Lhs, Rhs, LazyProduct>, DiagIndex> > Base;
  
  typedef evaluator type;
  typedef evaluator nestedType;

  evaluator(const XprType& xpr)
    : Base(Diagonal<const Product<Lhs, Rhs, LazyProduct>, DiagIndex>(
        Product<Lhs, Rhs, LazyProduct>(xpr.nestedExpression().lhs(), xpr.nestedExpression().rhs()),
        xpr.index() ))
  {}
};


// Helper class to perform a matrix product with the destination at hand.
// Depending on the sizes of the factors, there are different evaluation strategies
// as controlled by internal::product_type.
template< typename Lhs, typename Rhs,
          typename LhsShape = typename evaluator_traits<Lhs>::Shape,
          typename RhsShape = typename evaluator_traits<Rhs>::Shape,
          int ProductType = internal::product_type<Lhs,Rhs>::value>
struct generic_product_impl;

template<typename Lhs, typename Rhs>
struct evaluator_traits<Product<Lhs, Rhs, DefaultProduct> > 
 : evaluator_traits_base<Product<Lhs, Rhs, DefaultProduct> >
{
  enum { AssumeAliasing = 1 };
};

// The evaluator for default dense products creates a temporary and call generic_product_impl
template<typename Lhs, typename Rhs, int ProductTag>
struct product_evaluator<Product<Lhs, Rhs, DefaultProduct>, ProductTag, DenseShape, DenseShape, typename Lhs::Scalar, typename Rhs::Scalar> 
  : public evaluator<typename Product<Lhs, Rhs, DefaultProduct>::PlainObject>::type
{
  typedef Product<Lhs, Rhs, DefaultProduct> XprType;
  typedef typename XprType::PlainObject PlainObject;
  typedef typename evaluator<PlainObject>::type Base;

  product_evaluator(const XprType& xpr)
    : m_result(xpr.rows(), xpr.cols())
  {
    ::new (static_cast<Base*>(this)) Base(m_result);
    generic_product_impl<Lhs, Rhs>::evalTo(m_result, xpr.lhs(), xpr.rhs());
  }
  
protected:  
  PlainObject m_result;
};

// Dense = Product
template< typename DstXprType, typename Lhs, typename Rhs, typename Scalar>
struct Assignment<DstXprType, Product<Lhs,Rhs,DefaultProduct>, internal::assign_op<Scalar>, Dense2Dense, Scalar>
{
  typedef Product<Lhs,Rhs,DefaultProduct> SrcXprType;
  static void run(DstXprType &dst, const SrcXprType &src, const internal::assign_op<Scalar> &)
  {
    generic_product_impl<Lhs, Rhs>::evalTo(dst, src.lhs(), src.rhs());
  }
};

// Dense += Product
template< typename DstXprType, typename Lhs, typename Rhs, typename Scalar>
struct Assignment<DstXprType, Product<Lhs,Rhs,DefaultProduct>, internal::add_assign_op<Scalar>, Dense2Dense, Scalar>
{
  typedef Product<Lhs,Rhs,DefaultProduct> SrcXprType;
  static void run(DstXprType &dst, const SrcXprType &src, const internal::add_assign_op<Scalar> &)
  {
    generic_product_impl<Lhs, Rhs>::addTo(dst, src.lhs(), src.rhs());
  }
};

// Dense -= Product
template< typename DstXprType, typename Lhs, typename Rhs, typename Scalar>
struct Assignment<DstXprType, Product<Lhs,Rhs,DefaultProduct>, internal::sub_assign_op<Scalar>, Dense2Dense, Scalar>
{
  typedef Product<Lhs,Rhs,DefaultProduct> SrcXprType;
  static void run(DstXprType &dst, const SrcXprType &src, const internal::sub_assign_op<Scalar> &)
  {
    generic_product_impl<Lhs, Rhs>::subTo(dst, src.lhs(), src.rhs());
  }
};


// Dense ?= scalar * Product
// TODO we should apply that rule if that's really helpful
// for instance, this is not good for inner products
template< typename DstXprType, typename Lhs, typename Rhs, typename AssignFunc, typename Scalar, typename ScalarBis>
struct Assignment<DstXprType, CwiseUnaryOp<internal::scalar_multiple_op<ScalarBis>,
                                           const Product<Lhs,Rhs,DefaultProduct> >, AssignFunc, Dense2Dense, Scalar>
{
  typedef CwiseUnaryOp<internal::scalar_multiple_op<ScalarBis>,
                       const Product<Lhs,Rhs,DefaultProduct> > SrcXprType;
  static void run(DstXprType &dst, const SrcXprType &src, const AssignFunc& func)
  {
    // TODO use operator* instead of prod() once we have made enough progress
    call_assignment(dst.noalias(), prod(src.functor().m_other * src.nestedExpression().lhs(), src.nestedExpression().rhs()), func);
  }
};


template<typename Lhs, typename Rhs>
struct generic_product_impl<Lhs,Rhs,DenseShape,DenseShape,InnerProduct>
{
  template<typename Dst>
  static inline void evalTo(Dst& dst, const Lhs& lhs, const Rhs& rhs)
  {
    dst.coeffRef(0,0) = (lhs.transpose().cwiseProduct(rhs)).sum();
  }
  
  template<typename Dst>
  static inline void addTo(Dst& dst, const Lhs& lhs, const Rhs& rhs)
  {
    dst.coeffRef(0,0) += (lhs.transpose().cwiseProduct(rhs)).sum();
  }
  
  template<typename Dst>
  static void subTo(Dst& dst, const Lhs& lhs, const Rhs& rhs)
  { dst.coeffRef(0,0) -= (lhs.transpose().cwiseProduct(rhs)).sum(); }
};


/***********************************************************************
*  Implementation of outer dense * dense vector product
***********************************************************************/

// Column major result
template<typename Dst, typename Lhs, typename Rhs, typename Func>
EIGEN_DONT_INLINE void outer_product_selector_run(Dst& dst, const Lhs &lhs, const Rhs &rhs, const Func& func, const false_type&)
{
  typedef typename Dst::Index Index;
  // FIXME make sure lhs is sequentially stored
  // FIXME not very good if rhs is real and lhs complex while alpha is real too
  // FIXME we should probably build an evaluator for dst and rhs
  const Index cols = dst.cols();
  for (Index j=0; j<cols; ++j)
    func(dst.col(j), rhs.coeff(j) * lhs);
}

// Row major result
template<typename Dst, typename Lhs, typename Rhs, typename Func>
EIGEN_DONT_INLINE void outer_product_selector_run(Dst& dst, const Lhs &lhs, const Rhs &rhs, const Func& func, const true_type&) {
  typedef typename Dst::Index Index;
  // FIXME make sure rhs is sequentially stored
  // FIXME not very good if lhs is real and rhs complex while alpha is real too
  // FIXME we should probably build an evaluator for dst and lhs
  const Index rows = dst.rows();
  for (Index i=0; i<rows; ++i)
    func(dst.row(i), lhs.coeff(i) * rhs);
}

template<typename Lhs, typename Rhs>
struct generic_product_impl<Lhs,Rhs,DenseShape,DenseShape,OuterProduct>
{
  template<typename T> struct IsRowMajor : internal::conditional<(int(T::Flags)&RowMajorBit), internal::true_type, internal::false_type>::type {};
  typedef typename Product<Lhs,Rhs>::Scalar Scalar;
  
  // TODO it would be nice to be able to exploit our *_assign_op functors for that purpose
  struct set  { template<typename Dst, typename Src> void operator()(const Dst& dst, const Src& src) const { dst.const_cast_derived()  = src; } };
  struct add  { template<typename Dst, typename Src> void operator()(const Dst& dst, const Src& src) const { dst.const_cast_derived() += src; } };
  struct sub  { template<typename Dst, typename Src> void operator()(const Dst& dst, const Src& src) const { dst.const_cast_derived() -= src; } };
  struct adds {
    Scalar m_scale;
    adds(const Scalar& s) : m_scale(s) {}
    template<typename Dst, typename Src> void operator()(const Dst& dst, const Src& src) const {
      dst.const_cast_derived() += m_scale * src;
    }
  };
  
  template<typename Dst>
  static inline void evalTo(Dst& dst, const Lhs& lhs, const Rhs& rhs)
  {
    internal::outer_product_selector_run(dst, lhs, rhs, set(), IsRowMajor<Dst>());
  }
  
  template<typename Dst>
  static inline void addTo(Dst& dst, const Lhs& lhs, const Rhs& rhs)
  {
    internal::outer_product_selector_run(dst, lhs, rhs, add(), IsRowMajor<Dst>());
  }
  
  template<typename Dst>
  static inline void subTo(Dst& dst, const Lhs& lhs, const Rhs& rhs)
  {
    internal::outer_product_selector_run(dst, lhs, rhs, sub(), IsRowMajor<Dst>());
  }
  
  template<typename Dst>
  static inline void scaleAndAddTo(Dst& dst, const Lhs& lhs, const Rhs& rhs, const Scalar& alpha)
  {
    internal::outer_product_selector_run(dst, lhs, rhs, adds(alpha), IsRowMajor<Dst>());
  }
  
};


// This base class provides default implementations for evalTo, addTo, subTo, in terms of scaleAndAddTo
template<typename Lhs, typename Rhs, typename Derived>
struct generic_product_impl_base
{
  typedef typename Product<Lhs,Rhs>::Scalar Scalar;
  
  template<typename Dst>
  static void evalTo(Dst& dst, const Lhs& lhs, const Rhs& rhs)
  { dst.setZero(); scaleAndAddTo(dst, lhs, rhs, Scalar(1)); }

  template<typename Dst>
  static void addTo(Dst& dst, const Lhs& lhs, const Rhs& rhs)
  { scaleAndAddTo(dst,lhs, rhs, Scalar(1)); }

  template<typename Dst>
  static void subTo(Dst& dst, const Lhs& lhs, const Rhs& rhs)
  { scaleAndAddTo(dst, lhs, rhs, Scalar(-1)); }
  
  template<typename Dst>
  static void scaleAndAddTo(Dst& dst, const Lhs& lhs, const Rhs& rhs, const Scalar& alpha)
  { Derived::scaleAndAddTo(dst,lhs,rhs,alpha); }

};

template<typename Lhs, typename Rhs>
struct generic_product_impl<Lhs,Rhs,DenseShape,DenseShape,GemvProduct>
  : generic_product_impl_base<Lhs,Rhs,generic_product_impl<Lhs,Rhs,DenseShape,DenseShape,GemvProduct> >
{
  typedef typename Product<Lhs,Rhs>::Scalar Scalar;
  enum { Side = Lhs::IsVectorAtCompileTime ? OnTheLeft : OnTheRight };
  typedef typename internal::conditional<int(Side)==OnTheRight,Lhs,Rhs>::type MatrixType;

  template<typename Dest>
  static void scaleAndAddTo(Dest& dst, const Lhs& lhs, const Rhs& rhs, const Scalar& alpha)
  {
    internal::gemv_dense_sense_selector<Side,
                            (int(MatrixType::Flags)&RowMajorBit) ? RowMajor : ColMajor,
                            bool(internal::blas_traits<MatrixType>::HasUsableDirectAccess)
                           >::run(lhs, rhs, dst, alpha);
  }
};

template<typename Lhs, typename Rhs>
struct generic_product_impl<Lhs,Rhs,DenseShape,DenseShape,CoeffBasedProductMode> 
{
  typedef typename Product<Lhs,Rhs>::Scalar Scalar;
  
  template<typename Dst>
  static inline void evalTo(Dst& dst, const Lhs& lhs, const Rhs& rhs)
  {
    // TODO: use the following instead of calling call_assignment, same for the other methods
    // dst = lazyprod(lhs,rhs);
    call_assignment(dst, lazyprod(lhs,rhs), internal::assign_op<Scalar>());
  }
  
  template<typename Dst>
  static inline void addTo(Dst& dst, const Lhs& lhs, const Rhs& rhs)
  {
    // dst += lazyprod(lhs,rhs);
    call_assignment(dst, lazyprod(lhs,rhs), internal::add_assign_op<Scalar>());
  }
  
  template<typename Dst>
  static inline void subTo(Dst& dst, const Lhs& lhs, const Rhs& rhs)
  {
    // dst -= lazyprod(lhs,rhs);
    call_assignment(dst, lazyprod(lhs,rhs), internal::sub_assign_op<Scalar>());
  }
  
//   template<typename Dst>
//   static inline void scaleAndAddTo(Dst& dst, const Lhs& lhs, const Rhs& rhs, const Scalar& alpha)
//   { dst += alpha * lazyprod(lhs,rhs); }
};

// This specialization enforces the use of a coefficient-based evaluation strategy
template<typename Lhs, typename Rhs>
struct generic_product_impl<Lhs,Rhs,DenseShape,DenseShape,LazyCoeffBasedProductMode>
  : generic_product_impl<Lhs,Rhs,DenseShape,DenseShape,CoeffBasedProductMode> {};

// Case 2: Evaluate coeff by coeff
//
// This is mostly taken from CoeffBasedProduct.h
// The main difference is that we add an extra argument to the etor_product_*_impl::run() function
// for the inner dimension of the product, because evaluator object do not know their size.

template<int Traversal, int UnrollingIndex, typename Lhs, typename Rhs, typename RetScalar>
struct etor_product_coeff_impl;

template<int StorageOrder, int UnrollingIndex, typename Lhs, typename Rhs, typename Packet, int LoadMode>
struct etor_product_packet_impl;

template<typename Lhs, typename Rhs, int ProductTag>
struct product_evaluator<Product<Lhs, Rhs, LazyProduct>, ProductTag, DenseShape, DenseShape, typename Lhs::Scalar, typename Rhs::Scalar > 
    : evaluator_base<Product<Lhs, Rhs, LazyProduct> >
{
  typedef Product<Lhs, Rhs, LazyProduct> XprType;
  typedef CoeffBasedProduct<Lhs, Rhs, 0> CoeffBasedProductType;

  product_evaluator(const XprType& xpr) 
    : m_lhs(xpr.lhs()),
      m_rhs(xpr.rhs()),
      m_lhsImpl(m_lhs),     // FIXME the creation of the evaluator objects should result in a no-op, but check that!
      m_rhsImpl(m_rhs),     //       Moreover, they are only useful for the packet path, so we could completely disable them when not needed,
                            //       or perhaps declare them on the fly on the packet method... We have experiment to check what's best.
      m_innerDim(xpr.lhs().cols())
  { }

  typedef typename XprType::Index Index;
  typedef typename XprType::Scalar Scalar;
  typedef typename XprType::CoeffReturnType CoeffReturnType;
  typedef typename XprType::PacketScalar PacketScalar;
  typedef typename XprType::PacketReturnType PacketReturnType;

  // Everything below here is taken from CoeffBasedProduct.h

  enum {
    RowsAtCompileTime = traits<CoeffBasedProductType>::RowsAtCompileTime,
    PacketSize = packet_traits<Scalar>::size,
    InnerSize  = traits<CoeffBasedProductType>::InnerSize,
    CoeffReadCost = traits<CoeffBasedProductType>::CoeffReadCost,
    Unroll = CoeffReadCost != Dynamic && CoeffReadCost <= EIGEN_UNROLLING_LIMIT,
    CanVectorizeInner = traits<CoeffBasedProductType>::CanVectorizeInner,
    Flags = traits<CoeffBasedProductType>::Flags
  };
  
  typedef typename internal::nested_eval<Lhs,Rhs::ColsAtCompileTime>::type LhsNested;
  typedef typename internal::nested_eval<Rhs,Lhs::RowsAtCompileTime>::type RhsNested;
  
  typedef typename internal::remove_all<LhsNested>::type LhsNestedCleaned;
  typedef typename internal::remove_all<RhsNested>::type RhsNestedCleaned;

  typedef typename evaluator<LhsNestedCleaned>::type LhsEtorType;
  typedef typename evaluator<RhsNestedCleaned>::type RhsEtorType;
  
  const CoeffReturnType coeff(Index row, Index col) const
  {
    // TODO check performance regression wrt to Eigen 3.2 which has special handling of this function
    return (m_lhs.row(row).transpose().cwiseProduct( m_rhs.col(col) )).sum();
  }

  /* Allow index-based non-packet access. It is impossible though to allow index-based packed access,
   * which is why we don't set the LinearAccessBit.
   * TODO: this seems possible when the result is a vector
   */
  const CoeffReturnType coeff(Index index) const
  {
    const Index row = RowsAtCompileTime == 1 ? 0 : index;
    const Index col = RowsAtCompileTime == 1 ? index : 0;
    // TODO check performance regression wrt to Eigen 3.2 which has special handling of this function
    return (m_lhs.row(row).transpose().cwiseProduct( m_rhs.col(col) )).sum();
  }

  template<int LoadMode>
  const PacketReturnType packet(Index row, Index col) const
  {
    PacketScalar res;
    typedef etor_product_packet_impl<Flags&RowMajorBit ? RowMajor : ColMajor,
                                     Unroll ? InnerSize-1 : Dynamic,
                                     LhsEtorType, RhsEtorType, PacketScalar, LoadMode> PacketImpl;

    PacketImpl::run(row, col, m_lhsImpl, m_rhsImpl, m_innerDim, res);
    return res;
  }

protected:
  const LhsNested m_lhs;
  const RhsNested m_rhs;
  
  LhsEtorType m_lhsImpl;
  RhsEtorType m_rhsImpl;

  // TODO: Get rid of m_innerDim if known at compile time
  Index m_innerDim;
};

template<typename Lhs, typename Rhs>
struct product_evaluator<Product<Lhs, Rhs, DefaultProduct>, LazyCoeffBasedProductMode, DenseShape, DenseShape, typename Lhs::Scalar, typename Rhs::Scalar > 
  : product_evaluator<Product<Lhs, Rhs, LazyProduct>, CoeffBasedProductMode, DenseShape, DenseShape, typename Lhs::Scalar, typename Rhs::Scalar >
{
  typedef Product<Lhs, Rhs, DefaultProduct> XprType;
  typedef Product<Lhs, Rhs, LazyProduct> BaseProduct;
  typedef product_evaluator<BaseProduct, CoeffBasedProductMode, DenseShape, DenseShape, typename Lhs::Scalar, typename Rhs::Scalar > Base;
  product_evaluator(const XprType& xpr) 
    : Base(BaseProduct(xpr.lhs(),xpr.rhs()))
  {}
};

/****************************************
*** Coeff based product, Packet path  ***
****************************************/

template<int UnrollingIndex, typename Lhs, typename Rhs, typename Packet, int LoadMode>
struct etor_product_packet_impl<RowMajor, UnrollingIndex, Lhs, Rhs, Packet, LoadMode>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Index innerDim, Packet &res)
  {
    etor_product_packet_impl<RowMajor, UnrollingIndex-1, Lhs, Rhs, Packet, LoadMode>::run(row, col, lhs, rhs, innerDim, res);
    res =  pmadd(pset1<Packet>(lhs.coeff(row, UnrollingIndex)), rhs.template packet<LoadMode>(UnrollingIndex, col), res);
  }
};

template<int UnrollingIndex, typename Lhs, typename Rhs, typename Packet, int LoadMode>
struct etor_product_packet_impl<ColMajor, UnrollingIndex, Lhs, Rhs, Packet, LoadMode>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Index innerDim, Packet &res)
  {
    etor_product_packet_impl<ColMajor, UnrollingIndex-1, Lhs, Rhs, Packet, LoadMode>::run(row, col, lhs, rhs, innerDim, res);
    res =  pmadd(lhs.template packet<LoadMode>(row, UnrollingIndex), pset1<Packet>(rhs.coeff(UnrollingIndex, col)), res);
  }
};

template<typename Lhs, typename Rhs, typename Packet, int LoadMode>
struct etor_product_packet_impl<RowMajor, 0, Lhs, Rhs, Packet, LoadMode>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Index /*innerDim*/, Packet &res)
  {
    res = pmul(pset1<Packet>(lhs.coeff(row, 0)),rhs.template packet<LoadMode>(0, col));
  }
};

template<typename Lhs, typename Rhs, typename Packet, int LoadMode>
struct etor_product_packet_impl<ColMajor, 0, Lhs, Rhs, Packet, LoadMode>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Index /*innerDim*/, Packet &res)
  {
    res = pmul(lhs.template packet<LoadMode>(row, 0), pset1<Packet>(rhs.coeff(0, col)));
  }
};

template<typename Lhs, typename Rhs, typename Packet, int LoadMode>
struct etor_product_packet_impl<RowMajor, Dynamic, Lhs, Rhs, Packet, LoadMode>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Index innerDim, Packet& res)
  {
    eigen_assert(innerDim>0 && "you are using a non initialized matrix");
    res = pmul(pset1<Packet>(lhs.coeff(row, 0)),rhs.template packet<LoadMode>(0, col));
    for(Index i = 1; i < innerDim; ++i)
      res =  pmadd(pset1<Packet>(lhs.coeff(row, i)), rhs.template packet<LoadMode>(i, col), res);
  }
};

template<typename Lhs, typename Rhs, typename Packet, int LoadMode>
struct etor_product_packet_impl<ColMajor, Dynamic, Lhs, Rhs, Packet, LoadMode>
{
  typedef typename Lhs::Index Index;
  static EIGEN_STRONG_INLINE void run(Index row, Index col, const Lhs& lhs, const Rhs& rhs, Index innerDim, Packet& res)
  {
    eigen_assert(innerDim>0 && "you are using a non initialized matrix");
    res = pmul(lhs.template packet<LoadMode>(row, 0), pset1<Packet>(rhs.coeff(0, col)));
    for(Index i = 1; i < innerDim; ++i)
      res =  pmadd(lhs.template packet<LoadMode>(row, i), pset1<Packet>(rhs.coeff(i, col)), res);
  }
};


/***************************************************************************
* Triangular products
***************************************************************************/
template<int Mode, bool LhsIsTriangular,
         typename Lhs, bool LhsIsVector,
         typename Rhs, bool RhsIsVector>
struct triangular_product_impl;

template<typename Lhs, typename Rhs, int ProductTag>
struct generic_product_impl<Lhs,Rhs,TriangularShape,DenseShape,ProductTag>
  : generic_product_impl_base<Lhs,Rhs,generic_product_impl<Lhs,Rhs,TriangularShape,DenseShape,ProductTag> >
{
  typedef typename Product<Lhs,Rhs>::Scalar Scalar;
  
  template<typename Dest>
  static void scaleAndAddTo(Dest& dst, const Lhs& lhs, const Rhs& rhs, const Scalar& alpha)
  {
    triangular_product_impl<Lhs::Mode,true,typename Lhs::MatrixType,false,Rhs, Rhs::IsVectorAtCompileTime>
        ::run(dst, lhs.nestedExpression(), rhs, alpha);
  }
};

template<typename Lhs, typename Rhs, int ProductTag>
struct product_evaluator<Product<Lhs, Rhs, DefaultProduct>, ProductTag, TriangularShape, DenseShape, typename Lhs::Scalar, typename Rhs::Scalar> 
  : public evaluator<typename Product<Lhs, Rhs, DefaultProduct>::PlainObject>::type
{
  typedef Product<Lhs, Rhs, DefaultProduct> XprType;
  typedef typename XprType::PlainObject PlainObject;
  typedef typename evaluator<PlainObject>::type Base;

  product_evaluator(const XprType& xpr)
    : m_result(xpr.rows(), xpr.cols())
  {
    ::new (static_cast<Base*>(this)) Base(m_result);
    generic_product_impl<Lhs, Rhs, TriangularShape, DenseShape, ProductTag>::evalTo(m_result, xpr.lhs(), xpr.rhs());
  }
  
protected:  
  PlainObject m_result;
};


template<typename Lhs, typename Rhs, int ProductTag>
struct generic_product_impl<Lhs,Rhs,DenseShape,TriangularShape,ProductTag>
: generic_product_impl_base<Lhs,Rhs,generic_product_impl<Lhs,Rhs,DenseShape,TriangularShape,ProductTag> >
{
  typedef typename Product<Lhs,Rhs>::Scalar Scalar;
  
  template<typename Dest>
  static void scaleAndAddTo(Dest& dst, const Lhs& lhs, const Rhs& rhs, const Scalar& alpha)
  {
    triangular_product_impl<Rhs::Mode,false,Lhs,Lhs::IsVectorAtCompileTime, typename Rhs::MatrixType, false>::run(dst, lhs, rhs.nestedExpression(), alpha);
  }
};

template<typename Lhs, typename Rhs, int ProductTag>
struct product_evaluator<Product<Lhs, Rhs, DefaultProduct>, ProductTag, DenseShape, TriangularShape, typename Lhs::Scalar, typename Rhs::Scalar> 
  : public evaluator<typename Product<Lhs, Rhs, DefaultProduct>::PlainObject>::type
{
  typedef Product<Lhs, Rhs, DefaultProduct> XprType;
  typedef typename XprType::PlainObject PlainObject;
  typedef typename evaluator<PlainObject>::type Base;

  product_evaluator(const XprType& xpr)
    : m_result(xpr.rows(), xpr.cols())
  {
    ::new (static_cast<Base*>(this)) Base(m_result);
    generic_product_impl<Lhs, Rhs, DenseShape, TriangularShape, ProductTag>::evalTo(m_result, xpr.lhs(), xpr.rhs());
  }
  
protected:  
  PlainObject m_result;
};



/***************************************************************************
* SelfAdjoint products
***************************************************************************/
template <typename Lhs, int LhsMode, bool LhsIsVector,
          typename Rhs, int RhsMode, bool RhsIsVector>
struct selfadjoint_product_impl;

template<typename Lhs, typename Rhs, int ProductTag>
struct generic_product_impl<Lhs,Rhs,SelfAdjointShape,DenseShape,ProductTag>
  : generic_product_impl_base<Lhs,Rhs,generic_product_impl<Lhs,Rhs,SelfAdjointShape,DenseShape,ProductTag> >
{
  typedef typename Product<Lhs,Rhs>::Scalar Scalar;
  
  template<typename Dest>
  static void scaleAndAddTo(Dest& dst, const Lhs& lhs, const Rhs& rhs, const Scalar& alpha)
  {
    selfadjoint_product_impl<typename Lhs::MatrixType,Lhs::Mode,false,Rhs,0,Rhs::IsVectorAtCompileTime>::run(dst, lhs.nestedExpression(), rhs, alpha);
  }
};

template<typename Lhs, typename Rhs, int ProductTag>
struct product_evaluator<Product<Lhs, Rhs, DefaultProduct>, ProductTag, SelfAdjointShape, DenseShape, typename Lhs::Scalar, typename Rhs::Scalar> 
  : public evaluator<typename Product<Lhs, Rhs, DefaultProduct>::PlainObject>::type
{
  typedef Product<Lhs, Rhs, DefaultProduct> XprType;
  typedef typename XprType::PlainObject PlainObject;
  typedef typename evaluator<PlainObject>::type Base;

  product_evaluator(const XprType& xpr)
    : m_result(xpr.rows(), xpr.cols())
  {
    ::new (static_cast<Base*>(this)) Base(m_result);
    generic_product_impl<Lhs, Rhs, SelfAdjointShape, DenseShape, ProductTag>::evalTo(m_result, xpr.lhs(), xpr.rhs());
  }
  
protected:  
  PlainObject m_result;
};


template<typename Lhs, typename Rhs, int ProductTag>
struct generic_product_impl<Lhs,Rhs,DenseShape,SelfAdjointShape,ProductTag>
: generic_product_impl_base<Lhs,Rhs,generic_product_impl<Lhs,Rhs,DenseShape,SelfAdjointShape,ProductTag> >
{
  typedef typename Product<Lhs,Rhs>::Scalar Scalar;
  
  template<typename Dest>
  static void scaleAndAddTo(Dest& dst, const Lhs& lhs, const Rhs& rhs, const Scalar& alpha)
  {
    selfadjoint_product_impl<Lhs,0,Lhs::IsVectorAtCompileTime,typename Rhs::MatrixType,Rhs::Mode,false>::run(dst, lhs, rhs.nestedExpression(), alpha);
  }
};

template<typename Lhs, typename Rhs, int ProductTag>
struct product_evaluator<Product<Lhs, Rhs, DefaultProduct>, ProductTag, DenseShape, SelfAdjointShape, typename Lhs::Scalar, typename Rhs::Scalar> 
  : public evaluator<typename Product<Lhs, Rhs, DefaultProduct>::PlainObject>::type
{
  typedef Product<Lhs, Rhs, DefaultProduct> XprType;
  typedef typename XprType::PlainObject PlainObject;
  typedef typename evaluator<PlainObject>::type Base;

  product_evaluator(const XprType& xpr)
    : m_result(xpr.rows(), xpr.cols())
  {
    ::new (static_cast<Base*>(this)) Base(m_result);
    generic_product_impl<Lhs, Rhs, DenseShape, SelfAdjointShape, ProductTag>::evalTo(m_result, xpr.lhs(), xpr.rhs());
  }
  
protected:  
  PlainObject m_result;
};

/***************************************************************************
* Diagonal products
***************************************************************************/
  
template<typename MatrixType, typename DiagonalType, typename Derived>
struct diagonal_product_evaluator_base
  : evaluator_base<Derived>
{
   typedef typename MatrixType::Index Index;
   typedef typename scalar_product_traits<typename MatrixType::Scalar, typename DiagonalType::Scalar>::ReturnType Scalar;
   typedef typename internal::packet_traits<Scalar>::type PacketScalar;
public:
  diagonal_product_evaluator_base(const MatrixType &mat, const DiagonalType &diag)
    : m_diagImpl(diag), m_matImpl(mat)
  {
  }
  
  EIGEN_STRONG_INLINE const Scalar coeff(Index idx) const
  {
    return m_diagImpl.coeff(idx) * m_matImpl.coeff(idx);
  }
  
protected:
  template<int LoadMode>
  EIGEN_STRONG_INLINE PacketScalar packet_impl(Index row, Index col, Index id, internal::true_type) const
  {
    return internal::pmul(m_matImpl.template packet<LoadMode>(row, col),
                          internal::pset1<PacketScalar>(m_diagImpl.coeff(id)));
  }
  
  template<int LoadMode>
  EIGEN_STRONG_INLINE PacketScalar packet_impl(Index row, Index col, Index id, internal::false_type) const
  {
    enum {
      InnerSize = (MatrixType::Flags & RowMajorBit) ? MatrixType::ColsAtCompileTime : MatrixType::RowsAtCompileTime,
      DiagonalPacketLoadMode = (LoadMode == Aligned && (((InnerSize%16) == 0) || (int(DiagonalType::Flags)&AlignedBit)==AlignedBit) ? Aligned : Unaligned)
    };
    return internal::pmul(m_matImpl.template packet<LoadMode>(row, col),
                          m_diagImpl.template packet<DiagonalPacketLoadMode>(id));
  }
  
  typename evaluator<DiagonalType>::nestedType m_diagImpl;
  typename evaluator<MatrixType>::nestedType   m_matImpl;
};

// diagonal * dense
template<typename Lhs, typename Rhs, int ProductKind, int ProductTag>
struct product_evaluator<Product<Lhs, Rhs, ProductKind>, ProductTag, DiagonalShape, DenseShape, typename Lhs::Scalar, typename Rhs::Scalar> 
  : diagonal_product_evaluator_base<Rhs, typename Lhs::DiagonalVectorType, Product<Lhs, Rhs, LazyProduct> >
{
  typedef diagonal_product_evaluator_base<Rhs, typename Lhs::DiagonalVectorType, Product<Lhs, Rhs, LazyProduct> > Base;
  using Base::m_diagImpl;
  using Base::m_matImpl;
  using Base::coeff;
  using Base::packet_impl;
  typedef typename Base::Scalar Scalar;
  typedef typename Base::Index Index;
  typedef typename Base::PacketScalar PacketScalar;
  
  typedef Product<Lhs, Rhs, ProductKind> XprType;
  typedef typename XprType::PlainObject PlainObject;
  
  enum { StorageOrder = int(Rhs::Flags) & RowMajorBit ? RowMajor : ColMajor };

  product_evaluator(const XprType& xpr)
    : Base(xpr.rhs(), xpr.lhs().diagonal())
  {
  }
  
  EIGEN_STRONG_INLINE const Scalar coeff(Index row, Index col) const
  {
    return m_diagImpl.coeff(row) * m_matImpl.coeff(row, col);
  }
  
  template<int LoadMode>
  EIGEN_STRONG_INLINE PacketScalar packet(Index row, Index col) const
  {
    return this->template packet_impl<LoadMode>(row,col, row,
                                 typename internal::conditional<int(StorageOrder)==RowMajor, internal::true_type, internal::false_type>::type());
  }
  
  template<int LoadMode>
  EIGEN_STRONG_INLINE PacketScalar packet(Index idx) const
  {
    return packet<LoadMode>(int(StorageOrder)==ColMajor?idx:0,int(StorageOrder)==ColMajor?0:idx);
  }
  
};

// dense * diagonal
template<typename Lhs, typename Rhs, int ProductKind, int ProductTag>
struct product_evaluator<Product<Lhs, Rhs, ProductKind>, ProductTag, DenseShape, DiagonalShape, typename Lhs::Scalar, typename Rhs::Scalar> 
  : diagonal_product_evaluator_base<Lhs, typename Rhs::DiagonalVectorType, Product<Lhs, Rhs, LazyProduct> >
{
  typedef diagonal_product_evaluator_base<Lhs, typename Rhs::DiagonalVectorType, Product<Lhs, Rhs, LazyProduct> > Base;
  using Base::m_diagImpl;
  using Base::m_matImpl;
  using Base::coeff;
  using Base::packet_impl;
  typedef typename Base::Scalar Scalar;
  typedef typename Base::Index Index;
  typedef typename Base::PacketScalar PacketScalar;
  
  typedef Product<Lhs, Rhs, ProductKind> XprType;
  typedef typename XprType::PlainObject PlainObject;
  
  enum { StorageOrder = int(Rhs::Flags) & RowMajorBit ? RowMajor : ColMajor };

  product_evaluator(const XprType& xpr)
    : Base(xpr.lhs(), xpr.rhs().diagonal())
  {
  }
  
  EIGEN_STRONG_INLINE const Scalar coeff(Index row, Index col) const
  {
    return m_matImpl.coeff(row, col) * m_diagImpl.coeff(col);
  }
  
  template<int LoadMode>
  EIGEN_STRONG_INLINE PacketScalar packet(Index row, Index col) const
  {
    return this->template packet_impl<LoadMode>(row,col, col,
                                 typename internal::conditional<int(StorageOrder)==ColMajor, internal::true_type, internal::false_type>::type());
  }
  
  template<int LoadMode>
  EIGEN_STRONG_INLINE PacketScalar packet(Index idx) const
  {
    return packet<LoadMode>(int(StorageOrder)==ColMajor?idx:0,int(StorageOrder)==ColMajor?0:idx);
  }
  
};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_PRODUCT_EVALUATORS_H
