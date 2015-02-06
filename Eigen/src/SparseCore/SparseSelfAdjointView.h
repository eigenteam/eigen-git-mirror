// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_SPARSE_SELFADJOINTVIEW_H
#define EIGEN_SPARSE_SELFADJOINTVIEW_H

namespace Eigen { 
  
/** \ingroup SparseCore_Module
  * \class SparseSelfAdjointView
  *
  * \brief Pseudo expression to manipulate a triangular sparse matrix as a selfadjoint matrix.
  *
  * \param MatrixType the type of the dense matrix storing the coefficients
  * \param Mode can be either \c #Lower or \c #Upper
  *
  * This class is an expression of a sefladjoint matrix from a triangular part of a matrix
  * with given dense storage of the coefficients. It is the return type of MatrixBase::selfadjointView()
  * and most of the time this is the only way that it is used.
  *
  * \sa SparseMatrixBase::selfadjointView()
  */
namespace internal {
  
template<typename MatrixType, unsigned int Mode>
struct traits<SparseSelfAdjointView<MatrixType,Mode> > : traits<MatrixType> {
};

template<int SrcMode,int DstMode,typename MatrixType,int DestOrder>
void permute_symm_to_symm(const MatrixType& mat, SparseMatrix<typename MatrixType::Scalar,DestOrder,typename MatrixType::Index>& _dest, const typename MatrixType::Index* perm = 0);

template<int Mode,typename MatrixType,int DestOrder>
void permute_symm_to_fullsymm(const MatrixType& mat, SparseMatrix<typename MatrixType::Scalar,DestOrder,typename MatrixType::Index>& _dest, const typename MatrixType::Index* perm = 0);

}

template<typename MatrixType, unsigned int _Mode> class SparseSelfAdjointView
  : public EigenBase<SparseSelfAdjointView<MatrixType,_Mode> >
{
  public:
    
    enum { Mode = _Mode };

    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Index Index;
    typedef Matrix<Index,Dynamic,1> VectorI;
    typedef typename MatrixType::Nested MatrixTypeNested;
    typedef typename internal::remove_all<MatrixTypeNested>::type _MatrixTypeNested;
    
    explicit inline SparseSelfAdjointView(const MatrixType& matrix) : m_matrix(matrix)
    {
      eigen_assert(rows()==cols() && "SelfAdjointView is only for squared matrices");
    }

    inline Index rows() const { return m_matrix.rows(); }
    inline Index cols() const { return m_matrix.cols(); }

    /** \internal \returns a reference to the nested matrix */
    const _MatrixTypeNested& matrix() const { return m_matrix; }
    _MatrixTypeNested& matrix() { return m_matrix.const_cast_derived(); }

    /** \returns an expression of the matrix product between a sparse self-adjoint matrix \c *this and a sparse matrix \a rhs.
      *
      * Note that there is no algorithmic advantage of performing such a product compared to a general sparse-sparse matrix product.
      * Indeed, the SparseSelfadjointView operand is first copied into a temporary SparseMatrix before computing the product.
      */
    template<typename OtherDerived>
    Product<SparseSelfAdjointView, OtherDerived>
    operator*(const SparseMatrixBase<OtherDerived>& rhs) const
    {
      return Product<SparseSelfAdjointView, OtherDerived>(*this, rhs.derived());
    }

    /** \returns an expression of the matrix product between a sparse matrix \a lhs and a sparse self-adjoint matrix \a rhs.
      *
      * Note that there is no algorithmic advantage of performing such a product compared to a general sparse-sparse matrix product.
      * Indeed, the SparseSelfadjointView operand is first copied into a temporary SparseMatrix before computing the product.
      */
    template<typename OtherDerived> friend
    Product<OtherDerived, SparseSelfAdjointView>
    operator*(const SparseMatrixBase<OtherDerived>& lhs, const SparseSelfAdjointView& rhs)
    {
      return Product<OtherDerived, SparseSelfAdjointView>(lhs.derived(), rhs);
    }
    
    /** Efficient sparse self-adjoint matrix times dense vector/matrix product */
    template<typename OtherDerived>
    Product<SparseSelfAdjointView,OtherDerived>
    operator*(const MatrixBase<OtherDerived>& rhs) const
    {
      return Product<SparseSelfAdjointView,OtherDerived>(*this, rhs.derived());
    }

    /** Efficient dense vector/matrix times sparse self-adjoint matrix product */
    template<typename OtherDerived> friend
    Product<OtherDerived,SparseSelfAdjointView>
    operator*(const MatrixBase<OtherDerived>& lhs, const SparseSelfAdjointView& rhs)
    {
      return Product<OtherDerived,SparseSelfAdjointView>(lhs.derived(), rhs);
    }

    /** Perform a symmetric rank K update of the selfadjoint matrix \c *this:
      * \f$ this = this + \alpha ( u u^* ) \f$ where \a u is a vector or matrix.
      *
      * \returns a reference to \c *this
      *
      * To perform \f$ this = this + \alpha ( u^* u ) \f$ you can simply
      * call this function with u.adjoint().
      */
    template<typename DerivedU>
    SparseSelfAdjointView& rankUpdate(const SparseMatrixBase<DerivedU>& u, const Scalar& alpha = Scalar(1));
    
    /** \internal triggered by sparse_matrix = SparseSelfadjointView; */
    template<typename DestScalar,int StorageOrder> void evalTo(SparseMatrix<DestScalar,StorageOrder,Index>& _dest) const
    {
      internal::permute_symm_to_fullsymm<Mode>(m_matrix, _dest);
    }
    
    template<typename DestScalar> void evalTo(DynamicSparseMatrix<DestScalar,ColMajor,Index>& _dest) const
    {
      // TODO directly evaluate into _dest;
      SparseMatrix<DestScalar,ColMajor,Index> tmp(_dest.rows(),_dest.cols());
      internal::permute_symm_to_fullsymm<Mode>(m_matrix, tmp);
      _dest = tmp;
    }
    
    /** \returns an expression of P H P^-1 */
    // TODO implement twists in a more evaluator friendly fashion
    SparseSymmetricPermutationProduct<_MatrixTypeNested,Mode> twistedBy(const PermutationMatrix<Dynamic,Dynamic,Index>& perm) const
    {
      return SparseSymmetricPermutationProduct<_MatrixTypeNested,Mode>(m_matrix, perm);
    }

    template<typename SrcMatrixType,int SrcMode>
    SparseSelfAdjointView& operator=(const SparseSymmetricPermutationProduct<SrcMatrixType,SrcMode>& permutedMatrix)
    {
      permutedMatrix.evalTo(*this);
      return *this;
    }

    SparseSelfAdjointView& operator=(const SparseSelfAdjointView& src)
    {
      PermutationMatrix<Dynamic> pnull;
      return *this = src.twistedBy(pnull);
    }

    template<typename SrcMatrixType,unsigned int SrcMode>
    SparseSelfAdjointView& operator=(const SparseSelfAdjointView<SrcMatrixType,SrcMode>& src)
    {
      PermutationMatrix<Dynamic> pnull;
      return *this = src.twistedBy(pnull);
    }
    
  protected:

    typename MatrixType::Nested m_matrix;
    //mutable VectorI m_countPerRow;
    //mutable VectorI m_countPerCol;
};

/***************************************************************************
* Implementation of SparseMatrixBase methods
***************************************************************************/

template<typename Derived>
template<unsigned int Mode>
const SparseSelfAdjointView<const Derived, Mode> SparseMatrixBase<Derived>::selfadjointView() const
{
  return SparseSelfAdjointView<const Derived, Mode>(derived());
}

template<typename Derived>
template<unsigned int Mode>
SparseSelfAdjointView<Derived, Mode> SparseMatrixBase<Derived>::selfadjointView()
{
  return SparseSelfAdjointView<Derived, Mode>(derived());
}

/***************************************************************************
* Implementation of SparseSelfAdjointView methods
***************************************************************************/

template<typename MatrixType, unsigned int Mode>
template<typename DerivedU>
SparseSelfAdjointView<MatrixType,Mode>&
SparseSelfAdjointView<MatrixType,Mode>::rankUpdate(const SparseMatrixBase<DerivedU>& u, const Scalar& alpha)
{
  SparseMatrix<Scalar,(MatrixType::Flags&RowMajorBit)?RowMajor:ColMajor> tmp = u * u.adjoint();
  if(alpha==Scalar(0))
    m_matrix.const_cast_derived() = tmp.template triangularView<Mode>();
  else
    m_matrix.const_cast_derived() += alpha * tmp.template triangularView<Mode>();

  return *this;
}

/***************************************************************************
* Implementation of sparse self-adjoint time dense matrix
***************************************************************************/

namespace internal {

template<int Mode, typename SparseLhsType, typename DenseRhsType, typename DenseResType, typename AlphaType>
inline void sparse_selfadjoint_time_dense_product(const SparseLhsType& lhs, const DenseRhsType& rhs, DenseResType& res, const AlphaType& alpha)
{
  EIGEN_ONLY_USED_FOR_DEBUG(alpha);
  // TODO use alpha
  eigen_assert(alpha==AlphaType(1) && "alpha != 1 is not implemented yet, sorry");
  
  typedef typename evaluator<SparseLhsType>::type LhsEval;
  typedef typename evaluator<SparseLhsType>::InnerIterator LhsIterator;
  typedef typename SparseLhsType::Index Index;
  typedef typename SparseLhsType::Scalar LhsScalar;
  
  enum {
    LhsIsRowMajor = (LhsEval::Flags&RowMajorBit)==RowMajorBit,
    ProcessFirstHalf =
              ((Mode&(Upper|Lower))==(Upper|Lower))
          || ( (Mode&Upper) && !LhsIsRowMajor)
          || ( (Mode&Lower) && LhsIsRowMajor),
    ProcessSecondHalf = !ProcessFirstHalf
  };
  
  LhsEval lhsEval(lhs);
  
  for (Index j=0; j<lhs.outerSize(); ++j)
  {
    LhsIterator i(lhsEval,j);
    if (ProcessSecondHalf)
    {
      while (i && i.index()<j) ++i;
      if(i && i.index()==j)
      {
        res.row(j) += i.value() * rhs.row(j);
        ++i;
      }
    }
    for(; (ProcessFirstHalf ? i && i.index() < j : i) ; ++i)
    {
      Index a = LhsIsRowMajor ? j : i.index();
      Index b = LhsIsRowMajor ? i.index() : j;
      LhsScalar v = i.value();
      res.row(a) += (v) * rhs.row(b);
      res.row(b) += numext::conj(v) * rhs.row(a);
    }
    if (ProcessFirstHalf && i && (i.index()==j))
      res.row(j) += i.value() * rhs.row(j);
  }
}
  
// TODO currently a selfadjoint expression has the form SelfAdjointView<.,.>
//      in the future selfadjoint-ness should be defined by the expression traits
//      such that Transpose<SelfAdjointView<.,.> > is valid. (currently TriangularBase::transpose() is overloaded to make it work)
template<typename MatrixType, unsigned int Mode>
struct evaluator_traits<SparseSelfAdjointView<MatrixType,Mode> >
{
  typedef typename storage_kind_to_evaluator_kind<typename MatrixType::StorageKind>::Kind Kind;
  typedef SparseSelfAdjointShape Shape;
  
  static const int AssumeAliasing = 0;
};

template<typename LhsView, typename Rhs, int ProductType>
struct generic_product_impl<LhsView, Rhs, SparseSelfAdjointShape, DenseShape, ProductType>
{
  template<typename Dest>
  static void evalTo(Dest& dst, const LhsView& lhsView, const Rhs& rhs)
  {
    typedef typename LhsView::_MatrixTypeNested Lhs;
    typedef typename nested_eval<Lhs,Dynamic>::type LhsNested;
    typedef typename nested_eval<Rhs,Dynamic>::type RhsNested;
    LhsNested lhsNested(lhsView.matrix());
    RhsNested rhsNested(rhs);
    
    dst.setZero();
    internal::sparse_selfadjoint_time_dense_product<LhsView::Mode>(lhsNested, rhsNested, dst, typename Dest::Scalar(1));
  }
};

template<typename Lhs, typename RhsView, int ProductType>
struct generic_product_impl<Lhs, RhsView, DenseShape, SparseSelfAdjointShape, ProductType>
{
  template<typename Dest>
  static void evalTo(Dest& dst, const Lhs& lhs, const RhsView& rhsView)
  {
    typedef typename RhsView::_MatrixTypeNested Rhs;
    typedef typename nested_eval<Lhs,Dynamic>::type LhsNested;
    typedef typename nested_eval<Rhs,Dynamic>::type RhsNested;
    LhsNested lhsNested(lhs);
    RhsNested rhsNested(rhsView.matrix());
    
    dst.setZero();
    // transpoe everything
    Transpose<Dest> dstT(dst);
    internal::sparse_selfadjoint_time_dense_product<RhsView::Mode>(rhsNested.transpose(), lhsNested.transpose(), dstT, typename Dest::Scalar(1));
  }
};

// NOTE: these two overloads are needed to evaluate the sparse sefladjoint view into a full sparse matrix
// TODO: maybe the copy could be handled by generic_product_impl so that these overloads would not be needed anymore

template<typename LhsView, typename Rhs, int ProductTag>
struct product_evaluator<Product<LhsView, Rhs, DefaultProduct>, ProductTag, SparseSelfAdjointShape, SparseShape, typename traits<LhsView>::Scalar, typename traits<Rhs>::Scalar>
  : public evaluator<typename Product<typename Rhs::PlainObject, Rhs, DefaultProduct>::PlainObject>::type
{
  typedef Product<LhsView, Rhs, DefaultProduct> XprType;
  typedef typename XprType::PlainObject PlainObject;
  typedef typename evaluator<PlainObject>::type Base;

  product_evaluator(const XprType& xpr)
    : m_lhs(xpr.lhs()), m_result(xpr.rows(), xpr.cols())
  {
    ::new (static_cast<Base*>(this)) Base(m_result);
    generic_product_impl<typename Rhs::PlainObject, Rhs, SparseShape, SparseShape, ProductTag>::evalTo(m_result, m_lhs, xpr.rhs());
  }
  
protected:
  typename Rhs::PlainObject m_lhs;
  PlainObject m_result;
};

template<typename Lhs, typename RhsView, int ProductTag>
struct product_evaluator<Product<Lhs, RhsView, DefaultProduct>, ProductTag, SparseShape, SparseSelfAdjointShape, typename traits<Lhs>::Scalar, typename traits<RhsView>::Scalar>
  : public evaluator<typename Product<Lhs, typename Lhs::PlainObject, DefaultProduct>::PlainObject>::type
{
  typedef Product<Lhs, RhsView, DefaultProduct> XprType;
  typedef typename XprType::PlainObject PlainObject;
  typedef typename evaluator<PlainObject>::type Base;

  product_evaluator(const XprType& xpr)
    : m_rhs(xpr.rhs()), m_result(xpr.rows(), xpr.cols())
  {
    ::new (static_cast<Base*>(this)) Base(m_result);
    generic_product_impl<Lhs, typename Lhs::PlainObject, SparseShape, SparseShape, ProductTag>::evalTo(m_result, xpr.lhs(), m_rhs);
  }
  
protected:
  typename Lhs::PlainObject m_rhs;
  PlainObject m_result;
};

} // namespace internal

/***************************************************************************
* Implementation of symmetric copies and permutations
***************************************************************************/
namespace internal {

template<int Mode,typename MatrixType,int DestOrder>
void permute_symm_to_fullsymm(const MatrixType& mat, SparseMatrix<typename MatrixType::Scalar,DestOrder,typename MatrixType::Index>& _dest, const typename MatrixType::Index* perm)
{
  typedef typename MatrixType::Index Index;
  typedef typename MatrixType::Scalar Scalar;
  typedef SparseMatrix<Scalar,DestOrder,Index> Dest;
  typedef Matrix<Index,Dynamic,1> VectorI;
  
  Dest& dest(_dest.derived());
  enum {
    StorageOrderMatch = int(Dest::IsRowMajor) == int(MatrixType::IsRowMajor)
  };
  
  Index size = mat.rows();
  VectorI count;
  count.resize(size);
  count.setZero();
  dest.resize(size,size);
  for(Index j = 0; j<size; ++j)
  {
    Index jp = perm ? perm[j] : j;
    for(typename MatrixType::InnerIterator it(mat,j); it; ++it)
    {
      Index i = it.index();
      Index r = it.row();
      Index c = it.col();
      Index ip = perm ? perm[i] : i;
      if(Mode==(Upper|Lower))
        count[StorageOrderMatch ? jp : ip]++;
      else if(r==c)
        count[ip]++;
      else if(( Mode==Lower && r>c) || ( Mode==Upper && r<c))
      {
        count[ip]++;
        count[jp]++;
      }
    }
  }
  Index nnz = count.sum();
  
  // reserve space
  dest.resizeNonZeros(nnz);
  dest.outerIndexPtr()[0] = 0;
  for(Index j=0; j<size; ++j)
    dest.outerIndexPtr()[j+1] = dest.outerIndexPtr()[j] + count[j];
  for(Index j=0; j<size; ++j)
    count[j] = dest.outerIndexPtr()[j];
  
  // copy data
  for(Index j = 0; j<size; ++j)
  {
    for(typename MatrixType::InnerIterator it(mat,j); it; ++it)
    {
      Index i = it.index();
      Index r = it.row();
      Index c = it.col();
      
      Index jp = perm ? perm[j] : j;
      Index ip = perm ? perm[i] : i;
      
      if(Mode==(Upper|Lower))
      {
        Index k = count[StorageOrderMatch ? jp : ip]++;
        dest.innerIndexPtr()[k] = StorageOrderMatch ? ip : jp;
        dest.valuePtr()[k] = it.value();
      }
      else if(r==c)
      {
        Index k = count[ip]++;
        dest.innerIndexPtr()[k] = ip;
        dest.valuePtr()[k] = it.value();
      }
      else if(( (Mode&Lower)==Lower && r>c) || ( (Mode&Upper)==Upper && r<c))
      {
        if(!StorageOrderMatch)
          std::swap(ip,jp);
        Index k = count[jp]++;
        dest.innerIndexPtr()[k] = ip;
        dest.valuePtr()[k] = it.value();
        k = count[ip]++;
        dest.innerIndexPtr()[k] = jp;
        dest.valuePtr()[k] = numext::conj(it.value());
      }
    }
  }
}

template<int _SrcMode,int _DstMode,typename MatrixType,int DstOrder>
void permute_symm_to_symm(const MatrixType& mat, SparseMatrix<typename MatrixType::Scalar,DstOrder,typename MatrixType::Index>& _dest, const typename MatrixType::Index* perm)
{
  typedef typename MatrixType::Index Index;
  typedef typename MatrixType::Scalar Scalar;
  SparseMatrix<Scalar,DstOrder,Index>& dest(_dest.derived());
  typedef Matrix<Index,Dynamic,1> VectorI;
  enum {
    SrcOrder = MatrixType::IsRowMajor ? RowMajor : ColMajor,
    StorageOrderMatch = int(SrcOrder) == int(DstOrder),
    DstMode = DstOrder==RowMajor ? (_DstMode==Upper ? Lower : Upper) : _DstMode,
    SrcMode = SrcOrder==RowMajor ? (_SrcMode==Upper ? Lower : Upper) : _SrcMode
  };
  
  Index size = mat.rows();
  VectorI count(size);
  count.setZero();
  dest.resize(size,size);
  for(Index j = 0; j<size; ++j)
  {
    Index jp = perm ? perm[j] : j;
    for(typename MatrixType::InnerIterator it(mat,j); it; ++it)
    {
      Index i = it.index();
      if((int(SrcMode)==int(Lower) && i<j) || (int(SrcMode)==int(Upper) && i>j))
        continue;
                  
      Index ip = perm ? perm[i] : i;
      count[int(DstMode)==int(Lower) ? (std::min)(ip,jp) : (std::max)(ip,jp)]++;
    }
  }
  dest.outerIndexPtr()[0] = 0;
  for(Index j=0; j<size; ++j)
    dest.outerIndexPtr()[j+1] = dest.outerIndexPtr()[j] + count[j];
  dest.resizeNonZeros(dest.outerIndexPtr()[size]);
  for(Index j=0; j<size; ++j)
    count[j] = dest.outerIndexPtr()[j];
  
  for(Index j = 0; j<size; ++j)
  {
    
    for(typename MatrixType::InnerIterator it(mat,j); it; ++it)
    {
      Index i = it.index();
      if((int(SrcMode)==int(Lower) && i<j) || (int(SrcMode)==int(Upper) && i>j))
        continue;
                  
      Index jp = perm ? perm[j] : j;
      Index ip = perm? perm[i] : i;
      
      Index k = count[int(DstMode)==int(Lower) ? (std::min)(ip,jp) : (std::max)(ip,jp)]++;
      dest.innerIndexPtr()[k] = int(DstMode)==int(Lower) ? (std::max)(ip,jp) : (std::min)(ip,jp);
      
      if(!StorageOrderMatch) std::swap(ip,jp);
      if( ((int(DstMode)==int(Lower) && ip<jp) || (int(DstMode)==int(Upper) && ip>jp)))
        dest.valuePtr()[k] = numext::conj(it.value());
      else
        dest.valuePtr()[k] = it.value();
    }
  }
}

}

// TODO implement twists in a more evaluator friendly fashion

namespace internal {

template<typename MatrixType, int Mode>
struct traits<SparseSymmetricPermutationProduct<MatrixType,Mode> > : traits<MatrixType> {
};

}

template<typename MatrixType,int Mode>
class SparseSymmetricPermutationProduct
  : public EigenBase<SparseSymmetricPermutationProduct<MatrixType,Mode> >
{
  public:
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Index Index;
  protected:
    typedef PermutationMatrix<Dynamic,Dynamic,Index> Perm;
  public:
    typedef Matrix<Index,Dynamic,1> VectorI;
    typedef typename MatrixType::Nested MatrixTypeNested;
    typedef typename internal::remove_all<MatrixTypeNested>::type _MatrixTypeNested;
    
    SparseSymmetricPermutationProduct(const MatrixType& mat, const Perm& perm)
      : m_matrix(mat), m_perm(perm)
    {}
    
    inline Index rows() const { return m_matrix.rows(); }
    inline Index cols() const { return m_matrix.cols(); }
    
    template<typename DestScalar, int Options, typename DstIndex>
    void evalTo(SparseMatrix<DestScalar,Options,DstIndex>& _dest) const
    {
//       internal::permute_symm_to_fullsymm<Mode>(m_matrix,_dest,m_perm.indices().data());
      SparseMatrix<DestScalar,(Options&RowMajor)==RowMajor ? ColMajor : RowMajor, DstIndex> tmp;
      internal::permute_symm_to_fullsymm<Mode>(m_matrix,tmp,m_perm.indices().data());
      _dest = tmp;
    }
    
    template<typename DestType,unsigned int DestMode> void evalTo(SparseSelfAdjointView<DestType,DestMode>& dest) const
    {
      internal::permute_symm_to_symm<Mode,DestMode>(m_matrix,dest.matrix(),m_perm.indices().data());
    }
    
  protected:
    MatrixTypeNested m_matrix;
    const Perm& m_perm;

};

} // end namespace Eigen

#endif // EIGEN_SPARSE_SELFADJOINTVIEW_H
