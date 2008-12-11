// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_SPARSEPRODUCT_H
#define EIGEN_SPARSEPRODUCT_H

// sparse product return type specialization
template<typename Lhs, typename Rhs>
struct ProductReturnType<Lhs,Rhs,SparseProduct>
{
  typedef typename ei_traits<Lhs>::Scalar Scalar;
  enum {
    LhsRowMajor = ei_traits<Lhs>::Flags & RowMajorBit,
    RhsRowMajor = ei_traits<Rhs>::Flags & RowMajorBit,
    TransposeRhs = (!LhsRowMajor) && RhsRowMajor,
    TransposeLhs = LhsRowMajor && (!RhsRowMajor)
  };

  // FIXME if we transpose let's evaluate to a LinkedVectorMatrix since it is the
  // type of the temporary to perform the transpose op
  typedef typename ei_meta_if<TransposeLhs,
    SparseMatrix<Scalar,0>,
    const typename ei_nested<Lhs,Rhs::RowsAtCompileTime>::type>::ret LhsNested;

  typedef typename ei_meta_if<TransposeRhs,
    SparseMatrix<Scalar,0>,
    const typename ei_nested<Rhs,Lhs::RowsAtCompileTime>::type>::ret RhsNested;

  typedef Product<LhsNested,
                  RhsNested, SparseProduct> Type;
};

template<typename LhsNested, typename RhsNested>
struct ei_traits<Product<LhsNested, RhsNested, SparseProduct> >
{
  // clean the nested types:
  typedef typename ei_cleantype<LhsNested>::type _LhsNested;
  typedef typename ei_cleantype<RhsNested>::type _RhsNested;
  typedef typename _LhsNested::Scalar Scalar;

  enum {
    LhsCoeffReadCost = _LhsNested::CoeffReadCost,
    RhsCoeffReadCost = _RhsNested::CoeffReadCost,
    LhsFlags = _LhsNested::Flags,
    RhsFlags = _RhsNested::Flags,

    RowsAtCompileTime = _LhsNested::RowsAtCompileTime,
    ColsAtCompileTime = _RhsNested::ColsAtCompileTime,
    InnerSize = EIGEN_ENUM_MIN(_LhsNested::ColsAtCompileTime, _RhsNested::RowsAtCompileTime),

    MaxRowsAtCompileTime = _LhsNested::MaxRowsAtCompileTime,
    MaxColsAtCompileTime = _RhsNested::MaxColsAtCompileTime,

    LhsRowMajor = LhsFlags & RowMajorBit,
    RhsRowMajor = RhsFlags & RowMajorBit,

    EvalToRowMajor = (RhsFlags & LhsFlags & RowMajorBit),

    RemovedBits = ~(EvalToRowMajor ? 0 : RowMajorBit),

    Flags = (int(LhsFlags | RhsFlags) & HereditaryBits & RemovedBits)
          | EvalBeforeAssigningBit
          | EvalBeforeNestingBit,

    CoeffReadCost = Dynamic
  };
};

template<typename LhsNested, typename RhsNested> class Product<LhsNested,RhsNested,SparseProduct> : ei_no_assignment_operator,
  public MatrixBase<Product<LhsNested, RhsNested, SparseProduct> >
{
  public:

    EIGEN_GENERIC_PUBLIC_INTERFACE(Product)

  private:

    typedef typename ei_traits<Product>::_LhsNested _LhsNested;
    typedef typename ei_traits<Product>::_RhsNested _RhsNested;

  public:

    template<typename Lhs, typename Rhs>
    inline Product(const Lhs& lhs, const Rhs& rhs)
      : m_lhs(lhs), m_rhs(rhs)
    {
      ei_assert(lhs.cols() == rhs.rows());
    }

    Scalar coeff(int, int) const { ei_assert(false && "eigen internal error"); }
    Scalar& coeffRef(int, int) { ei_assert(false && "eigen internal error"); }

    inline int rows() const { return m_lhs.rows(); }
    inline int cols() const { return m_rhs.cols(); }

    const _LhsNested& lhs() const { return m_lhs; }
    const _LhsNested& rhs() const { return m_rhs; }

  protected:
    LhsNested m_lhs;
    RhsNested m_rhs;
};

template<typename Lhs, typename Rhs, typename ResultType,
  int LhsStorageOrder = ei_traits<Lhs>::Flags&RowMajorBit,
  int RhsStorageOrder = ei_traits<Rhs>::Flags&RowMajorBit,
  int ResStorageOrder = ei_traits<ResultType>::Flags&RowMajorBit>
struct ei_sparse_product_selector;

template<typename Lhs, typename Rhs, typename ResultType>
struct ei_sparse_product_selector<Lhs,Rhs,ResultType,ColMajor,ColMajor,ColMajor>
{
  typedef typename ei_traits<typename ei_cleantype<Lhs>::type>::Scalar Scalar;

  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res)
  {
    // make sure to call innerSize/outerSize since we fake the storage order.
    int rows = lhs.innerSize();
    int cols = rhs.outerSize();
    //int size = lhs.outerSize();
    ei_assert(lhs.outerSize() == rhs.innerSize());

    // allocate a temporary buffer
    AmbiVector<Scalar> tempVector(rows);

    // estimate the number of non zero entries
    float ratioLhs = float(lhs.nonZeros())/float(lhs.rows()*lhs.cols());
    float avgNnzPerRhsColumn = float(rhs.nonZeros())/float(cols);
    float ratioRes = std::min(ratioLhs * avgNnzPerRhsColumn, 1.f);

    res.resize(rows, cols);
    res.startFill(ratioRes*rows*cols);
    for (int j=0; j<cols; ++j)
    {
      // let's do a more accurate determination of the nnz ratio for the current column j of res
      //float ratioColRes = std::min(ratioLhs * rhs.innerNonZeros(j), 1.f);
      // FIXME find a nice way to get the number of nonzeros of a sub matrix (here an inner vector)
      float ratioColRes = ratioRes;
      tempVector.init(ratioColRes);
      tempVector.setZero();
      for (typename Rhs::InnerIterator rhsIt(rhs, j); rhsIt; ++rhsIt)
      {
        // FIXME should be written like this: tmp += rhsIt.value() * lhs.col(rhsIt.index())
        tempVector.restart();
        Scalar x = rhsIt.value();
        for (typename Lhs::InnerIterator lhsIt(lhs, rhsIt.index()); lhsIt; ++lhsIt)
        {
          tempVector.coeffRef(lhsIt.index()) += lhsIt.value() * x;
        }
      }
      for (typename AmbiVector<Scalar>::Iterator it(tempVector); it; ++it)
        if (ResultType::Flags&RowMajorBit)
          res.fill(j,it.index()) = it.value();
        else
          res.fill(it.index(), j) = it.value();
    }
    res.endFill();
  }
};

template<typename Lhs, typename Rhs, typename ResultType>
struct ei_sparse_product_selector<Lhs,Rhs,ResultType,ColMajor,ColMajor,RowMajor>
{
  typedef SparseMatrix<typename ResultType::Scalar> SparseTemporaryType;
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res)
  {
    SparseTemporaryType _res(res.rows(), res.cols());
    ei_sparse_product_selector<Lhs,Rhs,SparseTemporaryType,ColMajor,ColMajor,ColMajor>::run(lhs, rhs, _res);
    res = _res;
  }
};

template<typename Lhs, typename Rhs, typename ResultType>
struct ei_sparse_product_selector<Lhs,Rhs,ResultType,RowMajor,RowMajor,RowMajor>
{
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res)
  {
    // let's transpose the product to get a column x column product
    ei_sparse_product_selector<Rhs,Lhs,ResultType,ColMajor,ColMajor,ColMajor>::run(rhs, lhs, res);
  }
};

template<typename Lhs, typename Rhs, typename ResultType>
struct ei_sparse_product_selector<Lhs,Rhs,ResultType,RowMajor,RowMajor,ColMajor>
{
  typedef SparseMatrix<typename ResultType::Scalar> SparseTemporaryType;
  static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res)
  {
    // let's transpose the product to get a column x column product
    SparseTemporaryType _res(res.cols(), res.rows());
    ei_sparse_product_selector<Rhs,Lhs,ResultType,ColMajor,ColMajor,ColMajor>
      ::run(rhs, lhs, _res);
    res = _res.transpose();
  }
};

// NOTE eventually let's transpose one argument even in this case since it might be expensive if
// the result is not dense.
// template<typename Lhs, typename Rhs, typename ResultType, int ResStorageOrder>
// struct ei_sparse_product_selector<Lhs,Rhs,ResultType,RowMajor,ColMajor,ResStorageOrder>
// {
//   static void run(const Lhs& lhs, const Rhs& rhs, ResultType& res)
//   {
//     // trivial product as lhs.row/rhs.col dot products
//     // loop over the preferred order of the result
//   }
// };

// NOTE the 2 others cases (col row *) must never occurs since they are caught
// by ProductReturnType which transform it to (col col *) by evaluating rhs.


template<typename Derived>
template<typename Lhs, typename Rhs>
inline Derived& MatrixBase<Derived>::lazyAssign(const Product<Lhs,Rhs,SparseProduct>& product)
{
//   std::cout << "sparse product to dense\n";
  ei_sparse_product_selector<
    typename ei_cleantype<Lhs>::type,
    typename ei_cleantype<Rhs>::type,
    typename ei_cleantype<Derived>::type>::run(product.lhs(),product.rhs(),derived());
  return derived();
}

template<typename Derived>
template<typename Lhs, typename Rhs>
inline Derived& SparseMatrixBase<Derived>::operator=(const Product<Lhs,Rhs,SparseProduct>& product)
{
//   std::cout << "sparse product to sparse\n";
  ei_sparse_product_selector<
    typename ei_cleantype<Lhs>::type,
    typename ei_cleantype<Rhs>::type,
    Derived>::run(product.lhs(),product.rhs(),derived());
  return derived();
}

#endif // EIGEN_SPARSEPRODUCT_H
