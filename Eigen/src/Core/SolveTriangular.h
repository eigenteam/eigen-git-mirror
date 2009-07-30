// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_SOLVETRIANGULAR_H
#define EIGEN_SOLVETRIANGULAR_H

template<typename Lhs, typename Rhs,
  int Mode, // can be Upper/Lower | UnitDiag
  int Side, // can be OnTheLeft/OnTheRight
  int Unrolling = Rhs::IsVectorAtCompileTime && Rhs::SizeAtCompileTime <= 8 // FIXME
                ? CompleteUnrolling : NoUnrolling,
  int StorageOrder = int(Lhs::Flags) & RowMajorBit,
  int RhsCols = Rhs::ColsAtCompileTime
  >
struct ei_triangular_solver_selector;

// forward and backward substitution, row-major, rhs is a vector
template<typename Lhs, typename Rhs, int Mode>
struct ei_triangular_solver_selector<Lhs,Rhs,OnTheLeft,Mode,NoUnrolling,RowMajor,1>
{
  typedef typename Rhs::Scalar Scalar;
  typedef ei_blas_traits<Lhs> LhsProductTraits;
  typedef typename LhsProductTraits::ExtractType ActualLhsType;
  enum {
    IsLowerTriangular = ((Mode&LowerTriangularBit)==LowerTriangularBit)
  };
  static void run(const Lhs& lhs, Rhs& other)
  {
    static const int PanelWidth = EIGEN_TUNE_TRIANGULAR_PANEL_WIDTH;
    ActualLhsType actualLhs = LhsProductTraits::extract(lhs);
    
    const int size = lhs.cols();
    for(int pi=IsLowerTriangular ? 0 : size;
        IsLowerTriangular ? pi<size : pi>0;
        IsLowerTriangular ? pi+=PanelWidth : pi-=PanelWidth)
    {
      int actualPanelWidth = std::min(IsLowerTriangular ? size - pi : pi, PanelWidth);

      int r = IsLowerTriangular ? pi : size - pi; // remaining size
      if (r > 0)
      {
        // let's directly call the low level product function because:
        // 1 - it is faster to compile
        // 2 - it is slighlty faster at runtime
        int startRow = IsLowerTriangular ? pi : pi-actualPanelWidth;
        int startCol = IsLowerTriangular ? 0 : pi;
        VectorBlock<Rhs,Dynamic> target(other,startRow,actualPanelWidth);

        ei_cache_friendly_product_rowmajor_times_vector<LhsProductTraits::NeedToConjugate,false>(
          &(actualLhs.const_cast_derived().coeffRef(startRow,startCol)), actualLhs.stride(),
          &(other.coeffRef(startCol)), r,
          target, Scalar(-1));
      }

      for(int k=0; k<actualPanelWidth; ++k)
      {
        int i = IsLowerTriangular ? pi+k : pi-k-1;
        int s = IsLowerTriangular ? pi : i+1;
        if (k>0)
          other.coeffRef(i) -= ((lhs.row(i).segment(s,k).transpose())
                              .cwise()*(other.segment(s,k))).sum();

        if(!(Mode & UnitDiagBit))
          other.coeffRef(i) /= lhs.coeff(i,i);
      }
    }
  }
};

// forward and backward substitution, column-major, rhs is a vector
template<typename Lhs, typename Rhs, int Mode>
struct ei_triangular_solver_selector<Lhs,Rhs,OnTheLeft,Mode,NoUnrolling,ColMajor,1>
{
  typedef typename Rhs::Scalar Scalar;
  typedef typename ei_packet_traits<Scalar>::type Packet;
  typedef ei_blas_traits<Lhs> LhsProductTraits;
  typedef typename LhsProductTraits::ExtractType ActualLhsType;
  enum {
    PacketSize =  ei_packet_traits<Scalar>::size,
    IsLowerTriangular = ((Mode&LowerTriangularBit)==LowerTriangularBit)
  };

  static void run(const Lhs& lhs, Rhs& other)
  {
    static const int PanelWidth = EIGEN_TUNE_TRIANGULAR_PANEL_WIDTH;
    ActualLhsType actualLhs = LhsProductTraits::extract(lhs);

    const int size = lhs.cols();
    for(int pi=IsLowerTriangular ? 0 : size;
        IsLowerTriangular ? pi<size : pi>0;
        IsLowerTriangular ? pi+=PanelWidth : pi-=PanelWidth)
    {
      int actualPanelWidth = std::min(IsLowerTriangular ? size - pi : pi, PanelWidth);
      int startBlock = IsLowerTriangular ? pi : pi-actualPanelWidth;
      int endBlock = IsLowerTriangular ? pi + actualPanelWidth : 0;

      for(int k=0; k<actualPanelWidth; ++k)
      {
        int i = IsLowerTriangular ? pi+k : pi-k-1;
        if(!(Mode & UnitDiagBit))
          other.coeffRef(i) /= lhs.coeff(i,i);

        int r = actualPanelWidth - k - 1; // remaining size
        int s = IsLowerTriangular ? i+1 : i-r;
        if (r>0)
          other.segment(s,r) -= other.coeffRef(i) * Block<Lhs,Dynamic,1>(lhs, s, i, r, 1);
      }
      int r = IsLowerTriangular ? size - endBlock : startBlock; // remaining size
      if (r > 0)
      {
        // let's directly call the low level product function because:
        // 1 - it is faster to compile
        // 2 - it is slighlty faster at runtime
        ei_cache_friendly_product_colmajor_times_vector<LhsProductTraits::NeedToConjugate,false>(
          r,
          &(actualLhs.const_cast_derived().coeffRef(endBlock,startBlock)), actualLhs.stride(),
          other.segment(startBlock, actualPanelWidth),
          &(other.coeffRef(endBlock, 0)),
          Scalar(-1));
      }
    }
  }
};

template <typename Scalar, int Side, int Mode, bool Conjugate, int TriStorageOrder, int OtherStorageOrder>
struct ei_triangular_solve_matrix;

// the rhs is a matrix
template<typename Lhs, typename Rhs, int Side, int Mode, int StorageOrder, int RhsCols>
struct ei_triangular_solver_selector<Lhs,Rhs,Side,Mode,NoUnrolling,StorageOrder,RhsCols>
{
  typedef typename Rhs::Scalar Scalar;
  typedef ei_blas_traits<Lhs> LhsProductTraits;
  typedef typename LhsProductTraits::DirectLinearAccessType ActualLhsType;
  static void run(const Lhs& lhs, Rhs& rhs)
  {
    const ActualLhsType actualLhs = LhsProductTraits::extract(lhs);
    ei_triangular_solve_matrix<Scalar,Side,Mode,LhsProductTraits::NeedToConjugate,StorageOrder,
                               Rhs::Flags&RowMajorBit>
      ::run(lhs.rows(), rhs.cols(), &actualLhs.coeff(0,0), actualLhs.stride(), &rhs.coeffRef(0,0), rhs.stride());
  }
};

/***************************************************************************
* meta-unrolling implementation
***************************************************************************/

template<typename Lhs, typename Rhs, int Mode, int Index, int Size,
         bool Stop = Index==Size>
struct ei_triangular_solver_unroller;

template<typename Lhs, typename Rhs, int Mode, int Index, int Size>
struct ei_triangular_solver_unroller<Lhs,Rhs,Mode,Index,Size,false> {
  enum {
    IsLowerTriangular = ((Mode&LowerTriangularBit)==LowerTriangularBit),
    I = IsLowerTriangular ? Index : Size - Index - 1,
    S = IsLowerTriangular ? 0     : I+1
  };
  static void run(const Lhs& lhs, Rhs& rhs)
  {
    if (Index>0)
      rhs.coeffRef(I) -=      ((lhs.row(I).template segment<Index>(S).transpose())
                      .cwise()*(rhs.template segment<Index>(S))).sum();

    if(!(Mode & UnitDiagBit))
      rhs.coeffRef(I) /= lhs.coeff(I,I);

    ei_triangular_solver_unroller<Lhs,Rhs,Mode,Index+1,Size>::run(lhs,rhs);
  }
};

template<typename Lhs, typename Rhs, int Mode, int Index, int Size>
struct ei_triangular_solver_unroller<Lhs,Rhs,Mode,Index,Size,true> {
  static void run(const Lhs&, Rhs&) {}
};

template<typename Lhs, typename Rhs, int Mode, int StorageOrder>
struct ei_triangular_solver_selector<Lhs,Rhs,OnTheLeft,Mode,CompleteUnrolling,StorageOrder,1> {
  static void run(const Lhs& lhs, Rhs& rhs)
  { ei_triangular_solver_unroller<Lhs,Rhs,Mode,0,Rhs::SizeAtCompileTime>::run(lhs,rhs); }
};

/***************************************************************************
* TriangularView methods
***************************************************************************/

/** "in-place" version of MatrixBase::solveTriangular() where the result is written in \a other
  *
  * \nonstableyet
  *
  * \warning The parameter is only marked 'const' to make the C++ compiler accept a temporary expression here.
  * This function will const_cast it, so constness isn't honored here.
  *
  * See TriangularView:solve() for the details.
  */
template<typename MatrixType, unsigned int Mode>
template<int Side, typename RhsDerived>
void TriangularView<MatrixType,Mode>::solveInPlace(const MatrixBase<RhsDerived>& _rhs) const
{
  RhsDerived& rhs = _rhs.const_cast_derived();
  ei_assert(cols() == rows());
  ei_assert(cols() == rhs.rows());
  ei_assert(!(Mode & ZeroDiagBit));
  ei_assert(Mode & (UpperTriangularBit|LowerTriangularBit));

  enum { copy = ei_traits<RhsDerived>::Flags & RowMajorBit };
  typedef typename ei_meta_if<copy,
    typename ei_plain_matrix_type_column_major<RhsDerived>::type, RhsDerived&>::ret RhsCopy;
  RhsCopy rhsCopy(rhs);

  ei_triangular_solver_selector<MatrixType, typename ei_unref<RhsCopy>::type,
    Side, Mode>::run(_expression(), rhsCopy);

  if (copy)
    rhs = rhsCopy;
}

/** \returns the product of the inverse of \c *this with \a other, \a *this being triangular.
  *
  * \nonstableyet
  *
  * This function computes the inverse-matrix matrix product inverse(\c *this) * \a other.
  * The matrix \c *this must be triangular and invertible (i.e., all the coefficients of the
  * diagonal must be non zero). It works as a forward (resp. backward) substitution if \c *this
  * is an upper (resp. lower) triangular matrix.
  *
  * It is required that \c *this be marked as either an upper or a lower triangular matrix, which
  * can be done by marked(), and that is automatically the case with expressions such as those returned
  * by extract().
  *
  * Example: \include MatrixBase_marked.cpp
  * Output: \verbinclude MatrixBase_marked.out
  *
  * This function is essentially a wrapper to the faster solveTriangularInPlace() function creating
  * a temporary copy of \a other, calling solveTriangularInPlace() on the copy and returning it.
  * Therefore, if \a other is not needed anymore, it is quite faster to call solveTriangularInPlace()
  * instead of solveTriangular().
  *
  * For users coming from BLAS, this function (and more specifically solveTriangularInPlace()) offer
  * all the operations supported by the \c *TRSV and \c *TRSM BLAS routines.
  *
  * \b Tips: to perform a \em "right-inverse-multiply" you can simply transpose the operation, e.g.:
  * \code
  * M * T^1  <=>  T.transpose().solveInPlace(M.transpose());
  * \endcode
  *
  * \sa TriangularView::solveInPlace()
  */
template<typename Derived, unsigned int Mode>
template<int Side, typename RhsDerived>
typename ei_plain_matrix_type_column_major<RhsDerived>::type
TriangularView<Derived,Mode>::solve(const MatrixBase<RhsDerived>& rhs) const
{
  typename ei_plain_matrix_type_column_major<RhsDerived>::type res(rhs);
  solveInPlace<Side>(res);
  return res;
}

#endif // EIGEN_SOLVETRIANGULAR_H
