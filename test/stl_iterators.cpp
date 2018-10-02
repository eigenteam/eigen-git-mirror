// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <numeric>
#include "main.h"

template< class Iterator >
std::reverse_iterator<Iterator>
make_reverse_iterator( Iterator i )
{
  return std::reverse_iterator<Iterator>(i);
}

template<typename XprType>
bool is_PointerBasedStlIterator(const PointerBasedStlIterator<XprType> &) { return true; }

template<typename XprType>
bool is_DenseStlIterator(const DenseStlIterator<XprType> &) { return true; }

template<typename Scalar, int Rows, int Cols>
void test_range_for_loop(int rows=Rows, int cols=Cols)
{
  using std::begin;
  using std::end;

  typedef Matrix<Scalar,Rows,1> VectorType;
  typedef Matrix<Scalar,Rows,Cols,ColMajor> ColMatrixType;
  typedef Matrix<Scalar,Rows,Cols,RowMajor> RowMatrixType;
  VectorType v = VectorType::Random(rows);
  const VectorType& cv(v);
  ColMatrixType A = ColMatrixType::Random(rows,cols);
  const ColMatrixType& cA(A);
  RowMatrixType B = RowMatrixType::Random(rows,cols);
  
  Index i, j;

  VERIFY( is_PointerBasedStlIterator(v.begin()) );
  VERIFY( is_PointerBasedStlIterator(v.end()) );
  VERIFY( is_PointerBasedStlIterator(cv.begin()) );
  VERIFY( is_PointerBasedStlIterator(cv.end()) );

  j = internal::random<Index>(0,A.cols()-1);
  VERIFY( is_PointerBasedStlIterator(A.col(j).begin()) );
  VERIFY( is_PointerBasedStlIterator(A.col(j).end()) );
  VERIFY( is_PointerBasedStlIterator(cA.col(j).begin()) );
  VERIFY( is_PointerBasedStlIterator(cA.col(j).end()) );

  i = internal::random<Index>(0,A.rows()-1);
  VERIFY( is_PointerBasedStlIterator(A.row(i).begin()) );
  VERIFY( is_PointerBasedStlIterator(A.row(i).end()) );
  VERIFY( is_PointerBasedStlIterator(cA.row(i).begin()) );
  VERIFY( is_PointerBasedStlIterator(cA.row(i).end()) );

  VERIFY( is_PointerBasedStlIterator(A.reshaped().begin()) );
  VERIFY( is_PointerBasedStlIterator(A.reshaped().end()) );
  VERIFY( is_PointerBasedStlIterator(cA.reshaped().begin()) );
  VERIFY( is_PointerBasedStlIterator(cA.reshaped().end()) );

  VERIFY( is_DenseStlIterator(A.template reshaped<RowMajor>().begin()) );
  VERIFY( is_DenseStlIterator(A.template reshaped<RowMajor>().end()) );
  
#if EIGEN_HAS_CXX11
  i = 0;
  for(auto x : v) { VERIFY_IS_EQUAL(x,v[i++]); }

  j = internal::random<Index>(0,A.cols()-1);
  i = 0;
  for(auto x : A.col(j)) { VERIFY_IS_EQUAL(x,A(i++,j)); }

  i = 0;
  for(auto x : (v+A.col(j))) { VERIFY_IS_APPROX(x,v(i)+A(i,j)); ++i; }

  j = 0;
  i = internal::random<Index>(0,A.rows()-1);
  for(auto x : A.row(i)) { VERIFY_IS_EQUAL(x,A(i,j++)); }

  i = 0;
  for(auto x : A.reshaped()) { VERIFY_IS_EQUAL(x,A(i++)); }

  // check const_iterator
  {
    i = 0;
    for(auto x : cv) { VERIFY_IS_EQUAL(x,v[i++]); }

    i = 0;
    for(auto x : cA.reshaped()) { VERIFY_IS_EQUAL(x,A(i++)); }

    j = 0;
    i = internal::random<Index>(0,A.rows()-1);
    for(auto x : cA.row(i)) { VERIFY_IS_EQUAL(x,A(i,j++)); }
  }

  Matrix<Scalar,Dynamic,Dynamic,ColMajor> Bc = B;
  i = 0;
  for(auto x : B.reshaped()) { VERIFY_IS_EQUAL(x,Bc(i++)); }

  VectorType w(v.size());
  i = 0;
  for(auto& x : w) { x = v(i++); }
  VERIFY_IS_EQUAL(v,w);

  {
    j = internal::random<Index>(0,A.cols()-1);
    auto it = A.col(j).begin();
    for(i=0;i<rows;++i) {
      VERIFY_IS_EQUAL(it[i],A(i,j));
    }
  }

  {
    i = internal::random<Index>(0,A.rows()-1);
    auto it = A.row(i).begin();
    for(j=0;j<cols;++j) { VERIFY_IS_EQUAL(it[j],A(i,j)); }
  }

  {
    j = internal::random<Index>(0,A.cols()-1);
    // this would produce a dangling pointer:
    // auto it = (A+2*A).col(j).begin(); 
    // we need to name the temporary expression:
    auto tmp = (A+2*A).col(j);
    auto it = tmp.begin();
    for(i=0;i<rows;++i) {
      VERIFY_IS_APPROX(it[i],3*A(i,j));
    }
  }

  
  // {
  //   j = internal::random<Index>(0,A.cols()-1);
  //   auto it = (A+2*A).col(j).begin();
  //   for(i=0;i<rows;++i) {
  //     VERIFY_IS_APPROX(it[i],3*A(i,j));
  //   }
  // }
#endif

  if(rows>=3) {
    VERIFY_IS_EQUAL((v.begin()+rows/2)[1], v(rows/2+1));

    VERIFY_IS_EQUAL((A.allRows().begin()+rows/2)[1], A.row(rows/2+1));
  }

  if(cols>=3) {
    VERIFY_IS_EQUAL((A.allCols().begin()+cols/2)[1], A.col(cols/2+1));
  }

  if(rows>=2)
  {
    v(1) = v(0)-Scalar(1);
    VERIFY(!std::is_sorted(begin(v),end(v)));
  }
  std::sort(begin(v),end(v));
  VERIFY(std::is_sorted(begin(v),end(v)));
  VERIFY(!std::is_sorted(make_reverse_iterator(end(v)),make_reverse_iterator(begin(v))));

  // std::sort with pointer-based iterator and default increment
  {
    j = internal::random<Index>(0,A.cols()-1);
    // std::sort(begin(A.col(j)),end(A.col(j))); // does not compile because this returns const iterators
    typename ColMatrixType::ColXpr Acol = A.col(j);
    std::sort(begin(Acol),end(Acol));
    VERIFY(std::is_sorted(Acol.cbegin(),Acol.cend()));
    A.setRandom();

    std::sort(A.col(j).begin(),A.col(j).end());
    VERIFY(std::is_sorted(A.col(j).cbegin(),A.col(j).cend()));
    A.setRandom();
  }

  // std::sort with pointer-based iterator and runtime increment
  {
    i = internal::random<Index>(0,A.rows()-1);
    typename ColMatrixType::RowXpr Arow = A.row(i);
    VERIFY_IS_EQUAL( std::distance(begin(Arow),end(Arow)), cols);
    std::sort(begin(Arow),end(Arow));
    VERIFY(std::is_sorted(Arow.cbegin(),Arow.cend()));
    A.setRandom();

    std::sort(A.row(i).begin(),A.row(i).end());
    VERIFY(std::is_sorted(A.row(i).cbegin(),A.row(i).cend()));
    A.setRandom();
  }

  // std::sort with generic iterator
  {
    auto B1 = B.reshaped();
    std::sort(begin(B1),end(B1));
    VERIFY(std::is_sorted(B1.cbegin(),B1.cend()));
    B.setRandom();

    // assertion because nested expressions are different
    // std::sort(B.reshaped().begin(),B.reshaped().end());
    // VERIFY(std::is_sorted(B.reshaped().cbegin(),B.reshaped().cend()));
    // B.setRandom();
  }

  {
    j = internal::random<Index>(0,A.cols()-1);
    typename ColMatrixType::ColXpr Acol = A.col(j);
    std::partial_sum(begin(Acol), end(Acol), begin(v));
    VERIFY_IS_EQUAL(v(seq(1,last)), v(seq(0,last-1))+Acol(seq(1,last)));

    // inplace
    std::partial_sum(begin(Acol), end(Acol), begin(Acol));
    VERIFY_IS_EQUAL(v, Acol);
  }

  if(rows>=3)
  {
    // stress random access
    v.setRandom();
    VectorType v1 = v;
    std::sort(begin(v1),end(v1));
    std::nth_element(v.begin(), v.begin()+rows/2, v.end());
    VERIFY_IS_APPROX(v1(rows/2), v(rows/2));

    v.setRandom();
    v1 = v;
    std::sort(begin(v1)+rows/2,end(v1));
    std::nth_element(v.begin()+rows/2, v.begin()+rows/4, v.end());
    VERIFY_IS_APPROX(v1(rows/4), v(rows/4));
  }

#if EIGEN_HAS_CXX11
  j = 0;
  for(auto c : A.allCols()) { VERIFY_IS_APPROX(c.sum(), A.col(j).sum()); ++j; }
  j = 0;
  for(auto c : B.allCols()) { VERIFY_IS_APPROX(c.sum(), B.col(j).sum()); ++j; }

  j = 0;
  for(auto c : B.allCols()) {
    i = 0;
    for(auto& x : c) {
      VERIFY_IS_EQUAL(x, B(i,j));
      x = A(i,j);
      ++i;
    }
    ++j;
  }
  VERIFY_IS_APPROX(A,B);
  B = Bc; // restore B
  
  i = 0;
  for(auto r : A.allRows()) { VERIFY_IS_APPROX(r.sum(), A.row(i).sum()); ++i; }
  i = 0;
  for(auto r : B.allRows()) { VERIFY_IS_APPROX(r.sum(), B.row(i).sum()); ++i; }

#endif
}

EIGEN_DECLARE_TEST(stl_iterators)
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(( test_range_for_loop<double,2,3>() ));
    CALL_SUBTEST_1(( test_range_for_loop<float,7,5>() ));
    CALL_SUBTEST_1(( test_range_for_loop<int,Dynamic,Dynamic>(internal::random<int>(5,10), internal::random<int>(5,10)) ));
    CALL_SUBTEST_1(( test_range_for_loop<int,Dynamic,Dynamic>(internal::random<int>(10,200), internal::random<int>(10,200)) ));
  }
}
