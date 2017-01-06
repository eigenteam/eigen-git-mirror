// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <valarray>
#include <vector>
#include "main.h"

#if EIGEN_HAS_CXX11
#include <array>
#endif

typedef std::pair<Index,Index> IndexPair;

int encode(Index i, Index j) {
  return int(i*100 + j);
}

IndexPair decode(Index ij) {
  return IndexPair(ij / 100, ij % 100);
}

template<typename T>
bool match(const T& xpr, std::string ref, std::string str_xpr = "") {
  EIGEN_UNUSED_VARIABLE(str_xpr);
  //std::cout << str_xpr << "\n" << xpr << "\n\n";
  std::stringstream str;
  str << xpr;
  return str.str() == ref;
}

#define MATCH(X,R) match(X, R, #X)

void check_indexed_view()
{
  Index n = 10;

  ArrayXXi A = ArrayXXi::NullaryExpr(n,n, std::ptr_fun(encode));

  for(Index i=0; i<n; ++i)
    for(Index j=0; j<n; ++j)
      VERIFY( decode(A(i,j)) == IndexPair(i,j) );

  ArrayXd eia(10); eia.setRandom();
  Array4i eii(4); eii << 3, 1, 6, 5;
  std::valarray<double> vala(10); Map<ArrayXd>(&vala[0],10) = eia;
  std::valarray<int> vali(4); Map<ArrayXi>(&vali[0],4) = eii;
  std::vector<int> veci(4); Map<ArrayXi>(veci.data(),4) = eii;

  VERIFY( MATCH( A(3, range(9,3,-1)),
    "309  308  307  306  305  304  303")
  );

  VERIFY( MATCH( A(span(2,5), range(9,3,-1)),
    "209  208  207  206  205  204  203\n"
    "309  308  307  306  305  304  303\n"
    "409  408  407  406  405  404  403\n"
    "509  508  507  506  505  504  503\n"
    "609  608  607  606  605  604  603")
  );

  VERIFY( MATCH( A(span(2,5), 5),
    "205\n"
    "305\n"
    "405\n"
    "505\n"
    "605")
  );

  VERIFY( MATCH( A(span(last,5,-1), range(2,last)),
    "902  903  904  905  906  907  908  909\n"
    "802  803  804  805  806  807  808  809\n"
    "702  703  704  705  706  707  708  709\n"
    "602  603  604  605  606  607  608  609\n"
    "502  503  504  505  506  507  508  509")
  );

  VERIFY( MATCH( A(eii, veci),
    "303  301  306  305\n"
    "103  101  106  105\n"
    "603  601  606  605\n"
    "503  501  506  505")
  );

  VERIFY( MATCH( A(eii, all),
    "300  301  302  303  304  305  306  307  308  309\n"
    "100  101  102  103  104  105  106  107  108  109\n"
    "600  601  602  603  604  605  606  607  608  609\n"
    "500  501  502  503  504  505  506  507  508  509")
  );

  // takes the row numer 3, and repeat it 5 times
  VERIFY( MATCH( A(span(3,5,0), all),
    "300  301  302  303  304  305  306  307  308  309\n"
    "300  301  302  303  304  305  306  307  308  309\n"
    "300  301  302  303  304  305  306  307  308  309\n"
    "300  301  302  303  304  305  306  307  308  309\n"
    "300  301  302  303  304  305  306  307  308  309")
  );

  Array44i B;
  B.setRandom();
  VERIFY( (A(span(2,5), 5)).ColsAtCompileTime == 1);
  VERIFY( (A(span(2,5), 5)).RowsAtCompileTime == Dynamic);
  VERIFY_IS_EQUAL( (A(span(2,5), 5)).InnerStrideAtCompileTime , A.InnerStrideAtCompileTime);
  VERIFY_IS_EQUAL( (A(span(2,5), 5)).OuterStrideAtCompileTime , A.col(5).OuterStrideAtCompileTime);

  VERIFY_IS_EQUAL( (A(5,span(2,5))).InnerStrideAtCompileTime , A.row(5).InnerStrideAtCompileTime);
  VERIFY_IS_EQUAL( (A(5,span(2,5))).OuterStrideAtCompileTime , A.row(5).OuterStrideAtCompileTime);
  VERIFY_IS_EQUAL( (B(1,span(1,2))).InnerStrideAtCompileTime , B.row(1).InnerStrideAtCompileTime);
  VERIFY_IS_EQUAL( (B(1,span(1,2))).OuterStrideAtCompileTime , B.row(1).OuterStrideAtCompileTime);

  VERIFY_IS_EQUAL( (A(span(2,5), range(1,3))).InnerStrideAtCompileTime , A.InnerStrideAtCompileTime);
  VERIFY_IS_EQUAL( (A(span(2,5), range(1,3))).OuterStrideAtCompileTime , A.OuterStrideAtCompileTime);
  VERIFY_IS_EQUAL( (B(span(1,2), range(1,3))).InnerStrideAtCompileTime , B.InnerStrideAtCompileTime);
  VERIFY_IS_EQUAL( (B(span(1,2), range(1,3))).OuterStrideAtCompileTime , B.OuterStrideAtCompileTime);
  VERIFY_IS_EQUAL( (A(span(2,5,2), range(1,3,2))).InnerStrideAtCompileTime , Dynamic);
  VERIFY_IS_EQUAL( (A(span(2,5,2), range(1,3,2))).OuterStrideAtCompileTime , Dynamic);
  VERIFY_IS_EQUAL( (A(span(2,5,fix<2>), range(1,3,fix<3>))).InnerStrideAtCompileTime , 2);
  VERIFY_IS_EQUAL( (A(span(2,5,fix<2>), range(1,3,fix<3>))).OuterStrideAtCompileTime , Dynamic);
  VERIFY_IS_EQUAL( (B(span(1,2,fix<2>), range(1,3,fix<3>))).InnerStrideAtCompileTime , 2);
  VERIFY_IS_EQUAL( (B(span(1,2,fix<2>), range(1,3,fix<3>))).OuterStrideAtCompileTime , 3*4);

  VERIFY( (A(span(2,fix<5>), 5)).RowsAtCompileTime == 5);
  VERIFY( (A(4, all)).ColsAtCompileTime == Dynamic);
  VERIFY( (A(4, all)).RowsAtCompileTime == 1);
  VERIFY( (B(1, all)).ColsAtCompileTime == 4);
  VERIFY( (B(1, all)).RowsAtCompileTime == 1);
  VERIFY( (B(all,1)).ColsAtCompileTime == 1);
  VERIFY( (B(all,1)).RowsAtCompileTime == 4);

  VERIFY( (A(all, eii)).ColsAtCompileTime == eii.SizeAtCompileTime);
  VERIFY_IS_EQUAL( (A(eii, eii)).Flags&DirectAccessBit, (unsigned int)(0));
  VERIFY_IS_EQUAL( (A(eii, eii)).InnerStrideAtCompileTime, 0);
  VERIFY_IS_EQUAL( (A(eii, eii)).OuterStrideAtCompileTime, 0);



#if EIGEN_HAS_CXX11
  VERIFY( (A(all, std::array<int,4>{{1,3,2,4}})).ColsAtCompileTime == 4);

  VERIFY_IS_APPROX( (A(std::array<int,3>{{1,3,5}}, std::array<int,4>{{9,6,3,0}})), A(span(1,3,2), span(9,4,-3)) );

#if (!EIGEN_COMP_CLANG) || (EIGEN_COMP_CLANG>=308 && !defined(__apple_build_version__))
  VERIFY_IS_APPROX( A({3, 1, 6, 5}, all), A(std::array<int,4>{{3, 1, 6, 5}}, all) );
#endif

#endif

}

void test_indexed_view()
{
//   for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( check_indexed_view() );
//   }
}
