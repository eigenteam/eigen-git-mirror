// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2019 David Tellenbach <david.tellenbach@tellnotes.org>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_NO_STATIC_ASSERT

#include "main.h"

template<typename Scalar, bool is_integer = NumTraits<Scalar>::IsInteger>
struct TestMethodDispatching {
  static void run() {}
};

template<typename Scalar>
struct TestMethodDispatching<Scalar, 1> {
  static void run()
  {
    {
      Matrix<Scalar, Dynamic, Dynamic> m {3, 4};
      Array<Scalar, Dynamic, Dynamic> a {3, 4};
      VERIFY(m.rows() == 3);
      VERIFY(m.cols() == 4);
      VERIFY(a.rows() == 3);
      VERIFY(a.cols() == 4);
    }
    {
      Matrix<Scalar, 1, 2> m {3, 4};
      Array<Scalar, 1, 2> a {3, 4};
      VERIFY(m(0) == 3);
      VERIFY(m(1) == 4);
      VERIFY(a(0) == 3);
      VERIFY(a(1) == 4);
    }
    {
      Matrix<Scalar, 2, 1> m {3, 4};
      Array<Scalar, 2, 1> a {3, 4};
      VERIFY(m(0) == 3);
      VERIFY(m(1) == 4);
      VERIFY(a(0) == 3);
      VERIFY(a(1) == 4);
    }
  }
};

template<typename Scalar> void singleInitializerListVectorConstruction()
{
  Scalar raw[4];
  for(int k = 0; k < 4; ++k) {
    raw[k] = internal::random<Scalar>();
  }
  {
    Matrix<Scalar, 1, 4> m { raw[0], raw[1], raw[2], raw[3] };
    Array<Scalar, 1, 4> a { raw[0], raw[1], raw[2], raw[3] };
    for(int k = 0; k < 4; ++k) {
      VERIFY(m(k) == raw[k]);
    }
    for(int k = 0; k < 4; ++k) {
      VERIFY(a(k) == raw[k]);
    }
    VERIFY_IS_EQUAL(m, (Matrix<Scalar, 1, 4>(raw[0], raw[1], raw[2], raw[3])));
    VERIFY_IS_EQUAL(m, (Matrix<Scalar, 1, 4>({raw[0], raw[1], raw[2], raw[3]})));
    VERIFY((a == (Array<Scalar, 1, 4>(raw[0], raw[1], raw[2], raw[3]))).all());
    VERIFY((a == (Array<Scalar, 1, 4>({raw[0], raw[1], raw[2], raw[3]}))).all());
  }
  {
    Matrix<Scalar, 4, 1> m { raw[0], raw[1], raw[2], raw[3] };
    Array<Scalar, 4, 1> a { raw[0], raw[1], raw[2], raw[3] };
    for(int k = 0; k < 4; ++k) {
      VERIFY(m(k) == raw[k]);
    }
    for(int k = 0; k < 4; ++k) {
      VERIFY(a(k) == raw[k]);
    }
    VERIFY_IS_EQUAL(m, (Matrix<Scalar, 4, 1>(raw[0], raw[1], raw[2], raw[3])));
    VERIFY_IS_EQUAL(m, (Matrix<Scalar, 4, 1>({raw[0], raw[1], raw[2], raw[3]})));
    VERIFY((a == (Array<Scalar, 4, 1>(raw[0], raw[1], raw[2], raw[3]))).all());
    VERIFY((a == (Array<Scalar, 4, 1>({raw[0], raw[1], raw[2], raw[3]}))).all());
  }
}


template<typename Scalar> void initializerListVectorConstruction()
{
  Scalar raw[4];
  for(int k = 0; k < 4; ++k) {
    raw[k] = internal::random<Scalar>();
  }
  {
    Matrix<Scalar, 4, 1> m { {raw[0]}, {raw[1]},{raw[2]},{raw[3]} };
    Array<Scalar, 4, 1> a { {raw[0]}, {raw[1]}, {raw[2]}, {raw[3]} };
    for(int k = 0; k < 4; ++k) {
      VERIFY(m(k) == raw[k]);
    }
    for(int k = 0; k < 4; ++k) {
      VERIFY(a(k) == raw[k]);
    }
    VERIFY_IS_EQUAL(m, (Matrix<Scalar,4,1>({ {raw[0]}, {raw[1]}, {raw[2]}, {raw[3]} })));
    VERIFY((a == (Array<Scalar,4,1>({ {raw[0]}, {raw[1]}, {raw[2]}, {raw[3]} }))).all());
  }
  {
    Matrix<Scalar, 1, 4> m { {raw[0], raw[1], raw[2], raw[3]} };
    Array<Scalar, 1, 4> a { {raw[0], raw[1], raw[2], raw[3]} };
    for(int k = 0; k < 4; ++k) {
      VERIFY(m(k) == raw[k]);
    }
    for(int k = 0; k < 4; ++k) {
      VERIFY(a(k) == raw[k]);
    }
    VERIFY_IS_EQUAL(m, (Matrix<Scalar, 1, 4>({{raw[0],raw[1],raw[2],raw[3]}})));
    VERIFY((a == (Array<Scalar, 1, 4>({{raw[0],raw[1],raw[2],raw[3]}}))).all());
  }
  {
    Matrix<Scalar, 4, Dynamic> m { {raw[0]}, {raw[1]}, {raw[2]}, {raw[3]} };
    Array<Scalar, 4, Dynamic> a { {raw[0]}, {raw[1]}, {raw[2]}, {raw[3]} };
    for(int k=0; k < 4; ++k) {
      VERIFY(m(k) == raw[k]);
    }
    for(int k=0; k < 4; ++k) {
      VERIFY(a(k) == raw[k]);
    }
    VERIFY_IS_EQUAL(m, (Matrix<Scalar, 4, Dynamic>({ {raw[0]}, {raw[1]}, {raw[2]}, {raw[3]} })));
    VERIFY((a == (Array<Scalar, 4, Dynamic>({ {raw[0]}, {raw[1]}, {raw[2]}, {raw[3]} }))).all());
  }
  {
    Matrix<Scalar, Dynamic, 4> m {{raw[0],raw[1],raw[2],raw[3]}};
    Array<Scalar, Dynamic, 4> a {{raw[0],raw[1],raw[2],raw[3]}};
    for(int k=0; k < 4; ++k) {
      VERIFY(m(k) == raw[k]);
    }
    for(int k=0; k < 4; ++k) {
      VERIFY(a(k) == raw[k]);
    }
    VERIFY_IS_EQUAL(m, (Matrix<Scalar, Dynamic, 4>({{raw[0],raw[1],raw[2],raw[3]}})));
    VERIFY((a == (Array<Scalar, Dynamic, 4>({{raw[0],raw[1],raw[2],raw[3]}}))).all());
  }
}

template<typename Scalar> void initializerListMatrixConstruction()
{
  const Index RowsAtCompileTime = 5;
  const Index ColsAtCompileTime = 4;
  const Index SizeAtCompileTime = RowsAtCompileTime * ColsAtCompileTime;

  Scalar raw[SizeAtCompileTime];
  for (int i = 0; i < SizeAtCompileTime; ++i) {
    raw[i] = internal::random<Scalar>();
  }
  {
    Matrix<Scalar, Dynamic, Dynamic> m {};
    VERIFY(m.cols() == 0);
    VERIFY(m.rows() == 0);
    VERIFY_IS_EQUAL(m, (Matrix<Scalar, Dynamic, Dynamic>()));
  }
  {
    Matrix<Scalar, 5, 4> m {
      {raw[0], raw[1], raw[2], raw[3]},
      {raw[4], raw[5], raw[6], raw[7]},
      {raw[8], raw[9], raw[10], raw[11]},
      {raw[12], raw[13], raw[14], raw[15]},
      {raw[16], raw[17], raw[18], raw[19]}
    };

    Matrix<Scalar, 5, 4> m2;
    m2 << raw[0], raw[1], raw[2], raw[3],
          raw[4], raw[5], raw[6], raw[7],
          raw[8], raw[9], raw[10], raw[11],
          raw[12], raw[13], raw[14], raw[15],
          raw[16], raw[17], raw[18], raw[19];

    int k = 0;
    for(int i = 0; i < RowsAtCompileTime; ++i) {
      for (int j = 0; j < ColsAtCompileTime; ++j) {
        VERIFY(m(i, j) == raw[k]);
        ++k;
      }
    }
    VERIFY_IS_EQUAL(m, m2);
  }
  {
    Matrix<Scalar, Dynamic, Dynamic> m{
      {raw[0], raw[1], raw[2], raw[3]},
      {raw[4], raw[5], raw[6], raw[7]},
      {raw[8], raw[9], raw[10], raw[11]},
      {raw[12], raw[13], raw[14], raw[15]},
      {raw[16], raw[17], raw[18], raw[19]}
    };

    VERIFY(m.cols() == 4);
    VERIFY(m.rows() == 5);
    int k = 0;
    for(int i = 0; i < RowsAtCompileTime; ++i) {
      for (int j = 0; j < ColsAtCompileTime; ++j) {
        VERIFY(m(i, j) == raw[k]);
        ++k;
      }
    }

    Matrix<Scalar, Dynamic, Dynamic> m2(RowsAtCompileTime, ColsAtCompileTime);
    k = 0;
    for(int i = 0; i < RowsAtCompileTime; ++i) {
      for (int j = 0; j < ColsAtCompileTime; ++j) {
        m2(i, j) = raw[k];
        ++k;
      }
    }
    VERIFY_IS_EQUAL(m, m2);
  }
}

template<typename Scalar> void initializerListArrayConstruction()
{
  const Index RowsAtCompileTime = 5;
  const Index ColsAtCompileTime = 4;
  const Index SizeAtCompileTime = RowsAtCompileTime * ColsAtCompileTime;

  Scalar raw[SizeAtCompileTime];
  for (int i = 0; i < SizeAtCompileTime; ++i) {
    raw[i] = internal::random<Scalar>();
  }
  {
    Array<Scalar, Dynamic, Dynamic> a {};
    VERIFY(a.cols() == 0);
    VERIFY(a.rows() == 0);
  }
  {
    Array<Scalar, 5, 4> m {
      {raw[0], raw[1], raw[2], raw[3]},
      {raw[4], raw[5], raw[6], raw[7]},
      {raw[8], raw[9], raw[10], raw[11]},
      {raw[12], raw[13], raw[14], raw[15]},
      {raw[16], raw[17], raw[18], raw[19]}
    };

    Array<Scalar, 5, 4> m2;
    m2 << raw[0], raw[1], raw[2], raw[3],
          raw[4], raw[5], raw[6], raw[7],
          raw[8], raw[9], raw[10], raw[11],
          raw[12], raw[13], raw[14], raw[15],
          raw[16], raw[17], raw[18], raw[19];

    int k = 0;
    for(int i = 0; i < RowsAtCompileTime; ++i) {
      for (int j = 0; j < ColsAtCompileTime; ++j) {
        VERIFY(m(i, j) == raw[k]);
        ++k;
      }
    }
    VERIFY_IS_APPROX(m, m2);
  }
  {
    Array<Scalar, Dynamic, Dynamic> m {
      {raw[0], raw[1], raw[2], raw[3]},
      {raw[4], raw[5], raw[6], raw[7]},
      {raw[8], raw[9], raw[10], raw[11]},
      {raw[12], raw[13], raw[14], raw[15]},
      {raw[16], raw[17], raw[18], raw[19]}
    };

    VERIFY(m.cols() == 4);
    VERIFY(m.rows() == 5);
    int k = 0;
    for(int i = 0; i < RowsAtCompileTime; ++i) {
      for (int j = 0; j < ColsAtCompileTime; ++j) {
        VERIFY(m(i, j) == raw[k]);
        ++k;
      }
    }

    Array<Scalar, Dynamic, Dynamic> m2(RowsAtCompileTime, ColsAtCompileTime);
    k = 0;
    for(int i = 0; i < RowsAtCompileTime; ++i) {
      for (int j = 0; j < ColsAtCompileTime; ++j) {
        m2(i, j) = raw[k];
        ++k;
      }
    }
    VERIFY_IS_APPROX(m, m2);
  }
}

template<typename Scalar> void dynamicVectorConstruction()
{
  const Index size = 4;
  Scalar raw[size];
  for (int i = 0; i < size; ++i) {
    raw[i] = internal::random<Scalar>();
  }

  typedef Matrix<Scalar, Dynamic, 1>  VectorX;

  {
    VectorX v {{raw[0], raw[1], raw[2], raw[3]}};
    for (int i = 0; i < size; ++i) {
      VERIFY(v(i) == raw[i]);
    }
    VERIFY(v.rows() == size);
    VERIFY(v.cols() == 1);
    VERIFY_IS_EQUAL(v, (VectorX {{raw[0], raw[1], raw[2], raw[3]}}));
  }

  {
    VERIFY_RAISES_ASSERT((VectorX {raw[0], raw[1], raw[2], raw[3]}));
  }
  {
    VERIFY_RAISES_ASSERT((VectorX  {
      {raw[0], raw[1], raw[2], raw[3]},
      {raw[0], raw[1], raw[2], raw[3]},
    }));
  }
}

EIGEN_DECLARE_TEST(initializer_list_construction)
{
  CALL_SUBTEST_1(initializerListVectorConstruction<unsigned char>());
  CALL_SUBTEST_1(initializerListVectorConstruction<float>());
  CALL_SUBTEST_1(initializerListVectorConstruction<double>());
  CALL_SUBTEST_1(initializerListVectorConstruction<int>());
  CALL_SUBTEST_1(initializerListVectorConstruction<long int>());
  CALL_SUBTEST_1(initializerListVectorConstruction<std::ptrdiff_t>());
  CALL_SUBTEST_1(initializerListVectorConstruction<std::complex<int>>());
  CALL_SUBTEST_1(initializerListVectorConstruction<std::complex<double>>());
  CALL_SUBTEST_1(initializerListVectorConstruction<std::complex<float>>());

  CALL_SUBTEST_2(initializerListMatrixConstruction<unsigned char>());
  CALL_SUBTEST_2(initializerListMatrixConstruction<float>());
  CALL_SUBTEST_2(initializerListMatrixConstruction<double>());
  CALL_SUBTEST_2(initializerListMatrixConstruction<int>());
  CALL_SUBTEST_2(initializerListMatrixConstruction<long int>());
  CALL_SUBTEST_2(initializerListMatrixConstruction<std::ptrdiff_t>());
  CALL_SUBTEST_2(initializerListMatrixConstruction<std::complex<int>>());
  CALL_SUBTEST_2(initializerListMatrixConstruction<std::complex<double>>());
  CALL_SUBTEST_2(initializerListMatrixConstruction<std::complex<float>>());

  CALL_SUBTEST_3(initializerListArrayConstruction<unsigned char>());
  CALL_SUBTEST_3(initializerListArrayConstruction<float>());
  CALL_SUBTEST_3(initializerListArrayConstruction<double>());
  CALL_SUBTEST_3(initializerListArrayConstruction<int>());
  CALL_SUBTEST_3(initializerListArrayConstruction<long int>());
  CALL_SUBTEST_3(initializerListArrayConstruction<std::ptrdiff_t>());
  CALL_SUBTEST_3(initializerListArrayConstruction<std::complex<int>>());
  CALL_SUBTEST_3(initializerListArrayConstruction<std::complex<double>>());
  CALL_SUBTEST_3(initializerListArrayConstruction<std::complex<float>>());

  CALL_SUBTEST_4(singleInitializerListVectorConstruction<unsigned char>());
  CALL_SUBTEST_4(singleInitializerListVectorConstruction<float>());
  CALL_SUBTEST_4(singleInitializerListVectorConstruction<double>());
  CALL_SUBTEST_4(singleInitializerListVectorConstruction<int>());
  CALL_SUBTEST_4(singleInitializerListVectorConstruction<long int>());
  CALL_SUBTEST_4(singleInitializerListVectorConstruction<std::ptrdiff_t>());
  CALL_SUBTEST_4(singleInitializerListVectorConstruction<std::complex<int>>());
  CALL_SUBTEST_4(singleInitializerListVectorConstruction<std::complex<double>>());
  CALL_SUBTEST_4(singleInitializerListVectorConstruction<std::complex<float>>());

  CALL_SUBTEST_5(TestMethodDispatching<int>::run());
  CALL_SUBTEST_5(TestMethodDispatching<long int>::run());

  CALL_SUBTEST_6(dynamicVectorConstruction<unsigned char>());
  CALL_SUBTEST_6(dynamicVectorConstruction<float>());
  CALL_SUBTEST_6(dynamicVectorConstruction<double>());
  CALL_SUBTEST_6(dynamicVectorConstruction<int>());
  CALL_SUBTEST_6(dynamicVectorConstruction<long int>());
  CALL_SUBTEST_6(dynamicVectorConstruction<std::ptrdiff_t>());
  CALL_SUBTEST_6(dynamicVectorConstruction<std::complex<int>>());
  CALL_SUBTEST_6(dynamicVectorConstruction<std::complex<double>>());
  CALL_SUBTEST_6(dynamicVectorConstruction<std::complex<float>>());
}