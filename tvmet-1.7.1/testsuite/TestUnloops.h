/*
 * Tiny Vector Matrix Library
 * Dense Vector Matrix Libary of Tiny size using Expression Templates
 *
 * Copyright (C) 2001 - 2003 Olaf Petzold <opetzold@users.sourceforge.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * $Id: TestUnloops.h,v 1.1 2004/04/24 11:55:15 opetzold Exp $
 */

#ifndef TVMET_TEST_UNLOOPS_H
#define TVMET_TEST_UNLOOPS_H

#include <algorithm>

#include <cppunit/extensions/HelperMacros.h>

#include <tvmet/Vector.h>
#include <tvmet/Matrix.h>
#include <tvmet/util/Incrementor.h>


template <class T>
class TestUnloops : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( TestUnloops );
  CPPUNIT_TEST( Mx );
  CPPUNIT_TEST( Mtx );
  CPPUNIT_TEST( MM );
//   CPPUNIT_TEST( MtM );
//   CPPUNIT_TEST( MMt );
//   CPPUNIT_TEST( tMM );
  CPPUNIT_TEST_SUITE_END();

public:
  TestUnloops() { }

public: // cppunit interface
  /** cppunit hook for fixture set up. */
  void setUp();

  /** cppunit hook for fixture tear down. */
  void tearDown();

protected:
  template<class A, class B, class C>
  void mv_product(const A&, const B&, C&);

  template<class A, class B, class C>
  void mm_product(const A&, const B&, C&);

  template<class A, class B, class C>
  void mtm_product(const A&, const B&, C&);

  template<class A, class B, class C>
  void mmt_product(const A&, const B&, C&);

protected:
  void Mx();
  void Mtx();
  void MM();
  void MtM();
  void MMt();
  void tMM();

public:
  typedef T				value_type;

private:
  enum {
    dim = 8,
    foo = 2
  };
};

/*****************************************************************************
 * Implementation part I (cppunit part)
 ****************************************************************************/

template <class T>
void TestUnloops<T>::setUp() { }

template <class T>
void TestUnloops<T>::tearDown() { }

/*****************************************************************************
 * Implementation part II (reference loops)
 ****************************************************************************/
template<class T>
template<class LHS, class RHS, class RES>
void TestUnloops<T>::mv_product(const LHS& A, const RHS& B, RES& X) {
  assert(int(LHS::Rows) == int(RES::Size));
  assert(int(LHS::Cols) == int(RHS::Size));

  enum {
    M = LHS::Rows,
    N = RHS::Size // is Vector
  };

  for (int i = 0; i < M; i++){
    value_type sum(0);
    for (int j = 0; j < N; j++){
      sum += A(i, j) * B(j);
    }
    X(i) = sum;
  }
}

template<class T>
template<class LHS, class RHS, class RES>
void TestUnloops<T>::mm_product(const LHS& A, const RHS& B, RES& X) {
  assert(int(LHS::Rows) == int(RES::Rows));
  assert(int(LHS::Cols) == int(RHS::Rows));
  assert(int(RHS::Cols) == int(RES::Cols));

  enum {
    M = LHS::Rows,
    N = RHS::Cols,
    K = RHS::Rows
  };

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      value_type sum(0);
      for (int k = 0; k < K; ++k) {
	sum += A(i, k) * B(k, j);
      }
      X(i, j) = sum;
    }
  }
}

template<class T>
template<class LHS, class RHS, class RES>
void TestUnloops<T>::mtm_product(const LHS& A, const RHS& B, RES& X) {
  assert(int(LHS::Rows) == int(RHS::Rows));
  assert(int(LHS::Cols) == int(RES::Rows));
  assert(int(RHS::Cols) == int(RES::Cols));

  enum {
    M = LHS::Cols,
    N = RHS::Cols,
    K = RHS::Rows
  };

  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      value_type sum(0);
      for (int k = 0; k < K; k++){
	sum += A(k, i) * B(k, j);
      }
      X(i, j) = sum;
    }
  }
}

template<class T>
template<class LHS, class RHS, class RES>
void TestUnloops<T>::mmt_product(const LHS& A, const RHS& B, RES& X) {
  assert(int(LHS::Rows) == int(RES::Rows));
  assert(int(LHS::Cols) == int(RHS::Cols));
  assert(int(RHS::Rows) == int(RES::Cols));

  enum {
    M = LHS::Rows,
    N = RHS::Rows,
    K = LHS::Cols
  };

  for (int i = 0;i < N; i++){
    for (int j = 0;j < N; j++){
      value_type sum(0);
      for (int k = 0;k < N; k++){
	sum += A(i, k)*A(j, k);
      }
      X(i, j) = sum;
    }
  }
}

/*****************************************************************************
 * Implementation part III
 ****************************************************************************/

template <class T>
void TestUnloops<T>::Mx() {
  using namespace tvmet;

  enum {
    Rows = dim-foo,
    Cols = dim+foo,
  };

  typedef Matrix<T, Rows, Cols>          	matrix1_type;
  typedef Vector<T, matrix1_type::Cols>		vector1_type;
  typedef Vector<T, matrix1_type::Rows>		vector2_type;

  matrix1_type          			M;
  vector1_type          			x;

  std::generate(M.begin(), M.end(),
		tvmet::util::Incrementor<typename vector1_type::value_type>());
  std::generate(x.begin(), x.end(),
		tvmet::util::Incrementor<typename vector2_type::value_type>());

  vector2_type          			r;
  mv_product(M, x, r);

  vector2_type          			y;
  y = prod(M, x);

  CPPUNIT_ASSERT( all_elements( y == r ) );
}


template <class T>
void TestUnloops<T>::Mtx() {
  using namespace tvmet;

  enum {
    Rows = dim-foo,
    Cols = dim+foo,
  };

  typedef Matrix<T, Rows, Cols>          	matrix1_type;
  typedef Matrix<T,
    matrix1_type::Cols, matrix1_type::Rows>    	matrix1t_type;
  typedef Vector<T, matrix1t_type::Cols>        vector1_type;
  typedef Vector<T, matrix1t_type::Rows>	vector2_type;

  matrix1_type          			M;
  vector1_type          			x;

  std::generate(M.begin(), M.end(),
		tvmet::util::Incrementor<typename matrix1_type::value_type>());
  std::generate(x.begin(), x.end(),
		tvmet::util::Incrementor<typename vector1_type::value_type>());

  vector2_type          			r;
  matrix1t_type					Mt(trans(M));
  mv_product(Mt, x, r);

  vector2_type          			y;
  y = Mtx_prod(M, x);

  CPPUNIT_ASSERT( all_elements( y == r ) );
}


template <class T>
void TestUnloops<T>::MM() {
  using namespace tvmet;

  enum {
    Rows1 = dim-foo,
    Cols1 = dim+foo,
    Cols2 = dim
  };

  typedef Matrix<T, Rows1, Cols1>          	matrix1_type;
  typedef Matrix<T, Cols1, Cols2>          	matrix2_type;
  typedef Matrix<T,
    matrix1_type::Rows, matrix2_type::Cols>	matrix3_type;

  matrix1_type          			M1;
  matrix2_type          			M2;

  std::generate(M1.begin(), M1.end(),
		tvmet::util::Incrementor<typename matrix1_type::value_type>());
  std::generate(M2.begin(), M2.end(),
		tvmet::util::Incrementor<typename matrix2_type::value_type>());

  matrix3_type          			R;
  mm_product(M1, M2, R);

  matrix3_type          			M3;
  M3 = prod(M1, M2);

  CPPUNIT_ASSERT( all_elements( M3 == R ) );
}


template <class T>
void TestUnloops<T>::MtM() {
  using namespace tvmet;

  enum {
    Rows1 = dim-foo,
    Cols1 = dim+foo,
    Cols2 = dim
  };

  typedef Matrix<T, Rows1, Cols1>          	matrix1_type;
  typedef Matrix<T, Rows1, Cols2>          	matrix2_type;
  typedef Matrix<T,
    matrix1_type::Cols, matrix2_type::Cols>	matrix3_type;

  matrix1_type          			M1;
  matrix2_type          			M2;

  std::generate(M1.begin(), M1.end(),
		tvmet::util::Incrementor<typename matrix1_type::value_type>());
  std::generate(M2.begin(), M2.end(),
		tvmet::util::Incrementor<typename matrix2_type::value_type>());

  matrix3_type          			R;
  mtm_product(M1, M2, R);

  matrix3_type          			M3;
  M3 = MtM_prod(M1, M2);

  std::cout << "M1=" << M1 << std::endl;
  std::cout << "M2=" << M2 << std::endl;
  std::cout << "M3=" << M3 << std::endl;
  std::cout << "R=" << R << std::endl;

  CPPUNIT_ASSERT( all_elements( M3 == R ) );
}


template <class T>
void TestUnloops<T>::MMt() {
  using namespace tvmet;

  enum {
    Rows1 = dim-foo,
    Cols1 = dim+foo,
    Rows2 = dim
  };

  typedef Matrix<T, Rows1, Cols1>          	matrix1_type;
  typedef Matrix<T, Rows2, Cols1>          	matrix2_type;
  typedef Matrix<T,
    matrix1_type::Rows, matrix2_type::Rows>	matrix3_type;

  matrix1_type          			M1;
  matrix2_type          			M2;

  std::generate(M1.begin(), M1.end(),
		tvmet::util::Incrementor<typename matrix1_type::value_type>());
  std::generate(M2.begin(), M2.end(),
		tvmet::util::Incrementor<typename matrix2_type::value_type>());

  matrix3_type          			R;
  mmt_product(M1, M2, R);

  matrix3_type          			M3;
  M3 = MMt_prod(M1, M2);

  std::cout << "M1=" << M1 << std::endl;
  std::cout << "M2=" << M2 << std::endl;
  std::cout << "M3=" << M3 << std::endl;
  std::cout << "R=" << R << std::endl;

  CPPUNIT_ASSERT( all_elements( M3 == R ) );
}


template <class T>
void TestUnloops<T>::tMM() {
  using namespace tvmet;

  enum {
    Rows1 = dim-foo,
    Cols1 = dim+foo,
    Cols2 = dim
  };

  typedef Matrix<T, Rows1, Cols1>          	matrix1_type;
  typedef Matrix<T, Cols1, Cols2>          	matrix2_type;
  typedef Matrix<T,
    matrix1_type::Rows, matrix2_type::Cols>	matrix3_type;
  typedef Matrix<T,
    matrix3_type::Cols, matrix3_type::Rows>	matrix3t_type;

  matrix1_type          			M1;
  matrix2_type          			M2;

  std::generate(M1.begin(), M1.end(),
		tvmet::util::Incrementor<typename matrix1_type::value_type>());
  std::generate(M2.begin(), M2.end(),
		tvmet::util::Incrementor<typename matrix2_type::value_type>());

  matrix3_type          			R;
  matrix3t_type          			Rt;
  mm_product(M1, M2, R);
  Rt = trans(R);

  matrix3t_type          			M3;
  M3 = trans_prod(M1, M2);

  std::cout << "M1=" << M1 << std::endl;
  std::cout << "M2=" << M2 << std::endl;
  std::cout << "M3=" << M3 << std::endl;
  std::cout << "Rt=" << Rt << std::endl;

  CPPUNIT_ASSERT( all_elements( M3 == Rt ) );
}

#endif // TVMET_TEST_UNLOOPS_H

// Local Variables:
// mode:C++
// End:
