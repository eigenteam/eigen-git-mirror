/*
 * Test file for checking meta swap feature.
 *
 * Swapping using std::swap is faster than meta template implementation.
 */

#include <iostream>
#include <algorithm>				// min, max

#include <tvmet/Matrix.h>
#include <tvmet/Vector.h>


using namespace tvmet;
using namespace std;

NS_TVMET_BEGIN

template<size_t Sz, size_t Idx=0>
class MetaSwap // later, should be MetaVector
{
private:
  enum {
    doIt = (Idx < (Sz-1)) ? 1 : 0		/**< recursive counter */
  };

public:
  template<class E1, class E2>
  static inline
  void
  swap(E1& e1, E2& e2) {
    // XXX BUG: const problem?, we have to use the operator() for Vectors
    fcnl_Swap<typename E1::value_type, typename E2::value_type>::applyOn(e1(Idx), e2(Idx));
    MetaSwap<Sz * doIt, (Idx+1) * doIt>::swap(e1, e2);
  }
};

template<>
class MetaSwap<0, 0>
{
public:
  template<class E1, class E2> static inline void swap(E1&, E2&) { }
};


/**
 * \fun swap
 * \brief swaps to vector expressions XprVector<E, Sz>
 */
template<class E1, class E2, size_t Sz>
inline
void swap(XprVector<E1, Sz> e1, XprVector<E2, Sz> e2) {
  MetaSwap<Sz>::swap(e1, e2);
}

/**
 * \fun swap
 * \brief swaps to vector
 */
template<class T1, class T2, size_t Sz>
inline
void swap(Vector<T1, Sz>& lhs, Vector<T2, Sz>& rhs) {
  swap(lhs.asXpr(), rhs.asXpr());
}


/**
 * \fun swap2
 * \brief swaps to vector expressions XprVector<E, Sz>
 */
template<class E1, class E2, size_t Sz>
inline
void swap2(XprVector<E1, Sz> e1, XprVector<E2, Sz> e2) {
  // loops are faster than meta templates
  for(size_t i = 0; i < Sz; ++i)
    std::swap(e1[i], e2[i]);
}

/**
 * \fun swap2
 * \brief swaps to vector
 */
template<class T1, class T2, size_t Sz>
inline
void swap2(Vector<T1, Sz>& lhs, Vector<T2, Sz>& rhs) {
  // loops are faster than meta templates
  for(size_t i = 0; i < Sz; ++i)
    std::swap(lhs[i], rhs[i]);
}


NS_TVMET_END




template<class V1, class V2>
void test_meta_swap(V1& v1, V2& v2) {
  tvmet::swap(v1, v2);
}

template<class V1, class V2>
void test_loop_swap(V1& v1, V2& v2) {
  tvmet::swap2(v1, v2);
}



template<class M1, class M2>
void test_meta_mswap(M1& m1, M2& m2) {
  tvmet::swap2(row(m1, 0), row(m2, 0));
  tvmet::swap2(col(m1, 0), col(m2, 0));
}


#define LOOPS	1000000

int main() {
  typedef Matrix<double, 4, 4>				matrix_type;
  typedef Vector<double, 4>				vector_type;

  //----------------------------------------------------------------
  vector_type v1(1);
  vector_type v2(4);

  cout << "\nSwap Vectors\n\n";
  cout << v1 << endl;
  cout << v2 << endl;
  for(size_t i = 0; i < LOOPS; ++i)
    test_meta_swap(v1, v2);
  cout << v1 << endl;
  cout << v2 << endl;
  for(size_t i = 0; i < LOOPS; ++i)
    test_loop_swap(v1, v2);
  cout << v1 << endl;
  cout << v2 << endl;

  //----------------------------------------------------------------
  matrix_type m1, m2;
  vector_type rv;

  m1 = 1,2,3,4,
       5,6,7,8,
       9,10,11,12,
       13,14,15,16;
  m2 = transpose(m1);

  cout << "\nSwap Matrix\n\n";
  cout << m1 << endl;
  cout << m2 << endl;
  test_meta_mswap(m1, m2);
  cout << m1 << endl;
  cout << m2 << endl;
}
