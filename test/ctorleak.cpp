#include "main.h"

#include <exception>  // std::exception

struct Foo
{
  static unsigned object_count;
  static unsigned object_limit;
  int dummy;

  Foo()
  {
#ifdef EIGEN_EXCEPTIONS
    // TODO: Is this the correct way to handle this?
    if (Foo::object_count > Foo::object_limit) { throw Foo::Fail(); }
#endif
    ++Foo::object_count;
  }

  ~Foo()
  {
    --Foo::object_count;
  }

  class Fail : public std::exception {};
};

unsigned Foo::object_count = 0;
unsigned Foo::object_limit = 0;


void test_ctorleak()
{
  typedef DenseIndex Index;
  Foo::object_count = 0;
  for(int i = 0; i < g_repeat; i++) {
    Index rows = internal::random<Index>(2,EIGEN_TEST_MAX_SIZE), cols = internal::random<Index>(2,EIGEN_TEST_MAX_SIZE);
    Foo::object_limit = internal::random(0, rows*cols - 2);
#ifdef EIGEN_EXCEPTIONS
    try
    {
#endif
      Matrix<Foo, Dynamic, Dynamic> m(rows, cols);
#ifdef EIGEN_EXCEPTIONS
      VERIFY(false);  // not reached if exceptions are enabled
    }
    catch (const Foo::Fail&) { /* ignore */ }
#endif
  }
  VERIFY_IS_EQUAL(static_cast<unsigned>(0), Foo::object_count);
}
