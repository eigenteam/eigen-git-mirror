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

namespace Eigen
{
  template<>
  struct NumTraits<Foo>
  {
    typedef double Real;
    typedef double NonInteger;
    typedef double Nested;
    enum
      {
        IsComplex             =  0,
        IsInteger             =  1,
        ReadCost              = -1,
        AddCost               = -1,
        MulCost               = -1,
        IsSigned              =  1,
        RequireInitialization =  1
      };
    static inline Real epsilon() { return 1.0; }
    static inline Real dummy_epsilon() { return 0.0; }
  };
}

void test_ctorleak()
{
  Foo::object_count = 0;
  Foo::object_limit = internal::random(0, 14 * 92 - 2);
#ifdef EIGEN_EXCEPTIONS
  try
#endif
    {
      Matrix<Foo, Dynamic, Dynamic> m(14, 92);
      eigen_assert(false);  // not reached
    }
#ifdef EIGEN_EXCEPTIONS
  catch (const Foo::Fail&) { /* ignore */ }
#endif
  VERIFY_IS_EQUAL(static_cast<unsigned>(0), Foo::object_count);
}
