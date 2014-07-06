#include "main.h"

#include <exception>  // std::exception

struct Foo
{
  int dummy;
  Foo() { if (!internal::random(0, 10)) throw Foo::Fail(); }
  class Fail : public std::exception {};
};

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
  try
    {
      Matrix<Foo, Dynamic, Dynamic> m(14, 92);
      eigen_assert(false);  // not reached
    }
  catch (const Foo::Fail&) { /* ignore */ }
}
