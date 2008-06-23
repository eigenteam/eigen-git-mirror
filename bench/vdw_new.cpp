#include <Eigen/Core>

namespace Eigen {

template<typename Scalar> struct pow12_op EIGEN_EMPTY_STRUCT {
  inline const Scalar operator() (const Scalar& a) const
  {
      Scalar b = a*a*a;
      Scalar c = b*b;
      return c*c;
  }
  template<typename PacketScalar>
  inline const PacketScalar packetOp(const PacketScalar& a) const
  {
      PacketScalar b = ei_pmul(a, ei_pmul(a, a));
      PacketScalar c = ei_pmul(b, b);
      return ei_pmul(c, c);
  }
};

template<typename Scalar>
struct ei_functor_traits<pow12_op<Scalar> >
{
  enum {
    Cost = 4 * NumTraits<Scalar>::MulCost,
    PacketAccess = int(ei_packet_traits<Scalar>::size) > 1
  };
};

} // namespace Eigen

using Eigen::pow12_op;
USING_PART_OF_NAMESPACE_EIGEN

#ifndef SCALAR
#define SCALAR float
#endif

#ifndef SIZE
#define SIZE 10000
#endif

#ifndef REPEAT
#define REPEAT 10000
#endif

typedef Matrix<SCALAR, Eigen::Dynamic, 1> Vec;

using namespace std;

SCALAR E_VDW(const Vec &interactions1, const Vec &interactions2)
{
  return interactions2
         .cwiseQuotient(interactions1)
         .cwise(pow12_op<SCALAR>())
         .sum();
}

int main() 
{
  //
  //          1   2   3   4  ... (interactions)
  // ka       .   .   .   .  ...
  // rab      .   .   .   .  ...
  // energy   .   .   .   .  ...
  // ...     ... ... ... ... ...
  // (variables
  //    for
  // interaction)
  //
  Vec interactions1(SIZE), interactions2(SIZE); // SIZE is the number of vdw interactions in our system
  // SetupCalculations()
  SCALAR rab = 1.0;  
  interactions1.setConstant(2.4);
  interactions2.setConstant(rab);
  
  // Energy()
  SCALAR energy = 0.0;
  for (unsigned int i = 0; i<REPEAT; ++i) {
    energy += E_VDW(interactions1, interactions2);
    energy *= 1 + 1e-20 * i; // prevent compiler from optimizing the loop
  }
  cout << "energy = " << energy << endl;
}
