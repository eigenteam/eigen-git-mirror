#include<Eigen/Core>

using namespace Eigen;

/** From Golub & van Loan Algorithm 5.1.1 page 210
 */
template<typename InputVector, typename OutputVector>
void ei_compute_householder(const InputVector& x, OutputVector *v, typename OutputVector::RealScalar *beta)
{
  EIGEN_STATIC_ASSERT(ei_is_same_type<typename InputVector::Scalar, typename OutputVector::Scalar>::ret,
        YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY)
  EIGEN_STATIC_ASSERT((InputVector::SizeAtCompileTime == OutputVector::SizeAtCompileTime+1)
                      || InputVector::SizeAtCompileTime == Dynamic
                      || OutputVector::SizeAtCompileTime == Dynamic)
  typedef typename OutputVector::RealScalar RealScalar;
  ei_assert(x.size() == v->size()+1);
  int n = x.size();
  RealScalar sigma = x.end(n-1).squaredNorm();
  *v = x.end(n-1);
  // the big assumption in this code is that ei_abs2(x->coeff(0)) is not much smaller than sigma.
  if(ei_isMuchSmallerThan(sigma, ei_abs2(x.coeff(0))))
  {
    // in this case x is approx colinear to (1,0,....,0)
    // fixme, implement this trivial case
  }
  else
  {
    RealScalar mu = ei_sqrt(ei_abs2(x.coeff(0)) + sigma);
    RealScalar kappa = -sigma/(x.coeff(0)+mu);
    *beta = 
  }
}