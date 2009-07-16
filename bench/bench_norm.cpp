#include <Eigen/Core>
#include "BenchTimer.h"
using namespace Eigen;
using namespace std;

template<typename T>
EIGEN_DONT_INLINE typename T::Scalar sqsumNorm(const T& v)
{
  return v.norm();
}

template<typename T>
EIGEN_DONT_INLINE typename T::Scalar hypotNorm(const T& v)
{
  return v.stableNorm();
}

template<typename T>
EIGEN_DONT_INLINE typename T::Scalar blueNorm(const T& v)
{
  return v.blueNorm();
}

template<typename T>
EIGEN_DONT_INLINE typename T::Scalar lapackNorm(T& v)
{
  typedef typename T::Scalar Scalar;
  int n = v.size();
  Scalar scale = 1;
  Scalar ssq = 0;
  for (int i=0;i<n;++i)
  {
    Scalar ax = ei_abs(v.coeff(i));
    if (scale < ax)
    {
      ssq = Scalar(1) + ssq * ei_abs2(scale/ax);
      scale = ax;
    }
    else
      ssq += ei_abs2(ax/scale);
  }
  return scale * ei_sqrt(ssq);
}

template<typename T>
EIGEN_DONT_INLINE typename T::Scalar divacNorm(T& v)
{
  int n =v.size() / 2;
  for (int i=0;i<n;++i)
    v(i) = v(2*i)*v(2*i) + v(2*i+1)*v(2*i+1);
  n = n/2;
  while (n>0)
  {
    for (int i=0;i<n;++i)
      v(i) = v(2*i) + v(2*i+1);
    n = n/2;
  }
  return ei_sqrt(v(0));
}

Packet4f ei_plt(const Packet4f& a, Packet4f& b) { return _mm_cmplt_ps(a,b); }
Packet2d ei_plt(const Packet2d& a, Packet2d& b) { return _mm_cmplt_pd(a,b); }

Packet4f ei_pandnot(const Packet4f& a, Packet4f& b) { return _mm_andnot_ps(a,b); }
Packet2d ei_pandnot(const Packet2d& a, Packet2d& b) { return _mm_andnot_pd(a,b); }

template<typename T>
EIGEN_DONT_INLINE typename T::Scalar pblueNorm(const T& v)
{
  typedef typename T::Scalar Scalar;

  static int nmax;
  static Scalar b1, b2, s1m, s2m, overfl, rbig, relerr;
  int n;

  if(nmax <= 0)
  {
    int nbig, ibeta, it, iemin, iemax, iexp;
    Scalar abig, eps;

    nbig  = std::numeric_limits<int>::max();            // largest integer
    ibeta = NumTraits<Scalar>::Base;                    // base for floating-point numbers
    it    = NumTraits<Scalar>::Mantissa;                // number of base-beta digits in mantissa
    iemin = std::numeric_limits<Scalar>::min_exponent;  // minimum exponent
    iemax = std::numeric_limits<Scalar>::max_exponent;  // maximum exponent
    rbig  = std::numeric_limits<Scalar>::max();         // largest floating-point number

    // Check the basic machine-dependent constants.
    if(iemin > 1 - 2*it || 1+it>iemax || (it==2 && ibeta<5)
      || (it<=4 && ibeta <= 3 ) || it<2)
    {
      ei_assert(false && "the algorithm cannot be guaranteed on this computer");
    }
    iexp  = -((1-iemin)/2);
    b1    = bexp<Scalar>(ibeta, iexp);  // lower boundary of midrange
    iexp  = (iemax + 1 - it)/2;
    b2    = bexp<Scalar>(ibeta,iexp);   // upper boundary of midrange

    iexp  = (2-iemin)/2;
    s1m   = bexp<Scalar>(ibeta,iexp);   // scaling factor for lower range
    iexp  = - ((iemax+it)/2);
    s2m   = bexp<Scalar>(ibeta,iexp);   // scaling factor for upper range

    overfl  = rbig*s2m;          // overfow boundary for abig
    eps     = bexp<Scalar>(ibeta, 1-it);
    relerr  = ei_sqrt(eps);      // tolerance for neglecting asml
    abig    = 1.0/eps - 1.0;
    if (Scalar(nbig)>abig)  nmax = abig;  // largest safe n
    else                    nmax = nbig;
  }
  
  typedef typename ei_packet_traits<Scalar>::type Packet;
  const int ps = ei_packet_traits<Scalar>::size;
  Packet pasml = ei_pset1(Scalar(0));
  Packet pamed = ei_pset1(Scalar(0));
  Packet pabig = ei_pset1(Scalar(0));
  Packet ps2m = ei_pset1(s2m);
  Packet ps1m = ei_pset1(s1m);
  Packet pb2  = ei_pset1(b2);
  Packet pb1  = ei_pset1(b1);
  for(int j=0; j<v.size(); j+=ps)
  {
    Packet ax = ei_pabs(v.template packet<Aligned>(j));
    Packet ax_s2m = ei_pmul(ax,ps2m);
    Packet ax_s1m = ei_pmul(ax,ps1m);
    Packet maskBig = ei_plt(pb2,ax);
    Packet maskSml = ei_plt(ax,pb1);
    pabig = ei_padd(pabig, ei_pand(maskBig, ei_pmul(ax_s2m,ax_s2m)));
    pasml = ei_padd(pasml, ei_pand(maskSml, ei_pmul(ax_s1m,ax_s1m)));
    pamed = ei_padd(pamed, ei_pandnot(ei_pmul(ax,ax),ei_pand(maskSml,maskBig)));
  }
  Scalar abig = ei_predux(pabig);
  Scalar asml = ei_predux(pasml);
  Scalar amed = ei_predux(pamed);
  if(abig > Scalar(0))
  {
    abig = ei_sqrt(abig);
    if(abig > overfl)
    {
      ei_assert(false && "overflow");
      return rbig;
    }
    if(amed > Scalar(0))
    {
      abig = abig/s2m;
      amed = ei_sqrt(amed);
    }
    else
    {
      return abig/s2m;
    }

  }
  else if(asml > Scalar(0))
  {
    if (amed > Scalar(0))
    {
      abig = ei_sqrt(amed);
      amed = ei_sqrt(asml) / s1m;
    }
    else
    {
      return ei_sqrt(asml)/s1m;
    }
  }
  else
  {
    return ei_sqrt(amed);
  }
  asml = std::min(abig, amed);
  abig = std::max(abig, amed);
  if(asml <= abig*relerr)
    return abig;
  else
    return abig * ei_sqrt(Scalar(1) + ei_abs2(asml/abig));
}

#define BENCH_PERF(NRM) { \
  Eigen::BenchTimer tf, td; tf.reset(); td.reset();\
  for (int k=0; k<tries; ++k) { \
    tf.start(); \
    for (int i=0; i<iters; ++i) NRM(vf); \
    tf.stop(); \
  } \
  for (int k=0; k<tries; ++k) { \
    td.start(); \
    for (int i=0; i<iters; ++i) NRM(vd); \
    td.stop(); \
  } \
  std::cout << #NRM << "\t" << tf.value() << "   " << td.value() << "\n"; \
}

void check_accuracy(double basef, double based, int s)
{
  double yf = basef * ei_abs(ei_random<double>());
  double yd = based * ei_abs(ei_random<double>());
  VectorXf vf = VectorXf::Ones(s) * yf;
  VectorXd vd = VectorXd::Ones(s) * yd;
  
  std::cout << "reference\t" << ei_sqrt(double(s))*yf << "\t" << ei_sqrt(double(s))*yd << "\n";
  std::cout << "sqsumNorm\t" << sqsumNorm(vf) << "\t" << sqsumNorm(vd) << "\n";
  std::cout << "hypotNorm\t" << hypotNorm(vf) << "\t" << hypotNorm(vd) << "\n";
  std::cout << "blueNorm\t" << blueNorm(vf) << "\t" << blueNorm(vd) << "\n";
  std::cout << "pblueNorm\t" << pblueNorm(vf) << "\t" << pblueNorm(vd) << "\n";
  std::cout << "lapackNorm\t" << lapackNorm(vf) << "\t" << lapackNorm(vd) << "\n";
}

int main(int argc, char** argv) 
{
  int tries = 5;
  int iters = 100000;
  double y = 1.1345743233455785456788e12 * ei_random<double>();
  VectorXf v = VectorXf::Ones(1024) * y;
  
//   std::cerr << "Performance (out of cache):\n";
//   {
//     int iters = 1;
//     VectorXf vf = VectorXf::Ones(1024*1024*32) * y;
//     VectorXd vd = VectorXd::Ones(1024*1024*32) * y;
//     BENCH_PERF(sqsumNorm);
//     BENCH_PERF(blueNorm);
//     BENCH_PERF(pblueNorm);
//     BENCH_PERF(lapackNorm);
//     BENCH_PERF(hypotNorm);
//   }
//   
//   std::cerr << "\nPerformance (in cache):\n";
//   {
//     int iters = 100000;
//     VectorXf vf = VectorXf::Ones(512) * y;
//     VectorXd vd = VectorXd::Ones(512) * y;
//     BENCH_PERF(sqsumNorm);
//     BENCH_PERF(blueNorm);
//     BENCH_PERF(pblueNorm);
//     BENCH_PERF(lapackNorm);
//     BENCH_PERF(hypotNorm);
//   }
  
  int s = 10000;
  double basef_ok = 1.1345743233455785456788e12;
  double based_ok = 1.1345743233455785456788e32;
  
  double basef_under = 1.1345743233455785456788e-23;
  double based_under = 1.1345743233455785456788e-180;
  
  double basef_over = 1.1345743233455785456788e+27;
  double based_over = 1.1345743233455785456788e+185;
  
  std::cout.precision(20);
  
  std::cerr << "\nNo under/overflow:\n";
  check_accuracy(basef_ok, based_ok, s);
  
  std::cerr << "\nUnderflow:\n";
  check_accuracy(basef_under, based_under, 1);
  
  std::cerr << "\nOverflow:\n";
  check_accuracy(basef_over, based_over, s);
}
