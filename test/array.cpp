// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

template<typename ArrayType> void array(const ArrayType& m)
{
  typedef typename ArrayType::Index Index;
  typedef typename ArrayType::Scalar Scalar;
  typedef Array<Scalar, ArrayType::RowsAtCompileTime, 1> ColVectorType;
  typedef Array<Scalar, 1, ArrayType::ColsAtCompileTime> RowVectorType;

  Index rows = m.rows();
  Index cols = m.cols(); 

  ArrayType m1 = ArrayType::Random(rows, cols),
             m2 = ArrayType::Random(rows, cols),
             m3(rows, cols);
  ArrayType m4 = m1; // copy constructor
  VERIFY_IS_APPROX(m1, m4);

  ColVectorType cv1 = ColVectorType::Random(rows);
  RowVectorType rv1 = RowVectorType::Random(cols);

  Scalar  s1 = internal::random<Scalar>(),
          s2 = internal::random<Scalar>();

  // scalar addition
  VERIFY_IS_APPROX(m1 + s1, s1 + m1);
  VERIFY_IS_APPROX(m1 + s1, ArrayType::Constant(rows,cols,s1) + m1);
  VERIFY_IS_APPROX(s1 - m1, (-m1)+s1 );
  VERIFY_IS_APPROX(m1 - s1, m1 - ArrayType::Constant(rows,cols,s1));
  VERIFY_IS_APPROX(s1 - m1, ArrayType::Constant(rows,cols,s1) - m1);
  VERIFY_IS_APPROX((m1*Scalar(2)) - s2, (m1+m1) - ArrayType::Constant(rows,cols,s2) );
  m3 = m1;
  m3 += s2;
  VERIFY_IS_APPROX(m3, m1 + s2);
  m3 = m1;
  m3 -= s1;
  VERIFY_IS_APPROX(m3, m1 - s1);  
  
  // scalar operators via Maps
  m3 = m1;
  ArrayType::Map(m1.data(), m1.rows(), m1.cols()) -= ArrayType::Map(m2.data(), m2.rows(), m2.cols());
  VERIFY_IS_APPROX(m1, m3 - m2);
  
  m3 = m1;
  ArrayType::Map(m1.data(), m1.rows(), m1.cols()) += ArrayType::Map(m2.data(), m2.rows(), m2.cols());
  VERIFY_IS_APPROX(m1, m3 + m2);
  
  m3 = m1;
  ArrayType::Map(m1.data(), m1.rows(), m1.cols()) *= ArrayType::Map(m2.data(), m2.rows(), m2.cols());
  VERIFY_IS_APPROX(m1, m3 * m2);
  
  m3 = m1;
  m2 = ArrayType::Random(rows,cols);
  m2 = (m2==0).select(1,m2);
  ArrayType::Map(m1.data(), m1.rows(), m1.cols()) /= ArrayType::Map(m2.data(), m2.rows(), m2.cols());  
  VERIFY_IS_APPROX(m1, m3 / m2);

  // reductions
  VERIFY_IS_APPROX(m1.abs().colwise().sum().sum(), m1.abs().sum());
  VERIFY_IS_APPROX(m1.abs().rowwise().sum().sum(), m1.abs().sum());
  using std::abs;
  VERIFY_IS_MUCH_SMALLER_THAN(abs(m1.colwise().sum().sum() - m1.sum()), m1.abs().sum());
  VERIFY_IS_MUCH_SMALLER_THAN(abs(m1.rowwise().sum().sum() - m1.sum()), m1.abs().sum());
  if (!internal::isMuchSmallerThan(abs(m1.sum() - (m1+m2).sum()), m1.abs().sum(), test_precision<Scalar>()))
      VERIFY_IS_NOT_APPROX(((m1+m2).rowwise().sum()).sum(), m1.sum());
  VERIFY_IS_APPROX(m1.colwise().sum(), m1.colwise().redux(internal::scalar_sum_op<Scalar>()));

  // vector-wise ops
  m3 = m1;
  VERIFY_IS_APPROX(m3.colwise() += cv1, m1.colwise() + cv1);
  m3 = m1;
  VERIFY_IS_APPROX(m3.colwise() -= cv1, m1.colwise() - cv1);
  m3 = m1;
  VERIFY_IS_APPROX(m3.rowwise() += rv1, m1.rowwise() + rv1);
  m3 = m1;
  VERIFY_IS_APPROX(m3.rowwise() -= rv1, m1.rowwise() - rv1);
  
  // Conversion from scalar
  VERIFY_IS_APPROX((m3 = s1), ArrayType::Constant(rows,cols,s1));
  VERIFY_IS_APPROX((m3 = 1),  ArrayType::Constant(rows,cols,1));
  VERIFY_IS_APPROX((m3.topLeftCorner(rows,cols) = 1),  ArrayType::Constant(rows,cols,1));
  typedef Array<Scalar,
                ArrayType::RowsAtCompileTime==Dynamic?2:ArrayType::RowsAtCompileTime,
                ArrayType::ColsAtCompileTime==Dynamic?2:ArrayType::ColsAtCompileTime,
                ArrayType::Options> FixedArrayType;
  FixedArrayType f1(s1);
  VERIFY_IS_APPROX(f1, FixedArrayType::Constant(s1));
  FixedArrayType f2(numext::real(s1));
  VERIFY_IS_APPROX(f2, FixedArrayType::Constant(numext::real(s1)));
  FixedArrayType f3((int)100*numext::real(s1));
  VERIFY_IS_APPROX(f3, FixedArrayType::Constant((int)100*numext::real(s1)));
  f1.setRandom();
  FixedArrayType f4(f1.data());
  VERIFY_IS_APPROX(f4, f1);
  
  // Check possible conflicts with 1D ctor
  typedef Array<Scalar, Dynamic, 1> OneDArrayType;
  OneDArrayType o1(rows);
  VERIFY(o1.size()==rows);
  OneDArrayType o4((int)rows);
  VERIFY(o4.size()==rows);
}

template<typename ArrayType> void comparisons(const ArrayType& m)
{
  using std::abs;
  typedef typename ArrayType::Index Index;
  typedef typename ArrayType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  Index rows = m.rows();
  Index cols = m.cols();

  Index r = internal::random<Index>(0, rows-1),
        c = internal::random<Index>(0, cols-1);

  ArrayType m1 = ArrayType::Random(rows, cols),
            m2 = ArrayType::Random(rows, cols),
            m3(rows, cols),
            m4 = m1;
  
  m4 = (m4.abs()==Scalar(0)).select(1,m4);

  VERIFY(((m1 + Scalar(1)) > m1).all());
  VERIFY(((m1 - Scalar(1)) < m1).all());
  if (rows*cols>1)
  {
    m3 = m1;
    m3(r,c) += 1;
    VERIFY(! (m1 < m3).all() );
    VERIFY(! (m1 > m3).all() );
  }
  VERIFY(!(m1 > m2 && m1 < m2).any());
  VERIFY((m1 <= m2 || m1 >= m2).all());

  // comparisons array to scalar
  VERIFY( (m1 != (m1(r,c)+1) ).any() );
  VERIFY( (m1 >  (m1(r,c)-1) ).any() );
  VERIFY( (m1 <  (m1(r,c)+1) ).any() );
  VERIFY( (m1 ==  m1(r,c)    ).any() );

  // comparisons scalar to array
  VERIFY( ( (m1(r,c)+1) != m1).any() );
  VERIFY( ( (m1(r,c)-1) <  m1).any() );
  VERIFY( ( (m1(r,c)+1) >  m1).any() );
  VERIFY( (  m1(r,c)    == m1).any() );

  // test Select
  VERIFY_IS_APPROX( (m1<m2).select(m1,m2), m1.cwiseMin(m2) );
  VERIFY_IS_APPROX( (m1>m2).select(m1,m2), m1.cwiseMax(m2) );
  Scalar mid = (m1.cwiseAbs().minCoeff() + m1.cwiseAbs().maxCoeff())/Scalar(2);
  for (int j=0; j<cols; ++j)
  for (int i=0; i<rows; ++i)
    m3(i,j) = abs(m1(i,j))<mid ? 0 : m1(i,j);
  VERIFY_IS_APPROX( (m1.abs()<ArrayType::Constant(rows,cols,mid))
                        .select(ArrayType::Zero(rows,cols),m1), m3);
  // shorter versions:
  VERIFY_IS_APPROX( (m1.abs()<ArrayType::Constant(rows,cols,mid))
                        .select(0,m1), m3);
  VERIFY_IS_APPROX( (m1.abs()>=ArrayType::Constant(rows,cols,mid))
                        .select(m1,0), m3);
  // even shorter version:
  VERIFY_IS_APPROX( (m1.abs()<mid).select(0,m1), m3);

  // count
  VERIFY(((m1.abs()+1)>RealScalar(0.1)).count() == rows*cols);

  // and/or
  VERIFY( (m1<RealScalar(0) && m1>RealScalar(0)).count() == 0);
  VERIFY( (m1<RealScalar(0) || m1>=RealScalar(0)).count() == rows*cols);
  RealScalar a = m1.abs().mean();
  VERIFY( (m1<-a || m1>a).count() == (m1.abs()>a).count());

  typedef Array<typename ArrayType::Index, Dynamic, 1> ArrayOfIndices;

  // TODO allows colwise/rowwise for array
  VERIFY_IS_APPROX(((m1.abs()+1)>RealScalar(0.1)).colwise().count(), ArrayOfIndices::Constant(cols,rows).transpose());
  VERIFY_IS_APPROX(((m1.abs()+1)>RealScalar(0.1)).rowwise().count(), ArrayOfIndices::Constant(rows, cols));
}

template<typename ArrayType> void array_real(const ArrayType& m)
{
  using std::abs;
  using std::sqrt;
  typedef typename ArrayType::Index Index;
  typedef typename ArrayType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  Index rows = m.rows();
  Index cols = m.cols();

  ArrayType m1 = ArrayType::Random(rows, cols),
            m2 = ArrayType::Random(rows, cols),
            m3(rows, cols),
            m4 = m1;

  m4 = (m4.abs()==Scalar(0)).select(1,m4);

  Scalar  s1 = internal::random<Scalar>();

  // these tests are mostly to check possible compilation issues with free-functions.
  VERIFY_IS_APPROX(m1.sin(), sin(m1));
  VERIFY_IS_APPROX(m1.cos(), cos(m1));
  VERIFY_IS_APPROX(m1.tan(), tan(m1));
  VERIFY_IS_APPROX(m1.asin(), asin(m1));
  VERIFY_IS_APPROX(m1.acos(), acos(m1));
  VERIFY_IS_APPROX(m1.atan(), atan(m1));
  VERIFY_IS_APPROX(m1.sinh(), sinh(m1));
  VERIFY_IS_APPROX(m1.cosh(), cosh(m1));
  VERIFY_IS_APPROX(m1.tanh(), tanh(m1));
#ifdef EIGEN_HAS_C99_MATH
  VERIFY_IS_APPROX(m1.lgamma(), lgamma(m1));
  VERIFY_IS_APPROX(m1.digamma(), digamma(m1));
  VERIFY_IS_APPROX(m1.erf(), erf(m1));
  VERIFY_IS_APPROX(m1.erfc(), erfc(m1));
#endif  // EIGEN_HAS_C99_MATH
  VERIFY_IS_APPROX(m1.arg(), arg(m1));
  VERIFY_IS_APPROX(m1.round(), round(m1));
  VERIFY_IS_APPROX(m1.floor(), floor(m1));
  VERIFY_IS_APPROX(m1.ceil(), ceil(m1));
  VERIFY((m1.isNaN() == (Eigen::isnan)(m1)).all());
  VERIFY((m1.isInf() == (Eigen::isinf)(m1)).all());
  VERIFY((m1.isFinite() == (Eigen::isfinite)(m1)).all());
  VERIFY_IS_APPROX(m1.inverse(), inverse(m1));
  VERIFY_IS_APPROX(m1.abs(), abs(m1));
  VERIFY_IS_APPROX(m1.abs2(), abs2(m1));
  VERIFY_IS_APPROX(m1.square(), square(m1));
  VERIFY_IS_APPROX(m1.cube(), cube(m1));
  VERIFY_IS_APPROX(cos(m1+RealScalar(3)*m2), cos((m1+RealScalar(3)*m2).eval()));
  VERIFY_IS_APPROX(m1.sign(), sign(m1));


  // avoid NaNs with abs() so verification doesn't fail
  m3 = m1.abs();
  VERIFY_IS_APPROX(m3.sqrt(), sqrt(abs(m1)));
  VERIFY_IS_APPROX(m3.rsqrt(), Scalar(1)/sqrt(abs(m1)));
  VERIFY_IS_APPROX(m3.log(), log(m3));
  VERIFY_IS_APPROX(m3.log10(), log10(m3));


  VERIFY((!(m1>m2) == (m1<=m2)).all());

  VERIFY_IS_APPROX(sin(m1.asin()), m1);
  VERIFY_IS_APPROX(cos(m1.acos()), m1);
  VERIFY_IS_APPROX(tan(m1.atan()), m1);
  VERIFY_IS_APPROX(sinh(m1), 0.5*(exp(m1)-exp(-m1)));
  VERIFY_IS_APPROX(cosh(m1), 0.5*(exp(m1)+exp(-m1)));
  VERIFY_IS_APPROX(tanh(m1), (0.5*(exp(m1)-exp(-m1)))/(0.5*(exp(m1)+exp(-m1))));
  VERIFY_IS_APPROX(arg(m1), ((m1<0).template cast<Scalar>())*std::acos(-1.0));
  VERIFY((round(m1) <= ceil(m1) && round(m1) >= floor(m1)).all());
  VERIFY((Eigen::isnan)((m1*0.0)/0.0).all());
  VERIFY((Eigen::isinf)(m4/0.0).all());
  VERIFY(((Eigen::isfinite)(m1) && (!(Eigen::isfinite)(m1*0.0/0.0)) && (!(Eigen::isfinite)(m4/0.0))).all());
  VERIFY_IS_APPROX(inverse(inverse(m1)),m1);
  VERIFY((abs(m1) == m1 || abs(m1) == -m1).all());
  VERIFY_IS_APPROX(m3, sqrt(abs2(m1)));
  VERIFY_IS_APPROX( m1.sign(), -(-m1).sign() );
  VERIFY_IS_APPROX( m1*m1.sign(),m1.abs());
  VERIFY_IS_APPROX(m1.sign() * m1.abs(), m1);

  VERIFY_IS_APPROX(numext::abs2(numext::real(m1)) + numext::abs2(numext::imag(m1)), numext::abs2(m1));
  VERIFY_IS_APPROX(numext::abs2(real(m1)) + numext::abs2(imag(m1)), numext::abs2(m1));
  if(!NumTraits<Scalar>::IsComplex)
    VERIFY_IS_APPROX(numext::real(m1), m1);

  // shift argument of logarithm so that it is not zero
  Scalar smallNumber = NumTraits<Scalar>::dummy_precision();
  VERIFY_IS_APPROX((m3 + smallNumber).log() , log(abs(m1) + smallNumber));

  VERIFY_IS_APPROX(m1.exp() * m2.exp(), exp(m1+m2));
  VERIFY_IS_APPROX(m1.exp(), exp(m1));
  VERIFY_IS_APPROX(m1.exp() / m2.exp(),(m1-m2).exp());

  VERIFY_IS_APPROX(m1.pow(2), m1.square());
  VERIFY_IS_APPROX(pow(m1,2), m1.square());
  VERIFY_IS_APPROX(m1.pow(3), m1.cube());
  VERIFY_IS_APPROX(pow(m1,3), m1.cube());
  VERIFY_IS_APPROX((-m1).pow(3), -m1.cube());
  VERIFY_IS_APPROX(pow(2*m1,3), 8*m1.cube());

  ArrayType exponents = ArrayType::Constant(rows, cols, RealScalar(2));
  VERIFY_IS_APPROX(Eigen::pow(m1,exponents), m1.square());
  VERIFY_IS_APPROX(m1.pow(exponents), m1.square());
  VERIFY_IS_APPROX(Eigen::pow(2*m1,exponents), 4*m1.square());
  VERIFY_IS_APPROX((2*m1).pow(exponents), 4*m1.square());
  VERIFY_IS_APPROX(Eigen::pow(m1,2*exponents), m1.square().square());
  VERIFY_IS_APPROX(m1.pow(2*exponents), m1.square().square());
  VERIFY_IS_APPROX(pow(m1(0,0), exponents), ArrayType::Constant(rows,cols,m1(0,0)*m1(0,0)));

  VERIFY_IS_APPROX(m3.pow(RealScalar(0.5)), m3.sqrt());
  VERIFY_IS_APPROX(pow(m3,RealScalar(0.5)), m3.sqrt());

  VERIFY_IS_APPROX(m3.pow(RealScalar(-0.5)), m3.rsqrt());
  VERIFY_IS_APPROX(pow(m3,RealScalar(-0.5)), m3.rsqrt());

  VERIFY_IS_APPROX(log10(m3), log(m3)/log(10));

  // scalar by array division
  const RealScalar tiny = sqrt(std::numeric_limits<RealScalar>::epsilon());
  s1 += Scalar(tiny);
  m1 += ArrayType::Constant(rows,cols,Scalar(tiny));
  VERIFY_IS_APPROX(s1/m1, s1 * m1.inverse());



#ifdef EIGEN_HAS_C99_MATH
  // check special functions (comparing against numpy implementation)
  if (!NumTraits<Scalar>::IsComplex)
  {

    {
      // Test various propreties of igamma & igammac.  These are normalized
      // gamma integrals where
      //   igammac(a, x) = Gamma(a, x) / Gamma(a)
      //   igamma(a, x) = gamma(a, x) / Gamma(a)
      // where Gamma and gamma are considered the standard unnormalized
      // upper and lower incomplete gamma functions, respectively.
      ArrayType a = m1.abs() + 2;
      ArrayType x = m2.abs() + 2;
      ArrayType zero = ArrayType::Zero(rows, cols);
      ArrayType one = ArrayType::Constant(rows, cols, Scalar(1.0));
      ArrayType a_m1 = a - one;
      ArrayType Gamma_a_x = Eigen::igammac(a, x) * a.lgamma().exp();
      ArrayType Gamma_a_m1_x = Eigen::igammac(a_m1, x) * a_m1.lgamma().exp();
      ArrayType gamma_a_x = Eigen::igamma(a, x) * a.lgamma().exp();
      ArrayType gamma_a_m1_x = Eigen::igamma(a_m1, x) * a_m1.lgamma().exp();

      // Gamma(a, 0) == Gamma(a)
      VERIFY_IS_APPROX(Eigen::igammac(a, zero), one);

      // Gamma(a, x) + gamma(a, x) == Gamma(a)
      VERIFY_IS_APPROX(Gamma_a_x + gamma_a_x, a.lgamma().exp());

      // Gamma(a, x) == (a - 1) * Gamma(a-1, x) + x^(a-1) * exp(-x)
      VERIFY_IS_APPROX(Gamma_a_x, (a - 1) * Gamma_a_m1_x + x.pow(a-1) * (-x).exp());

      // gamma(a, x) == (a - 1) * gamma(a-1, x) - x^(a-1) * exp(-x)
      VERIFY_IS_APPROX(gamma_a_x, (a - 1) * gamma_a_m1_x - x.pow(a-1) * (-x).exp());
    }

    // Check exact values of igamma and igammac against a third party calculation.
    Scalar a_s[] = {Scalar(0), Scalar(1), Scalar(1.5), Scalar(4), Scalar(0.0001), Scalar(1000.5)};
    Scalar x_s[] = {Scalar(0), Scalar(1), Scalar(1.5), Scalar(4), Scalar(0.0001), Scalar(1000.5)};

    // location i*6+j corresponds to a_s[i], x_s[j].
    Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
    Scalar igamma_s[][6] = {{0.0, nan, nan, nan, nan, nan},
                            {0.0, 0.6321205588285578, 0.7768698398515702,
                             0.9816843611112658, 9.999500016666262e-05, 1.0},
                            {0.0, 0.4275932955291202, 0.608374823728911,
                             0.9539882943107686, 7.522076445089201e-07, 1.0},
                            {0.0, 0.01898815687615381, 0.06564245437845008,
                             0.5665298796332909, 4.166333347221828e-18, 1.0},
                            {0.0, 0.9999780593618628, 0.9999899967080838,
                             0.9999996219837988, 0.9991370418689945, 1.0},
                            {0.0, 0.0, 0.0, 0.0, 0.0, 0.5042041932513908}};
    Scalar igammac_s[][6] = {{nan, nan, nan, nan, nan, nan},
                             {1.0, 0.36787944117144233, 0.22313016014842982,
                              0.018315638888734182, 0.9999000049998333, 0.0},
                             {1.0, 0.5724067044708798, 0.3916251762710878,
                              0.04601170568923136, 0.9999992477923555, 0.0},
                             {1.0, 0.9810118431238462, 0.9343575456215499,
                              0.4334701203667089, 1.0, 0.0},
                             {1.0, 2.1940638138146658e-05, 1.0003291916285e-05,
                              3.7801620118431334e-07, 0.0008629581310054535,
                              0.0},
                             {1.0, 1.0, 1.0, 1.0, 1.0, 0.49579580674813944}};
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        if ((std::isnan)(igamma_s[i][j])) {
          VERIFY((std::isnan)(numext::igamma(a_s[i], x_s[j])));
        } else {
          VERIFY_IS_APPROX(numext::igamma(a_s[i], x_s[j]), igamma_s[i][j]);
        }

        if ((std::isnan)(igammac_s[i][j])) {
          VERIFY((std::isnan)(numext::igammac(a_s[i], x_s[j])));
        } else {
          VERIFY_IS_APPROX(numext::igammac(a_s[i], x_s[j]), igammac_s[i][j]);
        }
      }
    }
  }
#endif  // EIGEN_HAS_C99_MATH

  // check inplace transpose
  m3 = m1;
  m3.transposeInPlace();
  VERIFY_IS_APPROX(m3, m1.transpose());
  m3.transposeInPlace();
  VERIFY_IS_APPROX(m3, m1);
}

template<typename ArrayType> void array_complex(const ArrayType& m)
{
  typedef typename ArrayType::Index Index;
  typedef typename ArrayType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  Index rows = m.rows();
  Index cols = m.cols();

  ArrayType m1 = ArrayType::Random(rows, cols),
            m2(rows, cols),
            m4 = m1;
  
  m4.real() = (m4.real().abs()==RealScalar(0)).select(RealScalar(1),m4.real());
  m4.imag() = (m4.imag().abs()==RealScalar(0)).select(RealScalar(1),m4.imag());

  Array<RealScalar, -1, -1> m3(rows, cols);

  for (Index i = 0; i < m.rows(); ++i)
    for (Index j = 0; j < m.cols(); ++j)
      m2(i,j) = sqrt(m1(i,j));

  // these tests are mostly to check possible compilation issues with free-functions.
  VERIFY_IS_APPROX(m1.sin(), sin(m1));
  VERIFY_IS_APPROX(m1.cos(), cos(m1));
  VERIFY_IS_APPROX(m1.tan(), tan(m1));
  VERIFY_IS_APPROX(m1.sinh(), sinh(m1));
  VERIFY_IS_APPROX(m1.cosh(), cosh(m1));
  VERIFY_IS_APPROX(m1.tanh(), tanh(m1));
  VERIFY_IS_APPROX(m1.arg(), arg(m1));
  VERIFY((m1.isNaN() == (Eigen::isnan)(m1)).all());
  VERIFY((m1.isInf() == (Eigen::isinf)(m1)).all());
  VERIFY((m1.isFinite() == (Eigen::isfinite)(m1)).all());
  VERIFY_IS_APPROX(m1.inverse(), inverse(m1));
  VERIFY_IS_APPROX(m1.log(), log(m1));
  VERIFY_IS_APPROX(m1.log10(), log10(m1));
  VERIFY_IS_APPROX(m1.abs(), abs(m1));
  VERIFY_IS_APPROX(m1.abs2(), abs2(m1));
  VERIFY_IS_APPROX(m1.sqrt(), sqrt(m1));
  VERIFY_IS_APPROX(m1.square(), square(m1));
  VERIFY_IS_APPROX(m1.cube(), cube(m1));
  VERIFY_IS_APPROX(cos(m1+RealScalar(3)*m2), cos((m1+RealScalar(3)*m2).eval()));
  VERIFY_IS_APPROX(m1.sign(), sign(m1));


  VERIFY_IS_APPROX(m1.exp() * m2.exp(), exp(m1+m2));
  VERIFY_IS_APPROX(m1.exp(), exp(m1));
  VERIFY_IS_APPROX(m1.exp() / m2.exp(),(m1-m2).exp());

  VERIFY_IS_APPROX(sinh(m1), 0.5*(exp(m1)-exp(-m1)));
  VERIFY_IS_APPROX(cosh(m1), 0.5*(exp(m1)+exp(-m1)));
  VERIFY_IS_APPROX(tanh(m1), (0.5*(exp(m1)-exp(-m1)))/(0.5*(exp(m1)+exp(-m1))));

  for (Index i = 0; i < m.rows(); ++i)
    for (Index j = 0; j < m.cols(); ++j)
      m3(i,j) = std::atan2(imag(m1(i,j)), real(m1(i,j)));
  VERIFY_IS_APPROX(arg(m1), m3);

  std::complex<RealScalar> zero(0.0,0.0);
  VERIFY((Eigen::isnan)(m1*zero/zero).all());
#if EIGEN_COMP_MSVC
  // msvc complex division is not robust
  VERIFY((Eigen::isinf)(m4/RealScalar(0)).all());
#else
#if EIGEN_COMP_CLANG
  // clang's complex division is notoriously broken too
  if((numext::isinf)(m4(0,0)/RealScalar(0))) {
#endif
    VERIFY((Eigen::isinf)(m4/zero).all());
#if EIGEN_COMP_CLANG
  }
  else
  {
    VERIFY((Eigen::isinf)(m4.real()/zero.real()).all());
  }
#endif
#endif // MSVC

  VERIFY(((Eigen::isfinite)(m1) && (!(Eigen::isfinite)(m1*zero/zero)) && (!(Eigen::isfinite)(m1/zero))).all());

  VERIFY_IS_APPROX(inverse(inverse(m1)),m1);
  VERIFY_IS_APPROX(conj(m1.conjugate()), m1);
  VERIFY_IS_APPROX(abs(m1), sqrt(square(real(m1))+square(imag(m1))));
  VERIFY_IS_APPROX(abs(m1), sqrt(abs2(m1)));
  VERIFY_IS_APPROX(log10(m1), log(m1)/log(10));

  VERIFY_IS_APPROX( m1.sign(), -(-m1).sign() );
  VERIFY_IS_APPROX( m1.sign() * m1.abs(), m1);

  // scalar by array division
  Scalar  s1 = internal::random<Scalar>();
  const RealScalar tiny = std::sqrt(std::numeric_limits<RealScalar>::epsilon());
  s1 += Scalar(tiny);
  m1 += ArrayType::Constant(rows,cols,Scalar(tiny));
  VERIFY_IS_APPROX(s1/m1, s1 * m1.inverse());

  // check inplace transpose
  m2 = m1;
  m2.transposeInPlace();
  VERIFY_IS_APPROX(m2, m1.transpose());
  m2.transposeInPlace();
  VERIFY_IS_APPROX(m2, m1);

}

template<typename ArrayType> void min_max(const ArrayType& m)
{
  typedef typename ArrayType::Index Index;
  typedef typename ArrayType::Scalar Scalar;

  Index rows = m.rows();
  Index cols = m.cols();

  ArrayType m1 = ArrayType::Random(rows, cols);

  // min/max with array
  Scalar maxM1 = m1.maxCoeff();
  Scalar minM1 = m1.minCoeff();

  VERIFY_IS_APPROX(ArrayType::Constant(rows,cols, minM1), (m1.min)(ArrayType::Constant(rows,cols, minM1)));
  VERIFY_IS_APPROX(m1, (m1.min)(ArrayType::Constant(rows,cols, maxM1)));

  VERIFY_IS_APPROX(ArrayType::Constant(rows,cols, maxM1), (m1.max)(ArrayType::Constant(rows,cols, maxM1)));
  VERIFY_IS_APPROX(m1, (m1.max)(ArrayType::Constant(rows,cols, minM1)));

  // min/max with scalar input
  VERIFY_IS_APPROX(ArrayType::Constant(rows,cols, minM1), (m1.min)( minM1));
  VERIFY_IS_APPROX(m1, (m1.min)( maxM1));

  VERIFY_IS_APPROX(ArrayType::Constant(rows,cols, maxM1), (m1.max)( maxM1));
  VERIFY_IS_APPROX(m1, (m1.max)( minM1));

}

template<typename X, typename Y>
void verify_component_wise(const X& x, const Y& y)
{
  for(Index i=0; i<x.size(); ++i)
  {
    if((numext::isfinite)(y(i)))
      VERIFY_IS_APPROX( x(i), y(i) );
    else if((numext::isnan)(y(i)))
      VERIFY((numext::isnan)(x(i)));
    else
      VERIFY_IS_EQUAL( x(i), y(i) );
  }
}

// check special functions (comparing against numpy implementation)
template<typename ArrayType> void array_special_functions()
{
  using std::abs;
  using std::sqrt;
  typedef typename ArrayType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;

  Scalar plusinf = std::numeric_limits<Scalar>::infinity();
  Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();

  // Check the zeta function against scipy.special.zeta
  {
    ArrayType x(7), q(7), res(7), ref(7);
    x << 1.5,   4, 10.5, 10000.5,    3, 1,        0.9;
    q << 2,   1.5,    3,  1.0001, -2.5, 1.2345, 1.2345;
    ref << 1.61237534869, 0.234848505667, 1.03086757337e-5, 0.367879440865, 0.054102025820864097, plusinf, nan;
    CALL_SUBTEST( verify_component_wise(ref, ref); );
    CALL_SUBTEST( res = x.zeta(q); verify_component_wise(res, ref); );
    CALL_SUBTEST( res = zeta(x,q); verify_component_wise(res, ref); );
  }

  // digamma
  {
    ArrayType x(7), res(7), ref(7);
    x << 1, 1.5, 4, -10.5, 10000.5, 0, -1;
    ref << -0.5772156649015329, 0.03648997397857645, 1.2561176684318, 2.398239129535781, 9.210340372392849, plusinf, plusinf;
    CALL_SUBTEST( verify_component_wise(ref, ref); );

    CALL_SUBTEST( res = x.digamma(); verify_component_wise(res, ref); );
    CALL_SUBTEST( res = digamma(x);  verify_component_wise(res, ref); );
  }


  {
    ArrayType n(11), x(11), res(11), ref(11);
    n << 1, 1,    1, 1.5,   17,   31,   28,    8, 42, 147, 170;
    x << 2, 3, 25.5, 1.5,  4.7, 11.8, 17.7, 30.2, 15.8, 54.1, 64;
    ref << 0.644934066848, 0.394934066848, 0.0399946696496, nan, 293.334565435, 0.445487887616, -2.47810300902e-07, -8.29668781082e-09, -0.434562276666, 0.567742190178, -0.0108615497927;
    CALL_SUBTEST( verify_component_wise(ref, ref); );

    if(sizeof(RealScalar)>=64) {
//       CALL_SUBTEST( res = x.polygamma(n); verify_component_wise(res, ref); );
      CALL_SUBTEST( res = polygamma(n,x);  verify_component_wise(res, ref); );
    }
    else {
//       CALL_SUBTEST( res = x.polygamma(n); verify_component_wise(res.head(8), ref.head(8)); );
      CALL_SUBTEST( res = polygamma(n,x); verify_component_wise(res.head(8), ref.head(8)); );
    }

  }
}

void test_array()
{
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( array(Array<float, 1, 1>()) );
    CALL_SUBTEST_2( array(Array22f()) );
    CALL_SUBTEST_3( array(Array44d()) );
    CALL_SUBTEST_4( array(ArrayXXcf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_5( array(ArrayXXf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_6( array(ArrayXXi(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
  }
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( comparisons(Array<float, 1, 1>()) );
    CALL_SUBTEST_2( comparisons(Array22f()) );
    CALL_SUBTEST_3( comparisons(Array44d()) );
    CALL_SUBTEST_5( comparisons(ArrayXXf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_6( comparisons(ArrayXXi(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
  }
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( min_max(Array<float, 1, 1>()) );
    CALL_SUBTEST_2( min_max(Array22f()) );
    CALL_SUBTEST_3( min_max(Array44d()) );
    CALL_SUBTEST_5( min_max(ArrayXXf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
    CALL_SUBTEST_6( min_max(ArrayXXi(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
  }
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1( array_real(Array<float, 1, 1>()) );
    CALL_SUBTEST_2( array_real(Array22f()) );
    CALL_SUBTEST_3( array_real(Array44d()) );
    CALL_SUBTEST_5( array_real(ArrayXXf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
  }
  for(int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_4( array_complex(ArrayXXcf(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
  }

  VERIFY((internal::is_same< internal::global_math_functions_filtering_base<int>::type, int >::value));
  VERIFY((internal::is_same< internal::global_math_functions_filtering_base<float>::type, float >::value));
  VERIFY((internal::is_same< internal::global_math_functions_filtering_base<Array2i>::type, ArrayBase<Array2i> >::value));
  typedef CwiseUnaryOp<internal::scalar_multiple_op<double>, ArrayXd > Xpr;
  VERIFY((internal::is_same< internal::global_math_functions_filtering_base<Xpr>::type,
                           ArrayBase<Xpr>
                         >::value));

  CALL_SUBTEST_7(array_special_functions<ArrayXf>());
  CALL_SUBTEST_7(array_special_functions<ArrayXd>());
}
