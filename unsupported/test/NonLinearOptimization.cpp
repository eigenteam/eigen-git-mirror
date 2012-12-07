// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Thomas Capricelli <orzel@freehackers.org>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
#include <stdio.h>

#include "main.h"
#include <unsupported/Eigen/NonLinearOptimization>

// This disables some useless Warnings on MSVC.
// It is intended to be done for this test only.
#include <Eigen/src/Core/util/DisableStupidWarnings.h>

using std::sqrt;

int fcn_chkder(const VectorXd &x, VectorXd &fvec, MatrixXd &fjac, int iflag)
{
    /*      subroutine fcn for chkder example. */

    int i;
    assert(15 ==  fvec.size());
    assert(3 ==  x.size());
    double tmp1, tmp2, tmp3, tmp4;
    static const double y[15]={1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1,
        3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34, 2.1, 4.39};


    if (iflag == 0)
        return 0;

    if (iflag != 2)
        for (i=0; i<15; i++) {
            tmp1 = i+1;
            tmp2 = 16-i-1;
            tmp3 = tmp1;
            if (i >= 8) tmp3 = tmp2;
            fvec[i] = y[i] - (x[0] + tmp1/(x[1]*tmp2 + x[2]*tmp3));
        }
    else {
        for (i = 0; i < 15; i++) {
            tmp1 = i+1;
            tmp2 = 16-i-1;

            /* error introduced into next statement for illustration. */
            /* corrected statement should read    tmp3 = tmp1 . */

            tmp3 = tmp2;
            if (i >= 8) tmp3 = tmp2;
            tmp4 = (x[1]*tmp2 + x[2]*tmp3); tmp4=tmp4*tmp4;
            fjac(i,0) = -1.;
            fjac(i,1) = tmp1*tmp2/tmp4;
            fjac(i,2) = tmp1*tmp3/tmp4;
        }
    }
    return 0;
}


void testChkder()
{
  const int m=15, n=3;
  VectorXd x(n), fvec(m), xp, fvecp(m), err;
  MatrixXd fjac(m,n);
  VectorXi ipvt;

  /*      the following values should be suitable for */
  /*      checking the jacobian matrix. */
  x << 9.2e-1, 1.3e-1, 5.4e-1;

  internal::chkder(x, fvec, fjac, xp, fvecp, 1, err);
  fcn_chkder(x, fvec, fjac, 1);
  fcn_chkder(x, fvec, fjac, 2);
  fcn_chkder(xp, fvecp, fjac, 1);
  internal::chkder(x, fvec, fjac, xp, fvecp, 2, err);

  fvecp -= fvec;

  // check those
  VectorXd fvec_ref(m), fvecp_ref(m), err_ref(m);
  fvec_ref <<
      -1.181606, -1.429655, -1.606344,
      -1.745269, -1.840654, -1.921586,
      -1.984141, -2.022537, -2.468977,
      -2.827562, -3.473582, -4.437612,
      -6.047662, -9.267761, -18.91806;
  fvecp_ref <<
      -7.724666e-09, -3.432406e-09, -2.034843e-10,
      2.313685e-09,  4.331078e-09,  5.984096e-09,
      7.363281e-09,   8.53147e-09,  1.488591e-08,
      2.33585e-08,  3.522012e-08,  5.301255e-08,
      8.26666e-08,  1.419747e-07,   3.19899e-07;
  err_ref <<
      0.1141397,  0.09943516,  0.09674474,
      0.09980447,  0.1073116, 0.1220445,
      0.1526814, 1, 1,
      1, 1, 1,
      1, 1, 1;

  VERIFY_IS_APPROX(fvec, fvec_ref);
  VERIFY_IS_APPROX(fvecp, fvecp_ref);
  VERIFY_IS_APPROX(err, err_ref);
}

// Generic functor
template<typename _Scalar, int NX=Dynamic, int NY=Dynamic>
struct Functor
{
  typedef _Scalar Scalar;
  enum {
    InputsAtCompileTime = NX,
    ValuesAtCompileTime = NY
  };
  typedef Matrix<Scalar,InputsAtCompileTime,1> InputType;
  typedef Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
  typedef Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

  const int m_inputs, m_values;

  Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

  // you should define that in the subclass :
//  void operator() (const InputType& x, ValueType* v, JacobianType* _j=0) const;
};

struct hybrj_functor : Functor<double>
{
    hybrj_functor(void) : Functor<double>(9,9) {}

    int operator()(const VectorXd &x, VectorXd &fvec)
    {
        double temp, temp1, temp2;
        const int n = x.size();
        assert(fvec.size()==n);
        for (int k = 0; k < n; k++)
        {
            temp = (3. - 2.*x[k])*x[k];
            temp1 = 0.;
            if (k) temp1 = x[k-1];
            temp2 = 0.;
            if (k != n-1) temp2 = x[k+1];
            fvec[k] = temp - temp1 - 2.*temp2 + 1.;
        }
        return 0;
    }
    int df(const VectorXd &x, MatrixXd &fjac)
    {
        const int n = x.size();
        assert(fjac.rows()==n);
        assert(fjac.cols()==n);
        for (int k = 0; k < n; k++)
        {
            for (int j = 0; j < n; j++)
                fjac(k,j) = 0.;
            fjac(k,k) = 3.- 4.*x[k];
            if (k) fjac(k,k-1) = -1.;
            if (k != n-1) fjac(k,k+1) = -2.;
        }
        return 0;
    }
};


void testHybrj1()
{
  const int n=9;
  int info;
  VectorXd x(n);

  /* the following starting values provide a rough fit. */
  x.setConstant(n, -1.);

  // do the computation
  hybrj_functor functor;
  HybridNonLinearSolver<hybrj_functor> solver(functor);
  info = solver.hybrj1(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(solver.nfev, 11);
  VERIFY_IS_EQUAL(solver.njev, 1);

  // check norm
  VERIFY_IS_APPROX(solver.fvec.blueNorm(), 1.192636e-08);


// check x
  VectorXd x_ref(n);
  x_ref <<
     -0.5706545,    -0.6816283,    -0.7017325,
     -0.7042129,     -0.701369,    -0.6918656,
     -0.665792,    -0.5960342,    -0.4164121;
  VERIFY_IS_APPROX(x, x_ref);
}

void testHybrj()
{
  const int n=9;
  int info;
  VectorXd x(n);

  /* the following starting values provide a rough fit. */
  x.setConstant(n, -1.);


  // do the computation
  hybrj_functor functor;
  HybridNonLinearSolver<hybrj_functor> solver(functor);
  solver.diag.setConstant(n, 1.);
  solver.useExternalScaling = true;
  info = solver.solve(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(solver.nfev, 11);
  VERIFY_IS_EQUAL(solver.njev, 1);

  // check norm
  VERIFY_IS_APPROX(solver.fvec.blueNorm(), 1.192636e-08);


// check x
  VectorXd x_ref(n);
  x_ref <<
     -0.5706545,    -0.6816283,    -0.7017325,
     -0.7042129,     -0.701369,    -0.6918656,
     -0.665792,    -0.5960342,    -0.4164121;
  VERIFY_IS_APPROX(x, x_ref);

}

struct hybrd_functor : Functor<double>
{
    hybrd_functor(void) : Functor<double>(9,9) {}
    int operator()(const VectorXd &x, VectorXd &fvec) const
    {
        double temp, temp1, temp2;
        const int n = x.size();

        assert(fvec.size()==n);
        for (int k=0; k < n; k++)
        {
            temp = (3. - 2.*x[k])*x[k];
            temp1 = 0.;
            if (k) temp1 = x[k-1];
            temp2 = 0.;
            if (k != n-1) temp2 = x[k+1];
            fvec[k] = temp - temp1 - 2.*temp2 + 1.;
        }
        return 0;
    }
};

void testHybrd1()
{
  int n=9, info;
  VectorXd x(n);

  /* the following starting values provide a rough solution. */
  x.setConstant(n, -1.);

  // do the computation
  hybrd_functor functor;
  HybridNonLinearSolver<hybrd_functor> solver(functor);
  info = solver.hybrd1(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(solver.nfev, 20);

  // check norm
  VERIFY_IS_APPROX(solver.fvec.blueNorm(), 1.192636e-08);

  // check x
  VectorXd x_ref(n);
  x_ref << -0.5706545, -0.6816283, -0.7017325, -0.7042129, -0.701369, -0.6918656, -0.665792, -0.5960342, -0.4164121;
  VERIFY_IS_APPROX(x, x_ref);
}

void testHybrd()
{
  const int n=9;
  int info;
  VectorXd x;

  /* the following starting values provide a rough fit. */
  x.setConstant(n, -1.);

  // do the computation
  hybrd_functor functor;
  HybridNonLinearSolver<hybrd_functor> solver(functor);
  solver.parameters.nb_of_subdiagonals = 1;
  solver.parameters.nb_of_superdiagonals = 1;
  solver.diag.setConstant(n, 1.);
  solver.useExternalScaling = true;
  info = solver.solveNumericalDiff(x);

  // check return value
  VERIFY_IS_EQUAL(info, 1);
  VERIFY_IS_EQUAL(solver.nfev, 14);

  // check norm
  VERIFY_IS_APPROX(solver.fvec.blueNorm(), 1.192636e-08);

  // check x
  VectorXd x_ref(n);
  x_ref <<
      -0.5706545,    -0.6816283,    -0.7017325,
      -0.7042129,     -0.701369,    -0.6918656,
      -0.665792,    -0.5960342,    -0.4164121;
  VERIFY_IS_APPROX(x, x_ref);
}



void test_NonLinearOptimization()
{
    // Tests using the examples provided by (c)minpack
    CALL_SUBTEST(testChkder());
    CALL_SUBTEST(testHybrj1());
    CALL_SUBTEST(testHybrj());
    CALL_SUBTEST(testHybrd1());
    CALL_SUBTEST(testHybrd());
}

/*
 * Can be useful for debugging...
  printf("info, nfev : %d, %d\n", info, lm.nfev);
  printf("info, nfev, njev : %d, %d, %d\n", info, solver.nfev, solver.njev);
  printf("info, nfev : %d, %d\n", info, solver.nfev);
  printf("x[0] : %.32g\n", x[0]);
  printf("x[1] : %.32g\n", x[1]);
  printf("x[2] : %.32g\n", x[2]);
  printf("x[3] : %.32g\n", x[3]);
  printf("fvec.blueNorm() : %.32g\n", solver.fvec.blueNorm());
  printf("fvec.blueNorm() : %.32g\n", lm.fvec.blueNorm());

  printf("info, nfev, njev : %d, %d, %d\n", info, lm.nfev, lm.njev);
  printf("fvec.squaredNorm() : %.13g\n", lm.fvec.squaredNorm());
  std::cout << x << std::endl;
  std::cout.precision(9);
  std::cout << x[0] << std::endl;
  std::cout << x[1] << std::endl;
  std::cout << x[2] << std::endl;
  std::cout << x[3] << std::endl;
*/

