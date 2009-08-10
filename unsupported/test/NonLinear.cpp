// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Thomas Capricelli <orzel@freehackers.org>

#include <stdio.h>

#include "main.h"
#include <unsupported/Eigen/NonLinear>

int fcn_chkder(int /*m*/, int /*n*/, const double *x, double *fvec, double *fjac, int ldfjac, int iflag)
{
  /*      subroutine fcn for chkder example. */

  int i;
  double tmp1, tmp2, tmp3, tmp4;
  double y[15]={1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1,
		3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34, 2.1, 4.39};

  
  if (iflag == 0) 
    {
      /*      insert print statements here when nprint is positive. */
      return 0;
    }

  if (iflag != 2) 

    for (i=1; i<=15; i++)
      {
	tmp1 = i;
	tmp2 = 16 - i;
	tmp3 = tmp1;
	if (i > 8) tmp3 = tmp2;
	fvec[i-1] = y[i-1] - (x[1-1] + tmp1/(x[2-1]*tmp2 + x[3-1]*tmp3));
      }
  else
    {
      for (i = 1; i <= 15; i++)
	{
	  tmp1 = i;
	  tmp2 = 16 - i;
	  
	  /* error introduced into next statement for illustration. */
	  /* corrected statement should read    tmp3 = tmp1 . */
	  
	  tmp3 = tmp2;
	  if (i > 8) tmp3 = tmp2;
	  tmp4 = (x[2-1]*tmp2 + x[3-1]*tmp3); tmp4=tmp4*tmp4;
	  fjac[i-1+ ldfjac*(1-1)] = -1.;
	  fjac[i-1+ ldfjac*(2-1)] = tmp1*tmp2/tmp4;
	  fjac[i-1+ ldfjac*(3-1)] = tmp1*tmp3/tmp4;
	}
    }
  return 0;
}

void testChkder()
{
  int i, m, n, ldfjac;
  double x[3], fvec[15], fjac[15*3], xp[3], fvecp[15], 
    err[15];

  m = 15;
  n = 3;

  /*      the following values should be suitable for */
  /*      checking the jacobian matrix. */

  x[1-1] = 9.2e-1;
  x[2-1] = 1.3e-1;
  x[3-1] = 5.4e-1;

  ldfjac = 15;

  chkder(m, n, x, fvec, fjac, ldfjac, xp, fvecp, 1, err);
  fcn_chkder(m, n, x, fvec, fjac, ldfjac, 1);
  fcn_chkder(m, n, x, fvec, fjac, ldfjac, 2);
  fcn_chkder(m, n, xp, fvecp, fjac, ldfjac, 1);
  chkder(m, n, x, fvec, fjac, ldfjac, xp, fvecp, 2, err);

  for (i=1; i<=m; i++)
    {
      fvecp[i-1] = fvecp[i-1] - fvec[i-1];
    }


  double fvec_ref[] = {
      -1.181606, -1.429655, -1.606344,
      -1.745269, -1.840654, -1.921586,
      -1.984141, -2.022537, -2.468977,
      -2.827562, -3.473582, -4.437612,
      -6.047662, -9.267761, -18.91806
  };
  double fvecp_ref[] = {
      -7.724666e-09, -3.432406e-09, -2.034843e-10,
      2.313685e-09,  4.331078e-09,  5.984096e-09,
      7.363281e-09,   8.53147e-09,  1.488591e-08,
      2.33585e-08,  3.522012e-08,  5.301255e-08,
      8.26666e-08,  1.419747e-07,   3.19899e-07
  };
  double err_ref[] = {
      0.1141397,  0.09943516,  0.09674474,
      0.09980447,  0.1073116, 0.1220445,
      0.1526814, 1, 1,
      1, 1, 1,
      1, 1, 1
  };

  for (i=1; i<=m; i++) VERIFY_IS_APPROX(fvec[i-1], fvec_ref[i-1]);
  for (i=1; i<=m; i++) VERIFY_IS_APPROX(fvecp[i-1], fvecp_ref[i-1]);
  for (i=1; i<=m; i++) VERIFY_IS_APPROX(err[i-1], err_ref[i-1]);
}


struct lmder1_functor {
    static int f(void * /*p*/, int /*m*/, int /*n*/, const double *x, double *fvec, double *fjac, int ldfjac, int iflag)
    {

        /*      subroutine fcn for lmder1 example. */

        int i;
        double tmp1, tmp2, tmp3, tmp4;
        double y[15] = {1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1,
            3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34, 2.1, 4.39};

        if (iflag != 2)
        {
            for (i = 1; i <= 15; i++)
            {
                tmp1 = i;
                tmp2 = 16 - i;
                tmp3 = tmp1;
                if (i > 8) tmp3 = tmp2;
                fvec[i-1] = y[i-1] - (x[1-1] + tmp1/(x[2-1]*tmp2 + x[3-1]*tmp3));
            }
        }
        else
        {
            for ( i = 1; i <= 15; i++)
            {
                tmp1 = i;
                tmp2 = 16 - i;
                tmp3 = tmp1;
                if (i > 8) tmp3 = tmp2;
                tmp4 = (x[2-1]*tmp2 + x[3-1]*tmp3); tmp4 = tmp4*tmp4;
                fjac[i-1 + ldfjac*(1-1)] = -1.;
                fjac[i-1 + ldfjac*(2-1)] = tmp1*tmp2/tmp4;
                fjac[i-1 + ldfjac*(3-1)] = tmp1*tmp3/tmp4;
            }
        }
        return 0;
    }
};


void testLmder1()
{
  int m=15, n=3, info;

  Eigen::VectorXd x(n), fvec(m);
  VectorXi ipvt;

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  info = ei_lmder1<lmder1_functor,double>(x, fvec, ipvt);

  // check return value
  VERIFY( 1 == info);

  // check norm
  VERIFY_IS_APPROX(fvec.norm(), 0.09063596);

  // check x
  VectorXd x_ref(n);
  x_ref << 0.08241058, 1.133037, 2.343695;
  VERIFY_IS_APPROX(x, x_ref);
}

struct lmder_functor {
    static int f(void * /*p*/, int /*m*/, int  /*n*/, const double *x, double *fvec, double *fjac, int ldfjac, int iflag)
    {      

        /*      subroutine fcn for lmder example. */

        int i;
        double tmp1, tmp2, tmp3, tmp4;
        double y[15]={1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1,
            3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34, 2.1, 4.39};

        if (iflag == 0) 
        {
            /*      insert print statements here when nprint is positive. */
            return 0;
        }

        if (iflag != 2) 
        {
            for (i=1; i <= 15; i++)
            {
                tmp1 = i;
                tmp2 = 16 - i;
                tmp3 = tmp1;
                if (i > 8) tmp3 = tmp2;
                fvec[i-1] = y[i-1] - (x[1-1] + tmp1/(x[2-1]*tmp2 + x[3-1]*tmp3));
            }
        }
        else
        {
            for (i=1; i<=15; i++)
            {
                tmp1 = i;
                tmp2 = 16 - i;
                tmp3 = tmp1;
                if (i > 8) tmp3 = tmp2;
                tmp4 = (x[2-1]*tmp2 + x[3-1]*tmp3); tmp4 = tmp4*tmp4;
                fjac[i-1 + ldfjac*(1-1)] = -1.;
                fjac[i-1 + ldfjac*(2-1)] = tmp1*tmp2/tmp4;
                fjac[i-1 + ldfjac*(3-1)] = tmp1*tmp3/tmp4;
            };
        }
        return 0;
    }
};

void testLmder()
{
  const int m=15, n=3;
  int info, nfev, njev;
  double fnorm, covfac, covar_ftol;
  Eigen::VectorXd x(n), fvec(m), diag(n), wa1;
  Eigen::MatrixXd fjac;
  VectorXi ipvt;

  /* the following starting values provide a rough fit. */
  x.setConstant(n, 1.);

  // do the computation
  info = ei_lmder<lmder_functor, double>(x, fvec, nfev, njev, fjac, ipvt, wa1, diag);

  // check return values
  VERIFY( 1 == info);
  VERIFY(nfev==6);
  VERIFY(njev==5);

  // check norm
  fnorm = fvec.norm();
  VERIFY_IS_APPROX(fnorm, 0.09063596);

  // check x
  VectorXd x_ref(n);
  x_ref << 0.08241058, 1.133037, 2.343695;
  VERIFY_IS_APPROX(x, x_ref);

  // check covariance
  covar_ftol = dpmpar(1);
  covfac = fnorm*fnorm/(m-n);
  covar(n, fjac.data(), m, ipvt.data(), covar_ftol, wa1.data());

  Eigen::MatrixXd cov_ref(n,n);
  cov_ref << 
      0.0001531202,   0.002869941,  -0.002656662,
      0.002869941,    0.09480935,   -0.09098995,
      -0.002656662,   -0.09098995,    0.08778727;

//  std::cout << fjac*covfac << std::endl;

  Eigen::MatrixXd cov;
  cov =  covfac*fjac.corner<n,n>(TopLeft);
  VERIFY_IS_APPROX( cov, cov_ref);
  // TODO: why isn't this allowed ? : 
  // VERIFY_IS_APPROX( covfac*fjac.corner<n,n>(TopLeft) , cov_ref);
}

int fcn_hybrj1(void * /*p*/, int n, const double *x, double *fvec, double *fjac, int ldfjac, 
	 int iflag)
{
  /*      subroutine fcn for hybrj1 example. */

  int j, k;
  double one=1, temp, temp1, temp2, three=3, two=2, zero=0, four=4;

  if (iflag != 2)
    {
      for (k = 1; k <= n; k++)
	{
	  temp = (three - two*x[k-1])*x[k-1];
	  temp1 = zero;
	  if (k != 1) temp1 = x[k-1-1];
	  temp2 = zero;
	  if (k != n) temp2 = x[k+1-1];
	  fvec[k-1] = temp - temp1 - two*temp2 + one;
	}
    }
  else
    {
     for (k = 1; k <= n; k++)
       {
	 for (j = 1; j <= n; j++)
	   {
	     fjac[k-1 + ldfjac*(j-1)] = zero;
	   }
         fjac[k-1 + ldfjac*(k-1)] = three - four*x[k-1];
         if (k != 1) fjac[k-1 + ldfjac*(k-1-1)] = -one;
         if (k != n) fjac[k-1 + ldfjac*(k+1-1)] = -two;
       }
    }
  return 0;
}


void testHybrj1()
{
  int j, n, ldfjac, info, lwa;
  double tol, fnorm;
  double x[9], fvec[9], fjac[9*9], wa[99];

  n = 9;

/*      the following starting values provide a rough solution. */

  for (j=1; j<=9; j++)
    {
      x[j-1] = -1.;
    }

  ldfjac = 9;
  lwa = 99;

/*      set tol to the square root of the machine precision. */
/*      unless high solutions are required, */
/*      this is the recommended setting. */

  tol = sqrt(dpmpar(1));

  info = hybrj1(fcn_hybrj1, 0, n, x, fvec, fjac, ldfjac, tol, wa, lwa);

  fnorm = enorm(n, fvec);

  VERIFY_IS_APPROX(fnorm, 1.192636e-08);
  VERIFY(info==1);
  double x_ref[] = { 
      -0.5706545,    -0.6816283,    -0.7017325,
      -0.7042129,     -0.701369,    -0.6918656,
      -0.665792,    -0.5960342,    -0.4164121
  };
  for (j=1; j<=n; j++) VERIFY_IS_APPROX(x[j-1], x_ref[j-1]);
}

int fcn_hybrj(void * /*p*/, int n, const double *x, double *fvec, double *fjac, int ldfjac, 
	 int iflag)
{
  
  /*      subroutine fcn for hybrj example. */

  int j, k;
  double one=1, temp, temp1, temp2, three=3, two=2, zero=0, four=4;

  if (iflag == 0)
    {
      /*      insert print statements here when nprint is positive. */
      return 0;
    }

  if (iflag != 2) 
    {
      for (k=1; k <= n; k++)
	{
	  temp = (three - two*x[k-1])*x[k-1];
	  temp1 = zero;
	  if (k != 1) temp1 = x[k-1-1];
	  temp2 = zero;
	  if (k != n) temp2 = x[k+1-1];
	  fvec[k-1] = temp - temp1 - two*temp2 + one;
	}
    }
  else
    {
      for (k = 1; k <= n; k++)
	{
	  for (j=1; j <= n; j++)
	    {
	      fjac[k-1 + ldfjac*(j-1)] = zero;
	    }
	  fjac[k-1 + ldfjac*(k-1)] = three - four*x[k-1];
	  if (k != 1) fjac[k-1 + ldfjac*(k-1-1)] = -one;
	  if (k != n) fjac[k-1 + ldfjac*(k+1-1)] = -two;
	}      
    }
  return 0;
}

void testHybrj()
{

  int j, n, ldfjac, maxfev, mode, nprint, info, nfev, njev, lr;
  double xtol, factor, fnorm;
  double x[9], fvec[9], fjac[9*9], diag[9], r[45], qtf[9],
    wa1[9], wa2[9], wa3[9], wa4[9];

  n = 9;

/*      the following starting values provide a rough solution. */

  for (j=1; j<=9; j++)
    {
      x[j-1] = -1.;
    }

  ldfjac = 9;
  lr = 45;

/*      set xtol to the square root of the machine precision. */
/*      unless high solutions are required, */
/*      this is the recommended setting. */

  xtol = sqrt(dpmpar(1));

  maxfev = 1000;
  mode = 2;
  for (j=1; j<=9; j++)
    {
      diag[j-1] = 1.;
    }
  factor = 1.e2;
  nprint = 0;

  info = hybrj(fcn_hybrj, 0, n, x, fvec, fjac, ldfjac, xtol, maxfev, diag, 
	mode, factor, nprint, &nfev, &njev, r, lr, qtf, 
	wa1, wa2, wa3, wa4);
 fnorm = enorm(n, fvec);

 VERIFY_IS_APPROX(fnorm, 1.192636e-08);
 VERIFY(nfev==11);
 VERIFY(njev==1);
 VERIFY(info==1);

 double x_ref[] = { 
     -0.5706545,    -0.6816283,    -0.7017325,
     -0.7042129,     -0.701369,    -0.6918656,
     -0.665792,    -0.5960342,    -0.4164121
 };
 for (j=1; j<=n; j++) VERIFY_IS_APPROX(x[j-1], x_ref[j-1]);
}

struct hybrd1_functor {
    static int f(void * /*p*/, int n, const double *x, double *fvec, int /*iflag*/)
    {
        /*      subroutine fcn for hybrd1 example. */

        int k;
        double one=1, temp, temp1, temp2, three=3, two=2, zero=0;

        for (k=1; k <= n; k++)
        {
            temp = (three - two*x[k-1])*x[k-1];
            temp1 = zero;
            if (k != 1) temp1 = x[k-1-1];
            temp2 = zero;
            if (k != n) temp2 = x[k+1-1];
            fvec[k-1] = temp - temp1 - two*temp2 + one;
        }
        return 0;
    }
};

void testHybrd1()
{
  int n=9, info;
  Eigen::VectorXd x(n), fvec(n);

  /* the following starting values provide a rough solution. */
  x.setConstant(n, -1.);

  // do the computation
  info = ei_hybrd1<hybrd1_functor,double>(x, fvec);

  // check return value
  VERIFY( 1 == info);

  // check norm
  VERIFY_IS_APPROX(fvec.norm(), 1.192636e-08);

  // check x
  VectorXd x_ref(n);
  x_ref << -0.5706545, -0.6816283, -0.7017325, -0.7042129, -0.701369, -0.6918656, -0.665792, -0.5960342, -0.4164121;
  VERIFY_IS_APPROX(x, x_ref);
}

struct hybrd_functor {
    static int f(void * /*p*/, int n, const double *x, double *fvec, int iflag)
    {
        /*      subroutine fcn for hybrd example. */

        int k;
        double one=1, temp, temp1, temp2, three=3, two=2, zero=0;

        if (iflag == 0)
        {
            /*      insert print statements here when nprint is positive. */
            return 0;
        }
        for (k=1; k<=n; k++)
        {

            temp = (three - two*x[k-1])*x[k-1];
            temp1 = zero;
            if (k != 1) temp1 = x[k-1-1];
            temp2 = zero;
            if (k != n) temp2 = x[k+1-1];
            fvec[k-1] = temp - temp1 - two*temp2 + one;
        }
        return 0;
    }
};

void testHybrd()
{
  const int n=9;
  int info, nfev, ml, mu, mode;
  Eigen::VectorXd x(n), fvec, diag(n), R, qtf;
  Eigen::MatrixXd fjac;
  VectorXi ipvt;

  /* the following starting values provide a rough fit. */
  x.setConstant(n, -1.);

  ml = 1;
  mu = 1;
  mode = 2;
  diag.setConstant(n, 1.);

  // do the computation
  info = ei_hybrd<hybrd_functor, double>(x,fvec, nfev, fjac, R, qtf, diag, mode, ml, mu);

  // check return value
  VERIFY( 1 == info);
  VERIFY(nfev==14);

  // check norm
  VERIFY_IS_APPROX(fvec.norm(), 1.192636e-08);

  // check x
  VectorXd x_ref(n);
  x_ref << 
      -0.5706545,    -0.6816283,    -0.7017325,
      -0.7042129,     -0.701369,    -0.6918656,
      -0.665792,    -0.5960342,    -0.4164121;
  VERIFY_IS_APPROX(x, x_ref);
}

int  fcn_lmstr1(void * /*p*/, int /*m*/, int /*n*/, const double *x, double *fvec, double *fjrow, int iflag)
{
  /*  subroutine fcn for lmstr1 example. */
  int i;
  double tmp1, tmp2, tmp3, tmp4;
  double y[15]={1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1,
		3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34, 2.1, 4.39};

  if (iflag < 2)
    {
      for (i=1; i<=15; i++)
	{
	  tmp1=i;
	  tmp2 = 16-i;
	  tmp3 = tmp1;
	  if (i > 8) tmp3 = tmp2;
	  fvec[i-1] = y[i-1] - (x[1-1] + tmp1/(x[2-1]*tmp2 + x[3-1]*tmp3));
	}
    }
  else
    {
      i = iflag - 1;
      tmp1 = i;
      tmp2 = 16 - i;
      tmp3 = tmp1;
      if (i > 8) tmp3 = tmp2;
      tmp4 = (x[2-1]*tmp2 + x[3-1]*tmp3); tmp4=tmp4*tmp4;
      fjrow[1-1] = -1;
      fjrow[2-1] = tmp1*tmp2/tmp4;
      fjrow[3-1] = tmp1*tmp3/tmp4;
    }
  return 0;
}

void testLmstr1()
{
  int m, n, ldfjac, info, lwa, ipvt[3];
  double tol, fnorm;
  double x[30], fvec[15], fjac[9], wa[30];

  m = 15;
  n = 3;

  /*     the following starting values provide a rough fit. */

  x[0] = 1.;
  x[1] = 1.;
  x[2] = 1.;

  ldfjac = 3;
  lwa = 30;

  /*     set tol to the square root of the machine precision.
     unless high precision solutions are required,
     this is the recommended setting. */

  tol = sqrt(dpmpar(1));

  info = lmstr1(fcn_lmstr1, 0, m, n, 
	  x, fvec, fjac, ldfjac, 
	  tol, ipvt, wa, lwa);

  fnorm = enorm(m, fvec);

  VERIFY_IS_APPROX(fnorm, 0.09063596);
  VERIFY(info==1);
  double x_ref[] = {0.08241058, 1.133037, 2.343695 };
  for (m=1; m<=3; m++) VERIFY_IS_APPROX(x[m-1], x_ref[m-1]);
}

int fcn_lmstr(void * /*p*/, int /*m*/, int /*n*/, const double *x, double *fvec, double *fjrow, int iflag)
{

  /*      subroutine fcn for lmstr example. */

  int i;
  double tmp1, tmp2, tmp3, tmp4;
  double y[15]={1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1,
		3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34, 2.1, 4.39};

  if (iflag == 0)
    {
      /*      insert print statements here when nprint is positive. */
      return 0;
    }
  if (iflag < 2)
    {
      for (i = 1; i <= 15; i++)
	{
	  tmp1 = i;
	  tmp2 = 16 - i;
	  tmp3 = tmp1;
	  if (i > 8) tmp3 = tmp2;
	  fvec[i-1] = y[i-1] - (x[1-1] + tmp1/(x[2-1]*tmp2 + x[3-1]*tmp3));
}
			 }
else
  {
    i = iflag - 1;
    tmp1 = i;
    tmp2 = 16 - i;
    tmp3 = tmp1;
    if (i > 8) tmp3 = tmp2;
    tmp4 = (x[2-1]*tmp2 + x[3-1]*tmp3); tmp4 = tmp4*tmp4;
    fjrow[1-1] = -1.;
    fjrow[2-1] = tmp1*tmp2/tmp4;
    fjrow[3-1] = tmp1*tmp3/tmp4;
  }
  return 0;
}

void testLmstr()
{
  int j, m, n, ldfjac, maxfev, mode, nprint, info, nfev, njev;
  int ipvt[3];
  double ftol, xtol, gtol, factor, fnorm;
  double x[3], fvec[15], fjac[3*3], diag[3], qtf[3], 
    wa1[3], wa2[3], wa3[3], wa4[15];

  m = 15;
  n = 3;

  /*      the following starting values provide a rough fit. */

  x[1-1] = 1.;
  x[2-1] = 1.;
  x[3-1] = 1.;

  ldfjac = 3;

  /*      set ftol and xtol to the square root of the machine */
  /*      and gtol to zero. unless high solutions are */
  /*      required, these are the recommended settings. */

  ftol = sqrt(dpmpar(1));
  xtol = sqrt(dpmpar(1));
  gtol = 0.;

  maxfev = 400;
  mode = 1;
  factor = 1.e2;
  nprint = 0;

  info = lmstr(fcn_lmstr, 0, m, n, x, fvec, fjac, ldfjac, ftol, xtol, gtol, 
	maxfev, diag, mode, factor, nprint, &nfev, &njev, 
	ipvt, qtf, wa1, wa2, wa3, wa4);
  fnorm = enorm(m, fvec);

  VERIFY_IS_APPROX(fnorm, 0.09063596);
  VERIFY(nfev==6);
  VERIFY(njev==5);
  VERIFY(info==1);

  double x_ref[] = {0.08241058, 1.133037, 2.343695 };
  for (j=1; j<=n; j++) VERIFY_IS_APPROX(x[j-1], x_ref[j-1]);
}

int fcn_lmdif1(void * /*p*/, int /*m*/, int /*n*/, const double *x, double *fvec, int /*iflag*/)
{
  /* function fcn for lmdif1 example */

  int i;
  double tmp1,tmp2,tmp3;
  double y[15]={1.4e-1,1.8e-1,2.2e-1,2.5e-1,2.9e-1,3.2e-1,3.5e-1,3.9e-1,
		3.7e-1,5.8e-1,7.3e-1,9.6e-1,1.34e0,2.1e0,4.39e0};

  for (i=0; i<15; i++)
    {
      tmp1 = i+1;
      tmp2 = 15 - i;
      tmp3 = tmp1;
      
      if (i >= 8) tmp3 = tmp2;
      fvec[i] = y[i] - (x[0] + tmp1/(x[1]*tmp2 + x[2]*tmp3));
    }
  return 0;
}

void testLmdif1()
{
  int m, n, info, lwa, iwa[3];
  double tol, fnorm, x[3], fvec[15], wa[75];

  m = 15;
  n = 3;

  /* the following starting values provide a rough fit. */

  x[0] = 1.e0;
  x[1] = 1.e0;
  x[2] = 1.e0;

  lwa = 75;

  /* set tol to the square root of the machine precision.  unless high
     precision solutions are required, this is the recommended
     setting. */

  tol = sqrt(dpmpar(1));

  info = lmdif1(fcn_lmdif1, 0, m, n, x, fvec, tol, iwa, wa, lwa);

  fnorm = enorm(m, fvec);

  VERIFY_IS_APPROX(fnorm, 0.09063596);
  VERIFY(info==1);
  double x_ref[] = {0.0824106, 1.1330366, 2.3436947 };
  int j;
  for (j=1; j<=n; j++) VERIFY_IS_APPROX(x[j-1], x_ref[j-1]);

}

int fcn_lmdif(void * /*p*/, int /*m*/, int /*n*/, const double *x, double *fvec, int iflag)
{

/*      subroutine fcn for lmdif example. */

  int i;
  double tmp1, tmp2, tmp3;
  double y[15]={1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1,
		3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34, 2.1, 4.39};

  if (iflag == 0)
    {
      /*      insert print statements here when nprint is positive. */
      return 0;
    }
  for (i = 1; i <= 15; i++)
    {
      tmp1 = i;
      tmp2 = 16 - i;
      tmp3 = tmp1;
      if (i > 8) tmp3 = tmp2;
      fvec[i-1] = y[i-1] - (x[1-1] + tmp1/(x[2-1]*tmp2 + x[3-1]*tmp3));
    }
  return 0;
}

void testLmdif()
{
  int i, j, m, n, maxfev, mode, nprint, info, nfev, ldfjac;
  int ipvt[3];
  double ftol, xtol, gtol, epsfcn, factor, fnorm;
  double x[3], fvec[15], diag[3], fjac[15*3], qtf[3], 
    wa1[3], wa2[3], wa3[3], wa4[15];
  double covfac;

  m = 15;
  n = 3;

/*      the following starting values provide a rough fit. */

  x[1-1] = 1.;
  x[2-1] = 1.;
  x[3-1] = 1.;

  ldfjac = 15;

  /*      set ftol and xtol to the square root of the machine */
  /*      and gtol to zero. unless high solutions are */
  /*      required, these are the recommended settings. */

  ftol = sqrt(dpmpar(1));
  xtol = sqrt(dpmpar(1));
  gtol = 0.;

  maxfev = 800;
  epsfcn = 0.;
  mode = 1;
  factor = 1.e2;
  nprint = 0;

  info = lmdif(fcn_lmdif, 0, m, n, x, fvec, ftol, xtol, gtol, maxfev, epsfcn, 
	 diag, mode, factor, nprint, &nfev, fjac, ldfjac, 
	 ipvt, qtf, wa1, wa2, wa3, wa4);

  fnorm = enorm(m, fvec);

  VERIFY_IS_APPROX(fnorm, 0.09063596);
  VERIFY(nfev==21);
  VERIFY(info==1);

  double x_ref[] = {0.08241058, 1.133037, 2.343695 };
  for (j=1; j<=n; j++) VERIFY_IS_APPROX(x[j-1], x_ref[j-1]);

  ftol = dpmpar(1);
  covfac = fnorm*fnorm/(m-n);
  covar(n, fjac, ldfjac, ipvt, ftol, wa1);

  double cov_ref[] = { 
      0.0001531202,   0.002869942,  -0.002656662,
      0.002869942,    0.09480937,   -0.09098997,
      -0.002656662,   -0.09098997,    0.08778729
  };

  for (i=1; i<=n; i++)
    for (j=1; j<=n; j++)
        VERIFY_IS_APPROX(fjac[(i-1)*ldfjac+j-1]*covfac, cov_ref[(i-1)*3+(j-1)]);
}

struct misra1a_functor {
    static int f(void * /*p*/, int m, int n, const double *b, double *fvec, double *fjac, int ldfjac, int iflag)
    {
        static const double x[14] = { 77.6E0, 114.9E0, 141.1E0, 190.8E0, 239.9E0, 289.0E0, 332.8E0, 378.4E0, 434.8E0, 477.3E0, 536.8E0, 593.1E0, 689.1E0, 760.0E0};
        static const double y[14] = { 10.07E0, 14.73E0, 17.94E0, 23.93E0, 29.61E0, 35.18E0, 40.02E0, 44.82E0, 50.76E0, 55.05E0, 61.01E0, 66.40E0, 75.47E0, 81.78E0};
        int i;

        assert(m==14);
        assert(n==2);
        assert(ldfjac==14);
        if (iflag != 2) {// compute fvec at b
            for(i=0; i<14; i++) {
                fvec[i] = b[0]*(1.-exp(-b[1]*x[i])) - y[i] ;
            }
        }
        else { // compute fjac at b
            for(i=0; i<14; i++) {
                fjac[i+ldfjac*0] = (1.-exp(-b[1]*x[i]));
                fjac[i+ldfjac*1] = (b[0]*x[i]*exp(-b[1]*x[i]));
            }
        }
        return 0;
    }
};

// http://www.itl.nist.gov/div898/strd/nls/data/misra1a.shtml
void testNistMisra1a(void)
{
  const int m=14, n=2;
  int info, nfev, njev;

  Eigen::VectorXd x(n), fvec(m), wa1, diag;
  Eigen::MatrixXd fjac;
  VectorXi ipvt;

  /*
   * First try
   */
  x<< 500., 0.0001;
  // do the computation
  info = ei_lmder<misra1a_functor, double>(x, fvec, nfev, njev, fjac, ipvt, wa1, diag);

  // check return value
  VERIFY( 1 == info); 
  VERIFY( 19 == nfev); 
  VERIFY( 15 == njev); 
  // check norm^2
  VERIFY_IS_APPROX(fvec.squaredNorm(), 1.2455138894E-01); 
  // check x
  VERIFY_IS_APPROX(x[0], 2.3894212918E+02);
  VERIFY_IS_APPROX(x[1], 5.5015643181E-04);

  /*
   * Second try
   */
  x<< 250., 0.0005;
  // do the computation
  info = ei_lmder<misra1a_functor, double>(x, fvec, nfev, njev, fjac, ipvt, wa1, diag);

  // check return value
  VERIFY( 1 == info); 
  VERIFY( 5 == nfev); 
  VERIFY( 4 == njev); 
  // check norm^2
  VERIFY_IS_APPROX(fvec.squaredNorm(), 1.2455138894E-01); 
  // check x
  VERIFY_IS_APPROX(x[0], 2.3894212918E+02);
  VERIFY_IS_APPROX(x[1], 5.5015643181E-04);
}

struct hahn1_functor {
    static int f(void * /*p*/, int m, int n, const double *b, double *fvec, double *fjac, int ldfjac, int iflag)
    {
        static const double _x[236] = { .591E0 , 1.547E0 , 2.902E0 , 2.894E0 , 4.703E0 , 6.307E0 , 7.03E0  , 7.898E0 , 9.470E0 , 9.484E0 , 10.072E0 , 10.163E0 , 11.615E0 , 12.005E0 , 12.478E0 , 12.982E0 , 12.970E0 , 13.926E0 , 14.452E0 , 14.404E0 , 15.190E0 , 15.550E0 , 15.528E0 , 15.499E0 , 16.131E0 , 16.438E0 , 16.387E0 , 16.549E0 , 16.872E0 , 16.830E0 , 16.926E0 , 16.907E0 , 16.966E0 , 17.060E0 , 17.122E0 , 17.311E0 , 17.355E0 , 17.668E0 , 17.767E0 , 17.803E0 , 17.765E0 , 17.768E0 , 17.736E0 , 17.858E0 , 17.877E0 , 17.912E0 , 18.046E0 , 18.085E0 , 18.291E0 , 18.357E0 , 18.426E0 , 18.584E0 , 18.610E0 , 18.870E0 , 18.795E0 , 19.111E0 , .367E0 , .796E0 , 0.892E0 , 1.903E0 , 2.150E0 , 3.697E0 , 5.870E0 , 6.421E0 , 7.422E0 , 9.944E0 , 11.023E0 , 11.87E0  , 12.786E0 , 14.067E0 , 13.974E0 , 14.462E0 , 14.464E0 , 15.381E0 , 15.483E0 , 15.59E0  , 16.075E0 , 16.347E0 , 16.181E0 , 16.915E0 , 17.003E0 , 16.978E0 , 17.756E0 , 17.808E0 , 17.868E0 , 18.481E0 , 18.486E0 , 19.090E0 , 16.062E0 , 16.337E0 , 16.345E0 , 16.388E0 , 17.159E0 , 17.116E0 , 17.164E0 , 17.123E0 , 17.979E0 , 17.974E0 , 18.007E0 , 17.993E0 , 18.523E0 , 18.669E0 , 18.617E0 , 19.371E0 , 19.330E0 , 0.080E0 , 0.248E0 , 1.089E0 , 1.418E0 , 2.278E0 , 3.624E0 , 4.574E0 , 5.556E0 , 7.267E0 , 7.695E0 , 9.136E0 , 9.959E0 , 9.957E0 , 11.600E0 , 13.138E0 , 13.564E0 , 13.871E0 , 13.994E0 , 14.947E0 , 15.473E0 , 15.379E0 , 15.455E0 , 15.908E0 , 16.114E0 , 17.071E0 , 17.135E0 , 17.282E0 , 17.368E0 , 17.483E0 , 17.764E0 , 18.185E0 , 18.271E0 , 18.236E0 , 18.237E0 , 18.523E0 , 18.627E0 , 18.665E0 , 19.086E0 , 0.214E0 , 0.943E0 , 1.429E0 , 2.241E0 , 2.951E0 , 3.782E0 , 4.757E0 , 5.602E0 , 7.169E0 , 8.920E0 , 10.055E0 , 12.035E0 , 12.861E0 , 13.436E0 , 14.167E0 , 14.755E0 , 15.168E0 , 15.651E0 , 15.746E0 , 16.216E0 , 16.445E0 , 16.965E0 , 17.121E0 , 17.206E0 , 17.250E0 , 17.339E0 , 17.793E0 , 18.123E0 , 18.49E0  , 18.566E0 , 18.645E0 , 18.706E0 , 18.924E0 , 19.1E0   , 0.375E0 , 0.471E0 , 1.504E0 , 2.204E0 , 2.813E0 , 4.765E0 , 9.835E0 , 10.040E0 , 11.946E0 , 12.596E0 , 13.303E0 , 13.922E0 , 14.440E0 , 14.951E0 , 15.627E0 , 15.639E0 , 15.814E0 , 16.315E0 , 16.334E0 , 16.430E0 , 16.423E0 , 17.024E0 , 17.009E0 , 17.165E0 , 17.134E0 , 17.349E0 , 17.576E0 , 17.848E0 , 18.090E0 , 18.276E0 , 18.404E0 , 18.519E0 , 19.133E0 , 19.074E0 , 19.239E0 , 19.280E0 , 19.101E0 , 19.398E0 , 19.252E0 , 19.89E0  , 20.007E0 , 19.929E0 , 19.268E0 , 19.324E0 , 20.049E0 , 20.107E0 , 20.062E0 , 20.065E0 , 19.286E0 , 19.972E0 , 20.088E0 , 20.743E0 , 20.83E0  , 20.935E0 , 21.035E0 , 20.93E0  , 21.074E0 , 21.085E0 , 20.935E0 };
        static const double _y[236] = { 24.41E0 , 34.82E0 , 44.09E0 , 45.07E0 , 54.98E0 , 65.51E0 , 70.53E0 , 75.70E0 , 89.57E0 , 91.14E0 , 96.40E0 , 97.19E0 , 114.26E0 , 120.25E0 , 127.08E0 , 133.55E0 , 133.61E0 , 158.67E0 , 172.74E0 , 171.31E0 , 202.14E0 , 220.55E0 , 221.05E0 , 221.39E0 , 250.99E0 , 268.99E0 , 271.80E0 , 271.97E0 , 321.31E0 , 321.69E0 , 330.14E0 , 333.03E0 , 333.47E0 , 340.77E0 , 345.65E0 , 373.11E0 , 373.79E0 , 411.82E0 , 419.51E0 , 421.59E0 , 422.02E0 , 422.47E0 , 422.61E0 , 441.75E0 , 447.41E0 , 448.7E0  , 472.89E0 , 476.69E0 , 522.47E0 , 522.62E0 , 524.43E0 , 546.75E0 , 549.53E0 , 575.29E0 , 576.00E0 , 625.55E0 , 20.15E0 , 28.78E0 , 29.57E0 , 37.41E0 , 39.12E0 , 50.24E0 , 61.38E0 , 66.25E0 , 73.42E0 , 95.52E0 , 107.32E0 , 122.04E0 , 134.03E0 , 163.19E0 , 163.48E0 , 175.70E0 , 179.86E0 , 211.27E0 , 217.78E0 , 219.14E0 , 262.52E0 , 268.01E0 , 268.62E0 , 336.25E0 , 337.23E0 , 339.33E0 , 427.38E0 , 428.58E0 , 432.68E0 , 528.99E0 , 531.08E0 , 628.34E0 , 253.24E0 , 273.13E0 , 273.66E0 , 282.10E0 , 346.62E0 , 347.19E0 , 348.78E0 , 351.18E0 , 450.10E0 , 450.35E0 , 451.92E0 , 455.56E0 , 552.22E0 , 553.56E0 , 555.74E0 , 652.59E0 , 656.20E0 , 14.13E0 , 20.41E0 , 31.30E0 , 33.84E0 , 39.70E0 , 48.83E0 , 54.50E0 , 60.41E0 , 72.77E0 , 75.25E0 , 86.84E0 , 94.88E0 , 96.40E0 , 117.37E0 , 139.08E0 , 147.73E0 , 158.63E0 , 161.84E0 , 192.11E0 , 206.76E0 , 209.07E0 , 213.32E0 , 226.44E0 , 237.12E0 , 330.90E0 , 358.72E0 , 370.77E0 , 372.72E0 , 396.24E0 , 416.59E0 , 484.02E0 , 495.47E0 , 514.78E0 , 515.65E0 , 519.47E0 , 544.47E0 , 560.11E0 , 620.77E0 , 18.97E0 , 28.93E0 , 33.91E0 , 40.03E0 , 44.66E0 , 49.87E0 , 55.16E0 , 60.90E0 , 72.08E0 , 85.15E0 , 97.06E0 , 119.63E0 , 133.27E0 , 143.84E0 , 161.91E0 , 180.67E0 , 198.44E0 , 226.86E0 , 229.65E0 , 258.27E0 , 273.77E0 , 339.15E0 , 350.13E0 , 362.75E0 , 371.03E0 , 393.32E0 , 448.53E0 , 473.78E0 , 511.12E0 , 524.70E0 , 548.75E0 , 551.64E0 , 574.02E0 , 623.86E0 , 21.46E0 , 24.33E0 , 33.43E0 , 39.22E0 , 44.18E0 , 55.02E0 , 94.33E0 , 96.44E0 , 118.82E0 , 128.48E0 , 141.94E0 , 156.92E0 , 171.65E0 , 190.00E0 , 223.26E0 , 223.88E0 , 231.50E0 , 265.05E0 , 269.44E0 , 271.78E0 , 273.46E0 , 334.61E0 , 339.79E0 , 349.52E0 , 358.18E0 , 377.98E0 , 394.77E0 , 429.66E0 , 468.22E0 , 487.27E0 , 519.54E0 , 523.03E0 , 612.99E0 , 638.59E0 , 641.36E0 , 622.05E0 , 631.50E0 , 663.97E0 , 646.9E0  , 748.29E0 , 749.21E0 , 750.14E0 , 647.04E0 , 646.89E0 , 746.9E0  , 748.43E0 , 747.35E0 , 749.27E0 , 647.61E0 , 747.78E0 , 750.51E0 , 851.37E0 , 845.97E0 , 847.54E0 , 849.93E0 , 851.61E0 , 849.75E0 , 850.98E0 , 848.23E0};
        int i;

//        static int called=0; printf("call hahn1_functor with  iflag=%d, called=%d\n", iflag, called); if (iflag==1) called++;
        assert(m==236);
        assert(n==7);
        assert(ldfjac==236);
        if (iflag != 2) {// compute fvec at x
            for(i=0; i<236; i++) {
                double x=_x[i], xx=x*x, xxx=xx*x;
                fvec[i] = (b[0]+b[1]*x+b[2]*xx+b[3]*xxx) / (1.+b[4]*x+b[5]*xx+b[6]*xxx) - _y[i];
            }
        }
        else { // compute fjac at x
            for(i=0; i<236; i++) {
                double x=_x[i], xx=x*x, xxx=xx*x;
                double fact = 1./(1.+b[4]*x+b[5]*xx+b[6]*xxx);
                fjac[i+ldfjac*0] = 1.*fact;
                fjac[i+ldfjac*1] = x*fact;
                fjac[i+ldfjac*2] = xx*fact;
                fjac[i+ldfjac*3] = xxx*fact;
                fact = - (b[0]+b[1]*x+b[2]*xx+b[3]*xxx) * fact * fact;
                fjac[i+ldfjac*4] = x*fact;
                fjac[i+ldfjac*5] = xx*fact;
                fjac[i+ldfjac*6] = xxx*fact;
            }
        }
        return 0;
    }
};

// http://www.itl.nist.gov/div898/strd/nls/data/hahn1.shtml
void testNistHahn1(void)
{
  const int m=236, n=7;
  int info, nfev, njev;

  Eigen::VectorXd x(n), fvec(m), wa1, diag;
  Eigen::MatrixXd fjac;
  VectorXi ipvt;

  /*
   * First try
   */
  x<< 10., -1., .05, -.00001, -.05, .001, -.000001;
  // do the computation
  info = ei_lmder<hahn1_functor, double>(x, fvec, nfev, njev, fjac, ipvt, wa1, diag);

  // check return value
  printf("info=%d, f,j: %d, %d\n", info, nfev, njev);
  printf("norm2 =  %.50g\n", fvec.squaredNorm());
  std::cout << x << std::endl;
  VERIFY( 1 == info); 
  VERIFY( 33== nfev); 
  VERIFY( 22== njev); 
  // check norm^2
  VERIFY_IS_APPROX(fvec.squaredNorm(), 1.5324382854E+00); 
  // check x
  VERIFY_IS_APPROX(x[0], 1.0776351733E+00  );
  VERIFY_IS_APPROX(x[1],-1.2269296921E-01  );
  VERIFY_IS_APPROX(x[2], 4.0863750610E-03  );
  VERIFY_IS_APPROX(x[3],-1.4262662514E-06  );
  VERIFY_IS_APPROX(x[4],-5.7609940901E-03  );
  VERIFY_IS_APPROX(x[5], 2.4053735503E-04  );
  VERIFY_IS_APPROX(x[6],-1.2314450199E-07  );

  /*
   * Second try
   */
  x<< .1, -.1, .005, -.000001, -.005, .0001, -.0000001;
  // do the computation
  info = ei_lmder<misra1a_functor, double>(x, fvec, nfev, njev, fjac, ipvt, wa1, diag);

  // check return value
  printf("info=%d, f,j: %d, %d\n", info, nfev, njev);
  VERIFY( 1 == info); 
  VERIFY( 5 == nfev); 
  VERIFY( 4 == njev); 
  // check norm^2
  VERIFY_IS_APPROX(fvec.squaredNorm(), 1.2455138894E-01); 
  VERIFY_IS_APPROX(fvec.squaredNorm(), 1.5324382854E+00); 
  // check x
  VERIFY_IS_APPROX(x[0], 1.0776351733E+00  );
  VERIFY_IS_APPROX(x[1],-1.2269296921E-01  );
  VERIFY_IS_APPROX(x[2], 4.0863750610E-03  );
  VERIFY_IS_APPROX(x[3],-1.4262662514E-06  );
  VERIFY_IS_APPROX(x[4],-5.7609940901E-03  );
  VERIFY_IS_APPROX(x[5], 2.4053735503E-04  );
  VERIFY_IS_APPROX(x[6],-1.2314450199E-07  );

}

void test_NonLinear()
{
  CALL_SUBTEST(testNistMisra1a());
//  CALL_SUBTEST(testNistHahn1());

  CALL_SUBTEST(testChkder());
  CALL_SUBTEST(testLmder1());
  CALL_SUBTEST(testLmder());
  CALL_SUBTEST(testHybrj1());
  CALL_SUBTEST(testHybrj());
  CALL_SUBTEST(testHybrd1());
  CALL_SUBTEST(testHybrd());
  CALL_SUBTEST(testLmstr1());
  CALL_SUBTEST(testLmstr());
  CALL_SUBTEST(testLmdif1());
  CALL_SUBTEST(testLmdif());
}

