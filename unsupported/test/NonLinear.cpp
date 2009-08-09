// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Thomas Capricelli <orzel@freehackers.org>


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


int fcn_lmder1(void * /*p*/, int /*m*/, int /*n*/, const double *x, double *fvec, double *fjac, 
	 int ldfjac, int iflag)
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

void testLmder1()
{
  int j, m, n, ldfjac, info, lwa;
  int ipvt[3];
  double tol, fnorm;
  double x[3], fvec[15], fjac[15*3], wa[30];

  m = 15;
  n = 3;

/*      the following starting values provide a rough fit. */

  x[1-1] = 1.;
  x[2-1] = 1.;
  x[3-1] = 1.;

  ldfjac = 15;
  lwa = 30;

/*      set tol to the square root of the machine precision. */
/*      unless high solutions are required, */
/*      this is the recommended setting. */

  tol = sqrt(dpmpar(1));

  info = lmder1(fcn_lmder1, 0, m, n, x, fvec, fjac, ldfjac, tol, 
	  ipvt, wa, lwa);
  fnorm = enorm(m, fvec);

  VERIFY_IS_APPROX(fnorm, 0.09063596);
  VERIFY(info ==  1);
  double x_ref[] = {0.08241058, 1.133037, 2.343695 };
  for (j=1; j<=n; j++) VERIFY_IS_APPROX(x[j-1], x_ref[j-1]);
}


int fcn_lmder(void * /*p*/, int /*m*/, int  /*n*/, const double *x, double *fvec, double *fjac, 
	 int ldfjac, int iflag)
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

void testLmder()
{
  int i, j, m, n, ldfjac, maxfev, mode, nprint, info, nfev, njev;
  int ipvt[3];
  double ftol, xtol, gtol, factor, fnorm;
  double x[3], fvec[15], fjac[15*3], diag[3], qtf[3], 
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
    
  maxfev = 400;
  mode = 1;
  factor = 1.e2;
  nprint = 0;

  info = lmder(fcn_lmder, 0, m, n, x, fvec, fjac, ldfjac, ftol, xtol, gtol, 
	maxfev, diag, mode, factor, nprint, &nfev, &njev, 
	ipvt, qtf, wa1, wa2, wa3, wa4);
  fnorm = enorm(m, fvec);

  VERIFY_IS_APPROX(fnorm, 0.09063596);

  VERIFY(nfev==6);
  VERIFY(njev==5);
  VERIFY(info==1);
  double x_ref[] = {0.08241058, 1.133037, 2.343695 };
  for (j=1; j<=n; j++) VERIFY_IS_APPROX(x[j-1], x_ref[j-1]);
  ftol = dpmpar(1);
  covfac = fnorm*fnorm/(m-n);
  covar(n, fjac, ldfjac, ipvt, ftol, wa1);

  double cov_ref[] = { 
      0.0001531202,   0.002869941,  -0.002656662,
      0.002869941,    0.09480935,   -0.09098995,
      -0.002656662,   -0.09098995,    0.08778727
  };

  for (i=1; i<=n; i++)
    for (j=1; j<=n; j++)
        VERIFY_IS_APPROX(fjac[(i-1)*ldfjac+j-1]*covfac, cov_ref[(i-1)*3+(j-1)]);
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

int fcn_hybrd1(void * /*p*/, int n, const double *x, double *fvec, int /*iflag*/)
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


struct myfunctor {
    static int f(void *p, int n, const double *x, double *fvec, int iflag )
    { return fcn_hybrd1(p,n,x,fvec,iflag) ; }
};

void testHybrd1()
{
  int n=9, info;
  Eigen::VectorXd x(n), fvec(n);

  /* the following starting values provide a rough solution. */
  x.setConstant(n, -1.);

/*      set tol to the square root of the machine precision. */
/*      unless high solutions are required, */
/*      this is the recommended setting. */

  info = ei_hybrd1<myfunctor,double>(x, fvec);

  // check return value
  VERIFY( 1 == info);

  // check norm
  VERIFY_IS_APPROX(fvec.norm(), 1.192636e-08);

  // check x
  VectorXd x_ref(n);
  x_ref << -0.5706545, -0.6816283, -0.7017325, -0.7042129, -0.701369, -0.6918656, -0.665792, -0.5960342, -0.4164121;
  VERIFY_IS_APPROX(x, x_ref);
}

int fcn_hybrd(void * /*p*/, int n, const double *x, double *fvec, int iflag)
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

void testHybrd()
{
  int j, n, maxfev, ml, mu, mode, nprint, info, nfev, ldfjac, lr;
  double xtol, epsfcn, factor, fnorm;
  double x[9], fvec[9], diag[9], fjac[9*9], r[45], qtf[9],
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

  maxfev = 2000;
  ml = 1;
  mu = 1;
  epsfcn = 0.;
  mode = 2;
  for (j=1; j<=9; j++)
    {
      diag[j-1] = 1.;
    }

  factor = 1.e2;
  nprint = 0;

  info = hybrd(fcn_hybrd, 0, n, x, fvec, xtol, maxfev, ml, mu, epsfcn,
	 diag, mode, factor, nprint, &nfev,
	 fjac, ldfjac, r, lr, qtf, wa1, wa2, wa3, wa4);
  fnorm = enorm(n, fvec);

  VERIFY_IS_APPROX(fnorm, 1.192636e-08);
  VERIFY(nfev==14);
  VERIFY(info==1);
  double x_ref[] = { 
      -0.5706545,    -0.6816283,    -0.7017325,
      -0.7042129,     -0.701369,    -0.6918656,
      -0.665792,    -0.5960342,    -0.4164121
  };
  for (j=1; j<=n; j++) VERIFY_IS_APPROX(x[j-1], x_ref[j-1]);
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

void test_NonLinear()
{
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
