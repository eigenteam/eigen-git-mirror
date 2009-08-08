// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Thomas Capricelli <orzel@freehackers.org>


#include "main.h"

#include <stdio.h>
#include <math.h>
#include <cminpack.h>


int fcn(int m, int n, const double *x, double *fvec, double *fjac, int ldfjac, int iflag);

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
  fcn(m, n, x, fvec, fjac, ldfjac, 1);
  fcn(m, n, x, fvec, fjac, ldfjac, 2);
  fcn(m, n, xp, fvecp, fjac, ldfjac, 1);
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
  return;
}

int fcn(int /*m*/, int /*n*/, const double *x, double *fvec,
	 double *fjac, int ldfjac, int iflag)
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


void test_NonLinear()
{
  CALL_SUBTEST(testChkder());
}
