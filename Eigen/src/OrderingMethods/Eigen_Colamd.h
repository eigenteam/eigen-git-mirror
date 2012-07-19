// // This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Desire Nuentsa Wakam <desire.nuentsa_wakam@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// This file is modified from the eigen_colamd/symamd library. The copyright is below

//   The authors of the code itself are Stefan I. Larimore and Timothy A.
//   Davis (davis@cise.ufl.edu), University of Florida.  The algorithm was
//   developed in collaboration with John Gilbert, Xerox PARC, and Esmond
//   Ng, Oak Ridge National Laboratory.
// 
//     Date:
// 
//   September 8, 2003.  Version 2.3.
// 
//     Acknowledgements:
// 
//   This work was supported by the National Science Foundation, under
//   grants DMS-9504974 and DMS-9803599.
// 
//     Notice:
// 
//   Copyright (c) 1998-2003 by the University of Florida.
//   All Rights Reserved.
// 
//   THIS MATERIAL IS PROVIDED AS IS, WITH ABSOLUTELY NO WARRANTY
//   EXPRESSED OR IMPLIED.  ANY USE IS AT YOUR OWN RISK.
// 
//   Permission is hereby granted to use, copy, modify, and/or distribute
//   this program, provided that the Copyright, this License, and the
//   Availability of the original version is retained on all copies and made
//   accessible to the end-user of any code or package that includes COLAMD
//   or any modified version of COLAMD. 
// 
//     Availability:
// 
//   The eigen_colamd/symamd library is available at
// 
//       http://www.cise.ufl.edu/research/sparse/eigen_colamd/

//   This is the http://www.cise.ufl.edu/research/sparse/eigen_colamd/eigen_colamd.h
//   file.  It is required by the eigen_colamd.c, colamdmex.c, and symamdmex.c
//   files, and by any C code that calls the routines whose prototypes are
//   listed below, or that uses the eigen_colamd/symamd definitions listed below.
  
#ifndef EIGEN_COLAMD_H
#define EIGEN_COLAMD_H

/* Ensure that debugging is turned off: */
#ifndef COLAMD_NDEBUG
#define COLAMD_NDEBUG
#endif /* NDEBUG */

/* ========================================================================== */
/* === Knob and statistics definitions ====================================== */
/* ========================================================================== */

/* size of the knobs [ ] array.  Only knobs [0..1] are currently used. */
#define EIGEN_COLAMD_KNOBS 20

/* number of output statistics.  Only stats [0..6] are currently used. */
#define EIGEN_COLAMD_STATS 20 

/* knobs [0] and stats [0]: dense row knob and output statistic. */
#define EIGEN_COLAMD_DENSE_ROW 0

/* knobs [1] and stats [1]: dense column knob and output statistic. */
#define EIGEN_COLAMD_DENSE_COL 1

/* stats [2]: memory defragmentation count output statistic */
#define EIGEN_COLAMD_DEFRAG_COUNT 2

/* stats [3]: eigen_colamd status:  zero OK, > 0 warning or notice, < 0 error */
#define EIGEN_COLAMD_STATUS 3

/* stats [4..6]: error info, or info on jumbled columns */ 
#define EIGEN_COLAMD_INFO1 4
#define EIGEN_COLAMD_INFO2 5
#define EIGEN_COLAMD_INFO3 6

/* error codes returned in stats [3]: */
#define EIGEN_COLAMD_OK       (0)
#define EIGEN_COLAMD_OK_BUT_JUMBLED     (1)
#define EIGEN_COLAMD_ERROR_A_not_present    (-1)
#define EIGEN_COLAMD_ERROR_p_not_present    (-2)
#define EIGEN_COLAMD_ERROR_nrow_negative    (-3)
#define EIGEN_COLAMD_ERROR_ncol_negative    (-4)
#define EIGEN_COLAMD_ERROR_nnz_negative   (-5)
#define EIGEN_COLAMD_ERROR_p0_nonzero     (-6)
#define EIGEN_COLAMD_ERROR_A_too_small    (-7)
#define EIGEN_COLAMD_ERROR_col_length_negative  (-8)
#define EIGEN_COLAMD_ERROR_row_index_out_of_bounds  (-9)
#define EIGEN_COLAMD_ERROR_out_of_memory    (-10)
#define EIGEN_COLAMD_ERROR_internal_error   (-999)

/* ========================================================================== */
/* === Definitions ========================================================== */
/* ========================================================================== */

#define COLAMD_MAX(a,b) (((a) > (b)) ? (a) : (b))
#define COLAMD_MIN(a,b) (((a) < (b)) ? (a) : (b))

#define EIGEN_ONES_COMPLEMENT(r) (-(r)-1)

/* -------------------------------------------------------------------------- */

#define EIGEN_COLAMD_EMPTY (-1)

/* Row and column status */
#define EIGEN_ALIVE (0)
#define EIGEN_DEAD  (-1)

/* Column status */
#define EIGEN_DEAD_PRINCIPAL    (-1)
#define EIGEN_DEAD_NON_PRINCIPAL  (-2)

/* Macros for row and column status update and checking. */
#define EIGEN_ROW_IS_DEAD(r)      EIGEN_ROW_IS_MARKED_DEAD (Row[r].shared2.mark)
#define EIGEN_ROW_IS_MARKED_DEAD(row_mark)  (row_mark < EIGEN_ALIVE)
#define EIGEN_ROW_IS_ALIVE(r)     (Row [r].shared2.mark >= EIGEN_ALIVE)
#define EIGEN_COL_IS_DEAD(c)      (Col [c].start < EIGEN_ALIVE)
#define EIGEN_COL_IS_ALIVE(c)     (Col [c].start >= EIGEN_ALIVE)
#define EIGEN_EIGEN_COL_IS_DEAD_PRINCIPAL(c)  (Col [c].start == EIGEN_DEAD_PRINCIPAL)
#define EIGEN_KILL_ROW(r)     { Row [r].shared2.mark = EIGEN_DEAD ; }
#define EIGEN_KILL_PRINCIPAL_COL(c)   { Col [c].start = EIGEN_DEAD_PRINCIPAL ; }
#define EIGEN_KILL_NON_PRINCIPAL_COL(c) { Col [c].start = EIGEN_DEAD_NON_PRINCIPAL ; }

/* ========================================================================== */
/* === Colamd reporting mechanism =========================================== */
/* ========================================================================== */

#ifdef MATLAB_MEX_FILE

/* use mexPrintf in a MATLAB mexFunction, for debugging and statistics output */
#define PRINTF mexPrintf

/* In MATLAB, matrices are 1-based to the user, but 0-based internally */
#define INDEX(i) ((i)+1)

#else

/* Use printf in standard C environment, for debugging and statistics output. */
/* Output is generated only if debugging is enabled at compile time, or if */
/* the caller explicitly calls eigen_colamd_report or symamd_report. */
#define PRINTF printf

/* In C, matrices are 0-based and indices are reported as such in *_report */
#define INDEX(i) (i)

#endif /* MATLAB_MEX_FILE */

    // == Row and Column structures ==
typedef struct EIGEN_Colamd_Col_struct
{
    int start ;   /* index for A of first row in this column, or EIGEN_DEAD */
      /* if column is dead */
    int length ;  /* number of rows in this column */
    union
    {
  int thickness ; /* number of original columns represented by this */
      /* col, if the column is alive */
  int parent ;  /* parent in parent tree super-column structure, if */
      /* the column is dead */
    } shared1 ;
    union
    {
  int score ; /* the score used to maintain heap, if col is alive */
  int order ; /* pivot ordering of this column, if col is dead */
    } shared2 ;
    union
    {
  int headhash ;  /* head of a hash bucket, if col is at the head of */
      /* a degree list */
  int hash ;  /* hash value, if col is not in a degree list */
  int prev ;  /* previous column in degree list, if col is in a */
      /* degree list (but not at the head of a degree list) */
    } shared3 ;
    union
    {
  int degree_next ; /* next column, if col is in a degree list */
  int hash_next ;   /* next column, if col is in a hash list */
    } shared4 ;

} EIGEN_Colamd_Col ;

typedef struct EIGEN_Colamd_Row_struct
{
    int start ;   /* index for A of first col in this row */
    int length ;  /* number of principal columns in this row */
    union
    {
  int degree ;  /* number of principal & non-principal columns in row */
  int p ;   /* used as a row pointer in eigen_init_rows_cols () */
    } shared1 ;
    union
    {
  int mark ;  /* for computing set differences and marking dead rows*/
  int first_column ;/* first column in row (used in garbage collection) */
    } shared2 ;

} EIGEN_Colamd_Row ;
    
/* ========================================================================== */
/* === Colamd recommended memory size ======================================= */
/* ========================================================================== */

/*
    The recommended length Alen of the array A passed to eigen_colamd is given by
    the EIGEN_COLAMD_RECOMMENDED (nnz, n_row, n_col) macro.  It returns -1 if any
    argument is negative.  2*nnz space is required for the row and column
    indices of the matrix. EIGEN_COLAMD_C (n_col) + EIGEN_COLAMD_R (n_row) space is
    required for the Col and Row arrays, respectively, which are internal to
    eigen_colamd.  An additional n_col space is the minimal amount of "elbow room",
    and nnz/5 more space is recommended for run time efficiency.

    This macro is not needed when using symamd.

    Explicit typecast to int added Sept. 23, 2002, COLAMD version 2.2, to avoid
    gcc -pedantic warning messages.
*/

#define EIGEN_COLAMD_C(n_col) ((int) (((n_col) + 1) * sizeof (EIGEN_Colamd_Col) / sizeof (int)))
#define EIGEN_COLAMD_R(n_row) ((int) (((n_row) + 1) * sizeof (EIGEN_Colamd_Row) / sizeof (int)))

#define EIGEN_COLAMD_RECOMMENDED(nnz, n_row, n_col)                                 \
(                                                                             \
((nnz) < 0 || (n_row) < 0 || (n_col) < 0)                                     \
?                                                                             \
    (-1)                                                                      \
:                                                                             \
    (2 * (nnz) + EIGEN_COLAMD_C (n_col) + EIGEN_COLAMD_R (n_row) + (n_col) + ((nnz) / 5)) \
)

    // Various routines
int eigen_colamd_recommended (int nnz, int n_row, int n_col) ;

void eigen_colamd_set_defaults (double knobs [EIGEN_COLAMD_KNOBS]) ;

bool eigen_colamd (int n_row, int n_col, int Alen, int A [], int p [], double knobs[EIGEN_COLAMD_KNOBS], int stats [EIGEN_COLAMD_STATS]) ;

void eigen_colamd_report (int stats [EIGEN_COLAMD_STATS]);

int eigen_init_rows_cols (int n_row, int n_col, EIGEN_Colamd_Row Row [], EIGEN_Colamd_Col col [], int A [], int p [], int stats[EIGEN_COLAMD_STATS] ); 

void eigen_init_scoring (int n_row, int n_col, EIGEN_Colamd_Row Row [], EIGEN_Colamd_Col Col [], int A [], int head [], double knobs[EIGEN_COLAMD_KNOBS], int *p_n_row2, int *p_n_col2, int *p_max_deg);

int eigen_find_ordering (int n_row, int n_col, int Alen, EIGEN_Colamd_Row Row [], EIGEN_Colamd_Col Col [], int A [], int head [], int n_col2, int max_deg, int pfree);

void eigen_order_children (int n_col, EIGEN_Colamd_Col Col [], int p []);

void eigen_detect_super_cols (
#ifndef COLAMD_NDEBUG
  int n_col,
  EIGEN_Colamd_Row Row [],
#endif /* COLAMD_NDEBUG */
  EIGEN_Colamd_Col Col [],
  int A [],
  int head [],
  int row_start,
  int row_length ) ;

 int eigen_garbage_collection (int n_row, int n_col, EIGEN_Colamd_Row Row [], EIGEN_Colamd_Col Col [], int A [], int *pfree) ;

 int eigen_clear_mark (int n_row, EIGEN_Colamd_Row Row [] ) ;

 void eigen_print_report (char *method, int stats [EIGEN_COLAMD_STATS]) ;

/* ========================================================================== */
/* === Debugging prototypes and definitions ================================= */
/* ========================================================================== */

#ifndef COLAMD_NDEBUG

/* colamd_debug is the *ONLY* global variable, and is only */
/* present when debugging */

 int colamd_debug ;  /* debug print level */

#define COLAMD_DEBUG0(params) { (void) PRINTF params ; }
#define COLAMD_DEBUG1(params) { if (colamd_debug >= 1) (void) PRINTF params ; }
#define COLAMD_DEBUG2(params) { if (colamd_debug >= 2) (void) PRINTF params ; }
#define COLAMD_DEBUG3(params) { if (colamd_debug >= 3) (void) PRINTF params ; }
#define COLAMD_DEBUG4(params) { if (colamd_debug >= 4) (void) PRINTF params ; }

#ifdef MATLAB_MEX_FILE
#define COLAMD_ASSERT(expression) (mxAssert ((expression), ""))
#else
#define COLAMD_ASSERT(expression) (assert (expression))
#endif /* MATLAB_MEX_FILE */

 void eigen_colamd_get_debug /* gets the debug print level from getenv */
(
    char *method
) ;

 void eigen_debug_deg_lists
(
    int n_row,
    int n_col,
    EIGEN_Colamd_Row Row [],
    EIGEN_Colamd_Col Col [],
    int head [],
    int min_score,
    int should,
    int max_deg
) ;

 void eigen_debug_mark
(
    int n_row,
    EIGEN_Colamd_Row Row [],
    int tag_mark,
    int max_mark
) ;

 void eigen_debug_matrix
(
    int n_row,
    int n_col,
    EIGEN_Colamd_Row Row [],
    EIGEN_Colamd_Col Col [],
    int A []
) ;

 void eigen_debug_structures
(
    int n_row,
    int n_col,
    EIGEN_Colamd_Row Row [],
    EIGEN_Colamd_Col Col [],
    int A [],
    int n_col2
) ;

#else /* COLAMD_NDEBUG */

/* === No debugging ========================================================= */

#define COLAMD_DEBUG0(params) ;
#define COLAMD_DEBUG1(params) ;
#define COLAMD_DEBUG2(params) ;
#define COLAMD_DEBUG3(params) ;
#define COLAMD_DEBUG4(params) ;

#define COLAMD_ASSERT(expression) ((void) 0)

#endif /* COLAMD_NDEBUG */



/**
 * \brief Returns the recommended value of Alen 
 * 
 * Returns recommended value of Alen for use by eigen_colamd.  
 * Returns -1 if any input argument is negative.  
 * The use of this routine or macro is optional.  
 * Note that the macro uses its arguments   more than once, 
 * so be careful for side effects, if you pass expressions as arguments to EIGEN_COLAMD_RECOMMENDED.  
 * 
 * \param nnz nonzeros in A
 * \param n_row number of rows in A
 * \param n_col number of columns in A
 * \return recommended value of Alen for use by eigen_colamd
 */
int eigen_colamd_recommended ( int nnz, int n_row, int n_col)
{
  
   return (EIGEN_COLAMD_RECOMMENDED (nnz, n_row, n_col)) ; 
}

/**
 * \brief set default parameters  The use of this routine is optional.
 * 
 * Colamd: rows with more than (knobs [EIGEN_COLAMD_DENSE_ROW] * n_col)
 * entries are removed prior to ordering.  Columns with more than
 * (knobs [EIGEN_COLAMD_DENSE_COL] * n_row) entries are removed prior to
 * ordering, and placed last in the output column ordering. 
 *
 * EIGEN_COLAMD_DENSE_ROW and EIGEN_COLAMD_DENSE_COL are defined as 0 and 1,
 * respectively, in eigen_colamd.h.  Default values of these two knobs
 * are both 0.5.  Currently, only knobs [0] and knobs [1] are
 * used, but future versions may use more knobs.  If so, they will
 * be properly set to their defaults by the future version of
 * eigen_colamd_set_defaults, so that the code that calls eigen_colamd will
 * not need to change, assuming that you either use
 * eigen_colamd_set_defaults, or pass a (double *) NULL pointer as the
 * knobs array to eigen_colamd or symamd.
 * 
 * \param knobs parameter settings for eigen_colamd
 */
void eigen_colamd_set_defaults(double knobs[EIGEN_COLAMD_KNOBS])
{
   /* === Local variables ================================================== */

    int i ;

    if (!knobs)
    {
  return ;      /* no knobs to initialize */
    }
    for (i = 0 ; i < EIGEN_COLAMD_KNOBS ; i++)
    {
  knobs [i] = 0 ;
    }
    knobs [EIGEN_COLAMD_DENSE_ROW] = 0.5 ;  /* ignore rows over 50% dense */
    knobs [EIGEN_COLAMD_DENSE_COL] = 0.5 ;  /* ignore columns over 50% dense */
}

/** 
 * \brief  Computes a column ordering using the column approximate minimum degree ordering
 * 
 * Computes a column ordering (Q) of A such that P(AQ)=LU or
 * (AQ)'AQ=LL' have less fill-in and require fewer floating point
 * operations than factorizing the unpermuted matrix A or A'A,
 * respectively.
 * 
 * 
 * \param n_row number of rows in A
 * \param n_col number of columns in A
 * \param Alen, size of the array A
 * \param A row indices of the matrix, of size ALen
 * \param p column pointers of A, of size n_col+1
 * \param knobs parameter settings for eigen_colamd
 * \param stats eigen_colamd output statistics and error codes
 */
bool eigen_colamd(int n_row, int n_col, int Alen, int *A, int *p, double knobs[EIGEN_COLAMD_KNOBS], int stats[EIGEN_COLAMD_STATS])
{
      /* === Local variables ================================================== */

    int i ;     /* loop index */
    int nnz ;     /* nonzeros in A */
    int Row_size ;    /* size of Row [], in integers */
    int Col_size ;    /* size of Col [], in integers */
    int need ;      /* minimum required length of A */
    EIGEN_Colamd_Row *Row ;   /* pointer into A of Row [0..n_row] array */
    EIGEN_Colamd_Col *Col ;   /* pointer into A of Col [0..n_col] array */
    int n_col2 ;    /* number of non-dense, non-empty columns */
    int n_row2 ;    /* number of non-dense, non-empty rows */
    int ngarbage ;    /* number of garbage collections performed */
    int max_deg ;   /* maximum row degree */
    double default_knobs [EIGEN_COLAMD_KNOBS] ; /* default knobs array */

#ifndef COLAMD_NDEBUG
    eigen_colamd_get_debug ("eigen_colamd") ;
#endif /* COLAMD_NDEBUG */

    /* === Check the input arguments ======================================== */

    if (!stats)
    {
  COLAMD_DEBUG0 (("eigen_colamd: stats not present\n")) ;
  return (false) ;
    }
    for (i = 0 ; i < EIGEN_COLAMD_STATS ; i++)
    {
  stats [i] = 0 ;
    }
    stats [EIGEN_COLAMD_STATUS] = EIGEN_COLAMD_OK ;
    stats [EIGEN_COLAMD_INFO1] = -1 ;
    stats [EIGEN_COLAMD_INFO2] = -1 ;

    if (!A)   /* A is not present */
    {
  stats [EIGEN_COLAMD_STATUS] = EIGEN_COLAMD_ERROR_A_not_present ;
  COLAMD_DEBUG0 (("eigen_colamd: A not present\n")) ;
  return (false) ;
    }

    if (!p)   /* p is not present */
    {
  stats [EIGEN_COLAMD_STATUS] = EIGEN_COLAMD_ERROR_p_not_present ;
  COLAMD_DEBUG0 (("eigen_colamd: p not present\n")) ;
      return (false) ;
    }

    if (n_row < 0)  /* n_row must be >= 0 */
    {
  stats [EIGEN_COLAMD_STATUS] = EIGEN_COLAMD_ERROR_nrow_negative ;
  stats [EIGEN_COLAMD_INFO1] = n_row ;
  COLAMD_DEBUG0 (("eigen_colamd: nrow negative %d\n", n_row)) ;
      return (false) ;
    }

    if (n_col < 0)  /* n_col must be >= 0 */
    {
  stats [EIGEN_COLAMD_STATUS] = EIGEN_COLAMD_ERROR_ncol_negative ;
  stats [EIGEN_COLAMD_INFO1] = n_col ;
  COLAMD_DEBUG0 (("eigen_colamd: ncol negative %d\n", n_col)) ;
      return (false) ;
    }

    nnz = p [n_col] ;
    if (nnz < 0)  /* nnz must be >= 0 */
    {
  stats [EIGEN_COLAMD_STATUS] = EIGEN_COLAMD_ERROR_nnz_negative ;
  stats [EIGEN_COLAMD_INFO1] = nnz ;
  COLAMD_DEBUG0 (("eigen_colamd: number of entries negative %d\n", nnz)) ;
  return (false) ;
    }

    if (p [0] != 0)
    {
  stats [EIGEN_COLAMD_STATUS] = EIGEN_COLAMD_ERROR_p0_nonzero ;
  stats [EIGEN_COLAMD_INFO1] = p [0] ;
  COLAMD_DEBUG0 (("eigen_colamd: p[0] not zero %d\n", p [0])) ;
  return (false) ;
    }

    /* === If no knobs, set default knobs =================================== */

    if (!knobs)
    {
  eigen_colamd_set_defaults (default_knobs) ;
  knobs = default_knobs ;
    }

    /* === Allocate the Row and Col arrays from array A ===================== */

    Col_size = EIGEN_COLAMD_C (n_col) ;
    Row_size = EIGEN_COLAMD_R (n_row) ;
    need = 2*nnz + n_col + Col_size + Row_size ;

    if (need > Alen)
    {
  /* not enough space in array A to perform the ordering */
  stats [EIGEN_COLAMD_STATUS] = EIGEN_COLAMD_ERROR_A_too_small ;
  stats [EIGEN_COLAMD_INFO1] = need ;
  stats [EIGEN_COLAMD_INFO2] = Alen ;
  COLAMD_DEBUG0 (("eigen_colamd: Need Alen >= %d, given only Alen = %d\n", need,Alen));
  return (false) ;
    }

    Alen -= Col_size + Row_size ;
    Col = (EIGEN_Colamd_Col *) &A [Alen] ;
    Row = (EIGEN_Colamd_Row *) &A [Alen + Col_size] ;

    /* === Construct the row and column data structures ===================== */

    if (!eigen_init_rows_cols (n_row, n_col, Row, Col, A, p, stats))
    {
  /* input matrix is invalid */
  COLAMD_DEBUG0 (("eigen_colamd: Matrix invalid\n")) ;
  return (false) ;
    }

    /* === Initialize scores, kill dense rows/columns ======================= */

    eigen_init_scoring (n_row, n_col, Row, Col, A, p, knobs,
  &n_row2, &n_col2, &max_deg) ;

    /* === Order the supercolumns =========================================== */

    ngarbage = eigen_find_ordering (n_row, n_col, Alen, Row, Col, A, p,
  n_col2, max_deg, 2*nnz) ;

    /* === Order the non-principal columns ================================== */

    eigen_order_children (n_col, Col, p) ;

    /* === Return statistics in stats ======================================= */

    stats [EIGEN_COLAMD_DENSE_ROW] = n_row - n_row2 ;
    stats [EIGEN_COLAMD_DENSE_COL] = n_col - n_col2 ;
    stats [EIGEN_COLAMD_DEFRAG_COUNT] = ngarbage ;
    COLAMD_DEBUG0 (("eigen_colamd: done.\n")) ; 
    return (true) ;
}

/* ========================================================================== */
/* === eigen_colamd_report ======================================================== */
/* ========================================================================== */

 void eigen_colamd_report
(
    int stats [EIGEN_COLAMD_STATS]
)
{
    eigen_print_report ("eigen_colamd", stats) ;
}


/* ========================================================================== */
/* === NON-USER-CALLABLE ROUTINES: ========================================== */
/* ========================================================================== */

/* There are no user-callable routines beyond this point in the file */


/* ========================================================================== */
/* === eigen_init_rows_cols ======================================================= */
/* ========================================================================== */

/*
    Takes the column form of the matrix in A and creates the row form of the
    matrix.  Also, row and column attributes are stored in the Col and Row
    structs.  If the columns are un-sorted or contain duplicate row indices,
    this routine will also sort and remove duplicate row indices from the
    column form of the matrix.  Returns false if the matrix is invalid,
    true otherwise.  Not user-callable.
*/

 int eigen_init_rows_cols  /* returns true if OK, or false otherwise */
(
    /* === Parameters ======================================================= */

    int n_row,      /* number of rows of A */
    int n_col,      /* number of columns of A */
    EIGEN_Colamd_Row Row [],    /* of size n_row+1 */
    EIGEN_Colamd_Col Col [],    /* of size n_col+1 */
    int A [],     /* row indices of A, of size Alen */
    int p [],     /* pointers to columns in A, of size n_col+1 */
    int stats [EIGEN_COLAMD_STATS]  /* eigen_colamd statistics */ 
)
{
    /* === Local variables ================================================== */

    int col ;     /* a column index */
    int row ;     /* a row index */
    int *cp ;     /* a column pointer */
    int *cp_end ;   /* a pointer to the end of a column */
    int *rp ;     /* a row pointer */
    int *rp_end ;   /* a pointer to the end of a row */
    int last_row ;    /* previous row */

    /* === Initialize columns, and check column pointers ==================== */

    for (col = 0 ; col < n_col ; col++)
    {
  Col [col].start = p [col] ;
  Col [col].length = p [col+1] - p [col] ;

  if (Col [col].length < 0)
  {
      /* column pointers must be non-decreasing */
      stats [EIGEN_COLAMD_STATUS] = EIGEN_COLAMD_ERROR_col_length_negative ;
      stats [EIGEN_COLAMD_INFO1] = col ;
      stats [EIGEN_COLAMD_INFO2] = Col [col].length ;
      COLAMD_DEBUG0 (("eigen_colamd: col %d length %d < 0\n", col, Col [col].length)) ;
      return (false) ;
  }

  Col [col].shared1.thickness = 1 ;
  Col [col].shared2.score = 0 ;
  Col [col].shared3.prev = EIGEN_COLAMD_EMPTY ;
  Col [col].shared4.degree_next = EIGEN_COLAMD_EMPTY ;
    }

    /* p [0..n_col] no longer needed, used as "head" in subsequent routines */

    /* === Scan columns, compute row degrees, and check row indices ========= */

    stats [EIGEN_COLAMD_INFO3] = 0 ;  /* number of duplicate or unsorted row indices*/

    for (row = 0 ; row < n_row ; row++)
    {
  Row [row].length = 0 ;
  Row [row].shared2.mark = -1 ;
    }

    for (col = 0 ; col < n_col ; col++)
    {
  last_row = -1 ;

  cp = &A [p [col]] ;
  cp_end = &A [p [col+1]] ;

  while (cp < cp_end)
  {
      row = *cp++ ;

      /* make sure row indices within range */
      if (row < 0 || row >= n_row)
      {
    stats [EIGEN_COLAMD_STATUS] = EIGEN_COLAMD_ERROR_row_index_out_of_bounds ;
    stats [EIGEN_COLAMD_INFO1] = col ;
    stats [EIGEN_COLAMD_INFO2] = row ;
    stats [EIGEN_COLAMD_INFO3] = n_row ;
    COLAMD_DEBUG0 (("eigen_colamd: row %d col %d out of bounds\n", row, col)) ;
    return (false) ;
      }

      if (row <= last_row || Row [row].shared2.mark == col)
      {
    /* row index are unsorted or repeated (or both), thus col */
    /* is jumbled.  This is a notice, not an error condition. */
    stats [EIGEN_COLAMD_STATUS] = EIGEN_COLAMD_OK_BUT_JUMBLED ;
    stats [EIGEN_COLAMD_INFO1] = col ;
    stats [EIGEN_COLAMD_INFO2] = row ;
    (stats [EIGEN_COLAMD_INFO3]) ++ ;
    COLAMD_DEBUG1 (("eigen_colamd: row %d col %d unsorted/duplicate\n",row,col));
      }

      if (Row [row].shared2.mark != col)
      {
    Row [row].length++ ;
      }
      else
      {
    /* this is a repeated entry in the column, */
    /* it will be removed */
    Col [col].length-- ;
      }

      /* mark the row as having been seen in this column */
      Row [row].shared2.mark = col ;

      last_row = row ;
  }
    }

    /* === Compute row pointers ============================================= */

    /* row form of the matrix starts directly after the column */
    /* form of matrix in A */
    Row [0].start = p [n_col] ;
    Row [0].shared1.p = Row [0].start ;
    Row [0].shared2.mark = -1 ;
    for (row = 1 ; row < n_row ; row++)
    {
  Row [row].start = Row [row-1].start + Row [row-1].length ;
  Row [row].shared1.p = Row [row].start ;
  Row [row].shared2.mark = -1 ;
    }

    /* === Create row form ================================================== */

    if (stats [EIGEN_COLAMD_STATUS] == EIGEN_COLAMD_OK_BUT_JUMBLED)
    {
  /* if cols jumbled, watch for repeated row indices */
  for (col = 0 ; col < n_col ; col++)
  {
      cp = &A [p [col]] ;
      cp_end = &A [p [col+1]] ;
      while (cp < cp_end)
      {
    row = *cp++ ;
    if (Row [row].shared2.mark != col)
    {
        A [(Row [row].shared1.p)++] = col ;
        Row [row].shared2.mark = col ;
    }
      }
  }
    }
    else
    {
  /* if cols not jumbled, we don't need the mark (this is faster) */
  for (col = 0 ; col < n_col ; col++)
  {
      cp = &A [p [col]] ;
      cp_end = &A [p [col+1]] ;
      while (cp < cp_end)
      {
    A [(Row [*cp++].shared1.p)++] = col ;
      }
  }
    }

    /* === Clear the row marks and set row degrees ========================== */

    for (row = 0 ; row < n_row ; row++)
    {
  Row [row].shared2.mark = 0 ;
  Row [row].shared1.degree = Row [row].length ;
    }

    /* === See if we need to re-create columns ============================== */

    if (stats [EIGEN_COLAMD_STATUS] == EIGEN_COLAMD_OK_BUT_JUMBLED)
    {
      COLAMD_DEBUG0 (("eigen_colamd: reconstructing column form, matrix jumbled\n")) ;

#ifndef COLAMD_NDEBUG
  /* make sure column lengths are correct */
  for (col = 0 ; col < n_col ; col++)
  {
      p [col] = Col [col].length ;
  }
  for (row = 0 ; row < n_row ; row++)
  {
      rp = &A [Row [row].start] ;
      rp_end = rp + Row [row].length ;
      while (rp < rp_end)
      {
    p [*rp++]-- ;
      }
  }
  for (col = 0 ; col < n_col ; col++)
  {
      COLAMD_ASSERT (p [col] == 0) ;
  }
  /* now p is all zero (different than when debugging is turned off) */
#endif /* COLAMD_NDEBUG */

  /* === Compute col pointers ========================================= */

  /* col form of the matrix starts at A [0]. */
  /* Note, we may have a gap between the col form and the row */
  /* form if there were duplicate entries, if so, it will be */
  /* removed upon the first garbage collection */
  Col [0].start = 0 ;
  p [0] = Col [0].start ;
  for (col = 1 ; col < n_col ; col++)
  {
      /* note that the lengths here are for pruned columns, i.e. */
      /* no duplicate row indices will exist for these columns */
      Col [col].start = Col [col-1].start + Col [col-1].length ;
      p [col] = Col [col].start ;
  }

  /* === Re-create col form =========================================== */

  for (row = 0 ; row < n_row ; row++)
  {
      rp = &A [Row [row].start] ;
      rp_end = rp + Row [row].length ;
      while (rp < rp_end)
      {
    A [(p [*rp++])++] = row ;
      }
  }
    }

    /* === Done.  Matrix is not (or no longer) jumbled ====================== */

    return (true) ;
}


/* ========================================================================== */
/* === eigen_init_scoring ========================================================= */
/* ========================================================================== */

/*
    Kills dense or empty columns and rows, calculates an initial score for
    each column, and places all columns in the degree lists.  Not user-callable.
*/

 void eigen_init_scoring
(
    /* === Parameters ======================================================= */

    int n_row,      /* number of rows of A */
    int n_col,      /* number of columns of A */
    EIGEN_Colamd_Row Row [],    /* of size n_row+1 */
    EIGEN_Colamd_Col Col [],    /* of size n_col+1 */
    int A [],     /* column form and row form of A */
    int head [],    /* of size n_col+1 */
    double knobs [EIGEN_COLAMD_KNOBS],/* parameters */
    int *p_n_row2,    /* number of non-dense, non-empty rows */
    int *p_n_col2,    /* number of non-dense, non-empty columns */
    int *p_max_deg    /* maximum row degree */
)
{
    /* === Local variables ================================================== */

    int c ;     /* a column index */
    int r, row ;    /* a row index */
    int *cp ;     /* a column pointer */
    int deg ;     /* degree of a row or column */
    int *cp_end ;   /* a pointer to the end of a column */
    int *new_cp ;   /* new column pointer */
    int col_length ;    /* length of pruned column */
    int score ;     /* current column score */
    int n_col2 ;    /* number of non-dense, non-empty columns */
    int n_row2 ;    /* number of non-dense, non-empty rows */
    int dense_row_count ; /* remove rows with more entries than this */
    int dense_col_count ; /* remove cols with more entries than this */
    int min_score ;   /* smallest column score */
    int max_deg ;   /* maximum row degree */
    int next_col ;    /* Used to add to degree list.*/

#ifndef COLAMD_NDEBUG
    int debug_count ;   /* debug only. */
#endif /* COLAMD_NDEBUG */

    /* === Extract knobs ==================================================== */

    dense_row_count = COLAMD_MAX (0, COLAMD_MIN (knobs [EIGEN_COLAMD_DENSE_ROW] * n_col, n_col)) ;
    dense_col_count = COLAMD_MAX (0, COLAMD_MIN (knobs [EIGEN_COLAMD_DENSE_COL] * n_row, n_row)) ;
    COLAMD_DEBUG1 (("eigen_colamd: densecount: %d %d\n", dense_row_count, dense_col_count)) ;
    max_deg = 0 ;
    n_col2 = n_col ;
    n_row2 = n_row ;

    /* === Kill empty columns =============================================== */

    /* Put the empty columns at the end in their natural order, so that LU */
    /* factorization can proceed as far as possible. */
    for (c = n_col-1 ; c >= 0 ; c--)
    {
  deg = Col [c].length ;
  if (deg == 0)
  {
      /* this is a empty column, kill and order it last */
      Col [c].shared2.order = --n_col2 ;
      EIGEN_KILL_PRINCIPAL_COL (c) ;
  }
    }
    COLAMD_DEBUG1 (("eigen_colamd: null columns killed: %d\n", n_col - n_col2)) ;

    /* === Kill dense columns =============================================== */

    /* Put the dense columns at the end, in their natural order */
    for (c = n_col-1 ; c >= 0 ; c--)
    {
  /* skip any dead columns */
  if (EIGEN_COL_IS_DEAD (c))
  {
      continue ;
  }
  deg = Col [c].length ;
  if (deg > dense_col_count)
  {
      /* this is a dense column, kill and order it last */
      Col [c].shared2.order = --n_col2 ;
      /* decrement the row degrees */
      cp = &A [Col [c].start] ;
      cp_end = cp + Col [c].length ;
      while (cp < cp_end)
      {
    Row [*cp++].shared1.degree-- ;
      }
      EIGEN_KILL_PRINCIPAL_COL (c) ;
  }
    }
    COLAMD_DEBUG1 (("eigen_colamd: Dense and null columns killed: %d\n", n_col - n_col2)) ;

    /* === Kill dense and empty rows ======================================== */

    for (r = 0 ; r < n_row ; r++)
    {
  deg = Row [r].shared1.degree ;
  COLAMD_ASSERT (deg >= 0 && deg <= n_col) ;
  if (deg > dense_row_count || deg == 0)
  {
      /* kill a dense or empty row */
      EIGEN_KILL_ROW (r) ;
      --n_row2 ;
  }
  else
  {
      /* keep track of max degree of remaining rows */
      max_deg = COLAMD_MAX (max_deg, deg) ;
  }
    }
    COLAMD_DEBUG1 (("eigen_colamd: Dense and null rows killed: %d\n", n_row - n_row2)) ;

    /* === Compute initial column scores ==================================== */

    /* At this point the row degrees are accurate.  They reflect the number */
    /* of "live" (non-dense) columns in each row.  No empty rows exist. */
    /* Some "live" columns may contain only dead rows, however.  These are */
    /* pruned in the code below. */

    /* now find the initial matlab score for each column */
    for (c = n_col-1 ; c >= 0 ; c--)
    {
  /* skip dead column */
  if (EIGEN_COL_IS_DEAD (c))
  {
      continue ;
  }
  score = 0 ;
  cp = &A [Col [c].start] ;
  new_cp = cp ;
  cp_end = cp + Col [c].length ;
  while (cp < cp_end)
  {
      /* get a row */
      row = *cp++ ;
      /* skip if dead */
      if (EIGEN_ROW_IS_DEAD (row))
      {
    continue ;
      }
      /* compact the column */
      *new_cp++ = row ;
      /* add row's external degree */
      score += Row [row].shared1.degree - 1 ;
      /* guard against integer overflow */
      score = COLAMD_MIN (score, n_col) ;
  }
  /* determine pruned column length */
  col_length = (int) (new_cp - &A [Col [c].start]) ;
  if (col_length == 0)
  {
      /* a newly-made null column (all rows in this col are "dense" */
      /* and have already been killed) */
      COLAMD_DEBUG2 (("Newly null killed: %d\n", c)) ;
      Col [c].shared2.order = --n_col2 ;
      EIGEN_KILL_PRINCIPAL_COL (c) ;
  }
  else
  {
      /* set column length and set score */
      COLAMD_ASSERT (score >= 0) ;
      COLAMD_ASSERT (score <= n_col) ;
      Col [c].length = col_length ;
      Col [c].shared2.score = score ;
  }
    }
    COLAMD_DEBUG1 (("eigen_colamd: Dense, null, and newly-null columns killed: %d\n",
      n_col-n_col2)) ;

    /* At this point, all empty rows and columns are dead.  All live columns */
    /* are "clean" (containing no dead rows) and simplicial (no supercolumns */
    /* yet).  Rows may contain dead columns, but all live rows contain at */
    /* least one live column. */

#ifndef COLAMD_NDEBUG
    eigen_debug_structures (n_row, n_col, Row, Col, A, n_col2) ;
#endif /* COLAMD_NDEBUG */

    /* === Initialize degree lists ========================================== */

#ifndef COLAMD_NDEBUG
    debug_count = 0 ;
#endif /* COLAMD_NDEBUG */

    /* clear the hash buckets */
    for (c = 0 ; c <= n_col ; c++)
    {
  head [c] = EIGEN_COLAMD_EMPTY ;
    }
    min_score = n_col ;
    /* place in reverse order, so low column indices are at the front */
    /* of the lists.  This is to encourage natural tie-breaking */
    for (c = n_col-1 ; c >= 0 ; c--)
    {
  /* only add principal columns to degree lists */
  if (EIGEN_COL_IS_ALIVE (c))
  {
      COLAMD_DEBUG4 (("place %d score %d minscore %d ncol %d\n",
    c, Col [c].shared2.score, min_score, n_col)) ;

      /* === Add columns score to DList =============================== */

      score = Col [c].shared2.score ;

      COLAMD_ASSERT (min_score >= 0) ;
      COLAMD_ASSERT (min_score <= n_col) ;
      COLAMD_ASSERT (score >= 0) ;
      COLAMD_ASSERT (score <= n_col) ;
      COLAMD_ASSERT (head [score] >= EIGEN_COLAMD_EMPTY) ;

      /* now add this column to dList at proper score location */
      next_col = head [score] ;
      Col [c].shared3.prev = EIGEN_COLAMD_EMPTY ;
      Col [c].shared4.degree_next = next_col ;

      /* if there already was a column with the same score, set its */
      /* previous pointer to this new column */
      if (next_col != EIGEN_COLAMD_EMPTY)
      {
    Col [next_col].shared3.prev = c ;
      }
      head [score] = c ;

      /* see if this score is less than current min */
      min_score = COLAMD_MIN (min_score, score) ;

#ifndef COLAMD_NDEBUG
      debug_count++ ;
#endif /* COLAMD_NDEBUG */

  }
    }

#ifndef COLAMD_NDEBUG
    COLAMD_DEBUG1 (("eigen_colamd: Live cols %d out of %d, non-princ: %d\n",
  debug_count, n_col, n_col-debug_count)) ;
    COLAMD_ASSERT (debug_count == n_col2) ;
    eigen_debug_deg_lists (n_row, n_col, Row, Col, head, min_score, n_col2, max_deg) ;
#endif /* COLAMD_NDEBUG */

    /* === Return number of remaining columns, and max row degree =========== */

    *p_n_col2 = n_col2 ;
    *p_n_row2 = n_row2 ;
    *p_max_deg = max_deg ;
}


/* ========================================================================== */
/* === eigen_find_ordering ======================================================== */
/* ========================================================================== */

/*
    Order the principal columns of the supercolumn form of the matrix
    (no supercolumns on input).  Uses a minimum approximate column minimum
    degree ordering method.  Not user-callable.
*/

 int eigen_find_ordering /* return the number of garbage collections */
(
    /* === Parameters ======================================================= */

    int n_row,      /* number of rows of A */
    int n_col,      /* number of columns of A */
    int Alen,     /* size of A, 2*nnz + n_col or larger */
    EIGEN_Colamd_Row Row [],    /* of size n_row+1 */
    EIGEN_Colamd_Col Col [],    /* of size n_col+1 */
    int A [],     /* column form and row form of A */
    int head [],    /* of size n_col+1 */
    int n_col2,     /* Remaining columns to order */
    int max_deg,    /* Maximum row degree */
    int pfree     /* index of first free slot (2*nnz on entry) */
)
{
    /* === Local variables ================================================== */

    int k ;     /* current pivot ordering step */
    int pivot_col ;   /* current pivot column */
    int *cp ;     /* a column pointer */
    int *rp ;     /* a row pointer */
    int pivot_row ;   /* current pivot row */
    int *new_cp ;   /* modified column pointer */
    int *new_rp ;   /* modified row pointer */
    int pivot_row_start ; /* pointer to start of pivot row */
    int pivot_row_degree ;  /* number of columns in pivot row */
    int pivot_row_length ;  /* number of supercolumns in pivot row */
    int pivot_col_score ; /* score of pivot column */
    int needed_memory ;   /* free space needed for pivot row */
    int *cp_end ;   /* pointer to the end of a column */
    int *rp_end ;   /* pointer to the end of a row */
    int row ;     /* a row index */
    int col ;     /* a column index */
    int max_score ;   /* maximum possible score */
    int cur_score ;   /* score of current column */
    unsigned int hash ;   /* hash value for supernode detection */
    int head_column ;   /* head of hash bucket */
    int first_col ;   /* first column in hash bucket */
    int tag_mark ;    /* marker value for mark array */
    int row_mark ;    /* Row [row].shared2.mark */
    int set_difference ;  /* set difference size of row with pivot row */
    int min_score ;   /* smallest column score */
    int col_thickness ;   /* "thickness" (no. of columns in a supercol) */
    int max_mark ;    /* maximum value of tag_mark */
    int pivot_col_thickness ; /* number of columns represented by pivot col */
    int prev_col ;    /* Used by Dlist operations. */
    int next_col ;    /* Used by Dlist operations. */
    int ngarbage ;    /* number of garbage collections performed */

#ifndef COLAMD_NDEBUG
    int debug_d ;   /* debug loop counter */
    int debug_step = 0 ;  /* debug loop counter */
#endif /* COLAMD_NDEBUG */

    /* === Initialization and clear mark ==================================== */

    max_mark = INT_MAX - n_col ;  /* INT_MAX defined in <limits.h> */
    tag_mark = eigen_clear_mark (n_row, Row) ;
    min_score = 0 ;
    ngarbage = 0 ;
    COLAMD_DEBUG1 (("eigen_colamd: Ordering, n_col2=%d\n", n_col2)) ;

    /* === Order the columns ================================================ */

    for (k = 0 ; k < n_col2 ; /* 'k' is incremented below */)
    {

#ifndef COLAMD_NDEBUG
  if (debug_step % 100 == 0)
  {
      COLAMD_DEBUG2 (("\n...       Step k: %d out of n_col2: %d\n", k, n_col2)) ;
  }
  else
  {
      COLAMD_DEBUG3 (("\n----------Step k: %d out of n_col2: %d\n", k, n_col2)) ;
  }
  debug_step++ ;
  eigen_debug_deg_lists (n_row, n_col, Row, Col, head,
    min_score, n_col2-k, max_deg) ;
  eigen_debug_matrix (n_row, n_col, Row, Col, A) ;
#endif /* COLAMD_NDEBUG */

  /* === Select pivot column, and order it ============================ */

  /* make sure degree list isn't empty */
  COLAMD_ASSERT (min_score >= 0) ;
  COLAMD_ASSERT (min_score <= n_col) ;
  COLAMD_ASSERT (head [min_score] >= EIGEN_COLAMD_EMPTY) ;

#ifndef COLAMD_NDEBUG
  for (debug_d = 0 ; debug_d < min_score ; debug_d++)
  {
      COLAMD_ASSERT (head [debug_d] == EIGEN_COLAMD_EMPTY) ;
  }
#endif /* COLAMD_NDEBUG */

  /* get pivot column from head of minimum degree list */
  while (head [min_score] == EIGEN_COLAMD_EMPTY && min_score < n_col)
  {
      min_score++ ;
  }
  pivot_col = head [min_score] ;
  COLAMD_ASSERT (pivot_col >= 0 && pivot_col <= n_col) ;
  next_col = Col [pivot_col].shared4.degree_next ;
  head [min_score] = next_col ;
  if (next_col != EIGEN_COLAMD_EMPTY)
  {
      Col [next_col].shared3.prev = EIGEN_COLAMD_EMPTY ;
  }

  COLAMD_ASSERT (EIGEN_COL_IS_ALIVE (pivot_col)) ;
  COLAMD_DEBUG3 (("Pivot col: %d\n", pivot_col)) ;

  /* remember score for defrag check */
  pivot_col_score = Col [pivot_col].shared2.score ;

  /* the pivot column is the kth column in the pivot order */
  Col [pivot_col].shared2.order = k ;

  /* increment order count by column thickness */
  pivot_col_thickness = Col [pivot_col].shared1.thickness ;
  k += pivot_col_thickness ;
  COLAMD_ASSERT (pivot_col_thickness > 0) ;

  /* === Garbage_collection, if necessary ============================= */

  needed_memory = COLAMD_MIN (pivot_col_score, n_col - k) ;
  if (pfree + needed_memory >= Alen)
  {
      pfree = eigen_garbage_collection (n_row, n_col, Row, Col, A, &A [pfree]) ;
      ngarbage++ ;
      /* after garbage collection we will have enough */
      COLAMD_ASSERT (pfree + needed_memory < Alen) ;
      /* garbage collection has wiped out the Row[].shared2.mark array */
      tag_mark = eigen_clear_mark (n_row, Row) ;

#ifndef COLAMD_NDEBUG
      eigen_debug_matrix (n_row, n_col, Row, Col, A) ;
#endif /* COLAMD_NDEBUG */
  }

  /* === Compute pivot row pattern ==================================== */

  /* get starting location for this new merged row */
  pivot_row_start = pfree ;

  /* initialize new row counts to zero */
  pivot_row_degree = 0 ;

  /* tag pivot column as having been visited so it isn't included */
  /* in merged pivot row */
  Col [pivot_col].shared1.thickness = -pivot_col_thickness ;

  /* pivot row is the union of all rows in the pivot column pattern */
  cp = &A [Col [pivot_col].start] ;
  cp_end = cp + Col [pivot_col].length ;
  while (cp < cp_end)
  {
      /* get a row */
      row = *cp++ ;
      COLAMD_DEBUG4 (("Pivot col pattern %d %d\n", EIGEN_ROW_IS_ALIVE (row), row)) ;
      /* skip if row is dead */
      if (EIGEN_ROW_IS_DEAD (row))
      {
    continue ;
      }
      rp = &A [Row [row].start] ;
      rp_end = rp + Row [row].length ;
      while (rp < rp_end)
      {
    /* get a column */
    col = *rp++ ;
    /* add the column, if alive and untagged */
    col_thickness = Col [col].shared1.thickness ;
    if (col_thickness > 0 && EIGEN_COL_IS_ALIVE (col))
    {
        /* tag column in pivot row */
        Col [col].shared1.thickness = -col_thickness ;
        COLAMD_ASSERT (pfree < Alen) ;
        /* place column in pivot row */
        A [pfree++] = col ;
        pivot_row_degree += col_thickness ;
    }
      }
  }

  /* clear tag on pivot column */
  Col [pivot_col].shared1.thickness = pivot_col_thickness ;
  max_deg = COLAMD_MAX (max_deg, pivot_row_degree) ;

#ifndef COLAMD_NDEBUG
  COLAMD_DEBUG3 (("check2\n")) ;
  eigen_debug_mark (n_row, Row, tag_mark, max_mark) ;
#endif /* COLAMD_NDEBUG */

  /* === Kill all rows used to construct pivot row ==================== */

  /* also kill pivot row, temporarily */
  cp = &A [Col [pivot_col].start] ;
  cp_end = cp + Col [pivot_col].length ;
  while (cp < cp_end)
  {
      /* may be killing an already dead row */
      row = *cp++ ;
      COLAMD_DEBUG3 (("Kill row in pivot col: %d\n", row)) ;
      EIGEN_KILL_ROW (row) ;
  }

  /* === Select a row index to use as the new pivot row =============== */

  pivot_row_length = pfree - pivot_row_start ;
  if (pivot_row_length > 0)
  {
      /* pick the "pivot" row arbitrarily (first row in col) */
      pivot_row = A [Col [pivot_col].start] ;
      COLAMD_DEBUG3 (("Pivotal row is %d\n", pivot_row)) ;
  }
  else
  {
      /* there is no pivot row, since it is of zero length */
      pivot_row = EIGEN_COLAMD_EMPTY ;
      COLAMD_ASSERT (pivot_row_length == 0) ;
  }
  COLAMD_ASSERT (Col [pivot_col].length > 0 || pivot_row_length == 0) ;

  /* === Approximate degree computation =============================== */

  /* Here begins the computation of the approximate degree.  The column */
  /* score is the sum of the pivot row "length", plus the size of the */
  /* set differences of each row in the column minus the pattern of the */
  /* pivot row itself.  The column ("thickness") itself is also */
  /* excluded from the column score (we thus use an approximate */
  /* external degree). */

  /* The time taken by the following code (compute set differences, and */
  /* add them up) is proportional to the size of the data structure */
  /* being scanned - that is, the sum of the sizes of each column in */
  /* the pivot row.  Thus, the amortized time to compute a column score */
  /* is proportional to the size of that column (where size, in this */
  /* context, is the column "length", or the number of row indices */
  /* in that column).  The number of row indices in a column is */
  /* monotonically non-decreasing, from the length of the original */
  /* column on input to eigen_colamd. */

  /* === Compute set differences ====================================== */

  COLAMD_DEBUG3 (("** Computing set differences phase. **\n")) ;

  /* pivot row is currently dead - it will be revived later. */

  COLAMD_DEBUG3 (("Pivot row: ")) ;
  /* for each column in pivot row */
  rp = &A [pivot_row_start] ;
  rp_end = rp + pivot_row_length ;
  while (rp < rp_end)
  {
      col = *rp++ ;
      COLAMD_ASSERT (EIGEN_COL_IS_ALIVE (col) && col != pivot_col) ;
      COLAMD_DEBUG3 (("Col: %d\n", col)) ;

      /* clear tags used to construct pivot row pattern */
      col_thickness = -Col [col].shared1.thickness ;
      COLAMD_ASSERT (col_thickness > 0) ;
      Col [col].shared1.thickness = col_thickness ;

      /* === Remove column from degree list =========================== */

      cur_score = Col [col].shared2.score ;
      prev_col = Col [col].shared3.prev ;
      next_col = Col [col].shared4.degree_next ;
      COLAMD_ASSERT (cur_score >= 0) ;
      COLAMD_ASSERT (cur_score <= n_col) ;
      COLAMD_ASSERT (cur_score >= EIGEN_COLAMD_EMPTY) ;
      if (prev_col == EIGEN_COLAMD_EMPTY)
      {
    head [cur_score] = next_col ;
      }
      else
      {
    Col [prev_col].shared4.degree_next = next_col ;
      }
      if (next_col != EIGEN_COLAMD_EMPTY)
      {
    Col [next_col].shared3.prev = prev_col ;
      }

      /* === Scan the column ========================================== */

      cp = &A [Col [col].start] ;
      cp_end = cp + Col [col].length ;
      while (cp < cp_end)
      {
    /* get a row */
    row = *cp++ ;
    row_mark = Row [row].shared2.mark ;
    /* skip if dead */
    if (EIGEN_ROW_IS_MARKED_DEAD (row_mark))
    {
        continue ;
    }
    COLAMD_ASSERT (row != pivot_row) ;
    set_difference = row_mark - tag_mark ;
    /* check if the row has been seen yet */
    if (set_difference < 0)
    {
        COLAMD_ASSERT (Row [row].shared1.degree <= max_deg) ;
        set_difference = Row [row].shared1.degree ;
    }
    /* subtract column thickness from this row's set difference */
    set_difference -= col_thickness ;
    COLAMD_ASSERT (set_difference >= 0) ;
    /* absorb this row if the set difference becomes zero */
    if (set_difference == 0)
    {
        COLAMD_DEBUG3 (("aggressive absorption. Row: %d\n", row)) ;
        EIGEN_KILL_ROW (row) ;
    }
    else
    {
        /* save the new mark */
        Row [row].shared2.mark = set_difference + tag_mark ;
    }
      }
  }

#ifndef COLAMD_NDEBUG
  eigen_debug_deg_lists (n_row, n_col, Row, Col, head,
    min_score, n_col2-k-pivot_row_degree, max_deg) ;
#endif /* COLAMD_NDEBUG */

  /* === Add up set differences for each column ======================= */

  COLAMD_DEBUG3 (("** Adding set differences phase. **\n")) ;

  /* for each column in pivot row */
  rp = &A [pivot_row_start] ;
  rp_end = rp + pivot_row_length ;
  while (rp < rp_end)
  {
      /* get a column */
      col = *rp++ ;
      COLAMD_ASSERT (EIGEN_COL_IS_ALIVE (col) && col != pivot_col) ;
      hash = 0 ;
      cur_score = 0 ;
      cp = &A [Col [col].start] ;
      /* compact the column */
      new_cp = cp ;
      cp_end = cp + Col [col].length ;

      COLAMD_DEBUG4 (("Adding set diffs for Col: %d.\n", col)) ;

      while (cp < cp_end)
      {
    /* get a row */
    row = *cp++ ;
    COLAMD_ASSERT(row >= 0 && row < n_row) ;
    row_mark = Row [row].shared2.mark ;
    /* skip if dead */
    if (EIGEN_ROW_IS_MARKED_DEAD (row_mark))
    {
        continue ;
    }
    COLAMD_ASSERT (row_mark > tag_mark) ;
    /* compact the column */
    *new_cp++ = row ;
    /* compute hash function */
    hash += row ;
    /* add set difference */
    cur_score += row_mark - tag_mark ;
    /* integer overflow... */
    cur_score = COLAMD_MIN (cur_score, n_col) ;
      }

      /* recompute the column's length */
      Col [col].length = (int) (new_cp - &A [Col [col].start]) ;

      /* === Further mass elimination ================================= */

      if (Col [col].length == 0)
      {
    COLAMD_DEBUG4 (("further mass elimination. Col: %d\n", col)) ;
    /* nothing left but the pivot row in this column */
    EIGEN_KILL_PRINCIPAL_COL (col) ;
    pivot_row_degree -= Col [col].shared1.thickness ;
    COLAMD_ASSERT (pivot_row_degree >= 0) ;
    /* order it */
    Col [col].shared2.order = k ;
    /* increment order count by column thickness */
    k += Col [col].shared1.thickness ;
      }
      else
      {
    /* === Prepare for supercolumn detection ==================== */

    COLAMD_DEBUG4 (("Preparing supercol detection for Col: %d.\n", col)) ;

    /* save score so far */
    Col [col].shared2.score = cur_score ;

    /* add column to hash table, for supercolumn detection */
    hash %= n_col + 1 ;

    COLAMD_DEBUG4 ((" Hash = %d, n_col = %d.\n", hash, n_col)) ;
    COLAMD_ASSERT (hash <= n_col) ;

    head_column = head [hash] ;
    if (head_column > EIGEN_COLAMD_EMPTY)
    {
        /* degree list "hash" is non-empty, use prev (shared3) of */
        /* first column in degree list as head of hash bucket */
        first_col = Col [head_column].shared3.headhash ;
        Col [head_column].shared3.headhash = col ;
    }
    else
    {
        /* degree list "hash" is empty, use head as hash bucket */
        first_col = - (head_column + 2) ;
        head [hash] = - (col + 2) ;
    }
    Col [col].shared4.hash_next = first_col ;

    /* save hash function in Col [col].shared3.hash */
    Col [col].shared3.hash = (int) hash ;
    COLAMD_ASSERT (EIGEN_COL_IS_ALIVE (col)) ;
      }
  }

  /* The approximate external column degree is now computed.  */

  /* === Supercolumn detection ======================================== */

  COLAMD_DEBUG3 (("** Supercolumn detection phase. **\n")) ;

  eigen_detect_super_cols (

#ifndef COLAMD_NDEBUG
    n_col, Row,
#endif /* COLAMD_NDEBUG */

    Col, A, head, pivot_row_start, pivot_row_length) ;

  /* === Kill the pivotal column ====================================== */

  EIGEN_KILL_PRINCIPAL_COL (pivot_col) ;

  /* === Clear mark =================================================== */

  tag_mark += (max_deg + 1) ;
  if (tag_mark >= max_mark)
  {
      COLAMD_DEBUG2 (("clearing tag_mark\n")) ;
      tag_mark = eigen_clear_mark (n_row, Row) ;
  }

#ifndef COLAMD_NDEBUG
  COLAMD_DEBUG3 (("check3\n")) ;
  eigen_debug_mark (n_row, Row, tag_mark, max_mark) ;
#endif /* COLAMD_NDEBUG */

  /* === Finalize the new pivot row, and column scores ================ */

  COLAMD_DEBUG3 (("** Finalize scores phase. **\n")) ;

  /* for each column in pivot row */
  rp = &A [pivot_row_start] ;
  /* compact the pivot row */
  new_rp = rp ;
  rp_end = rp + pivot_row_length ;
  while (rp < rp_end)
  {
      col = *rp++ ;
      /* skip dead columns */
      if (EIGEN_COL_IS_DEAD (col))
      {
    continue ;
      }
      *new_rp++ = col ;
      /* add new pivot row to column */
      A [Col [col].start + (Col [col].length++)] = pivot_row ;

      /* retrieve score so far and add on pivot row's degree. */
      /* (we wait until here for this in case the pivot */
      /* row's degree was reduced due to mass elimination). */
      cur_score = Col [col].shared2.score + pivot_row_degree ;

      /* calculate the max possible score as the number of */
      /* external columns minus the 'k' value minus the */
      /* columns thickness */
      max_score = n_col - k - Col [col].shared1.thickness ;

      /* make the score the external degree of the union-of-rows */
      cur_score -= Col [col].shared1.thickness ;

      /* make sure score is less or equal than the max score */
      cur_score = COLAMD_MIN (cur_score, max_score) ;
      COLAMD_ASSERT (cur_score >= 0) ;

      /* store updated score */
      Col [col].shared2.score = cur_score ;

      /* === Place column back in degree list ========================= */

      COLAMD_ASSERT (min_score >= 0) ;
      COLAMD_ASSERT (min_score <= n_col) ;
      COLAMD_ASSERT (cur_score >= 0) ;
      COLAMD_ASSERT (cur_score <= n_col) ;
      COLAMD_ASSERT (head [cur_score] >= EIGEN_COLAMD_EMPTY) ;
      next_col = head [cur_score] ;
      Col [col].shared4.degree_next = next_col ;
      Col [col].shared3.prev = EIGEN_COLAMD_EMPTY ;
      if (next_col != EIGEN_COLAMD_EMPTY)
      {
    Col [next_col].shared3.prev = col ;
      }
      head [cur_score] = col ;

      /* see if this score is less than current min */
      min_score = COLAMD_MIN (min_score, cur_score) ;

  }

#ifndef COLAMD_NDEBUG
  eigen_debug_deg_lists (n_row, n_col, Row, Col, head,
    min_score, n_col2-k, max_deg) ;
#endif /* COLAMD_NDEBUG */

  /* === Resurrect the new pivot row ================================== */

  if (pivot_row_degree > 0)
  {
      /* update pivot row length to reflect any cols that were killed */
      /* during super-col detection and mass elimination */
      Row [pivot_row].start  = pivot_row_start ;
      Row [pivot_row].length = (int) (new_rp - &A[pivot_row_start]) ;
      Row [pivot_row].shared1.degree = pivot_row_degree ;
      Row [pivot_row].shared2.mark = 0 ;
      /* pivot row is no longer dead */
  }
    }

    /* === All principal columns have now been ordered ====================== */

    return (ngarbage) ;
}


/* ========================================================================== */
/* === eigen_order_children ======================================================= */
/* ========================================================================== */

/*
    The eigen_find_ordering routine has ordered all of the principal columns (the
    representatives of the supercolumns).  The non-principal columns have not
    yet been ordered.  This routine orders those columns by walking up the
    parent tree (a column is a child of the column which absorbed it).  The
    final permutation vector is then placed in p [0 ... n_col-1], with p [0]
    being the first column, and p [n_col-1] being the last.  It doesn't look
    like it at first glance, but be assured that this routine takes time linear
    in the number of columns.  Although not immediately obvious, the time
    taken by this routine is O (n_col), that is, linear in the number of
    columns.  Not user-callable.
*/

 void eigen_order_children
(
    /* === Parameters ======================================================= */

    int n_col,      /* number of columns of A */
    EIGEN_Colamd_Col Col [],    /* of size n_col+1 */
    int p []      /* p [0 ... n_col-1] is the column permutation*/
)
{
    /* === Local variables ================================================== */

    int i ;     /* loop counter for all columns */
    int c ;     /* column index */
    int parent ;    /* index of column's parent */
    int order ;     /* column's order */

    /* === Order each non-principal column ================================== */

    for (i = 0 ; i < n_col ; i++)
    {
  /* find an un-ordered non-principal column */
  COLAMD_ASSERT (EIGEN_COL_IS_DEAD (i)) ;
  if (!EIGEN_EIGEN_COL_IS_DEAD_PRINCIPAL (i) && Col [i].shared2.order == EIGEN_COLAMD_EMPTY)
  {
      parent = i ;
      /* once found, find its principal parent */
      do
      {
    parent = Col [parent].shared1.parent ;
      } while (!EIGEN_EIGEN_COL_IS_DEAD_PRINCIPAL (parent)) ;

      /* now, order all un-ordered non-principal columns along path */
      /* to this parent.  collapse tree at the same time */
      c = i ;
      /* get order of parent */
      order = Col [parent].shared2.order ;

      do
      {
    COLAMD_ASSERT (Col [c].shared2.order == EIGEN_COLAMD_EMPTY) ;

    /* order this column */
    Col [c].shared2.order = order++ ;
    /* collaps tree */
    Col [c].shared1.parent = parent ;

    /* get immediate parent of this column */
    c = Col [c].shared1.parent ;

    /* continue until we hit an ordered column.  There are */
    /* guarranteed not to be anymore unordered columns */
    /* above an ordered column */
      } while (Col [c].shared2.order == EIGEN_COLAMD_EMPTY) ;

      /* re-order the super_col parent to largest order for this group */
      Col [parent].shared2.order = order ;
  }
    }

    /* === Generate the permutation ========================================= */

    for (c = 0 ; c < n_col ; c++)
    {
  p [Col [c].shared2.order] = c ;
    }
}


/* ========================================================================== */
/* === eigen_detect_super_cols ==================================================== */
/* ========================================================================== */

/*
    Detects supercolumns by finding matches between columns in the hash buckets.
    Check amongst columns in the set A [row_start ... row_start + row_length-1].
    The columns under consideration are currently *not* in the degree lists,
    and have already been placed in the hash buckets.

    The hash bucket for columns whose hash function is equal to h is stored
    as follows:

  if head [h] is >= 0, then head [h] contains a degree list, so:

    head [h] is the first column in degree bucket h.
    Col [head [h]].headhash gives the first column in hash bucket h.

  otherwise, the degree list is empty, and:

    -(head [h] + 2) is the first column in hash bucket h.

    For a column c in a hash bucket, Col [c].shared3.prev is NOT a "previous
    column" pointer.  Col [c].shared3.hash is used instead as the hash number
    for that column.  The value of Col [c].shared4.hash_next is the next column
    in the same hash bucket.

    Assuming no, or "few" hash collisions, the time taken by this routine is
    linear in the sum of the sizes (lengths) of each column whose score has
    just been computed in the approximate degree computation.
    Not user-callable.
*/

 void eigen_detect_super_cols
(
    /* === Parameters ======================================================= */

#ifndef COLAMD_NDEBUG
    /* these two parameters are only needed when debugging is enabled: */
    int n_col,      /* number of columns of A */
    EIGEN_Colamd_Row Row [],    /* of size n_row+1 */
#endif /* COLAMD_NDEBUG */

    EIGEN_Colamd_Col Col [],    /* of size n_col+1 */
    int A [],     /* row indices of A */
    int head [],    /* head of degree lists and hash buckets */
    int row_start,    /* pointer to set of columns to check */
    int row_length    /* number of columns to check */
)
{
    /* === Local variables ================================================== */

    int hash ;      /* hash value for a column */
    int *rp ;     /* pointer to a row */
    int c ;     /* a column index */
    int super_c ;   /* column index of the column to absorb into */
    int *cp1 ;      /* column pointer for column super_c */
    int *cp2 ;      /* column pointer for column c */
    int length ;    /* length of column super_c */
    int prev_c ;    /* column preceding c in hash bucket */
    int i ;     /* loop counter */
    int *rp_end ;   /* pointer to the end of the row */
    int col ;     /* a column index in the row to check */
    int head_column ;   /* first column in hash bucket or degree list */
    int first_col ;   /* first column in hash bucket */

    /* === Consider each column in the row ================================== */

    rp = &A [row_start] ;
    rp_end = rp + row_length ;
    while (rp < rp_end)
    {
  col = *rp++ ;
  if (EIGEN_COL_IS_DEAD (col))
  {
      continue ;
  }

  /* get hash number for this column */
  hash = Col [col].shared3.hash ;
  COLAMD_ASSERT (hash <= n_col) ;

  /* === Get the first column in this hash bucket ===================== */

  head_column = head [hash] ;
  if (head_column > EIGEN_COLAMD_EMPTY)
  {
      first_col = Col [head_column].shared3.headhash ;
  }
  else
  {
      first_col = - (head_column + 2) ;
  }

  /* === Consider each column in the hash bucket ====================== */

  for (super_c = first_col ; super_c != EIGEN_COLAMD_EMPTY ;
      super_c = Col [super_c].shared4.hash_next)
  {
      COLAMD_ASSERT (EIGEN_COL_IS_ALIVE (super_c)) ;
      COLAMD_ASSERT (Col [super_c].shared3.hash == hash) ;
      length = Col [super_c].length ;

      /* prev_c is the column preceding column c in the hash bucket */
      prev_c = super_c ;

      /* === Compare super_c with all columns after it ================ */

      for (c = Col [super_c].shared4.hash_next ;
     c != EIGEN_COLAMD_EMPTY ; c = Col [c].shared4.hash_next)
      {
    COLAMD_ASSERT (c != super_c) ;
    COLAMD_ASSERT (EIGEN_COL_IS_ALIVE (c)) ;
    COLAMD_ASSERT (Col [c].shared3.hash == hash) ;

    /* not identical if lengths or scores are different */
    if (Col [c].length != length ||
        Col [c].shared2.score != Col [super_c].shared2.score)
    {
        prev_c = c ;
        continue ;
    }

    /* compare the two columns */
    cp1 = &A [Col [super_c].start] ;
    cp2 = &A [Col [c].start] ;

    for (i = 0 ; i < length ; i++)
    {
        /* the columns are "clean" (no dead rows) */
        COLAMD_ASSERT (EIGEN_ROW_IS_ALIVE (*cp1))  ;
        COLAMD_ASSERT (EIGEN_ROW_IS_ALIVE (*cp2))  ;
        /* row indices will same order for both supercols, */
        /* no gather scatter nessasary */
        if (*cp1++ != *cp2++)
        {
      break ;
        }
    }

    /* the two columns are different if the for-loop "broke" */
    if (i != length)
    {
        prev_c = c ;
        continue ;
    }

    /* === Got it!  two columns are identical =================== */

    COLAMD_ASSERT (Col [c].shared2.score == Col [super_c].shared2.score) ;

    Col [super_c].shared1.thickness += Col [c].shared1.thickness ;
    Col [c].shared1.parent = super_c ;
    EIGEN_KILL_NON_PRINCIPAL_COL (c) ;
    /* order c later, in eigen_order_children() */
    Col [c].shared2.order = EIGEN_COLAMD_EMPTY ;
    /* remove c from hash bucket */
    Col [prev_c].shared4.hash_next = Col [c].shared4.hash_next ;
      }
  }

  /* === Empty this hash bucket ======================================= */

  if (head_column > EIGEN_COLAMD_EMPTY)
  {
      /* corresponding degree list "hash" is not empty */
      Col [head_column].shared3.headhash = EIGEN_COLAMD_EMPTY ;
  }
  else
  {
      /* corresponding degree list "hash" is empty */
      head [hash] = EIGEN_COLAMD_EMPTY ;
  }
    }
}


/* ========================================================================== */
/* === eigen_garbage_collection =================================================== */
/* ========================================================================== */

/*
    Defragments and compacts columns and rows in the workspace A.  Used when
    all avaliable memory has been used while performing row merging.  Returns
    the index of the first free position in A, after garbage collection.  The
    time taken by this routine is linear is the size of the array A, which is
    itself linear in the number of nonzeros in the input matrix.
    Not user-callable.
*/

 int eigen_garbage_collection  /* returns the new value of pfree */
(
    /* === Parameters ======================================================= */

    int n_row,      /* number of rows */
    int n_col,      /* number of columns */
    EIGEN_Colamd_Row Row [],    /* row info */
    EIGEN_Colamd_Col Col [],    /* column info */
    int A [],     /* A [0 ... Alen-1] holds the matrix */
    int *pfree      /* &A [0] ... pfree is in use */
)
{
    /* === Local variables ================================================== */

    int *psrc ;     /* source pointer */
    int *pdest ;    /* destination pointer */
    int j ;     /* counter */
    int r ;     /* a row index */
    int c ;     /* a column index */
    int length ;    /* length of a row or column */

#ifndef COLAMD_NDEBUG
    int debug_rows ;
    COLAMD_DEBUG2 (("Defrag..\n")) ;
    for (psrc = &A[0] ; psrc < pfree ; psrc++) COLAMD_ASSERT (*psrc >= 0) ;
    debug_rows = 0 ;
#endif /* COLAMD_NDEBUG */

    /* === Defragment the columns =========================================== */

    pdest = &A[0] ;
    for (c = 0 ; c < n_col ; c++)
    {
  if (EIGEN_COL_IS_ALIVE (c))
  {
      psrc = &A [Col [c].start] ;

      /* move and compact the column */
      COLAMD_ASSERT (pdest <= psrc) ;
      Col [c].start = (int) (pdest - &A [0]) ;
      length = Col [c].length ;
      for (j = 0 ; j < length ; j++)
      {
    r = *psrc++ ;
    if (EIGEN_ROW_IS_ALIVE (r))
    {
        *pdest++ = r ;
    }
      }
      Col [c].length = (int) (pdest - &A [Col [c].start]) ;
  }
    }

    /* === Prepare to defragment the rows =================================== */

    for (r = 0 ; r < n_row ; r++)
    {
  if (EIGEN_ROW_IS_ALIVE (r))
  {
      if (Row [r].length == 0)
      {
    /* this row is of zero length.  cannot compact it, so kill it */
    COLAMD_DEBUG3 (("Defrag row kill\n")) ;
    EIGEN_KILL_ROW (r) ;
      }
      else
      {
    /* save first column index in Row [r].shared2.first_column */
    psrc = &A [Row [r].start] ;
    Row [r].shared2.first_column = *psrc ;
    COLAMD_ASSERT (EIGEN_ROW_IS_ALIVE (r)) ;
    /* flag the start of the row with the one's complement of row */
    *psrc = EIGEN_ONES_COMPLEMENT (r) ;

#ifndef COLAMD_NDEBUG
    debug_rows++ ;
#endif /* COLAMD_NDEBUG */

      }
  }
    }

    /* === Defragment the rows ============================================== */

    psrc = pdest ;
    while (psrc < pfree)
    {
  /* find a negative number ... the start of a row */
  if (*psrc++ < 0)
  {
      psrc-- ;
      /* get the row index */
      r = EIGEN_ONES_COMPLEMENT (*psrc) ;
      COLAMD_ASSERT (r >= 0 && r < n_row) ;
      /* restore first column index */
      *psrc = Row [r].shared2.first_column ;
      COLAMD_ASSERT (EIGEN_ROW_IS_ALIVE (r)) ;

      /* move and compact the row */
      COLAMD_ASSERT (pdest <= psrc) ;
      Row [r].start = (int) (pdest - &A [0]) ;
      length = Row [r].length ;
      for (j = 0 ; j < length ; j++)
      {
    c = *psrc++ ;
    if (EIGEN_COL_IS_ALIVE (c))
    {
        *pdest++ = c ;
    }
      }
      Row [r].length = (int) (pdest - &A [Row [r].start]) ;

#ifndef COLAMD_NDEBUG
      debug_rows-- ;
#endif /* COLAMD_NDEBUG */

  }
    }
    /* ensure we found all the rows */
    COLAMD_ASSERT (debug_rows == 0) ;

    /* === Return the new value of pfree ==================================== */

    return ((int) (pdest - &A [0])) ;
}


/* ========================================================================== */
/* === eigen_clear_mark =========================================================== */
/* ========================================================================== */

/*
    Clears the Row [].shared2.mark array, and returns the new tag_mark.
    Return value is the new tag_mark.  Not user-callable.
*/

 int eigen_clear_mark  /* return the new value for tag_mark */
(
    /* === Parameters ======================================================= */

    int n_row,    /* number of rows in A */
    EIGEN_Colamd_Row Row [] /* Row [0 ... n_row-1].shared2.mark is set to zero */
)
{
    /* === Local variables ================================================== */

    int r ;

    for (r = 0 ; r < n_row ; r++)
    {
  if (EIGEN_ROW_IS_ALIVE (r))
  {
      Row [r].shared2.mark = 0 ;
  }
    }
    return (1) ;
}



/* ========================================================================== */
/* === eigen_print_report ========================================================= */
/* ========================================================================== */

 void eigen_print_report
(
    char *method,
    int stats [EIGEN_COLAMD_STATS]
)
{

    int i1, i2, i3 ;

    if (!stats)
    {
      PRINTF ("%s: No statistics available.\n", method) ;
  return ;
    }

    i1 = stats [EIGEN_COLAMD_INFO1] ;
    i2 = stats [EIGEN_COLAMD_INFO2] ;
    i3 = stats [EIGEN_COLAMD_INFO3] ;

    if (stats [EIGEN_COLAMD_STATUS] >= 0)
    {
      PRINTF ("%s: OK.  ", method) ;
    }
    else
    {
      PRINTF ("%s: ERROR.  ", method) ;
    }

    switch (stats [EIGEN_COLAMD_STATUS])
    {

  case EIGEN_COLAMD_OK_BUT_JUMBLED:

      PRINTF ("Matrix has unsorted or duplicate row indices.\n") ;

      PRINTF ("%s: number of duplicate or out-of-order row indices: %d\n",
      method, i3) ;

      PRINTF ("%s: last seen duplicate or out-of-order row index:   %d\n",
      method, INDEX (i2)) ;

      PRINTF ("%s: last seen in column:                             %d",
      method, INDEX (i1)) ;

      /* no break - fall through to next case instead */

  case EIGEN_COLAMD_OK:

      PRINTF ("\n") ;

      PRINTF ("%s: number of dense or empty rows ignored:           %d\n",
      method, stats [EIGEN_COLAMD_DENSE_ROW]) ;

      PRINTF ("%s: number of dense or empty columns ignored:        %d\n",
      method, stats [EIGEN_COLAMD_DENSE_COL]) ;

      PRINTF ("%s: number of garbage collections performed:         %d\n",
      method, stats [EIGEN_COLAMD_DEFRAG_COUNT]) ;
      break ;

  case EIGEN_COLAMD_ERROR_A_not_present:

      PRINTF ("Array A (row indices of matrix) not present.\n") ;
      break ;

  case EIGEN_COLAMD_ERROR_p_not_present:

      PRINTF ("Array p (column pointers for matrix) not present.\n") ;
      break ;

  case EIGEN_COLAMD_ERROR_nrow_negative:

      PRINTF ("Invalid number of rows (%d).\n", i1) ;
      break ;

  case EIGEN_COLAMD_ERROR_ncol_negative:

      PRINTF ("Invalid number of columns (%d).\n", i1) ;
      break ;

  case EIGEN_COLAMD_ERROR_nnz_negative:

      PRINTF ("Invalid number of nonzero entries (%d).\n", i1) ;
      break ;

  case EIGEN_COLAMD_ERROR_p0_nonzero:

      PRINTF ("Invalid column pointer, p [0] = %d, must be zero.\n", i1) ;
      break ;

  case EIGEN_COLAMD_ERROR_A_too_small:

      PRINTF ("Array A too small.\n") ;
      PRINTF ("        Need Alen >= %d, but given only Alen = %d.\n",
      i1, i2) ;
      break ;

  case EIGEN_COLAMD_ERROR_col_length_negative:

      PRINTF
      ("Column %d has a negative number of nonzero entries (%d).\n",
      INDEX (i1), i2) ;
      break ;

  case EIGEN_COLAMD_ERROR_row_index_out_of_bounds:

      PRINTF
      ("Row index (row %d) out of bounds (%d to %d) in column %d.\n",
      INDEX (i2), INDEX (0), INDEX (i3-1), INDEX (i1)) ;
      break ;

  case EIGEN_COLAMD_ERROR_out_of_memory:

      PRINTF ("Out of memory.\n") ;
      break ;

  case EIGEN_COLAMD_ERROR_internal_error:

      /* if this happens, there is a bug in the code */
      PRINTF
      ("Internal error! Please contact authors (davis@cise.ufl.edu).\n") ;
      break ;
    }
}




/* ========================================================================== */
/* === eigen_colamd debugging routines ============================================ */
/* ========================================================================== */

/* When debugging is disabled, the remainder of this file is ignored. */

#ifndef COLAMD_NDEBUG


/* ========================================================================== */
/* === eigen_debug_structures ===================================================== */
/* ========================================================================== */

/*
    At this point, all empty rows and columns are dead.  All live columns
    are "clean" (containing no dead rows) and simplicial (no supercolumns
    yet).  Rows may contain dead columns, but all live rows contain at
    least one live column.
*/

 void eigen_debug_structures
(
    /* === Parameters ======================================================= */

    int n_row,
    int n_col,
    EIGEN_Colamd_Row Row [],
    EIGEN_Colamd_Col Col [],
    int A [],
    int n_col2
)
{
    /* === Local variables ================================================== */

    int i ;
    int c ;
    int *cp ;
    int *cp_end ;
    int len ;
    int score ;
    int r ;
    int *rp ;
    int *rp_end ;
    int deg ;

    /* === Check A, Row, and Col ============================================ */

    for (c = 0 ; c < n_col ; c++)
    {
  if (EIGEN_COL_IS_ALIVE (c))
  {
      len = Col [c].length ;
      score = Col [c].shared2.score ;
      COLAMD_DEBUG4 (("initial live col %5d %5d %5d\n", c, len, score)) ;
      COLAMD_ASSERT (len > 0) ;
      COLAMD_ASSERT (score >= 0) ;
      COLAMD_ASSERT (Col [c].shared1.thickness == 1) ;
      cp = &A [Col [c].start] ;
      cp_end = cp + len ;
      while (cp < cp_end)
      {
    r = *cp++ ;
    COLAMD_ASSERT (EIGEN_ROW_IS_ALIVE (r)) ;
      }
  }
  else
  {
      i = Col [c].shared2.order ;
      COLAMD_ASSERT (i >= n_col2 && i < n_col) ;
  }
    }

    for (r = 0 ; r < n_row ; r++)
    {
  if (EIGEN_ROW_IS_ALIVE (r))
  {
      i = 0 ;
      len = Row [r].length ;
      deg = Row [r].shared1.degree ;
      COLAMD_ASSERT (len > 0) ;
      COLAMD_ASSERT (deg > 0) ;
      rp = &A [Row [r].start] ;
      rp_end = rp + len ;
      while (rp < rp_end)
      {
    c = *rp++ ;
    if (EIGEN_COL_IS_ALIVE (c))
    {
        i++ ;
    }
      }
      COLAMD_ASSERT (i > 0) ;
  }
    }
}


/* ========================================================================== */
/* === eigen_debug_deg_lists ====================================================== */
/* ========================================================================== */

/*
    Prints the contents of the degree lists.  Counts the number of columns
    in the degree list and compares it to the total it should have.  Also
    checks the row degrees.
*/

 void eigen_debug_deg_lists
(
    /* === Parameters ======================================================= */

    int n_row,
    int n_col,
    EIGEN_Colamd_Row Row [],
    EIGEN_Colamd_Col Col [],
    int head [],
    int min_score,
    int should,
    int max_deg
)
{
    /* === Local variables ================================================== */

    int deg ;
    int col ;
    int have ;
    int row ;

    /* === Check the degree lists =========================================== */

    if (n_col > 10000 && colamd_debug <= 0)
    {
  return ;
    }
    have = 0 ;
    COLAMD_DEBUG4 (("Degree lists: %d\n", min_score)) ;
    for (deg = 0 ; deg <= n_col ; deg++)
    {
  col = head [deg] ;
  if (col == EIGEN_COLAMD_EMPTY)
  {
      continue ;
  }
  COLAMD_DEBUG4 (("%d:", deg)) ;
  while (col != EIGEN_COLAMD_EMPTY)
  {
      COLAMD_DEBUG4 ((" %d", col)) ;
      have += Col [col].shared1.thickness ;
      COLAMD_ASSERT (EIGEN_COL_IS_ALIVE (col)) ;
      col = Col [col].shared4.degree_next ;
  }
  COLAMD_DEBUG4 (("\n")) ;
    }
    COLAMD_DEBUG4 (("should %d have %d\n", should, have)) ;
    COLAMD_ASSERT (should == have) ;

    /* === Check the row degrees ============================================ */

    if (n_row > 10000 && colamd_debug <= 0)
    {
  return ;
    }
    for (row = 0 ; row < n_row ; row++)
    {
  if (EIGEN_ROW_IS_ALIVE (row))
  {
      COLAMD_ASSERT (Row [row].shared1.degree <= max_deg) ;
  }
    }
}


/* ========================================================================== */
/* === eigen_debug_mark =========================================================== */
/* ========================================================================== */

/*
    Ensures that the tag_mark is less that the maximum and also ensures that
    each entry in the mark array is less than the tag mark.
*/

 void eigen_debug_mark
(
    /* === Parameters ======================================================= */

    int n_row,
    EIGEN_Colamd_Row Row [],
    int tag_mark,
    int max_mark
)
{
    /* === Local variables ================================================== */

    int r ;

    /* === Check the Row marks ============================================== */

    COLAMD_ASSERT (tag_mark > 0 && tag_mark <= max_mark) ;
    if (n_row > 10000 && colamd_debug <= 0)
    {
  return ;
    }
    for (r = 0 ; r < n_row ; r++)
    {
  COLAMD_ASSERT (Row [r].shared2.mark < tag_mark) ;
    }
}


/* ========================================================================== */
/* === eigen_debug_matrix ========================================================= */
/* ========================================================================== */

/*
    Prints out the contents of the columns and the rows.
*/

 void eigen_debug_matrix
(
    /* === Parameters ======================================================= */

    int n_row,
    int n_col,
    EIGEN_Colamd_Row Row [],
    EIGEN_Colamd_Col Col [],
    int A []
)
{
    /* === Local variables ================================================== */

    int r ;
    int c ;
    int *rp ;
    int *rp_end ;
    int *cp ;
    int *cp_end ;

    /* === Dump the rows and columns of the matrix ========================== */

    if (colamd_debug < 3)
    {
  return ;
    }
    COLAMD_DEBUG3 (("DUMP MATRIX:\n")) ;
    for (r = 0 ; r < n_row ; r++)
    {
  COLAMD_DEBUG3 (("Row %d alive? %d\n", r, EIGEN_ROW_IS_ALIVE (r))) ;
  if (EIGEN_ROW_IS_DEAD (r))
  {
      continue ;
  }
  COLAMD_DEBUG3 (("start %d length %d degree %d\n",
    Row [r].start, Row [r].length, Row [r].shared1.degree)) ;
  rp = &A [Row [r].start] ;
  rp_end = rp + Row [r].length ;
  while (rp < rp_end)
  {
      c = *rp++ ;
      COLAMD_DEBUG4 (("  %d col %d\n", EIGEN_COL_IS_ALIVE (c), c)) ;
  }
    }

    for (c = 0 ; c < n_col ; c++)
    {
  COLAMD_DEBUG3 (("Col %d alive? %d\n", c, EIGEN_COL_IS_ALIVE (c))) ;
  if (EIGEN_COL_IS_DEAD (c))
  {
      continue ;
  }
  COLAMD_DEBUG3 (("start %d length %d shared1 %d shared2 %d\n",
    Col [c].start, Col [c].length,
    Col [c].shared1.thickness, Col [c].shared2.score)) ;
  cp = &A [Col [c].start] ;
  cp_end = cp + Col [c].length ;
  while (cp < cp_end)
  {
      r = *cp++ ;
      COLAMD_DEBUG4 (("  %d row %d\n", EIGEN_ROW_IS_ALIVE (r), r)) ;
  }
    }
}

 void eigen_colamd_get_debug
(
    char *method
)
{
    colamd_debug = 0 ;    /* no debug printing */

    /* get "D" environment variable, which gives the debug printing level */
    if (getenv ("D"))
    {
      colamd_debug = atoi (getenv ("D")) ;
    }

    COLAMD_DEBUG0 (("%s: debug version, D = %d (THIS WILL BE SLOW!)\n",
      method, colamd_debug)) ;
}

#endif /* NDEBUG */
#endif
