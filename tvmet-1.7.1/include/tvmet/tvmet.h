/*
 * Tiny Vector Matrix Library
 * Dense Vector Matrix Libary of Tiny size using Expression Templates
 *
 * Copyright (C) 2001 - 2003 Olaf Petzold <opetzold@users.sourceforge.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * $Id: tvmet.h,v 1.17 2004/11/04 16:28:32 opetzold Exp $
 */

#ifndef TVMET_H
#define TVMET_H

#include <tvmet/config.h>


/***********************************************************************
 * Compiler specifics
 ***********************************************************************/
#if defined(__GNUC__)
#  include <tvmet/config/config-gcc.h>
#endif

#if defined(__ICC)
#  include <tvmet/config/config-icc.h>
#endif

#if defined(__KCC)
#  include <tvmet/config/config-kcc.h>
#endif

#if defined(__PGI)
#  include <tvmet/config/config-pgi.h>
#endif

// vc7.1: 1310 and vc7.0 1300
#if defined(_MSC_VER) && (_MSC_VER >= 1310)
#  include <tvmet/config/config-vc71.h>
#endif


// give up for these cases
#if !defined(TVMET_HAVE_MUTABLE)
#  error "Your compiler doesn't support the mutable keyword! Giving up."
#endif

#if !defined(TVMET_HAVE_TYPENAME)
#  error "Your compiler doesn't support the typename keyword! Giving up."
#endif

#if !defined(TVMET_HAVE_NAMESPACES)
#  error "Your compiler doesn't support the namespace concept! Giving up."
#endif

#if !defined(TVMET_HAVE_PARTIAL_SPECIALIZATION)
#  error "Your compiler doesn't support partial specialization! Giving up."
#endif


/*
 * other compiler specific stuff
 */

/**
 * \def TVMET_CXX_ALWAYS_INLINE
 * \brief Compiler specific stuff to force inline code if supported.
 *
 * Mainly, this declares the functions using g++'s
 * __attribute__((always_inline)). This features is enabled
 * on defined TVMET_OPTIMIZE.
 */
#if !defined(TVMET_CXX_ALWAYS_INLINE)
#define TVMET_CXX_ALWAYS_INLINE
#endif


/*
 * Complexity triggers, compiler and architecture specific.
 * If not defined, use defaults.
 */

/**
 * \def TVMET_COMPLEXITY_DEFAULT_TRIGGER
 * \brief Trigger for changing the matrix-product strategy.
 */
#if !defined(TVMET_COMPLEXITY_DEFAULT_TRIGGER)
#  define TVMET_COMPLEXITY_DEFAULT_TRIGGER	1000
#endif

/**
 * \def TVMET_COMPLEXITY_M_ASSIGN_TRIGGER
 * \brief Trigger for changing the matrix assign strategy.
 */
#if !defined(TVMET_COMPLEXITY_M_ASSIGN_TRIGGER)
#  define TVMET_COMPLEXITY_M_ASSIGN_TRIGGER	8*8
#endif

/**
 * \def TVMET_COMPLEXITY_MM_TRIGGER
 * \brief Trigger for changing the matrix-matrix-product strategy.
 * One strategy to build the matrix-matrix-product is to use
 * meta templates. The other to use looping.
 */
#if !defined(TVMET_COMPLEXITY_MM_TRIGGER)
#  define TVMET_COMPLEXITY_MM_TRIGGER		8*8
#endif

/**
 * \def TVMET_COMPLEXITY_V_ASSIGN_TRIGGER
 * \brief Trigger for changing the vector assign strategy.
 */
#if !defined(TVMET_COMPLEXITY_V_ASSIGN_TRIGGER)
#  define TVMET_COMPLEXITY_V_ASSIGN_TRIGGER	8
#endif

/**
 * \def TVMET_COMPLEXITY_MV_TRIGGER
 * \brief Trigger for changing the matrix-vector strategy.
 * One strategy to build the matrix-vector-product is to use
 * meta templates. The other to use looping.
 */
#if !defined(TVMET_COMPLEXITY_MV_TRIGGER)
#  define TVMET_COMPLEXITY_MV_TRIGGER		8*8
#endif


/***********************************************************************
 * other specials
 ***********************************************************************/
#if defined(TVMET_HAVE_IEEE_MATH)
#  define _ALL_SOURCE
#  if !defined(_XOPEN_SOURCE)
#    define _XOPEN_SOURCE
#  endif
#  if !defined(_XOPEN_SOURCE_EXTENDED)
#    define _XOPEN_SOURCE_EXTENDED
#  endif
#endif


/**
 * \def TVMET_DEBUG
 * This is defined if <code>DEBUG</code> is defined. This enables runtime error
 * bounds checking. If you compile %tvmet from another source directory
 * which defines <code>DEBUG</code>, then <code>TVMET_DEBUG</code> will be
 * <b>not</b> defined (This behavior differs from release less than 0.6.0).
 */


/**
 * \def TVMET_OPTIMIZE
 * If this is defined tvmet uses some compiler specific keywords.
 *  Mainly, this declares the functions using gcc's
 * <tt>__attribute__((always_inline))</tt>. This allows the
 * compiler to produce high efficient code even on less
 * optimization levels, like gcc's -O2 or even -O!
 * This is known to work with gcc v3.3.3 (and higher).
 * Using icc's v8 gnuc compatibility mode this may work, I've read
 * that it's using as an hint, this means you can have static inline
 * functions inside left.
 */
#if !defined(TVMET_OPTIMIZE)
#  undef  TVMET_CXX_ALWAYS_INLINE
#  define TVMET_CXX_ALWAYS_INLINE
#endif


/***********************************************************************
 * Namespaces
 ***********************************************************************/


/**
 * \namespace std
 * \brief Imported ISO/IEC 14882:1998 functions from std namespace.
 */

/**
 * \namespace tvmet
 * \brief The namespace for the Tiny %Vector %Matrix using Expression Templates Libary.
 */

/**
 * \namespace tvmet::meta
 * \brief Meta stuff inside here.
 */

/**
 * \namespace tvmet::loop
 * \brief Loop stuff inside here.
 */

/**
 * \namespace tvmet::element_wise
 * \brief Operators inside this namespace does elementwise operations.
 */

/**
 * \namespace tvmet::util
 * \brief Miscellaneous utility functions used.
 */


/***********************************************************************
 * forwards
 ***********************************************************************/
#if defined(TVMET_HAVE_COMPLEX)
namespace std {
  template<class T> class complex;
}
#endif


/***********************************************************************
 * other stuff
 ***********************************************************************/
#include <tvmet/TvmetBase.h>


#endif // TVMET_H

// Local Variables:
// mode:C++
// End:
// LocalWords:  gnuc gcc's icc's std
