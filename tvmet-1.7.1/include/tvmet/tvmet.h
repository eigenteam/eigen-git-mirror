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

// this might not work on all platforms. Specifically, Qt uses a
// slightly different solution in the case when compiler==intel &&
// os != windows. See in Qt's source code, the definition of Q_UNUSED
// in src/corelib/global/qglobal.h
#define TVMET_UNUSED(x) (void)x

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
#if defined(EIGEN_USE_COMPLEX)
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
