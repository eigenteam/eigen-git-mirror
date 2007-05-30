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
 * $Id: Functional.h,v 1.7 2004/09/16 09:14:18 opetzold Exp $
 */

#ifndef TVMET_FUNCTIONAL_H
#define TVMET_FUNCTIONAL_H

#include <tvmet/TypePromotion.h>

namespace tvmet {


/**
 * \class Functional Functional.h "tvmet/Functional.h"
 * \brief Base class for all binary und unary functionals.
 *
 * All functional operators and functions have a static apply
 * member function for evaluating the expressions inside.
 */
struct Functional { };


/**
 * \class BinaryFunctional Functional.h "tvmet/Functional.h"
 * \brief Base class for all binary functions.
 * \note Used for collecting classes for doxygen.
 */
struct BinaryFunctional : public Functional { };


/**
 * \class UnaryFunctional Functional.h "tvmet/Functional.h"
 * \brief Base class for all unary functions.
 * \note Used for collecting classes for doxygen.
 */
struct UnaryFunctional : public Functional { };


/*
 * some macro magic need below
 */

/**
 * \def TVMET_STD_SCOPE(x)
 * \brief Simple macro to allow using macros for namespace std functions.
 */
#define TVMET_STD_SCOPE(x) std::x


/**
 * \def TVMET_GLOBAL_SCOPE(x)
 * \brief Simple macro to allow using macros for global namespace functions.
 */
#define TVMET_GLOBAL_SCOPE(x) ::x


} // namespace tvmet


#include <tvmet/BinaryFunctionals.h>
#include <tvmet/UnaryFunctionals.h>


#endif // TVMET_FUNCTIONAL_H

// Local Variables:
// mode:C++
// End:
