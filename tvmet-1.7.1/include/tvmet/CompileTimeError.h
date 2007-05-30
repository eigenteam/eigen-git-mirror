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
 * $Id: CompileTimeError.h,v 1.7 2003/11/30 08:26:25 opetzold Exp $
 */

#ifndef TVMET_COMPILE_TIME_ERROR_H
#define TVMET_COMPILE_TIME_ERROR_H

namespace tvmet {

/**
 * \class CompileTimeError CompileTimeError.h "tvmet/CompileTimeError.h"
 * \brief Compile Time Assertation classes.
 */
template<bool> struct CompileTimeError;

/**
 * \class CompileTimeError<true> CompileTimeError.h "tvmet/CompileTimeError.h"
 * \brief Specialized Compile Time Assertation for successfully condition.
 * This results in a compiler pass.
 */
template<> struct CompileTimeError<true> { };


/**
 * \def TVMET_CT_CONDITION(XPR, MSG)
 * \brief Simplify the Compile Time Assertation by using an expression
 * Xpr and an error message MSG.
 */
#define TVMET_CT_CONDITION(XPR, MSG) {				\
  CompileTimeError<(XPR)> tvmet_ERROR_##MSG;			\
  (void)tvmet_ERROR_##MSG;					\
}

} // namespace tvmet

#endif // TVMET_COMPILE_TIME_ERROR_H

// Local Variables:
// mode:C++
// End:
