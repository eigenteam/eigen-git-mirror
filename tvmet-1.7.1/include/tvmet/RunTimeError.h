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
 * $Id: RunTimeError.h,v 1.9 2003/11/30 08:26:25 opetzold Exp $
 */

#ifndef TVMET_RUN_TIME_ERROR_H
#define TVMET_RUN_TIME_ERROR_H

#include <cassert>


namespace tvmet {


/**
 * \def TVMET_RT_CONDITION(XPR, MSG)
 * \brief If TVMET_DEBUG is defined it checks the condition XPR and prints
 * an error message MSG at runtime.
 */
#if defined(TVMET_DEBUG)

#define TVMET_RT_CONDITION(XPR, MSG) {					\
  if(!(XPR)) {								\
    std::cerr << "[tvmet] Precondition failure in " << __FILE__		\
              << ", line " << __LINE__ << ": "				\
              << MSG << std::endl;					\
    std::cerr.flush();							\
    assert(0);								\
  }									\
}

#else

#define TVMET_RT_CONDITION(XPR, MSG)

#endif


} // namespace tvmet


#endif // TVMET_RUN_TIME_ERROR_H


// Local Variables:
// mode:C++
// End:
