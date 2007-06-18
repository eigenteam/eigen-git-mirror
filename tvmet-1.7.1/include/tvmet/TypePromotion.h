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
 * $Id: TypePromotion.h,v 1.6 2003/11/30 08:26:25 opetzold Exp $
 */

#ifndef TVMET_TYPE_PROMOTION_H
#define TVMET_TYPE_PROMOTION_H

namespace tvmet {


/**
 * \class PrecisionTraits TypePromotion.h "tvmet/TypePromotion.h"
 * \brief Declaring ranks of types to avoid specializing
 *
 * All possible promoted types. For example, bool=1, int=2, float=3, double=4,
 * etc. We can use a traits class to map from a type such as float onto its
 * "precision rank". We will promote to whichever type has a higher
 * "precision rank". f there is no "precision rank" for a type, we'll
 * promote to whichever type requires more storage space
 * (and hopefully more precision).
 */
template<class T>
struct PrecisionTraits {
  enum {
    rank = 0,			/**< the rank of type. */
    known = 0 			/**< true, if the rank is specialized = known. */
  };
};


#define TVMET_PRECISION(T,R)      					\
template<>                          					\
struct PrecisionTraits< T > {        					\
  enum {                          					\
    rank = R,                   					\
    known = 1                   					\
  };                              					\
};


/*
 * pod types
 */
TVMET_PRECISION(int, 100)
TVMET_PRECISION(float, 700)
TVMET_PRECISION(double, 800)

/*
 * complex types
 */
#if defined(TVMET_HAVE_COMPLEX)
TVMET_PRECISION(std::complex<int>, 1000)
TVMET_PRECISION(std::complex<float>, 1600)
TVMET_PRECISION(std::complex<double>, 1700)
#endif // defined(TVMET_HAVE_COMPLEX)


/** \class PrecisionTraits<int>				TypePromotion.h "tvmet/TypePromotion.h" */
/** \class PrecisionTraits<unsigned int> 		TypePromotion.h "tvmet/TypePromotion.h" */
/** \class PrecisionTraits<long>			TypePromotion.h "tvmet/TypePromotion.h" */
/** \class PrecisionTraits<unsigned long> 		TypePromotion.h "tvmet/TypePromotion.h" */
/** \class PrecisionTraits<long long>			TypePromotion.h "tvmet/TypePromotion.h" */
/** \class PrecisionTraits<unsigned long long> 		TypePromotion.h "tvmet/TypePromotion.h" */
/** \class PrecisionTraits<float>			TypePromotion.h "tvmet/TypePromotion.h" */
/** \class PrecisionTraits<double>			TypePromotion.h "tvmet/TypePromotion.h" */
/** \class PrecisionTraits<long double>			TypePromotion.h "tvmet/TypePromotion.h" */
/** \class PrecisionTraits< std::complex<int> >		TypePromotion.h "tvmet/TypePromotion.h" */
/** \class PrecisionTraits< std::complex<unsigned int> > TypePromotion.h "tvmet/TypePromotion.h" */
/** \class PrecisionTraits< std::complex<long> >	TypePromotion.h "tvmet/TypePromotion.h" */
/** \class PrecisionTraits< std::complex<unsigned long> > TypePromotion.h "tvmet/TypePromotion.h" */
/** \class PrecisionTraits< std::complex<long long> >	TypePromotion.h "tvmet/TypePromotion.h" */
/** \class PrecisionTraits< std::complex<unsigned long long> > TypePromotion.h "tvmet/TypePromotion.h" */
/** \class PrecisionTraits< std::complex<float> >	TypePromotion.h "tvmet/TypePromotion.h" */
/** \class PrecisionTraits< std::complex<double> >	TypePromotion.h "tvmet/TypePromotion.h" */
/** \class PrecisionTraits< std::complex<long double> >	TypePromotion.h "tvmet/TypePromotion.h" */

#undef TVMET_PRECISION


/**
 * \class AutopromoteTraits TypePromotion.h "tvmet/TypePromotion.h"
 * \brief The promoted types traits.
 */
template<class T>
struct AutopromoteTraits {
  typedef T value_type;
};

/**
 * \class promoteTo TypePromotion.h "tvmet/TypePromotion.h"
 * \brief Promote to T1.
 */
template<class T1, class T2, int promoteToT1>
struct promoteTo {
  typedef T1 value_type;
};


/**
 * \class promoteTo<T1,T2,0> TypePromotion.h "tvmet/TypePromotion.h"
 * \brief Promote to T2
 */
template<class T1, class T2>
struct promoteTo<T1,T2,0> {
  typedef T2 value_type;
};


/**
 * \class PromoteTraits TypePromotion.h "tvmet/TypePromotion.h"
 * \brief Promote type traits
 */
template<class T1org, class T2org>
class PromoteTraits {
  // Handle promotion of small integers to int/unsigned int
  typedef typename AutopromoteTraits<T1org>::value_type T1;
  typedef typename AutopromoteTraits<T2org>::value_type T2;

  enum {
    // True if T1 is higher ranked
    T1IsBetter = int(PrecisionTraits<T1>::rank) > int(PrecisionTraits<T2>::rank),

    // True if we know ranks for both T1 and T2
    knowBothRanks = PrecisionTraits<T1>::known && PrecisionTraits<T2>::known,

    // True if we know T1 but not T2
    knowT1butNotT2 = PrecisionTraits<T1>::known && !(PrecisionTraits<T2>::known),

    // True if we know T2 but not T1
    knowT2butNotT1 =  PrecisionTraits<T2>::known && !(PrecisionTraits<T1>::known),

    // True if T1 is bigger than T2
    T1IsLarger = sizeof(T1) >= sizeof(T2),

    // We know T1 but not T2: true
    // We know T2 but not T1: false
    // Otherwise, if T1 is bigger than T2: true
    defaultPromotion = knowT1butNotT2 ? false : (knowT2butNotT1 ? true : T1IsLarger),

    // If we have both ranks, then use them.
    // If we have only one rank, then use the unknown type.
    // If we have neither rank, then promote to the larger type.
    promoteToT1 = (knowBothRanks ? T1IsBetter : defaultPromotion) ? 1 : 0
  };

 public:
  typedef typename promoteTo<T1,T2,promoteToT1>::value_type value_type;
};


} // namespace tvmet

#endif // TVMET_TYPE_PROMOTION_H

// Local Variables:
// mode:C++
// End:
