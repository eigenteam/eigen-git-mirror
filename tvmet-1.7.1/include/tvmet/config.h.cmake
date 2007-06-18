#ifndef _INCLUDE_TVMET_CONFIG_H
#define _INCLUDE_TVMET_CONFIG_H

/* Define to 1 if you have the <sys/time.h> header file. */
#cmakedefine TVMET_HAVE_SYS_TIME_H  1 

/* Define to 1 if you have the <unistd.h> header file. */
#cmakedefine TVMET_HAVE_UNISTD_H  1

#define _tvmet_restrict @TVMET_RESTRICT_KEYWORD@

#define _tvmet_always_inline @TVMET_ALWAYS_INLINE@

/* _INCLUDE_TVMET_CONFIG_H */
#endif
