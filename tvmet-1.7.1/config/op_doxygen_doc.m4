dnl $Id: op_doxygen_doc.m4,v 1.2 2004/04/23 23:41:52 opetzold Exp $
dnl
dnl OP_DOXYGEN_DOC
dnl

AC_DEFUN([OP_DOXYGEN_DOC],
[
  AC_MSG_CHECKING(for documentation)
  AC_ARG_ENABLE(docs,
    [  --enable-docs           build documentation],
    [case "$enableval" in
      y | yes) CONFIG_DOC=yes ;;
      *) unset CONFIG_DOC ;;
    esac])
  AC_MSG_RESULT([${CONFIG_DOC:-no}])

  AC_MSG_CHECKING(whether using LaTeX non-stop mode)
  AC_ARG_ENABLE(verbose-latex,
    [  --enable-verbose-latex  Uses LaTeX non-stop mode],
    [case "$enableval" in
      y | yes) CONFIG_DOC_LATEX_NONSTOP=yes ;;
      *) unset CONFIG_DOC_LATEX_NONSTOP ;;
    esac])
  AC_MSG_RESULT(${CONFIG_DOC_LATEX_NONSTOP:-no})

  if test x${CONFIG_DOC_LATEX_NONSTOP} = xyes; then
     LATEX_BATCHMODE=NO
     LATEX_MODE=nonstopmode
  else
     LATEX_BATCHMODE=YES
     LATEX_MODE=batchmode
  fi

  AC_CHECK_PROG(DOXYGEN, doxygen, doxygen)
  if test x${CONFIG_DOC} = xyes -a x"$DOXYGEN" = x; then
    AC_MSG_ERROR([missing the doxygen tools to generate the documentation.])
  fi

  AC_CHECK_PROG(DOXYGEN_HAVE_DOT, dot, yes, no)

  AM_CONDITIONAL(CONFIG_DOC,[test x"$CONFIG_DOC" = xyes])
  dnl force docs
  dnl AM_CONDITIONAL(CONFIG_DOC, [test x=x])
  AC_SUBST(DOXYGEN)
  AC_SUBST(DOXYGEN_HAVE_DOT)
  AC_SUBST(LATEX_BATCHMODE)
  AC_SUBST(LATEX_MODE)
])
