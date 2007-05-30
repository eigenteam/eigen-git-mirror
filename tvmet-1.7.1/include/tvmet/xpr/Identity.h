/*
 * $Id: Identity.h,v 1.3 2004/09/16 09:14:18 opetzold Exp $
 */

#ifndef TVMET_XPR_IDENTITY_H
#define TVMET_XPR_IDENTITY_H


namespace tvmet {


/**
 * \class XprIdentity Identity.h "tvmet/xpr/Identity.h"
 * \brief Expression for the identity matrix.
 *
 * This expression doesn't hold any other expression, it
 * simply returns 1 or 0 depends where the row and column
 * element excess is done.
 *
 * \since release 1.6.0
 * \sa identity
 */
template<class T, std::size_t Rows, std::size_t Cols>
struct XprIdentity
  : public TvmetBase< XprIdentity<T, Rows, Cols> >
{
  XprIdentity& operator=(const XprIdentity&);

public:
  typedef T				value_type;

public:
  /** Complexity counter. */
  enum {
    ops_assign = Rows * Cols,
    ops        = ops_assign
  };

public:
  /** access by index. */
  value_type operator()(std::size_t i, std::size_t j) const {
    return i==j ? 1 : 0;
  }

public: // debugging Xpr parse tree
  void print_xpr(std::ostream& os, std::size_t l=0) const {
    os << IndentLevel(l++)
       << "XprIdentity[O="<< ops << ")]<"
       << std::endl;
    os << IndentLevel(l)
       << typeid(T).name() << ","
       << "R=" << Rows << ", C=" << Cols << std::endl;
    os << IndentLevel(--l) << ">"
       << ((l != 0) ? "," : "") << std::endl;
  }
};


} // namespace tvmet


#endif // TVMET_XPR_IDENTITY_H


// Local Variables:
// mode:C++
// End:
