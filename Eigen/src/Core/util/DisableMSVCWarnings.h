
#ifdef _MSC_VER
  // 4273 - QtAlignedMalloc, inconsistent DLL linkage
  // 4100 - unreferenced formal parameter (occurred e.g. in aligned_allocator::destroy(pointer p))
  // 4101 - unreferenced local variable
  // 4324 - structure was padded due to declspec(align())
  // 4512 - assignment operator could not be generated
  #pragma warning( push )
  #pragma warning( disable : 4100 4101 4181 4244 4127 4211 4273 4324 4512 4522 4717 )
#endif
