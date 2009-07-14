Matrix3d m = Matrix3d::Random();
cout << "Here is the matrix m:" << endl << m << endl;
Matrix3d inv;
if(m.computeInverseWithCheck(&inv)) {
  cout << "It is invertible, and its inverse is:" << endl << inv << endl;
}
else {
  cout << "It is not invertible." << endl;
}
