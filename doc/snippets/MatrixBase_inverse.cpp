Matrix2d m = Matrix2d::random();
cout << "Here is the matrix m:" << endl << m << endl;
Matrix2d::InverseType m_inv = m.inverse();
if(m_inv.exists())
  cout << "m is invertible, and its inverse is:" << endl << m_inv << endl;
else
  cout << "m is not invertible." << endl;
