Matrix3i m1;
m1 <<= 1, 2, 3,
      4, 5, 6,
      7, 8, 9;
cout << m1 << endl << endl;
Matrix3i m2 = Matrix3i::identity();
m2.block(0,0, 2,2) <<= 10, 11, 12, 13;
cout << m2 << endl;