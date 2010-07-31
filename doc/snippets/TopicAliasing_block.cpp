Matrix3i mat; 
mat << 1, 2, 3,   4, 5, 6,   7, 8, 9;
cout << "Here is the matrix mat:\n" << mat << endl;
mat.bottomRightCorner(2,2) = mat.topLeftCorner(2,2);
cout << "After the assignment, mat = \n" << mat << endl;
