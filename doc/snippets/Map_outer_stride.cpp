int array[12];
for(int i = 0; i < 12; ++i) array[i] = i;
cout << Map<MatrixXi, 0, OuterStride<Dynamic> >
         (array, 3, 3, OuterStride<Dynamic>(4))
     << endl;
