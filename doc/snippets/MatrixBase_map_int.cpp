int data[4] = {1,3,0,2};
cout << VectorXi::map(data, 4) << endl;
VectorXi::map(data, 4) *= 2;
cout << "The data is now:" << endl;
for(int i = 0; i < 4; i++) cout << data[i] << endl;
