Matrix2d m; m << 1,2,3,4;
Matrix2d n;
n = (m*m).lazy(); // if we did "n = m*m;" then m*m would first be evaluated into
 // a temporary, because the Product expression has the EvalBeforeAssigningBit.
 // This temporary would then be copied into n. Introducing this temporary is
 // useless here and wastes time. Doing "n = (m*m).lazy();" evaluates m*m directly
 // into n, which is faster. But, beware! This is only correct because m and n
 // are two distinct matrices. Doing "m = (m*m).lazy();" would not produce the
 // expected result.
cout << n << endl;
