void test(int a, int b, int n)
{
if(n <= 0)
{
print(a);
return;
}
int c = b;
 n = n - 1;
 b = a + b;
test(c, b, n );
}

void main(){
    int a = 0, b = 1, n = 10;
    test(a, b, n);
}

//int a[1];
//a[3 < 4];
//if (a < 2){
//    d = b;
//}
//if (a > 2){
//    w = w;
//} else {
//    _A = 1;
//}
//
//if (a == 2)
//    a = 3;
//else
//    b = 5;
//
//while (b < 0)
//    b = b + 1;
//while (b < 0){
//    index = 1;
//    b = b - +1;
//}

 /*for (int a = 0; a < 2; a = a + 2){
     d = 0;
     d = 1;
 }

for (int a = 0; a < 2; a = a + 2) d = d + 1;
f = 2.9 + 1e4;
//int b = 1 % 2;

for (;;);

int a = 0;
a[0];
int retur = 0, a = 0, c;
retur = a + b;
void abc(int a, int b, int c)
{
example = 1;
}

if(d)
if(d)
int b = 0;
else
int c = 9;


func(a[1+b[1] + func(1,1)]);
for(;;);
return 1;
//int (1 < 2); //WARNING!!! it's not call!!!
//while (1 < 2);
//while(a);
//int b = 2;

*/