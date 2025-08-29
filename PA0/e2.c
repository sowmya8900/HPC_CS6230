#include <sys/time.h>
#include <sys/resource.h>
#include <stdio.h>

void walltime(double *s,double *mus)
{ struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  (*s) = Tp.tv_sec;
  (*mus) = Tp.tv_usec;
}

int main(){
double W,X,Y,Z;
int j;
double Sec1,Sec2,MUS1,MUS2,Time;


W = 0.999;
X = 1.0;
Y = 1.0001;
Z = 1.001;

walltime(&Sec1,&MUS1);

for(j=0;j<100000000;j++){
W = W * 0.99999999;
X = X * 0.99999999; 
Y = Y * 0.99999999;
Z = Z * 0.99999999;
}

walltime(&Sec2,&MUS2);

Time = (Sec2-Sec1+1.0E-6*(MUS2-MUS1));
  if (W+X+Y+Z < -1) printf("Bug!!! %f", X);
  printf("Time, Performance for 4*100000000 multiplications is %.2f Seconds, %.2f GFLOPs\n",
         Time,4.0E-1/Time);

}
