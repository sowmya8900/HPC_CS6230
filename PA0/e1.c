#include <sys/time.h>
#include <sys/resource.h>
#include <stdio.h>

void walltime(double *s, double *mus)
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


X = 1.0;

walltime(&Sec1,&MUS1);

for(j=0;j<100000000;j++){
  W = 0.999999*X; 	
  X = 0.999999*W;}

walltime(&Sec2,&MUS2);
Time = (Sec2-Sec1+1.0E-6*(MUS2-MUS1));
  if (X < -1) printf("Bug!!! %f", X);
// Above line is to prevent compiler from optimizing away the loop via dead-code elimination
  printf("Time, Performance for 2*100000000 multiplications is %.2f Seconds, %.2f GFLOPs\n",
         Time,2.0E-1/Time);

}
