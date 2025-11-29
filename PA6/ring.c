#include <unistd.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

int main(int argc, char *argv[]) {

double clkbegin, clkend;
double t, tmax, *tarr;
double rtclock();
int i,it,m;
int myid, nprocs;
int MsgLen,MaxMsgLen,Niter;
MPI_Status status[2];
MPI_Request req[2];
char name[MPI_MAX_PROCESSOR_NAME];
int procnamelen;
double *in,*out;

  MPI_Init( &argc, &argv );
  MPI_Comm_rank( MPI_COMM_WORLD, &myid );
  MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
  MPI_Get_processor_name(name, &procnamelen);
  printf("I am %d of %d on %s\n", myid, nprocs, name);fflush(stdout);

  int msgLen[] = {1,8,64,512,4096,32768,262144,1048576};
  MaxMsgLen = 1048576;
  in = (double *) malloc(MaxMsgLen*sizeof(double));
  out = (double *) malloc(MaxMsgLen*sizeof(double));
  for(i=0;i<MaxMsgLen;i++) out[i] = i;
  for(m = 0 ; m < 8 ; m++)
  {
    MsgLen = msgLen[m];
    Niter = (1000000 + MsgLen) / (100 + MsgLen);
    MPI_Barrier(MPI_COMM_WORLD);

    clkbegin = MPI_Wtime();
    for(it=0;it<nprocs*Niter;it++) {
        int right = (myid + 1) % nprocs;
	int left = (myid + nprocs - 1) % nprocs;
       
       	MPI_Irecv(in,MsgLen,MPI_DOUBLE,left,0,MPI_COMM_WORLD,&req[0]);
	MPI_Isend(out,MsgLen,MPI_DOUBLE,right,0,MPI_COMM_WORLD,&req[1]);
	MPI_Waitall(2,req,status);
        
	MPI_Irecv(out,MsgLen,MPI_DOUBLE,left,0,MPI_COMM_WORLD,&req[0]);
	MPI_Isend(in,MsgLen,MPI_DOUBLE,right,0,MPI_COMM_WORLD,&req[1]);
        MPI_Waitall(2,req,status);
    }
    clkend = MPI_Wtime();
    t = clkend-clkbegin;
    MPI_Reduce(&t, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); 
    if (myid == 0)
    {
     printf("Ring Communication: Message Size = %d; %.3f Gbytes/sec; Time = %.3f sec; \n",
          MsgLen,2.0*1e-9*sizeof(double)*MsgLen*nprocs*Niter/tmax,tmax);
     fflush(stdout);
    }
  }
  free(in);
  free(out);
  MPI_Finalize();
  return 0;
}

