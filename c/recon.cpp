#include	<math.h>
#include	<fftw3.h>
#include	<iostream>
#include	<fstream>
#include	<vector>
#include	<sstream>
#include	<iomanip>
#include	<omp.h>
#include	<gsl/gsl_histogram.h>

using namespace std;

struct particle{
	double pos[3];
};

struct Box {
	double min[3],max[3],position[3];
	double Length;
};

int N=512;
double bias =2;
double f = 0.76;
double smoothingScale  = 10;
struct Box box; 
double k;

struct particle CreateParticle(const double x, const double y, const double z){

	struct particle local;
	local.pos[0] = x;
	local.pos[1] = y;
	local.pos[2] = z;
	return local;
}

void PrintStat(fftw_complex* data,string name){
	double min,max,mean,rms;
	mean = rms = 0;
	min = max = 0;
	int number;
	for (int xin=0;xin<N;xin++){for (int yin=0;yin<N;yin++){for (int zin=0;zin<N;zin++){
	 int index = N*N*xin+N*yin+zin;
	 min = (min<data[index][0])?min:data[index][0];
	 max = (max>data[index][0])?max:data[index][0];
	 mean += data[index][0];
	 rms += data[index][0]*data[index][0];
	}}}
	 mean/=N*N*N;
	 rms/=N*N*N;
	 rms =sqrt(rms);
	 cout<<"#The mean of "<<name<<" is "<<mean<<endl;
	 cout<<"#The rms of "<<name<<" is "<<rms<<endl;
	 cout<<"#The maximum of "<<name<<" is "<<max<<endl;
	 cout<<"#The minimum of "<<name<<" is "<<min<<endl;
}

void PlotHistogram(fftw_complex* data,int direction){
	int index;
	gsl_histogram *h = gsl_histogram_alloc(N);
	gsl_histogram_set_ranges_uniform(h,0,N);
	for (int xin=0;xin<N;xin++){
	 for (int yin=0;yin<N;yin++){
	  for (int zin=0;zin<N;zin++){
	   index = N*N*xin+N*yin+zin;
	   switch(direction){
	   case 0: 
	   gsl_histogram_accumulate(h,xin,data[index][0]);
	   break;
	   case 1: 
	   gsl_histogram_accumulate(h,yin,data[index][0]);
	   break;
	   case 2: 
	   gsl_histogram_accumulate(h,zin,data[index][0]);
	   break;}
   	}}}
	gsl_histogram_fprintf(stdout,h,"%g","%g");
}

void SetBox(vector<struct particle> &input){
	double minmax[3][2],min,max;
	for (int i=0;i<3;i++){for (int j=0;j<2;j++){minmax[i][j]=0;}}
	for (int i=0;i<input.size();i++){
		minmax[0][0]= (minmax[0][0]<input[i].pos[0])?minmax[0][0]:input[i].pos[0];
		minmax[0][1]= (minmax[0][1]>input[i].pos[0])?minmax[0][1]:input[i].pos[0];
		minmax[1][0]= (minmax[1][0]<input[i].pos[1])?minmax[1][0]:input[i].pos[1];
		minmax[1][1]= (minmax[1][1]>input[i].pos[1])?minmax[1][1]:input[i].pos[1];
		minmax[2][0]= (minmax[2][0]<input[i].pos[2])?minmax[2][0]:input[i].pos[2];
		minmax[2][1]= (minmax[2][1]>input[i].pos[2])?minmax[2][1]:input[i].pos[2];
	}
	for (int i=0;i<3;i++){
	 box.min[i] = minmax[i][0];
	 box.max[i] = minmax[i][1];
	 box.position[i] = (box.max[i]+box.min[i])/2;
	 box.Length = (box.Length<(minmax[i][1]-minmax[i][0]))?(minmax[i][1]-minmax[i][0]):box.Length;
	}
	box.Length *= 1.5;
	cout<<"#The box size is "<<box.Length<<" Mpc/h"<<endl;	
	cout<<"#The center position is "<<box.position[0]<<" "<<box.position[1]<<" "<<box.position[2]<<endl;	
}

vector<struct particle> readfile(string name){
	ifstream input(name.c_str());
	string buf;
	double x,y,z;
	vector<struct particle> array;
	do{
	getline(input,buf);
	} while(!input.eof() && buf[0] =='#');
	while(!input.eof()){
	istringstream(buf) >> x >> y >> z;
	array.push_back(CreateParticle(x,y,z));	
	getline(input,buf);
	}
	return array;
}

void WriteFile(vector<struct particle> output,string name){

	ofstream file(name.c_str());
	for (int i=0;i<output.size();i++){
	file<<fixed<<setw(15)<<setprecision(4)<<output[i].pos[0]<<setw(15)<<setprecision(4)<<output[i].pos[1]<<setw(15)<<setprecision(4)<<output[i].pos[2]<<setw(15)<<setprecision(4)<<1<<endl;
	}
	file.close();
}
void MapPositiontoGrid(vector<struct particle> &input){
	for (int i=0;i<input.size();i++){
		input[i].pos[0]=((input[i].pos[0]-box.position[0])/box.Length+0.5)*N;
		input[i].pos[1]=((input[i].pos[1]-box.position[1])/box.Length+0.5)*N;
		input[i].pos[2]=((input[i].pos[2]-box.position[2])/box.Length+0.5)*N;
	}
}

void MapGridtoPosition(vector<struct particle> &input){
	for (int i=0;i<input.size();i++){
		input[i].pos[0]=((input[i].pos[0]/N-0.5)*box.Length+box.position[0]);
		input[i].pos[1]=((input[i].pos[1]/N-0.5)*box.Length+box.position[1]);
		input[i].pos[2]=((input[i].pos[2]/N-0.5)*box.Length+box.position[2]);
	}
}

void Masking(fftw_complex *data,fftw_complex *random){
	int index;
	double mean,rms,count,count2;
	for (int xin=0;xin<N;xin++){
	 for (int yin=0;yin<N;yin++){
	  for (int zin=0;zin<N;zin++){
	   index = N*N*xin+N*yin+zin;
	   if (random[index][0]>0 && random[index][0] < 0.75){
	   random[index][0]=data[index][0]=0;
	   count2 ++;
	   }
	   if (random[index][0]!=0){
	   count += 1 ;	   
	   data[index][0] = data[index][0]/random[index][0];
	   mean += data[index][0];
	   }}}}
	cout<<"#"<<count<<" grid are filled, "<<"the number contained is "<<mean<<endl; 
	cout<<"#Removed "<<count2<<" low number grid point"<<endl;
	mean/=count;
	cout<<"#The mean density is "<<setprecision(12)<<mean<<endl;
	for (int xin=0;xin<N;xin++){
	 for (int yin=0;yin<N;yin++){
	  for (int zin=0;zin<N;zin++){
	   index = N*N*xin+N*yin+zin;
	   if (random[index][0]==0){
	   data[index][0]= mean;	   
	   data[index][1]= 0;
	   }
 	   data[index][0] = ((data[index][0]/mean)-1);
	   data[index][1] = 0;
	   }
	}}

}

void CICassignment(const vector<struct particle> data,fftw_complex *grid){
//#pragma omp parallel for shared(grid)
	for (int i=0;i<data.size();i++){
		int index[6];
		float difference[3];
		index[0]=floor(data[i].pos[0]);index[3]=(index[0]+1);difference[0]=data[i].pos[0]-index[0];
		index[1]=floor(data[i].pos[1]);index[4]=(index[1]+1);difference[1]=data[i].pos[1]-index[1];
		index[2]=floor(data[i].pos[2]);index[5]=(index[2]+1);difference[2]=data[i].pos[2]-index[2];
		grid[(N*N*index[0]+N*index[1]+index[2])][0] += (1.0-difference[0])*(1.0-difference[1])*(1.0-difference[2]);
		grid[(N*N*index[3]+N*index[1]+index[2])][0] += difference[0]*(1.0-difference[1])*(1.0-difference[2]);
		grid[(N*N*index[0]+N*index[4]+index[2])][0] += (1.0-difference[0])*difference[1]*(1.0-difference[2]);
		grid[(N*N*index[0]+N*index[1]+index[5])][0] += (1.0-difference[0])*(1.0-difference[1])*difference[2];
		grid[(N*N*index[3]+N*index[4]+index[2])][0] += difference[0]*difference[1]*(1.0-difference[2]);
		grid[(N*N*index[3]+N*index[1]+index[5])][0] += difference[0]*(1.0-difference[1])*difference[2];
		grid[(N*N*index[0]+N*index[4]+index[5])][0] += (1.0-difference[0])*difference[1]*difference[2];
		grid[(N*N*index[3]+N*index[4]+index[5])][0] += difference[0]*difference[1]*difference[2];
	}
}

void CICinterpolation(vector<struct particle> &data,const fftw_complex *dx,const fftw_complex *dy,const fftw_complex *dz){
	double sum=0;
	double xmax,xmin,ymax,ymin,zmax,zmin;
//#pragma omp parallel for shared(data,dx,dy,dz)
	for (int i=0;i<data.size();i++){
		double px,py,pz,rx,ry,rz,pr,r2;
		int index[6];
		float difference[3];
		index[0]=data[i].pos[0];index[3]=(index[0]+1);difference[0]=data[i].pos[0]-index[0];
		index[1]=data[i].pos[1];index[4]=(index[1]+1);difference[1]=data[i].pos[1]-index[1];
		index[2]=data[i].pos[2];index[5]=(index[2]+1);difference[2]=data[i].pos[2]-index[2];
		px=dx[(N*N*index[0]+N*index[1]+index[2])][0]*(1.0-difference[0])*(1.0-difference[1])*(1.0-difference[2]);
		py=dy[(N*N*index[0]+N*index[1]+index[2])][0]*(1.0-difference[0])*(1.0-difference[1])*(1.0-difference[2]);
		pz=dz[(N*N*index[0]+N*index[1]+index[2])][0]*(1.0-difference[0])*(1.0-difference[1])*(1.0-difference[2]);
		px+=dx[(N*N*index[3]+N*index[1]+index[2])][0]*difference[0]*(1.0-difference[1])*(1.0-difference[2]);
		py+=dy[(N*N*index[3]+N*index[1]+index[2])][0]*difference[0]*(1.0-difference[1])*(1.0-difference[2]);
		pz+=dz[(N*N*index[3]+N*index[1]+index[2])][0]*difference[0]*(1.0-difference[1])*(1.0-difference[2]);
		px+=dx[(N*N*index[0]+N*index[4]+index[2])][0]*(1.0-difference[0])*difference[1]*(1.0-difference[2]);
		py+=dy[(N*N*index[0]+N*index[4]+index[2])][0]*(1.0-difference[0])*difference[1]*(1.0-difference[2]);
		pz+=dz[(N*N*index[0]+N*index[4]+index[2])][0]*(1.0-difference[0])*difference[1]*(1.0-difference[2]);
		px+=dx[(N*N*index[0]+N*index[1]+index[5])][0]*(1.0-difference[0])*(1.0-difference[1])*difference[2];
		py+=dy[(N*N*index[0]+N*index[1]+index[5])][0]*(1.0-difference[0])*(1.0-difference[1])*difference[2];
		pz+=dz[(N*N*index[0]+N*index[1]+index[5])][0]*(1.0-difference[0])*(1.0-difference[1])*difference[2];
		px+=dx[(N*N*index[3]+N*index[4]+index[2])][0]*difference[0]*difference[1]*(1.0-difference[2]);
		py+=dy[(N*N*index[3]+N*index[4]+index[2])][0]*difference[0]*difference[1]*(1.0-difference[2]);
		pz+=dz[(N*N*index[3]+N*index[4]+index[2])][0]*difference[0]*difference[1]*(1.0-difference[2]);
		px+=dx[(N*N*index[3]+N*index[1]+index[5])][0]*difference[0]*(1.0-difference[1])*difference[2];
		py+=dy[(N*N*index[3]+N*index[1]+index[5])][0]*difference[0]*(1.0-difference[1])*difference[2];
		pz+=dz[(N*N*index[3]+N*index[1]+index[5])][0]*difference[0]*(1.0-difference[1])*difference[2];
		px+=dx[(N*N*index[0]+N*index[4]+index[5])][0]*(1.0-difference[0])*difference[1]*difference[2];
		py+=dy[(N*N*index[0]+N*index[4]+index[5])][0]*(1.0-difference[0])*difference[1]*difference[2];
		pz+=dz[(N*N*index[0]+N*index[4]+index[5])][0]*(1.0-difference[0])*difference[1]*difference[2];
		px+=dx[(N*N*index[3]+N*index[4]+index[5])][0]*difference[0]*difference[1]*difference[2];
		py+=dy[(N*N*index[3]+N*index[4]+index[5])][0]*difference[0]*difference[1]*difference[2];
		pz+=dz[(N*N*index[3]+N*index[4]+index[5])][0]*difference[0]*difference[1]*difference[2];
		rx = (box.position[0]+(data[i].pos[0]-box.Length/2));
		ry = (box.position[1]+(data[i].pos[1]-box.Length/2));
		rz = (box.position[2]+(data[i].pos[2]-box.Length/2));
		r2 = rx*rx+ry*ry+rz*rz;
		pr = f*(px*rx+py*ry+pz*rz);
		data[i].pos[0]-=(px+pr*rx/r2)*N/box.Length;
		data[i].pos[1]-=(py+pr*ry/r2)*N/box.Length;
		data[i].pos[2]-=(pz+pz*rz/r2)*N/box.Length;
		sum+=px*px+py*py+pz*pz;
		xmin = (xmin<px)?xmin:px;
		xmax = (xmax>px)?xmax:px;
		ymin = (ymin<py)?ymin:py;
		ymax = (ymax>py)?ymax:py;
		zmin = (zmin<pz)?zmin:pz;
		zmax = (zmax>pz)?zmax:pz;
	}
	cout<<"#"<<xmin<<" "<<xmax<<" "<<ymin<<" "<<ymax<<" "<<zmin<<" "<<zmax<<endl;
	cout<<"#"<<sqrt(sum/(3*data.size()))<<endl;
}

void GaussianSmoothing(fftw_complex *delta,double scale){
	fftw_complex *local;
	local = (fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	fftw_plan deltaforward,deltabackward;
	deltaforward = fftw_plan_dft_3d(N,N,N,delta,local,FFTW_FORWARD,FFTW_ESTIMATE);
	fftw_execute(deltaforward);
//#pragma omg parallel for shared(delta,dx,dy,dz)
	for (int xin=0;xin<N;xin++){
	int x = (xin<=N/2)?xin:(xin-N);
	 for (int yin=0;yin<N;yin++){
	 int y = (yin<=N/2)?yin:(yin-N);
	  for (int zin=0;zin<N;zin++){
  	  int z = (zin<=N/2)?zin:(zin-N);
	  int index = N*N*xin+N*yin+zin;
	  const double k2 = (x*x+y*y+z*z);
	  local[index][0] *= exp(-0.5*scale*scale*(2*M_PI/box.Length)*(2*M_PI/box.Length)*k2);
	  local[index][1] *= exp(-0.5*scale*scale*(2*M_PI/box.Length)*(2*M_PI/box.Length)*k2);
	}}}
	deltabackward = fftw_plan_dft_3d(N,N,N,local,delta,FFTW_BACKWARD,FFTW_MEASURE);
	fftw_execute(deltabackward);
	for (int i =0;i<(N*N*N);i++){
	delta[i][0]/=N*N*N;
	delta[i][1]/=N*N*N;
	}
}

void FirstDisplacement(fftw_complex *delta,fftw_complex *dx,fftw_complex *dy,fftw_complex *dz){
	fftw_plan deltaforward,deltabackward,xbackward,ybackward,zbackward;
	deltaforward = fftw_plan_dft_3d(N,N,N,delta,delta,FFTW_FORWARD,FFTW_ESTIMATE);
	xbackward = fftw_plan_dft_3d(N,N,N,dx,dx,FFTW_BACKWARD,FFTW_MEASURE);
	ybackward = fftw_plan_dft_3d(N,N,N,dy,dy,FFTW_BACKWARD,FFTW_MEASURE);
	zbackward = fftw_plan_dft_3d(N,N,N,dz,dz,FFTW_BACKWARD,FFTW_MEASURE);
	fftw_execute(deltaforward);
//#pragma omg parallel for shared(delta,dx,dy,dz)
	for (int xin=0;xin<N;xin++){
	int x = (xin<=N/2)?xin:(xin-N);
	 for (int yin=0;yin<N;yin++){
	 int y = (yin<=N/2)?yin:(yin-N);
	  for (int zin=0;zin<N;zin++){
  	  int z = (zin<=N/2)?zin:(zin-N);
	  int index = N*N*xin+N*yin+zin;
	  const double k2 = k*(x*x+y*y+z*z);
	  if (k2!=0){
	   const double re = delta[index][0]/bias;
	   const double im = -delta[index][1]/bias;
	   dx[index][0] = (x/k2)*im;
	   dx[index][1] = (x/k2)*re;
	   dy[index][0] = (y/k2)*im;
	   dy[index][1] = (y/k2)*re;
	   dz[index][0] = (z/k2)*im;
	   dz[index][1] = (z/k2)*re;
	  }
	  else{
	  dx[index][0]=dx[index][1]=dy[index][0]=dy[index][1]=dz[index][0]=dz[index][1]=0;
	  }
	}}}
	deltabackward = fftw_plan_dft_3d(N,N,N,delta,delta,FFTW_BACKWARD,FFTW_MEASURE);
	fftw_execute(deltabackward);
	fftw_execute(xbackward);
	fftw_execute(ybackward);
	fftw_execute(zbackward);
	double xmax,xmin,ymax,ymin,zmax,zmin;
	for (int i =0;i<(N*N*N);i++){
	delta[i][0]/=N*N*N;
	delta[i][1]/=N*N*N;
	dx[i][0]/=N*N*N;
	dx[i][1]/=N*N*N;
	dy[i][0]/=N*N*N;
	dy[i][1]/=N*N*N;
	dz[i][0]/=N*N*N;
	dz[i][1]/=N*N*N;
	xmin = (xmin<dx[i][0])?xmin:dx[i][0];
	xmax = (xmax>dx[i][0])?xmax:dx[i][0];
	ymin = (ymin<dy[i][0])?ymin:dy[i][0];
	ymax = (ymax>dy[i][0])?ymax:dy[i][0];
	zmin = (zmin<dz[i][0])?zmin:dz[i][0];
	zmax = (zmax>dz[i][0])?zmax:dz[i][0];
	}
	cout<<"#The range of displacements are "<<xmin<<" "<<xmax<<" "<<ymin<<" "<<ymax<<" "<<zmin<<" "<<zmax<<endl;
}

void SH0(fftw_complex result){
	result[0]=sqrt(1/M_PI)/2;
	result[1]=0;

}

void SH2(fftw_complex result,double x,double y,double z,int m){
	double r = x*x+y*y+z*z;
	if (r!=0){
	double factor;
	switch(m){
	 case -2:
	  factor = sqrt(15/(2*M_PI))/(4*r);
	  result[0] = factor*(x*x-y*y);
	  result[1] = -factor*(2*x*y);
	  break;
	 case -1:
	  factor = sqrt(15/(2*M_PI))/(2*r);
	  result[0] = factor*z*x;
	  result[1] = -factor*z*y;
	  break;
	 case 0:
	  factor = sqrt(5/(2*M_PI))/(4*r);
	  result[0] = factor*(2*z*z-x*x-y*y);
	  result[1] = 0;
	  break;
	 case 1:
	  factor = -sqrt(15/(2*M_PI))/(2*r);
	  result[0] = factor*z*x;
	  result[1] = factor*z*y;
	  break;
	 case 2: 
	  factor = sqrt(15/(2*M_PI))/(4*r);
	  result[0] = factor*(x*x-y*y);
	  result[1] = factor*(2*x*y);
	  break;
	}}
}

void SecondDisplacement(fftw_complex *delta, fftw_complex *dx, fftw_complex *dy, fftw_complex *dz){
	fftw_complex *SH00,*SH22,*SH21,*SH20,*SH2n1,*SH2n2;
	fftw_plan SH00forward,SH22forward,SH21forward,SH20forward,SH2n1forward,SH2n2forward,SH00backward,SH22backward,SH21backward,SH20backward,deltaforward,deltabackward,xbackward,ybackward,zbackward;
	SH00 = (fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	SH22 = (fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	SH21 = (fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	SH20 = (fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	SH2n1 = (fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	SH2n2 = (fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	deltaforward = fftw_plan_dft_3d(N,N,N,delta,delta,FFTW_FORWARD,FFTW_ESTIMATE);
	SH00forward = fftw_plan_dft_3d(N,N,N,SH00,SH00,FFTW_FORWARD,FFTW_ESTIMATE);
	SH22forward = fftw_plan_dft_3d(N,N,N,SH22,SH22,FFTW_FORWARD,FFTW_ESTIMATE);
	SH21forward = fftw_plan_dft_3d(N,N,N,SH21,SH21,FFTW_FORWARD,FFTW_ESTIMATE);
	SH20forward = fftw_plan_dft_3d(N,N,N,SH20,SH20,FFTW_FORWARD,FFTW_ESTIMATE);
	SH2n1forward = fftw_plan_dft_3d(N,N,N,SH2n1,SH2n2,FFTW_FORWARD,FFTW_ESTIMATE);
	SH2n2forward = fftw_plan_dft_3d(N,N,N,SH2n2,SH2n2,FFTW_FORWARD,FFTW_ESTIMATE);
	SH00backward = fftw_plan_dft_3d(N,N,N,SH00,SH00,FFTW_BACKWARD,FFTW_MEASURE);
	SH22backward = fftw_plan_dft_3d(N,N,N,SH22,SH22,FFTW_BACKWARD,FFTW_MEASURE);
	SH21backward = fftw_plan_dft_3d(N,N,N,SH21,SH21,FFTW_BACKWARD,FFTW_MEASURE);
	SH20backward = fftw_plan_dft_3d(N,N,N,SH20,SH20,FFTW_BACKWARD,FFTW_MEASURE);
	deltabackward = fftw_plan_dft_3d(N,N,N,delta,delta,FFTW_BACKWARD,FFTW_MEASURE);
	xbackward = fftw_plan_dft_3d(N,N,N,dx,dx,FFTW_BACKWARD,FFTW_MEASURE);
	ybackward = fftw_plan_dft_3d(N,N,N,dy,dy,FFTW_BACKWARD,FFTW_MEASURE);
	zbackward = fftw_plan_dft_3d(N,N,N,dz,dz,FFTW_BACKWARD,FFTW_MEASURE);
	double xmax,xmin,ymax,ymin,zmax,zmin;
//#pragma omp parallel for
	for (int xin=0;xin<N;xin++){
	 for (int yin=0;yin<N;yin++){
	  for (int zin=0;zin<N;zin++){
	  int index;
	  index = N*N*xin+N*yin+zin;
	  SH0(SH00[index]);
	  SH2(SH22[index],xin,yin,zin,2);
	  SH2(SH21[index],xin,yin,zin,1);
	  SH2(SH20[index],xin,yin,zin,0);
	  SH2(SH2n1[index],xin,yin,zin,-1);
	  SH2(SH2n2[index],xin,yin,zin,-2);
	  }
	 }
	}
	fftw_execute(deltaforward);
	fftw_execute(SH00forward);
	fftw_execute(SH22forward);
	fftw_execute(SH21forward);
	fftw_execute(SH20forward);
	fftw_execute(SH2n1forward);
	fftw_execute(SH2n2forward);
	for (int xin=0;xin<N;xin++){
	 for (int yin=0;yin<N;yin++){
	  for (int zin=0;zin<N;zin++){
	  int index;
	  index = N*N*xin+N*yin+zin;
	  SH00[index][0] *= delta[index][0];
	  SH22[index][0] *= delta[index][0];
	  SH21[index][0] *= delta[index][0];
	  SH20[index][0] *= delta[index][0];
	  SH2n1[index][0] *= delta[index][0];
	  SH2n2[index][0] *= delta[index][0];
	  SH00[index][1] *= delta[index][1];
	  SH22[index][1] *= delta[index][1];
	  SH21[index][1] *= delta[index][1];
	  SH20[index][1] *= delta[index][1];
	  SH2n1[index][1] *= delta[index][1];
	  SH2n2[index][1] *= delta[index][1];
	  }
	 }
	}
	fftw_execute(SH00forward);
	fftw_execute(SH22forward);
	fftw_execute(SH21forward);
	fftw_execute(SH20forward);
	fftw_execute(SH2n1forward);
	fftw_execute(SH2n2forward);
	for (int xin=0;xin<N;xin++){
	 for (int yin=0;yin<N;yin++){
	  for (int zin=0;zin<N;zin++){
	  int index;
	  index = N*N*xin+N*yin+zin;
	  SH00[index][0] *= SH00[index][0];
	  SH22[index][0] *= SH2n2[index][0];
	  SH21[index][0] *= SH2n1[index][0];
	  SH20[index][0] *= SH20[index][0];
	  SH00[index][1] *= SH00[index][1];
	  SH22[index][1] *= SH2n2[index][1];
	  SH21[index][1] *= SH2n1[index][1];
	  SH20[index][1] *= SH20[index][1];
	  }
	 }
	}
	fftw_execute(SH00backward);
	fftw_execute(SH22backward);
	fftw_execute(SH21backward);
	fftw_execute(SH20backward);
	for (int xin=0;xin<N;xin++){
	 for (int yin=0;yin<N;yin++){
	  for (int zin=0;zin<N;zin++){
	  int index;
	  index = N*N*xin+N*yin+zin;
	  SH22[index][0] = 4*M_PI*2*(SH00[index][0]-(2*SH22[index][0]/5+2*SH21[index][0]/5+SH20[index][0]/5))/3;
	  SH22[index][1] = 4*M_PI*2*(SH00[index][1]-(2*SH22[index][1]/5+2*SH21[index][1]/5+SH20[index][1]/5))/3;
	  }
	 }
	}
	for (int xin=0;xin<N;xin++){
	double x = (xin<N/2)?xin:xin-N;
	 for (int yin=0;yin<N;yin++){
 	 double y = (yin<N/2)?yin:yin-N;
	  for (int zin=0;zin<N;zin++){
  	  double z = (zin<N/2)?zin:zin-N;
	  int index;
	  double k2,im,re;
	  index = N*N*xin+N*yin+zin;
	  k2 = k*(x*x+y*y+z*z);
	  if (k2==0){}
	  else{
	   im = SH22[index][1];
	   re = SH22[index][0];
	   dx[index][0] = 3*(-(double)x/k2)*im/14;
	   dx[index][1] = 3*((double)x/k2)*re/14;
	   dy[index][0] = 3*(-(double)y/k2)*im/14;
	   dy[index][1] = 3*((double)y/k2)*re/14;
	   dz[index][0] = 3*(-(double)z/k2)*im/14;
	   dz[index][1] = 3*((double)z/k2)*re/14;
	   }
	  }
	 }
	}
	fftw_execute(SH22backward);
	fftw_execute(xbackward);
	fftw_execute(ybackward);
	fftw_execute(zbackward);

	for (int xin=0;xin<N;xin++){
	 for (int yin=0;yin<N;yin++){
	  for (int zin=0;zin<N;zin++){
	  int index;
	  index = N*N*xin+N*yin+zin;
	  dx[index][0] /= (double) N*N*N*N*N*N*N*N*N*N*N*N;//*(8*M_PI*M_PI*M_PI); // The second forward FFT gives extra N3 factor, so we have 12 in total(3+3 from second time forward transform and 6 from two backward transform;
	  dy[index][0] /= (double) N*N*N*N*N*N*N*N*N*N*N*N;//*(8*M_PI*M_PI*M_PI); 
	  dz[index][0] /= (double) N*N*N*N*N*N*N*N*N*N*N*N;//*(8*M_PI*M_PI*M_PI); 
	  dx[index][1] /= (double) N*N*N*N*N*N*N*N*N*N*N*N;//*(8*M_PI*M_PI*M_PI);
	  dy[index][1] /= (double) N*N*N*N*N*N*N*N*N*N*N*N;//*(8*M_PI*M_PI*M_PI); 
	  dz[index][1] /= (double) N*N*N*N*N*N*N*N*N*N*N*N;//*(8*M_PI*M_PI*M_PI); 
	  }
	 }
	}
}


int main(int argc, char** argv)
{
	string name1=argv[1],name2=argv[2];
	string outname=argv[1],outname2=argv[1];
	outname.append("_recon");outname2.append("_reconrandom");
	cout<<"# The output is "<<outname<<endl;
	vector<struct particle> DataArray=readfile(name1);
	vector<struct particle> RandomArray=readfile(name2);
	SetBox(RandomArray);
	k = 2*M_PI/box.Length;
	MapPositiontoGrid(DataArray);
	MapPositiontoGrid(RandomArray);
	fftw_complex *data,*random,*dx,*dy,*dz;
	data =(fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	random =(fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	dx =(fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	dy =(fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	dz =(fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	CICassignment(DataArray,data);
//	PrintStat(data,"data");
	CICassignment(RandomArray,random);	
//	PrintStat(random,"random");
	Masking(data,random);
//	PrintStat(data,"overdensity");
	GaussianSmoothing(data,smoothingScale);
	FirstDisplacement(data,dx,dy,dz);
//	PrintStat(dx,"dx");
//	PrintStat(dy,"dy");
//	PrintStat(dz,"dz");
//	for (int ix=0;ix<N;ix++){
//		for (int iy=0;iy<N;iy++){
//			for (int iz=0;iz<N;iz++){
//				cout<<ix<<" "<<iy<<" "<<iz<<" "<<dx[N*N*ix+N*iy+iz][0]<<" "<<dy[N*N*ix+N*iy+iz][0]<<" "<<dz[N*N*ix+N*iy+iz][0]<<endl;
//	}}}
/*	fftw_complex *dx2,*dy2,*dz2;
	dx2 =(fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	dy2 =(fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	dz2 =(fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	SecondDisplacement(data,dx2,dy2,dz2);
	PrintStat(dx2,"dx2");
	for (int a=0;a<N;a++){for (int b=0;b<N;b++){for (int c=0;c<N;c++){
	dx[N*N*a+N*b+c][0]+=dx2[N*N*a+N*b+c][0];
	dy[N*N*a+N*b+c][0]+=dy2[N*N*a+N*b+c][0];
	dz[N*N*a+N*b+c][0]+=dz2[N*N*a+N*b+c][0];
	}}}*/
	CICinterpolation(DataArray,dx,dy,dz);
	CICinterpolation(RandomArray,dx,dy,dz);
	MapGridtoPosition(DataArray);
	MapGridtoPosition(RandomArray);
	WriteFile(DataArray,outname);
	WriteFile(RandomArray,outname2);
}



