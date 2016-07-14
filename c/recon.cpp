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

int N=256;
struct Box box; 
double k;

struct particle CreateParticle(const double x, const double y, const double z){

	struct particle local;
	local.pos[0] = x;
	local.pos[1] = y;
	local.pos[2] = z;
	return local;
}

void SetBox(vector<struct particle> &input){
	double minmax[3][2],min,max;
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
	 min = (min<minmax[i][0])?min:minmax[i][0];
	 max = (max>minmax[i][1])?max:minmax[i][1];
	}
	box.Length = (max-min)*1.5;
	
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
	double mean,count,count2;
	for (int xin=0;xin<N;xin++){
	 for (int yin=0;yin<N;yin++){
	  for (int zin=0;zin<N;zin++){
	   index = N*N*xin+N*yin+zin;
	   if (random[index][0]>0 && random[index][0] < 2){
	   random[index][0]=data[index][0]=0;
	   count2 ++;
	   }
	   data[index][0] = data[index][0]/random[index][0];
	   if (random[index][0]!=0){
	    count ++;
	    mean += data[index][0];
	    }
	   }
	  }
	 }
	cout<<"#"<<mean<<" "<<count<<" "<<count2<<endl;
	mean/=count;
	for (int xin=0;xin<N;xin++){
	 for (int yin=0;yin<N;yin++){
	  for (int zin=0;zin<N;zin++){
	   index = N*N*xin+N*yin+zin;
	   data[index][0]= (random[index][0]!=0)?(data[index][0]/mean-1):0;
	   }
	  }
	 }
	mean=0;
	count=0;
	for (int xin=0;xin<N;xin++){
	 for (int yin=0;yin<N;yin++){
	  for (int zin=0;zin<N;zin++){
	   index = N*N*xin+N*yin+zin;
	   if (random[index][0]!=0){
	    count ++;
	    mean += data[index][0]*data[index][0];
	    }
	   }
	  }
	 }
	cout<<"#"<<sqrt(mean/count)<<endl;
}

void CICassignment(const vector<struct particle> data,fftw_complex *grid){
#pragma omp parallel for shared(grid)
	for (int i=0;i<data.size();i++){
		int index[6];
		float difference[3];
		index[0]=data[i].pos[0];index[3]=(index[0]+1)%N;difference[0]=data[i].pos[0]-index[0];
		index[1]=data[i].pos[1];index[4]=(index[1]+1)%N;difference[1]=data[i].pos[1]-index[1];
		index[2]=data[i].pos[2];index[5]=(index[2]+1)%N;difference[2]=data[i].pos[2]-index[2];
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
#pragma omp parallel for shared(data,dx,dy,dz)
	for (int i=0;i<data.size();i++){
		double px,py,pz;
		int index[6];
		float difference[3];
		index[0]=data[i].pos[0];index[3]=(index[0]+1)%N;difference[0]=data[i].pos[0]-index[0];
		index[1]=data[i].pos[1];index[4]=(index[1]+1)%N;difference[1]=data[i].pos[1]-index[1];
		index[2]=data[i].pos[2];index[5]=(index[2]+1)%N;difference[2]=data[i].pos[2]-index[2];
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
		data[i].pos[0]-=px*N/box.Length;
		data[i].pos[1]-=py*N/box.Length;
		data[i].pos[2]-=pz*N/box.Length;
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

void FirstDisplacement(fftw_complex *delta,fftw_complex *dx,fftw_complex *dy,fftw_complex *dz){
	fftw_plan xforward,xbackward,yforward,ybackward,zforward,zbackward;
	xforward = fftw_plan_dft_3d(N,N,N,delta,dx,FFTW_FORWARD,FFTW_ESTIMATE);
	yforward = fftw_plan_dft_3d(N,N,N,delta,dy,FFTW_FORWARD,FFTW_ESTIMATE);
	zforward = fftw_plan_dft_3d(N,N,N,delta,dz,FFTW_FORWARD,FFTW_ESTIMATE);
	xbackward = fftw_plan_dft_3d(N,N,N,dx,dx,FFTW_BACKWARD,FFTW_MEASURE);
	ybackward = fftw_plan_dft_3d(N,N,N,dy,dy,FFTW_BACKWARD,FFTW_MEASURE);
	zbackward = fftw_plan_dft_3d(N,N,N,dz,dz,FFTW_BACKWARD,FFTW_MEASURE);
	fftw_execute(xforward);
	fftw_execute(yforward);
	fftw_execute(zforward);
//#pragma omg parallel for shared(delta,dx,dy,dz)
	for (int xin=0;xin<N;xin++){
	double x = (xin<N/2)?xin:xin-N;
	 for (int yin=0;yin<N;yin++){
	 double y = (yin<N/2)?yin:yin-N;
	  for (int zin=0;zin<N;zin++){
  	  double z = (zin<N/2)?zin:zin-N;
	  int index;
	  index = N*N*xin+N*yin+zin;
	  const double k2 = k*(x*x+y*y+z*z);
	  if (k2==0){}
	  else{
	   const double im = dx[index][1];
	   const double re = dx[index][0];
	   dx[index][0] = (-(double)x/k2)*im;
	   dx[index][1] = ((double)x/k2)*re;
	   dy[index][0] = (-(double)y/k2)*im;
	   dy[index][1] = ((double)y/k2)*re;
	   dz[index][0] = (-(double)z/k2)*im;
	   dz[index][1] = ((double)z/k2)*re;
	   if (index ==256) cout<<"256:"<<y<<" "<<z<<" "<<k2<<" "<<im<<" "<<re<<endl;
	   if (index ==1) cout<<"1:"<<y<<" "<<z<<" "<<k2<<" "<<im<<" "<<re<<endl;
	   }
	  }
	 }
	}
	fftw_execute(xbackward);
	fftw_execute(ybackward);
	fftw_execute(zbackward);
	double xmax,xmin,ymax,ymin,zmax,zmin;
	for (int i =0;i<(N*N*N);i++){
	dx[i][0]/=(N*N*N);
	dx[i][1]/=(N*N*N);
	dy[i][0]/=(N*N*N);
	dy[i][1]/=(N*N*N);
	dz[i][0]/=(N*N*N);
	dz[i][1]/=(N*N*N);
	xmin = (xmin<dx[i][0])?xmin:dx[i][0];
	xmax = (xmax>dx[i][0])?xmax:dx[i][0];
	ymin = (ymin<dy[i][0])?ymin:dy[i][0];
	ymax = (ymax>dy[i][0])?ymax:dy[i][0];
	zmin = (zmin<dz[i][0])?zmin:dz[i][0];
	zmax = (zmax>dz[i][0])?zmax:dz[i][0];
	}
	cout<<"#"<<xmin<<" "<<xmax<<" "<<ymin<<" "<<ymax<<" "<<zmin<<" "<<zmax<<endl;
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
	xmax=xmin= 0;
	for (int i =0;i<(N*N*N);i++){
	xmin = (xmin<SH22[i][0])?xmin:SH22[i][0];
	xmax = (xmax>SH22[i][0])?xmax:SH22[i][0];
	}
	cout<<xmin<<" "<<xmax<<endl;
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
	  dx[index][0] /=(double) N*N*N*N*N*N*N*N*N*N*N*N; // The second forward FFT gives extra N3 factor, so we have 12 in total(3+3 from second time forward transform and 6 from two backward transform;
	  dy[index][0] /= (double) N*N*N*N*N*N*N*N*N*N*N*N; 
	  dz[index][0] /= (double) N*N*N*N*N*N*N*N*N*N*N*N; 
	  dx[index][1] /=(double) N*N*N*N*N*N*N*N*N*N*N*N;
	  dy[index][1] /= (double) N*N*N*N*N*N*N*N*N*N*N*N; 
	  dz[index][1] /= (double) N*N*N*N*N*N*N*N*N*N*N*N; 
	  }
	 }
	}
}

int main()
{
	string name1="data_raw.xyzw",name2="rand_raw.xyzw";
	string outname="datarec.dat",outname2="randrec.dat";
	vector<struct particle> DataArray=readfile(name1);
	vector<struct particle> RandomArray=readfile(name2);
	SetBox(RandomArray);
	k = 2*M_PI/box.Length;
	MapPositiontoGrid(DataArray);
	MapPositiontoGrid(RandomArray);
	fftw_complex *data,*random,*dx,*dy,*dz,*dx2,*dy2,*dz2;
	data =(fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	random =(fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	dx =(fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	dy =(fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	dz =(fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	dx2 =(fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	dy2 =(fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	dz2 =(fftw_complex*) fftw_malloc(N*N*N*sizeof(fftw_complex));
	CICassignment(DataArray,data);
	CICassignment(RandomArray,random);	
	Masking(data,random);
	FirstDisplacement(data,dx,dy,dz);
	SecondDisplacement(data,dx2,dy2,dz2);
	for (int a=0;a<N;a++){for (int b=0;b<N;b++){for (int c=0;c<N;c++){
	dx[N*N*a+N*b+c][0]+=dx2[N*N*a+N*b+c][0];
	dy[N*N*a+N*b+c][0]+=dy2[N*N*a+N*b+c][0];
	dz[N*N*a+N*b+c][0]+=dz2[N*N*a+N*b+c][0];
	}}}
	CICinterpolation(DataArray,dx,dy,dz);
	CICinterpolation(RandomArray,dx,dy,dz);
	MapGridtoPosition(DataArray);
	MapGridtoPosition(RandomArray);
	return 0;
}
