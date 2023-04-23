#pragma once
#include <bitset>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <time.h>
#include <string>
#include <iostream>
#include <atomic> 
#include <cmath>
#include <functional>
#include <numeric>
#include <pthread.h>

using namespace std;

#define MACRO_D2 {{2,0},{1,1}}
#define MACRO_D4 {{2,0,0,0},{1,1,0,0},{1,0,1,0},{1,0,0,1}}
#define MACRO_D8                                                                           \
   {{2.0000,         0,         0,         0,         0,         0,         0,         0}, \
    {1.0000,    1.0000,         0,         0,         0,         0,         0,         0}, \
    {1.0000,         0,    1.0000,         0,         0,         0,         0,         0}, \
    {1.0000,         0,         0,    1.0000,         0,         0,         0,         0}, \
    {1.0000,         0,         0,         0,    1.0000,         0,         0,         0}, \
    {1.0000,         0,         0,         0,         0,    1.0000,         0,         0}, \
    {1.0000,         0,         0,         0,         0,         0,    1.0000,         0}, \
    {0.5000,    0.5000,    0.5000,    0.5000,    0.5000,    0.5000,    0.5000,    0.5000}}
#define MACRO_D16                                                                                                  \
    {{4,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0}, \
     {2,     2,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0}, \
     {2,     0,     2,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0}, \
     {2,     0,     0,     2,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0}, \
     {2,     0,     0,     0,     2,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0}, \
     {2,     0,     0,     0,     0,     2,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0}, \
     {2,     0,     0,     0,     0,     0,     2,     0,     0,     0,     0,     0,     0,     0,     0,     0}, \
     {2,     0,     0,     0,     0,     0,     0,     2,     0,     0,     0,     0,     0,     0,     0,     0}, \
     {2,     0,     0,     0,     0,     0,     0,     0,     2,     0,     0,     0,     0,     0,     0,     0}, \
     {2,     0,     0,     0,     0,     0,     0,     0,     0,     2,     0,     0,     0,     0,     0,     0}, \
     {2,     0,     0,     0,     0,     0,     0,     0,     0,     0,     2,     0,     0,     0,     0,     0}, \
     {1,     1,     1,     1,     0,     1,     0,     1,     1,     0,     0,     1,     0,     0,     0,     0}, \
     {0,     1,     1,     1,     1,     0,     1,     0,     1,     1,     0,     0,     1,     0,     0,     0}, \
     {0,     0,     1,     1,     1,     1,     0,     1,     0,     1,     1,     0,     0,     1,     0,     0}, \
     {0,     0,     0,     1,     1,     1,     1,     0,     1,     0,     1,     1,     0,     0,     1,     0}, \
     {1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1}}
#define MACRO_D24                                                             \
    {{8 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
     {4 ,4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
     {4 ,0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
     {4 ,0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
     {4 ,0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
     {4 ,0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
     {4 ,0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
     {2 ,2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
     {4 ,0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
     {4 ,0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
     {4 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
     {2 ,2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
     {4 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
     {2 ,2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
     {2 ,0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
     {2 ,0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0},\
     {4 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0},\
     {2 ,0, 2, 0, 2, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0},\
     {2 ,0, 0, 2, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0},\
     {2 ,2, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0},\
     {0 ,2, 2, 2, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0},\
     {0 ,0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 0, 0},\
     {0 ,0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0},\
     {-3,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}
#define MACRO_D32                                                                \
   {{4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0},\
    {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0},\
    {1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0},\
    {1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0},\
    {0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0},\
    {0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0},\
    {1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0},\
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}
/*voronoi constellation*/
class VC{
public:
  cv::Mat G; //64F
  cv::Mat Gs; //64F
  cv::Mat ShiftVector; //double
  unsigned int d; //numof dimention
  unsigned int r;
  double M; //constellation size
  cv::Mat diag; //64F integer range
  unsigned long Ns;
  bitset<64> *b;
  cv::Mat BitsArray; //8U
  cv::Mat c; //64F
  cv::Mat symbol; //double MD VC symbols
  double Es; //average symbol energy

  int nofthreadused;

  VC(unsigned int _d, unsigned int _r, unsigned long _Ns, int _nofthreadused){
    this->nofthreadused = _nofthreadused; 
    cv::RNG rng = cv::theRNG().state = cv::getTickCount();
    this->d = _d; this->r = _r;
    /*construct G*/
    this->G = cv::Mat::eye(_d, _d, CV_64FC1);
    /*construct Gs*/
    double tmp[8];
    switch(_d){
      case 2:
        tmp[0]=-0.5; tmp[1]=0;
        this->Gs = cv::Mat(_d, _d, CV_64FC1, D2) * pow(2,r);
        this->ShiftVector = cv::Mat(1,_d,CV_64FC1, tmp);
        break;
      case 4:
        this->Gs = cv::Mat(_d, _d, CV_64FC1, D4) * pow(2,r);
        switch(_r){
          case 0:
            tmp[0] = 0; tmp[1] = -0.5; tmp[2] = 0; tmp[3] = 0;
            this->ShiftVector = cv::Mat(1,_d,CV_64FC1,tmp);
            break;
          case 1:
            tmp[0] = 0.25; tmp[1] = -0.375; tmp[2] = 0.5; tmp[3] = 0;
            this->ShiftVector = cv::Mat(1,_d,CV_64FC1,tmp);
            break;
          case 2:
            tmp[0] = -0.3438; tmp[1] = 0; tmp[2] = -0.1875; tmp[3] = 0.5;
            this->ShiftVector = cv::Mat(1,_d,CV_64FC1,tmp);
            break;
          case 3:
            tmp[0] = 0; tmp[1] = -0.5; tmp[2] = -0.1719; tmp[3] = -0.3359;
            this->ShiftVector = cv::Mat(1,_d,CV_64FC1,tmp);
            break;
          case 4:
            tmp[0] = 0.5; tmp[1] = 0; tmp[2] = -0.1680; tmp[3] = 0.334;
            this->ShiftVector = cv::Mat(1,_d,CV_64FC1,tmp);
            break;
          default:
            this->ShiftVector = cv::Mat(1,_d,CV_64FC1);
            rng.fill(this->ShiftVector, cv::RNG::UNIFORM, 0, 1);
            this->ShiftVector = this->ShiftVector - (0.5*cv::Mat::ones(1,_d,CV_64FC1));
        }
        break;
      case 8:
        this->Gs = cv::Mat(_d, _d, CV_64FC1, D8) * pow(2,r);
        switch(_r){
          case 1:
            tmp[0]=0; tmp[1]=-0.4688; tmp[2]=0; tmp[3]=-0.4844; tmp[4]=-0.375; tmp[5]=-0.5; tmp[6]=0; tmp[7]=0.4375;
            this->ShiftVector = cv::Mat(1,_d,CV_64FC1);
            std::memcpy(this->ShiftVector.data, tmp, _d*sizeof(double));
            break;
          case 2:
            tmp[0]=0.4617; tmp[1]=0.4233; tmp[2]=0.5; tmp[3]=0.1299; tmp[4]=0; tmp[5]=-0.0317; tmp[6]=-0.0630; tmp[7]=-0.0952;
            this->ShiftVector = cv::Mat(1,_d,CV_64FC1);
            std::memcpy(this->ShiftVector.data, tmp, _d*sizeof(double));
            break;
          default:
            this->ShiftVector = cv::Mat(1,_d,CV_64FC1);
            rng.fill(this->ShiftVector, cv::RNG::UNIFORM, 0, 1);
            this->ShiftVector = this->ShiftVector - (0.5*cv::Mat::ones(1,_d,CV_64FC1));
        }
        break;
      case 16:
        this->Gs = cv::Mat(_d, _d, CV_64FC1, D16) * pow(2,r);
        this->ShiftVector = cv::Mat(1,_d,CV_64FC1);
        rng.fill(this->ShiftVector, cv::RNG::UNIFORM, 0, 1);
        this->ShiftVector = this->ShiftVector - (0.5*cv::Mat::ones(1,_d,CV_64FC1));
        break;
      case 24:
        this->Gs = cv::Mat(_d, _d, CV_64FC1, D24) * pow(2,r);
        this->ShiftVector = cv::Mat(1,_d,CV_64FC1);
        rng.fill(this->ShiftVector, cv::RNG::UNIFORM, 0, 1);
        this->ShiftVector = this->ShiftVector - (0.5*cv::Mat::ones(1,_d,CV_64FC1));
        break;
      case 32:
        this->Gs = cv::Mat(_d, _d, CV_64FC1, D32) * pow(2,r);
        this->ShiftVector = cv::Mat(1,_d,CV_64FC1);
        rng.fill(this->ShiftVector, cv::RNG::UNIFORM, 0, 1);
        this->ShiftVector = this->ShiftVector - (0.5*cv::Mat::ones(1,_d,CV_64FC1));
        break;
      default:
        cout<<"supported dimension is either 2, 4, 8, 16, 24, 32. Exiting"<<endl;
        exit(0);
    }
    M=cv::determinant(this->Gs);
    diag = Gs.diag(0);
    Ns = _Ns;

    /*generate random bits*/
    this->BitsArray = cv::Mat::zeros(Ns, (unsigned int)log2(M), CV_8UC1);
    rng.fill(this->BitsArray, cv::RNG::UNIFORM, 0+(uint)'0', 2+(uint)'0');
    m2c_Forney(this->BitsArray, this->diag, this->c, this->Ns);
    
    encoder_Forney(this->c, this->G, this->ShiftVector, this->r, this->d, this->symbol);

    //each symbol - ShiftVector
    parallel_for_(cv::Range(0, this->symbol.rows), [&](const cv::Range& range){
      for (int r = range.start; r < range.end; r++) {
        this->symbol.row(r) = this->symbol.row(r) - this->ShiftVector;
      }
    });
    cv::Mat abs_sqrt = cv::abs(this->symbol);
    cv::pow(abs_sqrt, 2, abs_sqrt);
    cv::Mat EnergySymbols;
    cv::reduce(abs_sqrt, EnergySymbols, 1, cv::REDUCE_SUM, -1);
    cv::Scalar Es_Scalar = cv::mean(EnergySymbols);
    this->Es = Es_Scalar(0);
  }
  void m2c_Forney(const cv::Mat& BitsArray, const cv::Mat& D, cv::Mat& c, const unsigned long& Ns){
    vector<double> d_log2;
    this->MatToVector(D, &d_log2);    
    //for_each(d_log2.begin(), d_log2.end(),[](double& n){cout<<n<<" "<<endl;});
    for_each(d_log2.begin(), d_log2.end(),[](double& n){n = log2(n);});
    //for_each(d_log2.begin(), d_log2.end(),[](double& n){cout<<n<<" "<<endl;});
    cv::Mat c_tmp[D.rows];
    for(uint i=0; i<d_log2.size(); i++){
      uint start_index=(uint)std::accumulate(d_log2.begin(), d_log2.begin() + i, 0);
      uint end_index = start_index + (uint)d_log2[i] - 1;
      cv::Mat tmpmat = BitsArray(cv::Rect(start_index, 0, (uint)d_log2[i], Ns));
      c_tmp[i] = cv::Mat::zeros(Ns, 1, CV_32SC1);
      cv::parallel_for_(cv::Range{0, tmpmat.rows}, MatRowParFor(tmpmat, [&](cv::Mat RowMat, int RowIdx)->void{
        //c_tmp[i].at<int>(RowIdx) = 
        vector<uchar> RowInVec;
        this->MatToVector(RowMat, &RowInVec);
        string RowInString(RowInVec.begin(), RowInVec.end());
        c_tmp[i].at<int>(RowIdx,0) = stoi(RowInString, 0, 2);
      }));
    }
    cv::hconcat(c_tmp, D.rows, c);
    c.convertTo(c, CV_64FC1);
  }
  void encoder_Forney(const cv::Mat& c, const cv::Mat& G, const cv::Mat& ShiftVector, const uint& r, const uint& d, cv::Mat& symbol){
    cv::Mat cG = c*G;
    cv::Mat y = cG - cv::Mat::ones(c.rows, 1, CV_64FC1) * ShiftVector;
    cv::Mat center[2];
    double tmp = pow(2,r);
    center[0] = y / tmp;
    switch (d) {
      case 2:
      case 4:
        CPA_DN(center);
        break;
      case 8:
        center[1] = CPA_E8(center[0]);
        break;
      case 16:
        center[1] = CPA_L16(center[0]);
        break;
      case 24:
        center[1] = CPA_L24(center[0]);
        break;
      default:
        cout<<"does support "<<d<<"-dimention. Exiting";
        exit(0);
    }
    center[1].convertTo(center[1], CV_64FC1);
    center[1] = center[1] * tmp;
    symbol = cG - center[1];
  } 
  cv::Mat CPA_L24(const cv:: Mat& y){
    const uint bitwidth = 12;
    const uint nof_cc_array_row = 4096; // pow(2, bitwidth)
    const uint nofcopy = 8192;
    const double copyratio=4;
    int G_coset_rep_array[bitwidth][24] =                                               \
      {{1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, \
       {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, \
       {1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, \
       {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}, \
       {1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}, \
       {1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0}, \
       {1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0}, \
       {1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0}, \
       {0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}, \
       {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0}, \
       {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, \
       {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
    cv::Mat G_coset_rep(bitwidth, 24, CV_32SC1, G_coset_rep_array);
    string BinaryString;
    cv::Mat cc_array[nof_cc_array_row];
    for(int i=0; i<nof_cc_array_row; i++){
      BinaryString = std::bitset<bitwidth>(i).to_string();
      cv::Mat tmp(1, 12, CV_8UC1, BinaryString.data());
      tmp.copyTo(cc_array[i]);
    }
    cv::Mat cc;
    cv::vconcat(cc_array, cv::pow(2, bitwidth), cc);
    cc = cc - 48;

    cc.convertTo(cc, CV_64FC1);
    G_coset_rep.convertTo(G_coset_rep, CV_64FC1);
    /*cc is like the following
    [ 0, 0, 0, 0, 0;
      0, 0, 0, 0, 1;
      0, 0, 0, 1, 0;
      0, 0, 0, 1, 1;
      ......
      1, 1, 1, 1, 0;
      1, 1, 1, 1, 1;
    */
    cv::Mat coset_rep = cc * G_coset_rep;
    coset_rep.convertTo(coset_rep, CV_32SC1);
    int u_array[24] ={-3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; 
    cv::Mat u(1, 24, CV_32SC1,u_array);
    cv::Mat coset_repx2plusu(coset_rep.size(), CV_32SC1);
    parallel_for_(cv::Range(0, coset_rep.rows), [&](const cv::Range& range){
      for(int r = range.start; r<range.end; r++){
        //coset_repx2plusu.row(r) = 2*coset_rep.row(r) + u;
        coset_repx2plusu.row(r) = 2 * coset_rep.row(r);
        coset_repx2plusu.row(r) = coset_repx2plusu.row(r) + u;
      }
    });
    cv::vconcat(2*coset_rep, coset_repx2plusu, coset_rep);
    coset_rep.convertTo(coset_rep, CV_64FC1);
    pthread_t threads[this->nofthreadused];
    int nofemitted=0;
    cv::Mat y_thread[nofcopy][2];


    while(nofemitted<nofcopy){
      int nofemitted_current_loop=0;
      for(int i=0; i<this->nofthreadused; i++){
        cv::Mat CurrentCopyOfy = y.clone();
        cv::parallel_for_(cv::Range{0, y.rows}, MatRowParFor(y, [&](cv::Mat RowMat, int RowIdx)->void{
          CurrentCopyOfy.row(RowIdx) = (RowMat - coset_rep.row(nofemitted))/copyratio;
        }));
        //cout<<y_thread[nofemitted][0].size()<<CurrentCopyOfy.size()<<endl;
        CurrentCopyOfy.copyTo(y_thread[nofemitted][0]);
        int tmp = pthread_create(&threads[i], NULL, CPA_DN, (void*)(y_thread[nofemitted]));
        if(tmp > 0){
          cout<<"failed at CPA_L24, the number of threads being used is "<<this->nofthreadused<<endl;
          cout<<"Maybe try a smaller number"<<endl;
          cout<<"Exiting"<<endl;
          exit(0);
        } else{
          nofemitted++;
          nofemitted_current_loop++;
          if(nofemitted>=nofcopy)
            break;
        }
      }
      for(int i=0; i<nofemitted_current_loop; i++){
        void(pthread_join(threads[i], NULL));
      }
      //cout<<"number of emitted threads is " << nofemitted<<endl;
    }
    //convert y_thread[i][1] to double, the result generated from CPA_DN is of int type
    for(int i=0; i<nofcopy; i++)
      y_thread[i][1].convertTo(y_thread[i][1], CV_64FC1);
    cv::Mat ret(y.size(), CV_64FC1);
    cv::parallel_for_(cv::Range(0, y.rows), [&](const cv::Range& range){
      for(int r = range.start; r<range.end; r++){
        uint MinIdx = 0;
        cv::Mat diff_sqr;
        cv::Mat d;

        y_thread[0][1].row(r) = y_thread[0][1].row(r) * copyratio;
        cv::add(y_thread[0][1].row(r), coset_rep.row(0), y_thread[0][1].row(r));
        cv::absdiff(y_thread[0][1].row(r), y.row(r), diff_sqr); 
        cv::pow(diff_sqr, 2, diff_sqr);
        cv::reduce(diff_sqr, d, 1, cv::REDUCE_SUM, -1);
        double d_init = d.at<double>(0,0);
        
        for(int i=1; i<nofcopy; i++){
          cv::add(y_thread[i][1].row(r) * copyratio, coset_rep.row(i), y_thread[i][1].row(r));
          cv::absdiff(y_thread[i][1].row(r), y.row(r), diff_sqr); 
          cv::pow(diff_sqr, 2, diff_sqr);
          cv::reduce(diff_sqr, d, 1, cv::REDUCE_SUM, -1);
          if(d.at<double>(0,0) < d_init){
            d_init = d.at<double>(0,0);
            MinIdx = i;
          }
        }
        //cout<<"r = "<<r<<endl;
        //cout<<"MinIdx = "<<MinIdx<<endl;
        //cout<<"y_thread[MinIdx][1].row(r) is "<<y_thread[MinIdx][1].row(r)<<endl;
        y_thread[MinIdx][1].row(r).copyTo(ret.row(r));
      }
    });
    return ret;
  }
  cv::Mat CPA_L16(const cv:: Mat& y){
    int G_coset_rep_array[5][16] =                             \
      {{1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0}, \
       {0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0}, \
       {0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0}, \
       {0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0}, \
       {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};
    cv::Mat G_coset_rep(5, 16, CV_32SC1, G_coset_rep_array);
    string BinaryString[32];
    cv::Mat cc_array[32];
    for(int i=0; i<32; i++){
      BinaryString[i] = std::bitset<5>(i).to_string();
      cv::Mat tmp(1, 5, CV_8UC1, BinaryString[i].data());
      tmp.copyTo(cc_array[i]);
    }
    cv::Mat cc;
    cv::vconcat(cc_array, 32, cc);
    cc = cc - 48;
    cc.convertTo(cc, CV_64FC1);
    G_coset_rep.convertTo(G_coset_rep, CV_64FC1);
    /*cc is like the following
    [ 0, 0, 0, 0, 0;
      0, 0, 0, 0, 1;
      0, 0, 0, 1, 0;
      0, 0, 0, 1, 1;
      ......
      1, 1, 1, 1, 0;
      1, 1, 1, 1, 1;
    */
    cv::Mat coset_rep = cc * G_coset_rep;
    coset_rep.convertTo(coset_rep, CV_32SC1);
    coset_rep.forEach<int>(
      [](int& i_coset_rep, const int* position)->void{
        i_coset_rep = i_coset_rep % 2;  
      }
    );
    coset_rep.convertTo(coset_rep, CV_64FC1);

    pthread_t threads[this->nofthreadused];
    int nofemitted=0;
    cv::Mat y_thread[32][2];
    while(nofemitted<32){
      int nofemitted_current_loop=0;
      for(int i=0; i<this->nofthreadused; i++){
        cv::Mat CurrentCopyOfy = y.clone();
        cv::parallel_for_(cv::Range{0, y.rows}, MatRowParFor(y, [&](cv::Mat RowMat, int RowIdx)->void{
          CurrentCopyOfy.row(RowIdx) = 0.5*(RowMat - coset_rep.row(nofemitted));
        }));
        y_thread[nofemitted][0] = CurrentCopyOfy;
        int tmp = pthread_create(&threads[i], NULL, CPA_DN, (void*)(y_thread[nofemitted]));
        if(tmp > 0){
          cout<<"failed at CPA_L16, the number of threads being used is "<<this->nofthreadused<<endl;
          cout<<"Maybe try a smaller number"<<endl;
          cout<<"Exiting"<<endl;
          exit(0);
        } else{
          nofemitted++;
          nofemitted_current_loop++;
          if(nofemitted>=32)
            break;
        }
      }
      for(int i=0; i<nofemitted_current_loop; i++){
        void(pthread_join(threads[i], NULL));
      }
    }
    /*convert y_thread[i][1] to double, the result generated from CPA_DN is of int type*/
    for(int i=0; i<32; i++)
      y_thread[i][1].convertTo(y_thread[i][1], CV_64FC1);
    cv::Mat ret(y.size(), CV_64FC1);
    cv::parallel_for_(cv::Range(0, y.rows), [&](const cv::Range& range){
      for(int r = range.start; r<range.end; r++){
        uint MinIdx = 0;
        cv::Mat diff_sqr;
        cv::Mat d;

        y_thread[0][1].row(r) = y_thread[0][1].row(r) * 2;
        cv::add(y_thread[0][1].row(r), coset_rep.row(0), y_thread[0][1].row(r));
        cv::absdiff(y_thread[0][1].row(r), y.row(r), diff_sqr); 
        cv::pow(diff_sqr, 2, diff_sqr);
        cv::reduce(diff_sqr, d, 1, cv::REDUCE_SUM, -1);
        double d_init = d.at<double>(0,0);

        for(int i=1; i<32; i++){
          cv::add(y_thread[i][1].row(r) * 2, coset_rep.row(i), y_thread[i][1].row(r));
          cv::absdiff(y_thread[i][1].row(r), y.row(r), diff_sqr); 
          cv::pow(diff_sqr, 2, diff_sqr);
          cv::reduce(diff_sqr, d, 1, cv::REDUCE_SUM, -1);
          if(d.at<double>(0,0) < d_init){
            d_init = d.at<double>(0,0);
            MinIdx = i;
          }
        }
        //cout<<"y_thread[MinIdx][1].row(r) is "<<y_thread[MinIdx][1].row(r)<<endl;
        y_thread[MinIdx][1].row(r).copyTo(ret.row(r));
      }
    });
    return ret;
    
  }


  /*return the closet corrodiates of y*, return value is in double*/
  cv::Mat CPA_E8(const cv:: Mat& y){
    cv::Mat y0[2]; 
    y0[0] = y;
    CPA_DN(y0);
    y0[1].convertTo(y0[1],CV_64FC1);

    cv::Mat y1[2];
    y1[0] = y-0.5;
    CPA_DN(y1);
    y1[1].convertTo(y1[1],CV_64FC1);
    y1[1] = y1[1] + 0.5;

    cv::Mat abs_d0_sqr;
    cv::Mat abs_d1_sqr;
    cv::absdiff(y0[1], y, abs_d0_sqr);
    cv::absdiff(y1[1], y, abs_d1_sqr);
    cv::pow(abs_d0_sqr, 2, abs_d0_sqr);
    cv::pow(abs_d1_sqr, 2, abs_d1_sqr);
    cv::Mat d0;
    cv::Mat d1;
    cv::reduce(abs_d0_sqr, d0, 1, cv::REDUCE_SUM, -1);
    cv::reduce(abs_d1_sqr, d1, 1, cv::REDUCE_SUM, -1);

    cv::Mat ret(y.size(), CV_64FC1);
    d0.forEach<double>(
      [&](double& i_d0, const int* position)->void{
        if(i_d0<d1.at<double>(position[0],0))
          y0[1].row(position[0]).copyTo(ret.row(position[0]));
        else
          y1[1].row(position[0]).copyTo(ret.row(position[0]));
      }
    );
    return ret;
  }

  /*
  return the closet corrodiates of y[0]
  to work with pthread_create(), input arg is cv::Mat y[2].
  y[0] is random constellation corrodinate in double
  y[1] is the closet corrodiates of y[0] in int
  */
  static void* CPA_DN(void* threadarg){
    cv::Mat * y = (cv::Mat *) threadarg;
    y[1] = cv::Mat::zeros(y[0].size(), CV_32SC1);
    cv::Mat round_y_double(y[0].size(), CV_64FC1);
    cv::Mat round_y_int(y[0].size(), CV_32SC1);
    y[0].forEach<double>(
      [&](double& y_i, const int* position) -> void{
        round_y_int.at<int>(position[0], position[1]) = round(y_i);
      }
    );
    round_y_int.convertTo(round_y_double, CV_64FC1);
    cv::Mat diff = y[0]-round_y_double;
    cv::Mat abs_diff;
    abs_diff = cv::abs(diff);
    cv::Mat g_y = round_y_int.clone();

    /*
    cout<<round_y<<endl;
    cout<<diff<<endl;
    cout<<abs_diff<<endl;
    */
    cv::parallel_for_(cv::Range{0, abs_diff.rows}, MatRowParFor(abs_diff, [&](cv::Mat RowMat, int RowIdx) -> void{
      vector<double> RowInVec;
      MatToVector(RowMat, &RowInVec);
      int SelectedPosition[2];
      SelectedPosition[0] = RowIdx;
      SelectedPosition[1] = std::distance(RowInVec.begin(),std::max_element(RowInVec.begin(), RowInVec.end()));
      if(diff.at<double>(SelectedPosition[0],SelectedPosition[1]) > 0)
        g_y.at<int>(SelectedPosition[0],SelectedPosition[1]) += 1;
      else if(diff.at<double>(SelectedPosition[0],SelectedPosition[1]) < 0)
        g_y.at<int>(SelectedPosition[0],SelectedPosition[1]) -= 1;
      else{
        cout<<"(diff.at<double>(SelectedPosition[0],SelectedPosition[1]) == 0), existing"<<endl;
        exit(0);
      }
    }));
    cv::Mat sum_round_y;
    cv::Mat sum_g_y;     
    sum_round_y = cv::Mat::zeros (round_y_int.rows, 1, CV_32SC1);
    sum_g_y = cv::Mat::zeros (g_y.rows, 1, CV_32SC1);

    /*due to the limitation of cv::reduce, perform row-sum on round_y_int*/
    cv::parallel_for_(cv::Range{0,round_y_int.rows}, MatRowParFor(round_y_int, [&](cv::Mat RowMat, int RowIdx)->void{
      vector<int> RowInVec;
      VC::MatToVector(RowMat, &RowInVec);
      sum_round_y.at<int>(RowIdx, 0) = std::accumulate(RowInVec.begin(), RowInVec.end(), 0);
    }));
    //cv::reduce(round_y_int, sum_round_y, 1, cv::REDUCE_SUM, CV_32SC1);

    /*due to the limitation of cv::reduce, perform row-sum on round_y_int*/
    cv::parallel_for_(cv::Range{0,g_y.rows}, MatRowParFor(g_y, [&](cv::Mat RowMat, int RowIdx)->void{
      vector<int> RowInVec;
      VC::MatToVector(RowMat, &RowInVec);
      sum_g_y.at<int>(RowIdx, 0) = std::accumulate(RowInVec.begin(), RowInVec.end(), 0);
    }));
    //cv::reduce(g_y, sum_g_y, 1, cv::REDUCE_SUM, CV_32SC1);
    /*
    cout<<"round_y_int is"<<endl;
    cout<<round_y_int<<endl;
    cout<<"sum_round_y is"<<endl;
    cout<< sum_round_y<<endl;
    cout<<"g_y is"<<endl;
    cout<<g_y<<endl;
    cout<<"sum_g_y is"<<endl;
    cout<< sum_g_y<<endl;
    */
    sum_round_y.forEach<int>(
      [&](int& i_sum_round_y, const int* position) -> void{
        if(i_sum_round_y % 2 == 0)
          round_y_int.row(position[0]).copyTo(y[1].row(position[0]));
      }
    );
    sum_g_y.forEach<int>(
      [&](int& i_sum_g_y, const int* position) -> void{
        if(i_sum_g_y % 2 == 0)
          g_y.row(position[0]).copyTo(y[1].row(position[0]));
      }
    );
    return NULL;
  }
  

  /*define a class where parallel_for is used to iterate rows of a cv::Mat*/
  //template<typename T>
  class MatRowParFor : public cv::ParallelLoopBody {
  public:
    cv::Mat mat;
    std::function<void(cv::Mat, int)> LambdaFunc;
    MatRowParFor(const cv::Mat& mat, std::function<void(cv::Mat, int)> LambdaFunc) : mat(mat), LambdaFunc(LambdaFunc) { }
    void operator()(const cv::Range& range) const override {
      for (int i = range.start; i < range.end; i++) {
        LambdaFunc(mat.row(i), i);
      }
    }
  };

  template<typename T>
  static void MatToVector(const cv::Mat& mat, vector<T> *array){
    if (mat.isContinuous()) {
      // array.assign((float*)mat.datastart, (float*)mat.dataend); // <- has problems for sub-matrix like mat = big_mat.row(i)
      array->assign((T*)mat.data, (T*)mat.data + mat.total()*mat.channels());
    } else {
      for (int i = 0; i < mat.rows; ++i) {
        array->insert(array->end(), mat.ptr<T>(i), mat.ptr<T>(i)+mat.cols*mat.channels());
      }
    }
  }
private:
  double D2[2][2] = MACRO_D2;
  double D4[4][4] = MACRO_D4;
  double D8[8][8] = MACRO_D8;
  double D16[16][16] = MACRO_D16;
  double D24[24][24] = MACRO_D24;
  double D32[32][32] = MACRO_D32;

};