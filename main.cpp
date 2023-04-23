#include "Vconstellation.h"
#include "util.h"
#include <iostream>

using namespace std;
using namespace cv;

int main(int argc, char* argv[]){
  /*
  VC(dimensions, scalling-factor, number of symbols, number of threads to be used)
  
  dimensions: unsigned int, currently supports 2, 4, 8, 16, 24
  scalling-factor: unsigned int
  number of symbols: unsigned long
  number of threads: int 
  */
  VC myVC(2, 3, 10000, 7);
  cout<<"M: "<<myVC.M<<endl;
  cout<<"Symbols: "<<myVC.symbol<<endl;
  cout<<"Es: "<<myVC.Es<<endl;
  matwrite("myVC_symbol.bin", myVC.symbol);
  return 0;
}
