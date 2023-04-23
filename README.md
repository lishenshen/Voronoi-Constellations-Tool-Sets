# Voronoi-Constellations-Tool-Sets

This repo provides tool-sets for Voronoi Constellations (VC) in digital communications, with intent to provide a license-free and effective way to try VC in simulation.

# Introduction to VC
Check the following materials
- placeholder1
- placeholder2
# Build this repo
The code responsible for generating VC is written in C++ based on OpenCV framework, which is mainly for the sake of efficiency. For demostration, figures are plotted in Python environment. To enable passing data from C++ to Python, utils are borrowed from [util.h](https://raw.githubusercontent.com/HaogeL/CommunicationOnOpenCV/main/util.h) and [PyUtils.py](https://raw.githubusercontent.com/HaogeL/CommunicationOnOpenCV/main/Python/PyUtils.py), which stores OpenCV matrices in binary files and also keeps precision of floating numbers.\
Install OpenCV first, refer to https://opencv.org/releases/ \
Steps to build this repo are shown below (in Linux for example)
```bash
git clone https://github.com/lishenshen/Voronoi-Constellations-Tool-Sets.git
cd Voronoi-Constellations-Tool-Sets
wget https://raw.githubusercontent.com/HaogeL/CommunicationOnOpenCV/main/util.h
cd Python
wget https://raw.githubusercontent.com/HaogeL/CommunicationOnOpenCV/main/Python/PyUtils.py
cd .. && mkdir build && cd build
cmake ..
cmake --build .
./VC
```
# Generate VC
VC class constructor: 
```cpp
VC(unsigned int _d, unsigned int _r, unsigned long _Ns, int _nofthreadused)
//_d : number of dimensions
//_r : scalling factor
//_Ns: number of symbols to be generated
//_nofthreadused: number of threads to be used
```
VC class public members:

| Member | Description | Format|
| --------- | --------- |---------|
| `Gs` | Generator matrix |cv::Mat CV_64FC1|
| `ShiftVector` | Shift vector |cv::Mat CV_64FC1|
| `d` | Dimentions |unsigned int|
| `r` | Scalling factor |unsigned int|
| `M` | constellation size |double|
| `BitsArray` | Transmit bits |cv::Mat CV_U8C1|
| `symbol` | Transmit symbols |cv::Mat CV_64FC1|
| `Es` | symbol power |double|

## Example: use 7 threads to generate 1000 symbols in a 2D VC with scalling-factor of 3
```cpp
int main(int argc, char* argv[]){
  VC myVC(2, 3, 10000, 7);
  cout<<"M: "<<myVC.M<<endl;
  cout<<"Symbols: "<<myVC.symbol<<endl;
  cout<<"Es: "<<myVC.Es<<endl;
  matwrite("myVC_symbol.bin", myVC.symbol);
  return 0;
}
```
plot the 1000 symbols in [VCplot.ipynb](https://github.com/lishenshen/Voronoi-Constellations-Tool-Sets/blob/main/Python/VCplot.ipynb)
```python
import numpy as np
import PyUtils
import matplotlib.pyplot as plt
#load VC symbols
VCsymbols = PyUtils.GetMatFromBin("../build/myVC_symbol.bin")
#plot
plt.scatter(VCsymbols[:,0,0],VCsymbols[:,1,0])
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.grid()
```
![](https://github.com/lishenshen/Voronoi-Constellations-Tool-Sets/blob/main/VSPlotExampleForREADME.png)
