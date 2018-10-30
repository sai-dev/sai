// This simple program generates a v1 weights file
// for Leela Zero and sends it to stdout.
// The parameter VHEADS states the number of value heads
// e.g. 1 for LZ and 2 for SAI (double head type W)

#include <iostream>
#include <random>
#include <cmath>

#define PLANES 17        // was 18
#define LAYERS 2
#define FILTERS 64
#define GOBAN_AREA 81    // now on 9x9
#define MOVES 82
#define VHEADFILTERS 64
#define VHEADS 2
 
int weights_line (int input, int output) // a line of n weights
 {
   std::random_device rd;
   std::mt19937 gen{rd()};
   std::normal_distribution<float> d{0,1};

   std::cout << std::endl;

   const int n=input*output;
   
   // standard deviation is as prescribed by Xavier initialization
   const float s=std::sqrt(2.0f/(input+output));
   
   for (int i = 0; i<n ; i++)
     {
       float x;

       // the weights are distributed as a truncated normal random
       // variable
       while ((std::abs(x = d(gen)))>2.0f);
       if (i == 0)
	   std::cout << x*s;
       else
	   std::cout << " " << x*s;
     }
   
   return 0;
 }

int const_line (int n, float x)
 {
     std::cout << std::endl
	       << x;
     for (int i = 1; i<n ; i++)
	 std::cout << " " << x;
     
     return 0;
 }


int main()
{
  // version is v1
    std::cout << "1";

  // first convolutional layer  
  weights_line(PLANES*9,FILTERS); // kernels
  const_line(FILTERS,0); // bias=0 because of batch normalization
  const_line(FILTERS,0); // BN mean
  const_line(FILTERS,1); // BN stddev

  // resconv layers -- each one is the same as two convolutional
  // layers
  for (int i=0 ; i<2*LAYERS ; i++)
    {
      weights_line(FILTERS*9,FILTERS);
      const_line(FILTERS,0);
      const_line(FILTERS,0);
      const_line(FILTERS,1);
    }

  // policy network head
  weights_line(FILTERS*1,2); // convolution
  const_line(2,0);
  const_line(2,0);
  const_line(2,1);
  weights_line(2*GOBAN_AREA,MOVES); // fully connected
  const_line(MOVES,0);

  // one or two value network heads
  for (int i=0 ; i<VHEADS ; i++)
    {
      weights_line(FILTERS*1,1); // convolution
      const_line(1,0);
      const_line(1,0);
      const_line(1,1);
      weights_line(GOBAN_AREA,VHEADFILTERS); // fully connected
      const_line(VHEADFILTERS,0);
      weights_line(VHEADFILTERS,1); // one last convolution
      const_line(1,0);
    }

  
  return 0;
}
