/*
  This program counts the (approximate) number of nodes created for
  all the games in a chunk.

  Usage:
  gunzip * -c | nodecount
*/

#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>
#include <vector>

#define VISITS 250
#define GOBAN_SIZE 7
#define LOW_THR 0.0005f
#define LRG_N 3000

const std::vector<int> default_visits={250,160,100,60,40};

unsigned long int gcd_denum(std::vector<float> q) {
    std::sort(q.begin(),q.end());

    unsigned long int den=1, m=0;
    for (auto &p : q) {
	if (p == 0.0f)
	    continue;
	for ( m=1 ; 0 != ((int)round(LRG_N*p*den*m) % LRG_N) ; m++);
	den *= m;
    }
    // if(den > 249) {
    // 	std::cout << "Denumerator " << den << std::endl;
    // 	for (auto &p : q) {
    // 	    std::cout << p << " ";
    // 	}
    // 	std::cout << std::endl;
    // }
    
    return den;
}


unsigned long int gcd_denum_bad(const std::vector<float> &x) {
    std::vector<float> q,dq;

    q = x;
    while (q.size()>1) {
	q.emplace_back(0.0f);
	std::sort(q.begin(),q.end());
	for (size_t i=0 ; i+1<q.size(); ++i) {
	    auto diff = q[i+1]-q[i];
	    if (diff > LOW_THR) {
		dq.emplace_back(diff);
	    }
	}
	q = dq;
	dq.clear();
    }

    auto y = 1.0f/q[0];
    auto denum = (unsigned long int)round(y);

    if (denum==250) {
	std::cout << "Denum rounded to " << denum
		  << " force changed to 249" << std::endl;
	denum--;
    }
    else if (denum==83) {
	std::cout << "Denum rounded to " << denum
		  << " force changed to 249" << std::endl;
	denum*=3;
	y*=3;
    }
    
    
    //    std::cout << "Denumerator " << denum << std::endl;
    if (std::abs(y-denum) > 0.1f) {
	std::cout << "Warning: denumerator " << denum
		  << " obtained from rough rounding of " << y
		  << std::endl;
    }

    for (auto &p : x) {
	if (0 != ((int)round(1000*p*denum) % 1000)) {
	    std::cout << "Warning: p=" << p
		      << " denum=" << denum
		      << " p*denum=" << p*denum
		      << std::endl;
	}
    }

    return denum;
}


struct freq_it {
    unsigned long int ind;
    size_t c;
};

void add_freq(unsigned long int denum, std::vector<freq_it> &freq) {
    size_t i;
    
    for (i=0 ; i<freq.size() ; ++i) {
	if (freq[i].ind==denum)
	    break;
    }
    if (i>=freq.size()) {
      	//std::cout << "Found new denominator: " << denum << std::endl;
	freq_it it;
	it.ind = denum;
	it.c = 1;
	freq.emplace_back(it);
    } else {
	++freq[i].c;
    }
}


int main(int argc, char* argv[]){
  std::string buf;
  std::string hexnil((GOBAN_SIZE*GOBAN_SIZE+3)/4,'0');
  std::vector<freq_it> freq;
  unsigned long int cfg_visits = VISITS;
  
  if(argc >= 2) {
      cfg_visits = std::stoi(argv[1]);
  }
  
  unsigned long int denum=0;
  
  // cumulative number of nodes created
  int nodessum = 0;
  
  // the single move with maximum number of nodes is (typically)
  // chosen: these nodes are shared in the analysis of the next move,
  // and hence not computed again
  int maxnodes = 0;
  
  while (std::cin >> buf) {

    // check whether the goban is empty
    int chknil = buf.compare(hexnil); // 0 if goban empty (1/16)
    for (int i=0; i<15; i++) {

      // skip the 16 lines describing the position
      std::cin >> buf;

    // check whether this is the starting position, otherwise 1 
      if (!chknil && buf.compare(hexnil))
	chknil = 1;
    }

    // if starting position, the former maxnodes should have never
    // been subtracted, because after the last position there is no
    // more tree re-use
    if (!chknil)
      nodessum += maxnodes;

    // skip line 17, with the player and komi
    std::getline(std::cin, buf);
    std::getline(std::cin, buf);
    //    std::cout << "-" << buf << "-" << std::endl;
    //    std::cin >> buf ;
    //    std::cout << "." << buf << "." << std::endl;
    
    // maximum probability of a move
    float maxprb=0;
    
    std::vector<float> polprb;
    for (int i=0; i<GOBAN_SIZE*GOBAN_SIZE+1; i++) {

      float prob;
	
      // line 18 has the 50 probabilities, all of which are fractions
      // like k/99
      std::cin >> prob;
      polprb.emplace_back(prob);

      // the one with the maximum probability will be typically chosen
      if (prob > maxprb)
	maxprb = prob;
    }

    //    std::cout << polprb.size() << " probabilities read." << std::endl;
    denum = gcd_denum(polprb);
    add_freq(denum, freq);
    if (denum != cfg_visits-1) {
	auto i = default_visits.size();
	for ( ; i>0 ; --i) {
	    const auto visits = default_visits[i-1];
	    if ((visits-1) % denum == 0) {
		denum = visits-1;
		break;
	    }
	}
	if (i==0) {
	    std::cerr << "Possibly wrong denominator " << denum << std::endl;
	}
    }

    nodessum += denum+1;

    // this number of nodes will be shared with the next position
    maxnodes = round(maxprb*denum);

    // so subtract it from the total
    nodessum -= maxnodes;

    // skip line 19, with the winner
    std::cin >> buf;
    polprb.clear();
  }

  for (auto &j : freq) {
      std::cerr << "Denominator " << j.ind
		<< " frequency " << j.c << std::endl;
  }
  
  std::cout << nodessum << std::endl;
  
  return 0;
}
