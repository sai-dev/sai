/*
  This program counts the (approximate) number of nodes created for
  all the games in a chunk.

  Usage:
  gunzip * -c | nodecount
*/

#include <iostream>
#include <string>
#include <cmath>
#include <cassert>
#include <vector>
#include <algorithm>

//#define VISITS 160
#define GOBAN_SIZE 7

struct one_komi_stats {
    float komi;
    unsigned int bwg;
    unsigned int bwm;
    unsigned int wwg;
    unsigned int wwm;
    unsigned int mvs;
};

bool compare(const one_komi_stats& a, const one_komi_stats& b) {
    return a.komi<b.komi;    
}


unsigned int komi_index(float komi, std::vector<one_komi_stats> &stats) {
    unsigned int j;
    
    for (j=0 ; j<stats.size() ; ++j) {
	if (stats[j].komi == komi)
	    break;
    }
    if (j == stats.size()) {
	one_komi_stats p;
	p.komi = komi;
	p.bwg = 0;
	p.bwm = 0;
	p.wwg = 0;
	p.wwm = 0;
	p.mvs = 0;
	stats.push_back(p);
	std::cout << "Found new komi (" << j
		  << "): " << komi << std::endl;
    }
    return j;
}


int main(){
    std::string buf;
    std::string hexnil((GOBAN_SIZE*GOBAN_SIZE+3)/4,'0');
    int games=0, line=0, moves=0, lastwinner=0, winner=0;
    float komi;
    unsigned int j = 0;
    std::vector<one_komi_stats> stats;
    
    while (std::cin >> buf) {
	++line;
      
	// check whether the goban is empty
	int chknil = buf.compare(hexnil); // 0 if goban empty (1/16)
	for (int i=0; i<15; i++) {
	    
	    // skip the 16 lines describing the position
	    std::cin >> buf;
	    ++line;
	    
	    // check whether this is the starting position, otherwise 1 
	    if (!chknil && buf.compare(hexnil))
		chknil = 1;
	}
	
	// skip line 17, with the player and komi
	int stm;
	std::cin >> stm;
	std::cin >> komi;
	++line;

	j = komi_index(komi, stats);
	++stats[j].mvs;
	
	for (int i=0; i<GOBAN_SIZE*GOBAN_SIZE+1; i++) {
	    float polprb;
	    
	    // line 18 has the 50 probabilities, all of which are fractions
	    // like k/99
	    std::cin >> polprb;
	    
	    //	    assert (0 == ((int)round(1000*polprb*(VISITS-1)) % 1000));
	}
	++line;
	
	// skip line 19, with the winner
	std::cin >> winner;
	++line;
	++moves;
	
	// if starting position, the former maxnodes should have never
	// been subtracted, because after the last position there is no
	// more tree re-use
	if (!chknil && stm == 0) {
	    assert (winner == 0 || winner == 1 || winner == -1);
	    ++games;
	    if (lastwinner == 1) {
		++stats[j].bwg;
		stats[j].bwm += moves;
	    }
	    else if (lastwinner == -1) {
		++stats[j].wwg;
		stats[j].wwm += moves;
	    }
	    moves = 0;
	    lastwinner = winner;
	}
	
    }
    if (lastwinner == 1) {
	++stats[j].bwg;
	stats[j].bwm += moves;
    }
    else if (lastwinner == -1) {
	++stats[j].wwg;
	stats[j].wwm += moves;
    }
    
    sort(stats.begin(), stats.end(), compare);
    
    std::cout << "Total games found: " << games << std::endl;
    for (unsigned int j=0 ; j<stats.size() ; ++j) {
	const auto den = stats[j].bwg+stats[j].wwg;
	std::cout << j << ". komi " << stats[j].komi
		  << ", games: " << den
		  << ", moves: " << stats[j].mvs
		  << ", black wins: " << stats[j].bwg/(float)den
		  << " (avg len) " << stats[j].bwm/(float)stats[j].bwg
		  << ", white wins: " << stats[j].wwg/(float)den
		  << " (avg len) " << stats[j].wwm/(float)stats[j].wwg
		  << std::endl;
    }
    
    return 0;
}
