#include <iostream>
#include <string>
#include <cmath>

#define MULT 20  // 40
#define COEFF 1.0f
#define RESERVE 0.3f
#define OFFICIAL_KOMI 9.5f
#define SHIFT 0.5f

int main (int argc, char* argv[]) {

    unsigned int totalgames = 5120;
    
    if(argc >= 2) {
	totalgames = std::stoi(argv[1]);
    }

    
    
    std::string tmp;
    std::cin >> tmp;
    const auto alpha = std::stof(tmp);

    std::cin >> tmp;
    const auto beta = std::stof(tmp);

    //    std::cout << "Read parameters: alpha=" << alpha
    //	      << ", beta=" << beta << std::endl
    //	      << "Sample size " << N
    //	      << ", coefficient " << COEFF << std::endl;

    const unsigned int N = std::lround(totalgames/MULT);
    const unsigned int n = std::lround(N*(1.0f-RESERVE));
    
    const float lowest = 0.5f + std::floor(alpha - std::log(2*n-1)
					   / beta / COEFF);

    const float highest = 0.5f + std::floor(alpha + std::log(2*n-1)
					   / beta / COEFF);

    //    std::cout << "Komi values between " << lowest
    //	      << " and " << highest << std::endl;

    for (auto komi = lowest ; komi < highest+0.5f ; komi+=1.0f) {
	const auto games = std::lround(n/(1.0f+std::exp(-beta*COEFF*(std::ceil(komi)-SHIFT-alpha))))
	    - std::lround(n/(1.0f+std::exp(-beta*COEFF*(std::floor(komi)-SHIFT-alpha))))
	    + (komi==OFFICIAL_KOMI ? N-n : 0);
	if (games>0) {
	    std::cout << "curl -F number_to_play=" << games * MULT
		      << " -F komi=" << komi
		      << " -F other_options=\"${SAI_OTHEROPTIONS}\" $KOMIS_CMD_STRING"
		      << std::endl;
	}
    }
    
    return 0;
}
