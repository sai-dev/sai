#include <iostream>
#include <string>
#include <cmath>

#define MULT 20  // 40 
#define COEFF 1.0f
#define RESERVE 0.0 // 0.3f
#define OFFICIAL_KOMI 9.0 // 9.5f
#define SHIFT 0.0 // 0.5f
constexpr double step=0.5;

double distrib (double x, double mean, double dev) {
    return 1.0/(1.0+std::exp(-(x-mean)/dev));
}

double inv_dist (double p, double mean, double dev) {
    return mean + dev * std::log(p/(1.0-p));
}

int main (int argc, char* argv[]) {

    unsigned int totalgames = 5120;
    
    if(argc >= 2) {
	totalgames = std::stoi(argv[1]);
    }

    
    
    std::string tmp;
    std::cin >> tmp;
    const auto alpha = std::stof(tmp);
    const auto mean = alpha + SHIFT;
    
    std::cin >> tmp;
    const auto beta = std::stof(tmp);
    const auto dev = 1.0/(beta*COEFF);

    //    std::cout << "Read parameters: alpha=" << alpha
    //	      << ", beta=" << beta << std::endl
    //	      << "Sample size " << N
    //	      << ", coefficient " << COEFF << std::endl;

    const unsigned int blocks = std::lround(totalgames*(1.0-RESERVE)/double(MULT));
    const unsigned int reserved_games = totalgames - blocks*MULT;

    const auto lowest_real  = inv_dist(0.5/blocks, mean, dev) - 0.5 * step;
    const auto highest_real = inv_dist(1.0 - 0.5/blocks, mean, dev) + 0.5 * step;
    const auto lowest_komi  = OFFICIAL_KOMI + step * std::ceil( (lowest_real - OFFICIAL_KOMI) / step );
    const auto highest_komi = OFFICIAL_KOMI + step * std::ceil( (highest_real - OFFICIAL_KOMI) / step - 1.0 );

    for (auto k = lowest_komi ; k < highest_komi + 0.5*step ; k += step) {
	const auto games = std::round(blocks * distrib(k + 0.5*step, mean, dev))
            -std::round(blocks * distrib(k - 0.5*step, mean, dev))
	    + (std::abs(k-OFFICIAL_KOMI) < 0.5*step ? reserved_games : 0);
	if (games>0) {
	    std::cout << "curl -F number_to_play=" << games * MULT
		      << " -F komi=" << k
		      << " -F other_options=\"${SAI_OTHEROPTIONS}\" $KOMIS_CMD_STRING"
		      << std::endl;
	}
    }
    
    return 0;
}
