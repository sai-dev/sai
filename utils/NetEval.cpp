#include <array>
#include <iostream>

std::array<float, 15> means = {0.9533274235, 0.9287059913, 0.8929619049, 0.7879189113, 0.7221856119,
			       0.5671358464, 0.6534111856, 0.2734502325, 0.165120555, 0.7405820716,
			       0.3864972585, 0.340824635, 0.6591981697, 0.3688741877, 0.2167708809};

std::array<float, 15> pca_w = {0.04, 0.0616, 0.0926, 0.1797, 0.2244,
			       0.3818, 0.3166, 0.3733, 0.1388, 0.1859,
			       0.3066, 0.2164, 0.3151, 0.4145, 0.2288};


int main() {
    float val = 0.0f;
    
    for (auto i = 0; i<15 ; ++i) {
	float wins, losses;
	std::cin >> wins >> losses;
	auto wr = 0.5f;
	if (wins+losses>=1.0f) {
	    wr = wins/(wins+losses);
	}
	val += (wr-means[i])*pca_w[i];
    }

    std::cout << val << std::endl;
    
    return 0;
}
