#include <iostream>
#include <string>
#include <array>

int main() {
    std::array<std::string, 15> panelhash = {
	"2991c83a",
	"bc627734",
	"bfb90b36",
	"2c5e522d",
	"d9c5d4fb",
	"1272232d",
	"3b0df457",
	"ad4f55da",
	"3c6c1c4b",
	"739dff8e",
	"ff16cc71",
	"83f30a3e",
	"344fb61f",
	"afd1c7e9",
	"12537226"
    };
    //    std::array<std::array<int, 4>, 15> result;

    for (size_t i=0 ; i<15 ; ++i) {
	std::string hash;
	std::cin >> hash;

	if (hash!=panelhash[i]) {
	    std::cerr << "Expected panel hash " << panelhash[i]
		      << ", observed" << hash << std::endl;
	    return 1;
	}

	std::array<int, 4> result;
	int games=0;

	for (size_t j=0 ; j<4 ; ++j) {
	    std::cin >> result[j];
	    games += result[j];
	}

	if (games < 100) {
	    std::cerr << "Panel hash " << panelhash[i]
		      << " has only " << games
		      << ", but at least 100 were expected." << std::endl;
	    return 1;
	}
	for (size_t j=0 ; j<4 ; ++j) {
	    std::cout << result[j];
	    if (i<14 || j<3) {
		std::cout << " ";
	    }
	}
    }
    std::cout << std::endl;

    return 0;
}
