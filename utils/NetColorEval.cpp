#include <array>
#include <iostream>
#include <string>

std::array<float, 15> weightswht = {0.001, 0.001, 0.001, 0.03088321, 0.01211115,
				    0.37131672, 0.02297181, 0.40003006, 0.49160695, 0.01789272,
				    0.08280381, 0.06084046, 0.02789337, 0.2503669, 0.62004198};
    
std::array<float, 15> weightsblk = {0.001, 0.00597199, 0.02775871, 0.10541281, 0.15726762,
				    0.37896146, 0.28796306, 0.48513024, 0.03057332, 0.05677957,
				    0.16795283, 0.12024328, 0.20015829, 0.60185867, 0.22877867};



int main(int argc, char* argv[]) {
    bool as_black_too=true;

    if(argc >= 2) {
	std::string arg = argv[1];
	if(0 == arg.compare(std::string{"-w"})) {
	    as_black_too=false;
	}
    }
    
    std::string buf;
    std::cin >> buf >> buf;
    
    float val = 0.0f;
    for (auto i = 0; i<15 ; ++i) {
	float winswht, losswht, winsblk, lossblk;
	std::cin >> winsblk >> winswht >> lossblk >> losswht;
	auto wrwht = 0.5f;
	if (winswht+losswht>=1.0f) {
	    wrwht = winswht/(winswht+losswht);
	}
	val += wrwht*weightswht[i];
	if (as_black_too) {
	    auto wrblk = 0.5f;
	    if (winsblk+lossblk>=1.0f) {
		wrblk = winsblk/(winsblk+lossblk);
	    }
	    val += wrblk*weightsblk[i];
	}
    }

    std::cout << val << std::endl;
    
    return 0;
}
