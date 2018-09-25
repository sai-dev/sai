#include <array>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <string>

const std::array<float, 15> weightswht = {0.001, 0.001, 0.001, 0.03088321, 0.01211115,
					  0.37131672, 0.02297181, 0.40003006, 0.49160695, 0.01789272,
					  0.08280381, 0.06084046, 0.02789337, 0.2503669, 0.62004198};
    
const std::array<float, 15> weightsblk = {0.001, 0.00597199, 0.02775871, 0.10541281, 0.15726762,
					  0.37896146, 0.28796306, 0.48513024, 0.03057332, 0.05677957,
					  0.16795283, 0.12024328, 0.20015829, 0.60185867, 0.22877867};

const std::array<float, 15> weightsold = {0.04, 0.0616, 0.0926, 0.1797, 0.2244,
					  0.3818, 0.3166, 0.3733, 0.1388, 0.1859,
					  0.3066, 0.2164, 0.3151, 0.4145, 0.2288};

const float oldavg = 1.77021488;


class rv {
public:
    int n() const { return m_num; }
    float mean() const { return m_mean; }
    float var() const { return m_var; }
    float sd() const { return m_sd; }
    void update(int wins, int loss, float wheight);
    
private:
    int m_num{0};
    float m_mean{0.0f};
    float m_var{0.0f};
    float m_sd{0.0f};
};


void rv::update(int wins, int loss, float weight) {
    auto winrate = 0.5f;
    auto wr_var = 1.0f;

    const auto num = std::max( 0.5f, float(wins * loss) );
    const auto den = wins + loss;

    if (den >= 1) {
	winrate = float(wins)/den;
	wr_var = num / (den*den*den); // X(n-X)/n^3 = p(1-p)/n
    }
    
    m_num++;
    m_mean += winrate * weight;
    m_var += wr_var * weight * weight;
    m_sd = std::sqrt(m_var);
}

int main() {
    std::string buf;
    std::cin >> buf >> buf;
    
    rv newwht, newblk, sumwht, sumblk, oldpca;

    for (auto i = 0; i<15 ; ++i) {
	float winswht, losswht, winsblk, lossblk;
	std::cin >> winsblk >> winswht >> lossblk >> losswht;

	newwht.update(winswht, losswht, weightswht[i]);
	newblk.update(winsblk, lossblk, weightsblk[i]);
	sumwht.update(winswht, losswht, 1.0f);
	sumblk.update(winsblk, lossblk, 1.0f);
	oldpca.update(winswht+winsblk, losswht+lossblk, weightsold[i]);
    }

    std::cout << newwht.mean() << " " << newwht.sd() << " "
	      << newwht.mean()+newblk.mean() << " " << std::sqrt(newwht.var()+newblk.var()) << " "
	      << sumwht.mean() << " " << sumwht.sd() << " "
	      << sumwht.mean()+sumblk.mean() << " " << std::sqrt(sumwht.var()+sumblk.var()) << " "
	      << oldpca.mean()-oldavg << " " << oldpca.sd()
	      << std::endl;
    
    return 0;
}
