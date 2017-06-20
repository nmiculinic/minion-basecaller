/*
https://github.com/BlueBrain/HighFive
clang++ -std=c++11 mincall/tools/align_ref.cpp -I ~/Desktop/HighFive/include -lhdf5 -o align_ref
 */

#include <iostream>
#undef H5_USE_BOOST
#define H5_USE_BOOST
// for perfomance production only #define BOOST_DISABLE_ASSERTS
#include <highfive/H5File.hpp>
#include <boost/multi_array.hpp>
#include <boost/algorithm/string.hpp>
#include <string>
#include <fstream>
#include <cctype>
#include <map>

using namespace HighFive;
using namespace boost;
using namespace std;

template<typename T>
std::ostream& operator<<(std::ostream& stream,
                     const multi_array<T, 1>& vec) {
    stream << "[" << vec.shape()[0] <<"]";
    for (int i = 0; i < vec.shape()[0]; ++i) {
        stream << vec[i] << ",";
        // if (i % 10 == 0)
            // stream << " ";
    }
    return stream << endl;
}

template<typename T>
std::ostream& operator<<(std::ostream& stream,
                     const multi_array<T, 2>& mat) {
    stream << "[" << mat.shape()[0] << ", " << mat.shape()[1] <<"]";
    for (int i = 0; i < mat.shape()[0]; ++i){
        for (int j = 0; j < mat.shape()[1]; ++j){
            stream << mat[i][j] << ",";
        }
        stream << endl;
    }
    return stream << endl;
}


const std::string DATASET_NAME("/Analyses/MinCall/Logits");

std::map<char, int> n2i {
    {'A', 0},
    {'G', 1},
    {'T', 2},
    {'C', 3},
};

multi_array<int, 1> read_ref(const std::string ref_file) {
    std::ifstream in_str(ref_file);
    std::string tmp;
    // Totally hacky way of getting 4th line.
    std::getline(in_str, tmp);
    std::getline(in_str, tmp);
    std::getline(in_str, tmp);
    std::getline(in_str, tmp);
    in_str.close();

    auto ref = to_upper_copy<std::string>(tmp);
    multi_array<int, 1> refe(extents[ref.size()]);
    for (int i = 0; i < ref.size(); ++i) {
        refe[i] = n2i[ref[i]];
        if (i > 0 && refe[i] == refe[i - 1])
            refe[i] += 4;
    }
    return refe;
}

void calc_transition_matrix(const multi_array<int, 1>& ref, const multi_array<double, 2>& logits) {
    multi_array<double, 2> sol(extents[ref.shape()[0]][logits.shape()[0]]);
    multi_array<int, 2>   back(extents[ref.shape()[0]][logits.shape()[0]]); // For backtracing dp matrix

    sol[0][0] = logits[0][ref[0]];
    back[0][0] = -1;

    // We're at first place on reference, forbidden
    for(int j = 1; j < logits.shape()[0]; ++j) {
        sol[0][j] = -1e9;
        back[0][j] = -1;
    }

    for (int i = 1; i < ref.shape()[0]; ++i){
        // Enforcing i <= j, since we must always move by one on reference
        // That is we disallow transition where we stay in same position on logits and move on reference. In other words each logit member has one and only one alignemtn on reference in dp table.

        for (int j = i; j < logits.shape()[0]; ++j) {

            // case we move on both logits and reference
            auto mov = sol[i - 1][j - 1] + logits[j][ref[i]];
            sol[i][j] = mov;
            back[i][j] = i - 1;

            // case We move on logits, but not on reference
            if (i <= j - 1) {
                auto same = sol[i][j - 1] +
                log(
                    exp(logits[j][ref[i]]) +
                    exp(logits[j][8])
                );
                // 8 is blank label.. This should properly add two probabilites together, I hope, without needing to renormalize.
                if (same > sol[i][j]) {
                    sol[i][j] = same;
                    back[i][j] = i;
                }
            }
        }
    }

    boost::multi_array<int, 1> path(extents[logits.shape()[0]]);
    int ref_pos = ref.shape()[0] - 1;
    for (int log_pos = logits.shape()[0] - 1; log_pos >= 0; --log_pos) {
        path[log_pos] = ref[ref_pos];
        cout << ref_pos - back[ref_pos][log_pos];
        ref_pos = back[ref_pos][log_pos];
        // std::cout << "(" << log_pos << ", " << ref_pos << "), ";
    }
    cout << endl << path;
}

void process(const std::string fast5_file) {
    size_t index = fast5_file.rfind(".fast5");
    if (index != fast5_file.size() - 6) {
        std::cerr << fast5_file << " not a fast5, ignorring" << std::endl;
        return;
    }
    std::string ref_file = fast5_file.substr(0, fast5_file.size() - 6) + ".ref";
    std::cout << "Processing: " << fast5_file << std::endl;

    try {
        File file(fast5_file, File::ReadOnly);
        multi_array<double, 2> logits;
        DataSet dataset = file.getDataSet(DATASET_NAME);
        dataset.read(logits);
        auto ref = read_ref(ref_file);

        std::cout << logits.shape()[0] << " " << logits.shape()[1] << std::endl;
        std::cout << ref << std::endl;

        calc_transition_matrix(ref, logits);

    } catch(Exception & err){
        std::cerr << "Error during processing " << fast5_file << " :" << err.what() << std::endl;
    }
}

int main (int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        process(argv[i]);
    }
    return 0;  // successfully terminated
}
