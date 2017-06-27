/*
https://github.com/BlueBrain/HighFive
clang++ -std=c++14 mincall/tools/align_ref.cpp -I ~/Desktop/HighFive/include -lhdf5 -o align_ref
 */

#include <iostream>
#undef H5_USE_BOOST
#define H5_USE_BOOST
// for perfomance production only #define BOOST_DISABLE_ASSERTS

#define EXTRA_ASSERT

#include <highfive/H5File.hpp>
#include <boost/multi_array.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <string>
#include <stdexcept>
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
const std::string ALIGNMENT_OUTPUT("/Analyses/MinCall/RefAlignment");
const int GAP_WIDTH=1000;

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
    if (refe.size() == 0) {
        throw std::runtime_error("Error while reading " + ref_file);
    }
    return refe;
}

bool test_valid_path(const multi_array<int, 1>& path, const multi_array<int, 1>& ref) {
    if (path[0] != ref[0]){
        return false;
    }

    int ref_pos = 0;
    for (int i = 0; i < path.size(); ++i) {
        if (path[i] != ref[ref_pos]) {
            if (path[i] != ref[++ref_pos]) {
                return false;
            }
        }
    }
    return ref_pos == ref.size() - 1;
}

auto calc_best_fit_path(const multi_array<int, 1>& ref, const multi_array<double, 2>& logits) {

    if (ref.size() > logits.shape()[0]) {
        throw std::invalid_argument("ref must be shorter than logits: (ref_size, logits_size) " + to_string(ref.size()) + " ! <=" + to_string(logits.shape()[0]));
    }

    multi_array<int, 1>   center(extents[ref.shape()[0]]); // For knowing center line
    for (int i = 0; i < ref.shape()[0]; ++i){
        center[i] = i * double(logits.shape()[0] - 1) / (ref.shape()[0] - 1);
    }

    multi_array<double, 2> sol(extents[ref.shape()[0]][2*GAP_WIDTH + 5]);
    multi_array<int, 2>   back(extents[ref.shape()[0]][2*GAP_WIDTH + 5]); // For backtracing dp matrix
    auto proxy = [&center](multi_array<double, 2>&arr, int i, int j)->double& {return arr[i][j - center[i] + GAP_WIDTH];};

    auto proxyI = [&center](multi_array<int, 2>&arr, int i, int j)->int& {return arr[i][j - center[i] + GAP_WIDTH];};

    auto is_valid = [&](int i, int j) {
        if (i < 0 || j < 0)
            throw std::runtime_error("indices must be non-negative");
        if (i >= ref.shape()[0]) {
            throw std::runtime_error("i > ref.shape()[0]");
        }
        if (j >= logits.shape()[0]) {
            throw std::runtime_error("j > logits.shape()[0]");
        }
        return i <= j && center[i] - GAP_WIDTH <= j && j <= center[i] + GAP_WIDTH;
    };

    fill(sol.data(), sol.data() + sol.size(), 1e-8);
    proxy(sol, 0, 0) = logits[0][ref[0]];
    proxyI(back, 0, 0) = -1;

    for (int i = 1; i < ref.shape()[0]; ++i){
        // Enforcing i <= j, since we must always move by one on reference
        // That is we disallow transition where we stay in same position on logits and move on reference. In other words each logit member has one and only one alignemtn on reference in dp table.
        const int lower_log_pos = max(i, center[i] - GAP_WIDTH);
        const int upper_log_pos = min((int) logits.shape()[0] - 1, center[i] + GAP_WIDTH);
#ifdef EXTRA_ASSERT
        if (lower_log_pos > upper_log_pos) {
            throw std::runtime_error("lower_log_pos must less than upper_log_pos");
        }

#endif
        for (int j = lower_log_pos;
                 j <= upper_log_pos;
            ++j) {

            if (!is_valid(i, j)) {
                throw std::runtime_error("Field not valid..This should be called");
            }

            bool ok = false;

            // case we move on both logits and reference
            if (is_valid(i - 1,j - 1)) {
                proxy(sol, i, j) = proxy(sol, i - 1, j - 1) + logits[j][ref[i]];
                proxyI(back, i, j) = i - 1;
                ok = true;
            }


            // case We move on logits, but not on reference
            if (is_valid(i, j - 1)) {
                auto same = proxy(sol, i, j - 1) +
                log(
                    exp(logits[j][ref[i]]) +
                    exp(logits[j][8])
                );
                // logits[j][ref[i]];
                // 8 is blank label.. This should properly add two probabilites together, I hope, without needing to renormalize.
                // It's speed improvement without additional factoring

                ok = true;
                if (same > proxy(sol, i, j)) {
                    proxy(sol, i, j)  = same;
                    proxyI(back, i, j) = i;
                }
            }
            if (!ok) {
                throw std::runtime_error("no field to propagate from. This shouldn't be called");
            }
            // cout << " " << proxyI(back, i, j) << ", " << j - 1 << endl;
        }
    }

    boost::multi_array<int, 1> path(extents[logits.shape()[0]]);
    int ref_pos = ref.shape()[0] - 1;
    for (int log_pos = logits.shape()[0] - 1; log_pos >= 0; --log_pos) {
        #ifdef EXTRA_ASSERT
        if (!(is_valid(ref_pos, log_pos))) {
            throw std::runtime_error("Invalid ref pos in backtracing!!!");
        }
        #endif

        path[log_pos] = ref[ref_pos];
        // cout << ref_pos - proxyI(back, ref_pos, log_pos);
        ref_pos = proxyI(back, ref_pos, log_pos);
        // std::cout << "(" << ref_pos << ", " << log_pos << ")" << endl;
    }

    if (!test_valid_path(path, ref)) {
        throw std::runtime_error("Invalid path found, internal error");
    }
    // cout << path << endl;
    return path;
}

void process(const std::string fast5_file) {
    size_t index = fast5_file.rfind(".fast5");
    if (index != fast5_file.size() - 6) {
        std::cerr << fast5_file << " not a fast5, ignorring" << std::endl;
        return;
    }
    std::string ref_file = fast5_file.substr(0, fast5_file.size() - 6) + ".ref";

    try {
        File file(fast5_file, File::ReadWrite);
        multi_array<double, 2> logits;
        DataSet dataset = file.getDataSet(DATASET_NAME);
        dataset.read(logits);
        auto ref = read_ref(ref_file);

        // std::cout << "Processing: " << fast5_file << std::endl;
        // std::cout << logits.shape()[0] << " " << logits.shape()[1] << std::endl;
        // std::cout << ref.size() << std::endl;

        auto path = calc_best_fit_path(ref, logits);
        if (file.exist(ALIGNMENT_OUTPUT)) {
            dataset = file.getDataSet(ALIGNMENT_OUTPUT);
        } else {
            dataset = file.createDataSet<int>(ALIGNMENT_OUTPUT, DataSpace::From(path));
        }
        dataset.write(path);

    } catch(std::exception& err){
        std::cerr << "Error during processing " << fast5_file << " :" << err.what() << std::endl;
    } catch(std::string& s) {
        cerr << "Error during processing " << fast5_file << s << endl;
    } catch(...) {
        cerr << "Error during processing " << fast5_file << "Unknown error happened" << endl;
    }
}

int main (int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        process(argv[i]);
    }
    return 0;  // successfully terminated
}
