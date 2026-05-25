#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>

// structura de baza pentru o imagine grayscale
// pixelii sunt stocati ca un vector 1D, accesat prin row*width+col
struct Image {
    int width = 0, height = 0;
    std::vector<uint8_t> data;

    uint8_t& at(int row, int col) {
        return data[row * width + col];
    }
    const uint8_t& at(int row, int col) const {
        return data[row * width + col];
    }
};

// genereaza o imagine sintetica cu pattern sinusoidal + zgomot
// folosim seed diferit pentru fiecare imagine ca sa arate diferit
Image generateImage(int width, int height, int seed = 42) {
    Image img;
    img.width = width;
    img.height = height;
    img.data.resize(width * height);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> noise(0, 30);

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            float base = 128.0f + 80.0f * std::sin(r * 0.05f) * std::cos(c * 0.05f);
            float pattern = 40.0f * ((r / 20 + c / 20) % 2 == 0 ? 1.0f : -1.0f);
            float n = (float)noise(rng);
            int val = (int)(base + pattern + n);
            img.data[r * width + c] = (uint8_t)(val < 0 ? 0 : (val > 255 ? 255 : val));
        }
    }
    return img;
}

// calculeaza imaginea integrala (summed area table)
// ne permite sa calculam suma oricarei ferestre in O(1) in loc de O(n^2)
// formula clasica: integral[r][c] = val + sus + stanga - diagonal
std::vector<long long> computeIntegral(const Image& img) {
    int W = img.width, H = img.height;
    std::vector<long long> integral(W * H, 0LL);

    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            long long val = img.data[r * W + c];
            long long above = (r > 0) ? integral[(r - 1) * W + c] : 0;
            long long left = (c > 0) ? integral[r * W + (c - 1)] : 0;
            long long diag = (r > 0 && c > 0) ? integral[(r - 1) * W + (c - 1)] : 0;
            integral[r * W + c] = val + above + left - diag;
        }
    }
    return integral;
}

// adaptive thresholding secvential (algoritmul Bradley-Roth)
// fiecare pixel devine alb sau negru in functie de media vecinilor sai
// fereastra de comparatie are dimensiunea windowSize x windowSize
Image applyThreshold(const Image& src, int windowSize = 15, float T = 0.15f) {
    int W = src.width;
    int H = src.height;
    int half = windowSize / 2;

    auto integral = computeIntegral(src);

    Image dst;
    dst.width = W;
    dst.height = H;
    dst.data.resize(W * H);

    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            // coordonatele ferestrei cu clamp la marginile imaginii
            int r1 = std::max(0, r - half);
            int r2 = std::min(H - 1, r + half);
            int c1 = std::max(0, c - half);
            int c2 = std::min(W - 1, c + half);

            int count = (r2 - r1 + 1) * (c2 - c1 + 1);

            // suma pixelilor din fereastra folosind imaginea integrala
            long long s = integral[r2 * W + c2];
            if (r1 > 0)           s -= integral[(r1 - 1) * W + c2];
            if (c1 > 0)           s -= integral[r2 * W + (c1 - 1)];
            if (r1 > 0 && c1 > 0) s += integral[(r1 - 1) * W + (c1 - 1)];

            float mean = (float)s / count;
            dst.data[r * W + c] = (src.data[r * W + c] > mean * (1.0f - T)) ? 255 : 0;
        }
    }
    return dst;
}

// aceeasi functie ca mai sus, dar liniile imaginii sunt procesate in paralel
// impartim intervalul [0, H) in blocuri si fiecare thread lucreaza pe blocul sau
// imaginea integrala tot secvential se calculeaza, ea are dependinte intre linii
Image applyThresholdParallelInside(const Image& src, int windowSize = 15, float T = 0.15f) {
    int W = src.width;
    int H = src.height;
    int half = windowSize / 2;

    auto integral = computeIntegral(src);

    Image dst;
    dst.width = W;
    dst.height = H;
    dst.data.resize(W * H);

    // fiecare thread primeste un subset de linii si le proceseaza independent
    // nu exista conflict intre thread-uri pentru ca scriu in locatii diferite din dst
    tbb::parallel_for(tbb::blocked_range<int>(0, H),
        [&](const tbb::blocked_range<int>& range) {
            for (int r = range.begin(); r < range.end(); r++) {
                for (int c = 0; c < W; c++) {
                    int r1 = std::max(0, r - half);
                    int r2 = std::min(H - 1, r + half);
                    int c1 = std::max(0, c - half);
                    int c2 = std::min(W - 1, c + half);

                    int count = (r2 - r1 + 1) * (c2 - c1 + 1);

                    long long s = integral[r2 * W + c2];
                    if (r1 > 0)           s -= integral[(r1 - 1) * W + c2];
                    if (c1 > 0)           s -= integral[r2 * W + (c1 - 1)];
                    if (r1 > 0 && c1 > 0) s += integral[(r1 - 1) * W + (c1 - 1)];

                    float mean = (float)s / count;
                    dst.data[r * W + c] = (src.data[r * W + c] > mean * (1.0f - T)) ? 255 : 0;
                }
            }
        });

    return dst;
}

// varianta secventiala - procesam imaginile una dupa alta pe un singur thread
// e varianta de referinta fata de care calculam speedup-ul
std::vector<Image> processSequential(const std::vector<Image>& dataset) {
    std::vector<Image> results;
    results.reserve(dataset.size());
    for (const auto& img : dataset) {
        results.push_back(applyThreshold(img));
    }
    return results;
}

// paralelism intre imagini - fiecare thread ia un grup de imagini si le proceseaza
// imaginile sunt independente intre ele, deci nu avem probleme de sincronizare
// functioneaza bine cand avem multe imagini
std::vector<Image> processParallelBetween(const std::vector<Image>& dataset) {
    std::vector<Image> results(dataset.size());

    tbb::parallel_for(tbb::blocked_range<size_t>(0, dataset.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i < range.end(); i++) {
                results[i] = applyThreshold(dataset[i]);
            }
        });

    return results;
}

// paralelism in interiorul imaginii - imaginile se proceseaza una cate una
// dar liniile fiecarei imagini sunt distribuite intre thread-uri
// mai putin eficient decat cel de mai sus din cauza overhead-ului per imagine
std::vector<Image> processParallelInside(const std::vector<Image>& dataset) {
    std::vector<Image> results;
    results.reserve(dataset.size());
    for (const auto& img : dataset) {
        results.push_back(applyThresholdParallelInside(img));
    }
    return results;
}

// varianta mixta - paralelism la ambele niveluri simultan
// thread-urile proceseaza imagini diferite SI linii diferite din aceeasi imagine
// in practica poate fi mai lent din cauza supraincarcarii thread pool-ului TBB
std::vector<Image> processParallelMixed(const std::vector<Image>& dataset) {
    std::vector<Image> results(dataset.size());

    tbb::parallel_for(tbb::blocked_range<size_t>(0, dataset.size()),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i < range.end(); i++) {
                results[i] = applyThresholdParallelInside(dataset[i]);
            }
        });

    return results;
}

// ruleaza functia de mai multe ori si returneaza cel mai bun timp
// folosim minimul ca sa eliminam variatiile datorate sistemului de operare
template<typename Func>
double benchmark(Func f, int repetitions = 3) {
    double best = 1e18;
    for (int i = 0; i < repetitions; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        f();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        best = std::min(best, ms);
    }
    return best;
}

// afiseaza timpul, speedup-ul si throughput-ul pentru o strategie
void printResults(const std::string& name, double ms, double seqMs, int numImages) {
    double speedup = seqMs / ms;
    double throughput = numImages / (ms / 1000.0);
    std::cout << "\n[" << name << "]\n";
    std::cout << "  Timp:       " << ms << " ms\n";
    std::cout << "  Speedup:    " << speedup << "x\n";
    std::cout << "  Throughput: " << (int)throughput << " imagini/sec\n";
}

int main() {
    const int IMG_W = 512;
    const int IMG_H = 512;
    const int NUM_IMAGES = 100;

    // generam 100 de imagini sintetice cu seed-uri diferite
    std::vector<Image> dataset;
    dataset.reserve(NUM_IMAGES);
    for (int i = 0; i < NUM_IMAGES; i++) {
        dataset.push_back(generateImage(IMG_W, IMG_H, i * 17 + 42));
    }

    std::cout << "Dataset: " << NUM_IMAGES << " imagini " << IMG_W << "x" << IMG_H << "\n";
    std::cout << "=========================================\n";

    // rulam fiecare strategie si masuram timpul
    double seqMs = benchmark([&]() { processSequential(dataset); });
    printResults("SECVENTIAL", seqMs, seqMs, NUM_IMAGES);

    double parBetweenMs = benchmark([&]() { processParallelBetween(dataset); });
    printResults("PARALEL INTRE IMAGINI", parBetweenMs, seqMs, NUM_IMAGES);

    double parInsideMs = benchmark([&]() { processParallelInside(dataset); });
    printResults("PARALEL IN INTERIORUL IMAGINII", parInsideMs, seqMs, NUM_IMAGES);

    double parMixedMs = benchmark([&]() { processParallelMixed(dataset); });
    printResults("PARALEL MIXT (intre imagini + interior)", parMixedMs, seqMs, NUM_IMAGES);

    // testam cum se comporta algoritmul cu 1/2/4/8/16 thread-uri
    // ne asteaptam ca speedup-ul sa creasca pana la numarul de nuclee fizice
    // dupa care eficienta scade pentru ca thread-urile se bat pe resurse
    std::cout << "\n=========================================\n";
    std::cout << "Analiza la diferite numere de thread-uri (paralel intre imagini):\n";
    std::cout << "Threads | Timp(ms) | Speedup | Throughput(img/sec) | Eficienta\n";
    std::cout << "--------|----------|---------|---------------------|----------\n";

    for (int nThreads : {1, 2, 4, 8, 16}) {
        // limitam TBB sa foloseasca exact nThreads thread-uri
        tbb::global_control gc(tbb::global_control::max_allowed_parallelism, nThreads);
        double ms = benchmark([&]() { processParallelBetween(dataset); });
        double speedup = seqMs / ms;
        double throughput = NUM_IMAGES / (ms / 1000.0);
        // eficienta = cat % din capacitatea teoretica folosim
        double efficiency = speedup / nThreads * 100.0;
        std::cout << "  " << nThreads
            << "     | " << (int)ms
            << "       | " << speedup
            << "x      | " << (int)throughput
            << "                  | " << efficiency << "%\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}