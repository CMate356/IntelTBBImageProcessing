#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>

// ---- Structura imagine ----
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

// ---- Generare imagine sintetica ----
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

// ---- Integral Image ----
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

// ---- Adaptive Thresholding (Bradley-Roth) ----
Image applyThreshold(const Image& src, int windowSize = 15, float T = 0.15f) {
    int W = src.width, H = src.height;
    int half = windowSize / 2;

    auto integral = computeIntegral(src);

    Image dst;
    dst.width = W;
    dst.height = H;
    dst.data.resize(W * H);

    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            int r1 = std::max(0, r - half);
            int r2 = std::min(H - 1, r + half);
            int c1 = std::max(0, c - half);
            int c2 = std::min(W - 1, c + half);

            int count = (r2 - r1 + 1) * (c2 - c1 + 1);

            long long s = integral[r2 * W + c2];
            if (r1 > 0) s -= integral[(r1 - 1) * W + c2];
            if (c1 > 0) s -= integral[r2 * W + (c1 - 1)];
            if (r1 > 0 && c1 > 0) s += integral[(r1 - 1) * W + (c1 - 1)];

            float mean = (float)s / count;
            dst.data[r * W + c] = (src.data[r * W + c] > mean * (1.0f - T)) ? 255 : 0;
        }
    }
    return dst;
}

// ---- Adaptive Thresholding PARALEL IN INTERIORUL IMAGINII ----
Image applyThresholdParallelInside(const Image& src, int windowSize = 15, float T = 0.15f) {
    int W = src.width, H = src.height;
    int half = windowSize / 2;

    auto integral = computeIntegral(src);

    Image dst;
    dst.width = W;
    dst.height = H;
    dst.data.resize(W * H);

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
                    if (r1 > 0) s -= integral[(r1 - 1) * W + c2];
                    if (c1 > 0) s -= integral[r2 * W + (c1 - 1)];
                    if (r1 > 0 && c1 > 0) s += integral[(r1 - 1) * W + (c1 - 1)];

                    float mean = (float)s / count;
                    dst.data[r * W + c] = (src.data[r * W + c] > mean * (1.0f - T)) ? 255 : 0;
                }
            }
        });

    return dst;
}

// ---- Procesare SECVENTIALA a dataset-ului ----
std::vector<Image> processSequential(const std::vector<Image>& dataset) {
    std::vector<Image> results;
    results.reserve(dataset.size());
    for (const auto& img : dataset) {
        results.push_back(applyThreshold(img));
    }
    return results;
}

// ---- Procesare PARALELA INTRE IMAGINI ----
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

// ---- Procesare PARALELA IN INTERIORUL IMAGINILOR ----
std::vector<Image> processParallelInside(const std::vector<Image>& dataset) {
    std::vector<Image> results;
    results.reserve(dataset.size());
    for (const auto& img : dataset) {
        results.push_back(applyThresholdParallelInside(img));
    }
    return results;
}

// ---- Benchmark helper ----
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

// ---- Afisare rezultate ----
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

    // Genereaza dataset
    std::vector<Image> dataset;
    dataset.reserve(NUM_IMAGES);
    for (int i = 0; i < NUM_IMAGES; i++) {
        dataset.push_back(generateImage(IMG_W, IMG_H, i * 17 + 42));
    }

    std::cout << "Dataset: " << NUM_IMAGES << " imagini " << IMG_W << "x" << IMG_H << "\n";
    std::cout << "=========================================\n";

    // 1. Secvential
    double seqMs = benchmark([&]() { processSequential(dataset); });
    printResults("SECVENTIAL", seqMs, seqMs, NUM_IMAGES);

    // 2. Paralel intre imagini
    double parBetweenMs = benchmark([&]() { processParallelBetween(dataset); });
    printResults("PARALEL INTRE IMAGINI", parBetweenMs, seqMs, NUM_IMAGES);

    // 3. Paralel in interiorul imaginilor
    double parInsideMs = benchmark([&]() { processParallelInside(dataset); });
    printResults("PARALEL IN INTERIORUL IMAGINII", parInsideMs, seqMs, NUM_IMAGES);

    // 4. Analiza la diferite numere de thread-uri
    std::cout << "\n=========================================\n";
    std::cout << "Analiza speedup la diferite thread-uri (paralel intre imagini):\n";
    std::cout << "Threads | Timp(ms) | Speedup | Throughput(img/sec) | Eficienta\n";
    std::cout << "--------|----------|---------|---------------------|----------\n";

    for (int nThreads : {1, 2, 4, 8, 16}) {
        tbb::global_control gc(tbb::global_control::max_allowed_parallelism, nThreads);
        double ms = benchmark([&]() { processParallelBetween(dataset); });
        double speedup = seqMs / ms;
        double throughput = NUM_IMAGES / (ms / 1000.0);
        double efficiency = speedup / nThreads * 100.0;
        std::cout << "  " << nThreads << "     | "
            << (int)ms << "       | "
            << speedup << "x      | "
            << (int)throughput << "                  | "
            << efficiency << "%\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}