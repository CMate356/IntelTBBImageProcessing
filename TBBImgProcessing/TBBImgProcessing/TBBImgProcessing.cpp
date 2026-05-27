#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>

// ---- Structura imagine ----
struct Image {
    int width=0, height=0;
    std::vector<uint8_t> data;

    uint8_t& at(int row, int col) {
        return data[row * width + col];
    }
    const uint8_t& at(int row, int col) const {
        return data[row * width + col];
    }
};

// ---- Generare imagine sintetica (inlocuieste dataset-ul) ----
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
           // img.data[r * width + c] = (uint8_t)std::clamp(val, 0, 255);
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

// ---- Procesare secventiala a unui dataset ----
std::vector<Image> processSequential(const std::vector<Image>& dataset) {
    std::vector<Image> results;
    results.reserve(dataset.size());
    for (const auto& img : dataset) {
        results.push_back(applyThreshold(img));
    }
    return results;
}

int main() {
    // Parametri
    const int IMG_W = 512;
    const int IMG_H = 512;
    const int NUM_IMAGES = 100;

    // Genereaza dataset
    std::vector<Image> dataset;
    for (int i = 0; i < NUM_IMAGES; i++) {
        dataset.push_back(generateImage(IMG_W, IMG_H, i * 17 + 42));
    }

    std::cout << "Dataset: " << NUM_IMAGES << " imagini "
        << IMG_W << "x" << IMG_H << std::endl;

    // Benchmark
    auto t0 = std::chrono::high_resolution_clock::now();

    auto results = processSequential(dataset);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Timp secvential: " << ms << " ms" << std::endl;
    std::cout << "Throughput: " << (int)(NUM_IMAGES / (ms / 1000.0))
        << " imagini/sec" << std::endl;

    return 0;
}