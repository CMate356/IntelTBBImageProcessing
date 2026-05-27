# IntelTBBImageProcessing

Procesarea paralela a seturilor de imagini folosind Intel TBB (Threading Building Blocks).
Proiectul compara mai multe strategii de paralelizare aplicate pe algoritmul de adaptive thresholding (Bradley-Roth).

## Descrierea problemei

Adaptive thresholding binarizeaza o imagine: fiecare pixel devine alb sau negru in functie de media pixelilor din fereastra din jurul sau. Algoritmul foloseste o imagine integrala (Summed Area Table) pentru a calcula media in O(1) per pixel.

Am implementat si comparat 4 strategii:
- **Secvential** - referinta, o imagine dupa alta pe un singur thread
- **Paralel intre imagini** - fiecare thread proceseaza imagini diferite din dataset
- **Paralel in interiorul imaginii** - liniile fiecarei imagini sunt distribuite intre thread-uri
- **Paralel mixt** - paralelism la ambele niveluri simultan

## Structura proiectului

```
TBBImgProcessing/
    TBBImgProcessing.cpp   - codul sursa cu toate cele 4 strategii
    TBBImgProcessing.sln   - solutia Visual Studio
```

## Rezultate experimentale

Dataset: 100 imagini sintetice de 512x512 pixeli
Masuratori facute in Release x64, cele mai bune timpi din 3 rulari.

### Comparatie strategii

| Strategie | Timp (ms) | Speedup | Throughput (img/sec) |
|-----------|-----------|---------|----------------------|
| Secvential | 823 | 1x | 121 |
| Paralel intre imagini | 228 | 3.61x | 438 |
| Paralel in interiorul imaginii | 482 | 1.71x | 207 |
| Paralel mixt | 275 | 2.99x | 363 |

### Analiza la diferite numere de thread-uri (paralel intre imagini)

| Thread-uri | Timp (ms) | Speedup | Throughput (img/sec) | Eficienta |
|------------|-----------|---------|----------------------|-----------|
| 1 | 511 | 1.61x | 195 | 161% |
| 2 | 302 | 2.73x | 331 | 136% |
| 4 | 207 | 3.97x | 482 | 99% |
| 8 | 204 | 4.04x | 489 | 50% |
| 16 | 202 | 4.07x | 494 | 25% |

## Interpretarea rezultatelor

**Paralel intre imagini** este cea mai eficienta strategie cu un speedup de 3.61x.
Imaginile sunt independente intre ele, deci nu exista conflicte intre thread-uri si overhead-ul de sincronizare e minim.

**Paralel in interiorul imaginii** da un speedup mai mic (1.71x) pentru ca granularitatea task-urilor e mai mica.
Overhead-ul TBB de a sincroniza thread-urile pentru fiecare imagine in parte consuma o parte din castig.

**Paralel mixt** este al doilea ca performanta (2.99x) dar mai lent decat paralelismul simplu intre imagini.
Paralelismul nested supraincarca thread pool-ul TBB si introduce overhead suplimentar.

**Analiza pe thread-uri** arata ca speedup-ul creste pana la 4 thread-uri (~100% eficienta),
dupa care platoul indica ca procesorul are 4 nuclee fizice si thread-urile suplimentare
nu aduc beneficii semnificative.

## Dependinte

- Intel TBB (oneAPI Threading Building Blocks) v2022.3.0
- Visual Studio 2022
- C++17

## Cum se ruleaza

1. Instaleaza TBB prin vcpkg: `vcpkg install tbb:x64-windows`
2. Deschide `TBBImgProcessing.sln` in Visual Studio
3. Selecteaza configuratia **Release x64**
4. Build si Run (Ctrl+F5)
