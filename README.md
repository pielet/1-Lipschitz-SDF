# 1-lipschitz network

to make hyperparameter tuning easier, and explore different nn architecture.

## Environments

Recursively clone the repo, then create and activate conda virtual environment:
```bash
conda env create -f environments.yaml
conda activate 1Lipchitz
```

## Dataset generation
Dataset are generated from mesh or explicit sdf

Install nanobind:

```bash
cd extern/nanobind
mkdir build && cd build
cmake .. \
  -DCMAKE_INSTALL_PREFIX=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
  -DPYTHON_EXECUTABLE=$(which python)
cmake --build . --target install
```

Install openvdb:

```bash
cd extern/openvdb
mkdir build && cd build
cmake -D OPENVDB_BUILD_PYTHON_MODULE=ON -D USE_NUMPY=ON -DCMAKE_PREFIX_PATH=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") ..
make -j8
cp openvdb/openvdb/python/openvdb.* `python -c 'import site; print(site.getsitepackages()[0])'`
```

**Reference repo:**
* [awesome-implicit-representations](https://github.com/vsitzmann/awesome-implicit-representations)
* [IGR](https://github.com/amosgropp/IGR)
* [siren](https://github.com/vsitzmann/siren)
* [ffn](https://github.com/tancik/fourier-feature-networks)
* [nerual-tanents](https://github.com/google/neural-tangents)
* [1-Lipschitz neural distance field](https://github.com/GCoiffier/1-Lipschitz-Neural-Distance-Fields)
* [deel-torchlip](https://github.com/deel-ai/deel-torchlip/)
* [lipschitz_mlp](https://github.com/ml-for-gp/jaxgptoolbox/tree/main/demos/lipschitz_mlp)

