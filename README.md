# 1-lipschitz network

to make hyperparameter tuning easier, and explore different nn architectures.

## Environment

Recursively clone the repo, then create and activate conda virtual environment:
```bash
conda env create -f environment.yaml
conda activate 1Lipchitz
```

## Dataset generation
Dataset are generated from mesh or explicit [sdf](https://github.com/fogleman/sdf), or binary svg for 2D.

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

Generate 2d samples from svg

```bash
python gen_2d_dataset.py config/butterfly.yaml
```

or from a pre-defined fractal shape. For instance, Von Koch snowflacks with 3 orders of recursions:

```bash
python gen_2d_dataset.py config/koch.yaml
```

## Training

Choose model from SIREN, FFN, or SLL. SLL uses semi-supervised loss which only requires inside/outside labels.

```bash
python train.py config/butterfly.yaml
```

**Reference repo:**
* [awesome-implicit-representations](https://github.com/vsitzmann/awesome-implicit-representations)
* [IGR](https://github.com/amosgropp/IGR)
* [siren](https://github.com/vsitzmann/siren)
* [ffn](https://github.com/tancik/fourier-feature-networks)
* [wire](https://github.com/vishwa91/wire/)
* [nerual-tanents](https://github.com/google/neural-tangents)
* [1-Lipschitz neural distance field](https://github.com/GCoiffier/1-Lipschitz-Neural-Distance-Fields)
* [deel-torchlip](https://github.com/deel-ai/deel-torchlip/)
* [lipschitz_mlp](https://github.com/ml-for-gp/jaxgptoolbox/tree/main/demos/lipschitz_mlp)

