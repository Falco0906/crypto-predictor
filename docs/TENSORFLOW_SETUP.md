# TensorFlow installation notes (platform specific)

This project ships core Python deps in `requirements.txt` but intentionally does NOT pin TensorFlow there because TensorFlow wheels are OS- and Python-version-specific. Follow the instructions below to install the correct TensorFlow package for your machine.

## Recommended general advice

- Use Python 3.10 for best compatibility (this repo was tested with 3.10).
- Create and activate a venv before installing packages.
- If you encounter binary incompatibility errors, reinstall TensorFlow that matches your Python minor version.

---

## Windows / Linux (x86_64) — CPU-only or GPU

- CPU-only (easy):

```powershell
# activate your venv first
pip install --upgrade pip
# install CPU TensorFlow (unified wheel on manylinux)
pip install tensorflow==2.10.0
```

- GPU (if you want to train on GPU): install matching CUDA 11.8 and cuDNN 8.6, then:

```powershell
# after installing CUDA & cuDNN and setting PATH
pip install tensorflow==2.10.0
```

If you don't want GPU complexity, prefer the CPU-only route above — the repo includes a pre-trained model so you can still run predictions.

---

## macOS (Intel / x86_64)

- Use the CPU-only path. TensorFlow mac wheels for older TF versions may not exist, so we recommend using the system CPU by installing the packages below (or using Python 3.10):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install a compatible TensorFlow build if available for your macOS + Python:
# If tensorflow==2.10 is not available, skip TF install and run on CPU without TF,
# or install a newer TF that your pip accepts. Alternatively, use the project with
# `tensorflow-cpu` if it is available for your platform.
pip install "numpy<2" --force-reinstall
```

If you need GPU on Apple Silicon, follow the Apple Silicon section below.

---

## macOS (Apple Silicon — M1 / M2)

Apple provides a Metal-backed TensorFlow build. Use `tensorflow-macos` and `tensorflow-metal`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install tensorflow-macos
pip install tensorflow-metal
# Then install the project deps (don't reinstall TensorFlow)
pip install -r requirements.txt --no-deps
```

Notes:
- `tensorflow-macos` and `tensorflow-metal` are Apple-specific and replace the standard `tensorflow` package.
- Performance and behavior may differ from NVIDIA GPU setups.

---

## Quick troubleshooting

- If `pip` fails with "No matching distribution found for tensorflow==2.10.0", the wheel isn't published for your OS/Python combo — either install a compatible TF build manually (see above) or run the project in CPU-only mode using the included pre-trained model.
- To run predictions without training you only need a working TensorFlow installation for inference. CPU-only installs are sufficient.

---

If you want, I can add small helper scripts (`scripts/setup_macos.sh` and `scripts/setup_linux.sh`) to automate these steps. Ask and I’ll add them.
