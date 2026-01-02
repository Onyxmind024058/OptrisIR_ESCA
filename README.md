# OptrisIR_ESCA

A Python-based graphical user interface (GUI) for **Optris infrared cameras**, developed for research and laboratory use at the **University of Basel**.

This application provides a workflow for thermal imaging, including live view, recording, ROI analysis, emissivity correction, scientific data export, and visualization tools.

## Installer

A compiled and compressed installer is available as OptrisIR_ESCA_setup.exe

## Camera and SDK

the SDK headers and library need to be download from: https://github.com/Optris/irdirectsdk_downloads/tree/main 
When download the structure is given below.

## License

This project is licensed under the **Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)** license.

Commercial use is not permitted without explicit permission from the author.


---

## ✨ Features

- 🔴 **Live thermal imaging**
- 📸 **Single-frame capture**
- 🎥 **Video recording & AVI export**
- 🔍 **Zoom / pan / cursor readout** (Matplotlib toolbar)
- 📐 **ROI selection** with live statistics
- 🌡️ **Emissivity & transmissivity correction**
- 🔬 **Planck-based emissivity estimation**
- 🎨 **Multiple color palettes** (Matplotlib colormaps)
- 📊 **Dynamic colorbar (auto / fixed scale)**
- 💾 **Export formats**:
  - CSV (RAW, temperature without emissivity, corrected temperature)
  - HDF5 datasets
  - Scientific TIFF (float32)
- 🌓 **Light / Dark mode**
- 🧭 **Menu bar** (File / Camera / View / Help)
- 📄 **Integrated user manual link**
- 📦 **Portable paths** (works across computers)
- 🪟 **Windows executable support (PyInstaller)**

---

## 🖥️ System Requirements

- **Windows 10 / 11**
- **Python 3.9+**
- Optris camera with **libirimager SDK**
- USB connection (Direct SDK mode)

---

## 📦 Python Dependencies

Install required packages with:

```bash
pip install PySide6 numpy h5py opencv-python matplotlib tifffile

IRCameraPython/
│
├── main.py
├── icon.ico
├── splash.png
├── README.md
├── LICENSE
│
└── sdk/
    ├── generic.xml
    ├── Formats.def
    ├── Califiles_SNxxxx/
    └── x64/
        └── libirimager.dll

```

## 📖 User Manual

A detailed user procedure is available here:

👉 IRCamera Imager UB – User Procedure
https://www.notion.so/paulhiret/IRCamera-Imager-UB-User-Procedure-2735aa0ff72f80b58024c86f0eb73271

You can also access it from within the app via:

Help → User manual

## 📄 Logging

Application logs are written to:

%APPDATA%\Optris IR GUI\logs\


(one log file per day)

Accessible from the menu:

Help → Open log folder

## 📦 Packaging as Windows Executable

The application can be bundled using PyInstaller.

Example (one-folder build):

pyinstaller --noconfirm --clean ^
  --name OptrisIR_GUI ^
  --windowed ^
  --icon icon.ico ^
  --add-data "sdk;sdk" ^
  --add-data "splash.png;." ^
  --add-data "icon.ico;." ^
  --add-binary "sdk\\x64\\libirimager.dll;sdk\\x64" ^
  main.py

## ⚖️ License

This project is licensed under the
Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0) license.

✔️ Free for academic & non-commercial use

❌ Commercial use requires explicit permission

See the LICENSE
 file for full terms.

## 👤 Author

Paul Hiret
University of Basel
2026

## 🧠 Disclaimer

This software is provided as-is for research purposes.
The author assumes no responsibility for incorrect measurements, misuse, or hardware damage.

