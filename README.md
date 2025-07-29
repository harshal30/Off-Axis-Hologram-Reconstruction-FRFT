# Off-Axis-Hologram-Reconstruction-FRFT

**Digital Hologram Reconstruction Algorithm**
This repository presents a digital hologram reconstruction algorithm based on the Fractional Fourier Transform (FRFT), specifically tailored for non-telecentric digital holographic microscopy. The optimal fractional order representing the recorded hologram is estimated based on an evaluation metric. The FRFT-based hologram reconstruction enables noise-robust amplitude and phase imaging with enhanced resolution. The project includes implementations in both Python and MATLAB. 


**Structure of the Repository**

Off-Axis-Hologram-Reconstruction-FRFT/
├── python/                 # Python implementation of the algorithm
│   ├── src/
│   │   ├── FRFT_Algorithm/  # Core FRFT computation modules
│   │   ├── Reconstruction/   # Hologram reconstruction scripts
│   │   └── utils/           # Utility functions for data processing
├── matlab/                 # MATLAB implementation of the algorithm
│   ├── src/
│   │   ├── FRFT_Algorithm/  # MATLAB FRFT computation modules
│   │   ├── Reconstruction/   # MATLAB hologram reconstruction scripts
│   │   └── utils/           # MATLAB utility functions
├── data/                   # Sample data and test cases
└── README.md               # This main project README

