# Off-Axis-Hologram-Reconstruction-FRFT

**Digital Hologram Reconstruction Algorithm**
This repository presents a digital hologram reconstruction algorithm based on the Fractional Fourier Transform (FRFT), specifically tailored for non-telecentric digital holographic microscopy. The optimal fractional order representing the recorded hologram is estimated based on an evaluation metric. The FRFT-based hologram reconstruction enables noise-robust amplitude and phase imaging with enhanced resolution. The project includes implementations in both Python and MATLAB. 


## Structure of the Repository
digital_hologram_reconstruction/
├── python/                 # Python implementation of the algorithm
│   ├── src/
│   │   ├── frft_algorithm/
│   │   ├── reconstruction/
│   │   └── utils/
│   ├── data/
│   ├── notebooks/
│   ├── requirements.txt
│   └── README.md
├── matlab/                 # MATLAB implementation of the algorithm
│   ├── src/
│   │   ├── frft_algorithm/
│   │   ├── reconstruction/
│   │   └── utils/
│   ├── data/
│   ├── examples/
│   └── README.md
├── .gitignore              # Specifies intentionally untracked files
├── README.md               # This main project README
└── LICENSE                 # Project license

