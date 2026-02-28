# Demenstration of the code for the paper: *'Digital adiabatic evolution is universally accurate'*

This repository contains the Python source code for the numerical simulations presented in our paper: *'Digital adiabatic evolution is universally accurate'* (https://arxiv.org/abs/2510.12237), including the Trotterized and GQSP-based Adiabatic State Preparation (ASP) for molecular systems, as well as the adiabatic algorithm for the Quantum Linear System Problem (QLSP).

---

## 📜 File Descriptions

The repository is organized as follows:

In folder Code:
* `Trotter_mole.py`: Main script to generate data for the Trotterized Adiabatic State Preparation (ASP) for molecules. This corresponds to **Fig 2(a)** and **Fig 3(a-c)** in our paper.
* `GQSP_mole.py`: Main script to generate data for the Generalized Quantum Signal Processing (GQSP)-based ASP for molecules. This corresponds to **Fig 3(d)** in our paper.
* `QLSP.py`: Main script to generate data for the adiabatic algorithm for the Quantum Linear System Problem (QLSP). This corresponds to **Fig 3(e-f)** in our paper.
* `utils.py`: A utility module containing helper functions used by the main scripts.
* `Func_for_QLSP.py`: A module containing specific functions required for the QLSP simulations.

In Fig.zip:
* `Figdrawer.ipynb`: A programme to draw the figures in the Numerical Simulation part.
* Other folders: Folders containing the data required for the figures calculated by programmes in Code folder.

---

## System Requirements

### Hardware requirements
This project requires only a standard computer with enough RAM to support the in-memory operations.

### Software requirements

* OS Requirements
This package is supported for Linux. The package has been tested on the following system:

```bash
Linux: Rocky 8.7
```

* Python Dependencies

```bash
Python 3.13
pyscf 2.9.0
openfermion 1.7.1
cirq 1.5.0
```


## ⚙️ Prerequisites & Installation

### 1. Before running the scripts, ensure you have Python 3 installed. You will also need to install the following required packages. This process will cost no more than 5 mins typically.

You can install them using pip:
```bash
pip3 install numpy scipy pyscf openfermion cirq
```



### 2. Create Data Directory:
You are advised to create a folder named `Data` in the root directory of this project. The simulation results will be automatically saved there.

```bash
mkdir Data
```

### 3. Run the Scripts:
Execute the Python scripts from your terminal. Each script corresponds to different figures in the paper as described below.

* To reproduce Fig 2(a) and Fig 3(a-c):
```bash
python3 Trotter_mole.py
```

* To reproduce Fig 3(d):
```bash
python3 GQSP_mole.py
```

* To reproduce Fig 3(e-f):
```bash
python3 QLSP.py
```

* To draw Fig 2-3:
Run `/Fig/Figdrawer.ipynb`

## 📊 Output
The scripts will run the simulations and save the output data files into the `Data/` directory. You can then use this data for analysis and plotting.

## License
This project is covered under the Apache 2.0 License.
