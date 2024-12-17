## Computational Setup

The computational setup for structure optimization and band structure calculations was consistent across all oxide systems. We made minor adjustments to the KPOINTS values, aligning them with similar, already converged values found in the [Materials Project repository](https://docs.materialsproject.org/methodology/materials-methodology/calculation-details/gga+u-calculations/parameters-and-convergence). The energy cutoff for the plane-wave basis set (ENCUT) was set to 520 eV for all metal oxides, approximately double the converged value obtained from the Materials Project repository. 

**VASP Input Parameters**

- **Precision Setting:** Normal
- **Stress Tensor Optimization:** Fully optimized (ISIF = 3)
- **Energy Convergence Criterion (EDIFF):** 1.0e-06
- **Force Convergence Criterion (EDIFFG):** -0.01
- **Spin Polarization (ISPIN):** 2
- **LDA+U Parameters:** 
  - **LDAU**
  - **LDAUTYPE**
  - **LDAUL**
  - **LDAUU**
  - **LDAUJ**
- **Up Values:** Integer steps from 0.00 eV to 10.00 eV
- **Ud/f Values:** Integer steps from 2.00 eV to 10.00 eV
- **Wavefunction Initialization:** From scratch (ISTART = 0, ICHARG = 2)

**Band Structure and DOS Calculations**

For band structure and density of states (DOS) calculations, the primary modifications included:

- **IBRION:** -1
- **Number of Ionic Steps (NSW):** 0
- **ISIF:** 2
- **Force Convergence Criterion (EDIFFG):** -0.02
- **Charge Density Calculation (LCHARG):** Enabled
- **Wavefunction Information (LWAVE):** Disabled

In both structure optimization and electronic structure calculations, the **U** values ranged from 0 eV to 10 eV, and **Ud/f** values ranged from 2 eV to 10 eV. This setup allowed for an extensive evaluation of their impact on the electronic structures (band gap) and lattice parameters (a, b, c) of the metal oxides.
<br></br>
## Sheet Name Descriptions

- **Orig:** Contains material data without extrapolation data.
- **Without 'Orig':** Includes both extrapolation data and the initial material data.
- **-Extra:** Consists exclusively of extrapolation data.
- **Without Chemical Formula or Shortform (e.g., C-ZnO):** Contains material data prepared specifically for Bayesian optimization.

**Examples:**

- **Rutile TiO2 Orig:** Original dataset for rutile TiO2 without extrapolation.
- **Rutile TiO2:** Dataset for rutile TiO2 including extrapolation data.
- **Rutile TiO2-Extra:** Dataset containing only extrapolation data for rutile TiO2.
- **Rutile:** Data for rutile TiO2 formatted for Bayesian optimization.
- **C-ZnO:** Cubic ZnO data formatted for Bayesian optimization.

**Other Sheet Names:**

- **All_Primary_System:** Comprehensive dataset of all primary metal oxides.
- **All Extra:** Extrapolation data for all primary metal oxides.
- **All_Primary_System+Extra:** Combined dataset of all primary metal oxides and their extrapolation data.
- **All_Systems:** Full dataset including both primary and secondary metal oxides.
- **All_Systems + Extra:** Full primary and secondary metal oxides dataset with extrapolation data.

<br></br>
## Descriptions of features considered in model training across metal oxides.

| Feature                                                | Description                                                                                                  |
|--------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| POSCAR Crystal System                                  | Crystal system of the metal oxide structure from POSCAR data                                                 |
| Oxide                                                  | Metal oxide formula                                                                                           |
| Ud_Value/eV                                            | Ud value applied to the d-orbitals of the metal (X)                                                           |
| Uf_Value/eV                                            | Uf value applied to the f-orbitals of the metal (X)                                                           |
| Up_Value/eV                                            | Up value applied to the p-orbitals of oxygen                                                                  |
| Number of X atoms                                      | Number of metal (X) atoms in the metal oxide unit cell                                                        |
| Number of O atoms                                      | Number of oxygen (O) atoms in the metal oxide unit cell                                                       |
| POSCAR α (°)                                           | Alpha angle of the metal oxide structure in degrees                                                           |
| POSCAR β (°)                                           | Beta angle of the metal oxide structure in degrees                                                            |
| POSCAR γ (°)                                           | Gamma angle of the metal oxide structure in degrees                                                           |
| POSCAR Volume (Å³)                                      | Volume of the metal oxide structure in cubic angstroms (Å³)                                                   |
| POSCAR a (Å)                                           | Lattice constant a of the metal oxide structure in angstroms (Å)                                              |
| POSCAR b (Å)                                           | Lattice constant b of the metal oxide structure in angstroms (Å)                                              |
| POSCAR c (Å)                                           | Lattice constant c of the metal oxide structure in angstroms (Å)                                              |
| POSCAR Space Group                                     | Space group of the metal oxide structure                                                                      |
| POSCAR Hall Number                                     | Hall number of the metal oxide structure                                                                      |
| POSCAR International Number                             | International number of the metal oxide structure                                                             |
| POSCAR Symbol                                          | Symbol of the metal oxide structure                                                                           |
| POSCAR Point Group                                     | Point group of the metal oxide structure                                                                      |
| Space_group_number_of_X                                | Space group number of the metal (X)                                                                           |
| Space_group_of_X                                       | Space group of the metal (X)                                                                                  |
| Structure_of_X                                         | Structure type of metal (X)                                                                                   |
| Lattice_constant_a_of_X_pm                             | Lattice constant a of the metal (X) structure in picometers (pm)                                              |
| Lattice_constant_b_of_X_pm                             | Lattice constant b of the metal (X) structure in picometers (pm)                                              |
| Lattice_constant_c_of_X_pm                             | Lattice constant c of the metal (X) structure in picometers (pm)                                              |
| Atomic_radius/pm_of_X                                  | Atomic radius of the metal (X) in picometers (pm)                                                             |
| Van_der_Waals_radius/pm_of_X                           | Van der Waals radius of the metal (X) in picometers (pm)                                                      |
| Atomic_No_of_X                                         | Atomic number of the metal (X)                                                                                |
| Atomic_Mass/amu_of_X                                   | Atomic mass of the metal (X) in atomic mass units (amu)                                                       |
| Period_of_X                                            | Period of the metal (X) in the periodic table                                                                 |
| First_ionization_energy/KJ/mol_of_X                    | First ionization energy of the metal (X) in kJ/mol                                                            |
| Density/Kg/m³_of_X                                     | Density of the metal (X) in kg/m³                                                                             |
| Electron_Affinity/eV_of_X                              | Electron affinity of the metal (X) in electron volts (eV)                                                     |
| Work_Function/eV_of_X                                  | Work function of the metal (X) in electron volts (eV)                                                         |
| Pauling_Electronegativity/units_of_X                   | Pauling electronegativity of the metal (X)                                                                    |
| d-shell_of_X                                           | Number of electrons in the d-shell of the metal (X)                                                           |
| Lattice_angle_alpha_of_X_degree                        | Alpha angle of the metal (X) structure in degrees                                                             |
| Lattice_angle_beta_of_X_degree                         | Beta angle of the metal (X) structure in degrees                                                              |
| Lattice_angle_gamma_of_X_degree                        | Gamma angle of the metal (X) structure in degrees                                                             |
| Pauling_Electronegativity_Difference/units_of_O_and_X  | Pauling electronegativity difference between oxygen (O) and metal (X)                                         |
| Mean Metal Valence Electrons                           | Mean number of valence electrons for metal (X)                                                                |
| Mean Metal Bond Length (Å)                              | Mean bond length between metal (X) and its neighbors in angstroms (Å)                                         |
| Min Metal Bond Length (Å)                               | Minimum bond length between metal (X) and its neighbors in angstroms (Å)                                      |
| Max Metal Bond Length (Å)                               | Maximum bond length between metal (X) and its neighbors in angstroms (Å)                                      |
| Std Metal Bond Length (Å)                               | Standard deviation of bond lengths for metal (X) and its neighbors in Å                                       |
| Sum Metal Bond Length (Å)                               | Sum of bond lengths for metal (X) and its neighbors in angstroms (Å)                                          |
| Mean Metal Coordination Number                          | Mean coordination number of metal (X)                                                                         |
| Min Metal Coordination Number                           | Minimum coordination number of metal (X)                                                                      |
| Max Metal Coordination Number                           | Maximum coordination number of metal (X)                                                                      |
| Std Metal Coordination Number                           | Standard deviation of coordination numbers for metal (X)                                                      |
| Sum Metal Coordination Number                           | Sum of coordination numbers for metal (X)                                                                     |
| Mean Metal Oxidation State                              | Mean oxidation state of metal (X)                                                                             |
| Mean Oxygen Valence Electrons                           | Mean number of valence electrons for oxygen atoms                                                             |
| Mean Oxygen Bond Length (Å)                             | Mean bond length between oxygen and its neighbors in angstroms (Å)                                            |
| Min Oxygen Bond Length (Å)                              | Minimum bond length between oxygen and its neighbors in angstroms (Å)                                         |
| Max Oxygen Bond Length (Å)                              | Maximum bond length between oxygen and its neighbors in angstroms (Å)                                         |
| Std Oxygen Bond Length (Å)                              | Standard deviation of bond lengths for oxygen and its neighbors in Å                                          |
| Sum Oxygen Bond Length (Å)                              | Sum of bond lengths for oxygen and its neighbors in angstroms (Å)                                             |
| Mean Oxygen Coordination Number                         | Mean coordination number of oxygen atoms                                                                       |
| Min Oxygen Coordination Number                          | Minimum coordination number of oxygen atoms                                                                    |
| Max Oxygen Coordination Number                          | Maximum coordination number of oxygen atoms                                                                    |
| Std Oxygen Coordination Number                          | Standard deviation of coordination numbers for oxygen atoms                                                    |
| Sum Oxygen Coordination Number                          | Sum of coordination numbers for oxygen atoms                                                                   |
| Mean Oxygen Oxidation State                             | Mean oxidation state of oxygen atoms                                                                           |

