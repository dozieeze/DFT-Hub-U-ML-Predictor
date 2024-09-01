**Sheet Name Descriptions:**

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
- **All_Systems + Extra:** Full dataset of primary and secondary metal oxides with extrapolation data.
- **All_Systems (-m-ZrO2)+ Extra:** Complete dataset of metal oxides (excluding monoclinic ZrO2) with extrapolation data included.

<br></br>
**Descriptions of features used in model training across metal oxides.**

| Features                               | Description                                                           |
|----------------------------------------|-----------------------------------------------------------------------|
| Up_Value/eV                            | Up                                                                    |
| Ud_Value/eV                            | Ud/f                                                                  |
| alpha_oxide/degree                     | Alpha angle of the metal oxide structure in degrees                   |
| beta_oxide/degree                      | Beta angle of the metal oxide structure in degrees                    |
| gamma_oxide/degree                     | Gamma angle of the metal oxide structure in degrees                   |
| Number of X atoms                      | Number of metal (X) atoms in the metal oxide unit cell                |
| Number of O atoms                      | Number of oxygen (O) atoms in the metal oxide unit cell               |
| Lattice_constant_a_of_X_pm             | Lattice constant a of the metal (X) structure in picometers (pm)      |
| Lattice_constant_b_of_X_pm             | Lattice constant b of the metal (X) structure in picometers (pm)      |
| Lattice_constant_c_of_X_pm             | Lattice constant c of the metal (X) structure in picometers (pm)      |
| Atomic_radius/pm_of_X                  | Atomic radius of the metal (X) in picometers (pm)                     |
| Van_der_Waals_radius/pm_of_X           | Van der Waals radius of the metal (X) in picometers (pm)              |
| Atomic_No_of_X                         | Atomic number of the metal (X)                                        |
| Atomic_Mass/amu_of_X                   | Atomic mass of the metal (X) in atomic mass units (amu)               |
| Period_of_X                            | Period of the metal (X) in the periodic table                         |
| First_ionization_energy/KJ/mol_of_X    | First ionization energy of the metal (X) in kJ/mol                    |
| Density/Kg/m^3_of_X                    | Density of the metal (X) in Kg/m^3                                    |
| Electron_Affinity/eV_of_X              | Electron affinity of the metal (X) in electron volts (eV)             |
| Work_Function/eV_of_X                  | Work function of the metal (X) in electron volts (eV)                 |
| Pauling_Electronegativity/units_of_X   | Pauling electronegativity of the metal (X)                            |
| d-shell_of_X                           | Number of electrons in the d-shell of the metal (X)                   |
| Lattice_angle_alpha_of_X_degree        | Lattice angle alpha of the metal (X) structure in degrees             |
| Lattice_angle_beta_of_X_degree         | Lattice angle beta of the metal (X) structure in degrees              |
| Lattice_angle_gamma_of_X_degree        | Lattice angle gamma of the metal (X) structure in degrees             |
