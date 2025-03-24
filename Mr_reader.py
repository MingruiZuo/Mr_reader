import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


"""
250305 Mingrui.Zuo
"""

class File:
    """
    A base file class that can be extended for different file types.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

class CIF(File):
    """
    A subclass of File that handles CIF (Crystallographic Information File) parsing.
    """
    def __init__(self, filepath):
        super().__init__(filepath)
        self.lattice_parameters = None
    
    def parse_lattice(self):
        """
        Parses the lattice parameters from the CIF file.
        """
        keywords = {
            "_cell_length_a": None,
            "_cell_length_b": None,
            "_cell_length_c": None,
            "_cell_angle_alpha": None,
            "_cell_angle_beta": None,
            "_cell_angle_gamma": None
        }
        
        with open(self.filepath, "r") as file:
            for line in file:
                for key in keywords:
                    if key in line:
                        try:
                            keywords[key] = float(line.split()[-1])
                        except ValueError:
                            raise ValueError(f"Invalid value for {key} in CIF file.")
        
        # Ensure all parameters were found
        if None in keywords.values():
            missing = [k for k, v in keywords.items() if v is None]
            raise ValueError(f"Missing lattice parameters in CIF file: {missing}")
        
        self.lattice_parameters = (
            keywords["_cell_length_a"],
            keywords["_cell_length_b"],
            keywords["_cell_length_c"],
            keywords["_cell_angle_alpha"],
            keywords["_cell_angle_beta"],
            keywords["_cell_angle_gamma"]
        )
        
        return self.lattice_parameters
    
    def parse_atoms(self, start_line=25, supercell=(1, 1, 1), col=(0, 2, -1), print_element="False"):
        """
        Parses atomic positions and charges from the CIF file and generates a supercell.
        """
        e_col = col[0]
        x_col = col[1]
        q_col = col[2]

        with open(self.filepath, "r") as file:
            cif_atom_lines = file.readlines()[start_line-1:]
        
        atom_len = len(cif_atom_lines)
        point_charges = []
        
        for i in range(atom_len):
            line_list = cif_atom_lines[i].split()
            
            x, y, z = map(float, line_list[x_col:x_col+3])  # x_col is the starting column of x values
            q = float(line_list[q_col])
            element = line_list[e_col]
            point_charges.append((element, x, y, z, q))
        
        supercell_charges = []
        for dx in range(supercell[0]):  # x axis
            for dy in range(supercell[1]):  # y axis
                for dz in range(supercell[2]):  # z axis
                    for element, x, y, z, q in point_charges:
                        new_x = x + dx
                        new_y = y + dy
                        new_z = z + dz
                        if print_element != "False":
                            supercell_charges.append((element, new_x, new_y, new_z, q))
                        else:
                            supercell_charges.append((new_x, new_y, new_z, q))

        return supercell_charges

    def formalize_cif(self, output_file, row=(17, 24), col=(0, 2, -1)):
        """
        Formalize a CIF file into a standardized format.
        """
        loop_line, start_line = row
        element_col, coordinate_col, charge_col = col

        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        # Save the lines before loop
        cleaned_lines = lines[:loop_line]
        
        atoms = []
        for line in lines[start_line - 1:]:
            parts = line.split()
            if len(parts) >= max(coordinate_col + 3, charge_col + 1):
                element = parts[element_col]
                x, y, z = [float(parts[coordinate_col + i]) % 1 for i in range(3)]
                charge = parts[charge_col]
                atoms.append((element, x, y, z, charge))
        
        # Sort atoms in descending order based on (x, y, z)
        atoms.sort(key=lambda atom: (atom[1], atom[2], atom[3]), reverse=True)
        
        # Remove duplicate atoms
        seen_particles = set()
        formatted_atoms = []
        for atom in atoms:
            key = (atom[1], atom[2], atom[3], atom[4])
            if key not in seen_particles:
                seen_particles.add(key)
                formatted_atoms.append(f"{atom[0]}   {atom[1]:.5f}  {atom[2]:.5f}  {atom[3]:.5f}   {atom[4]}\n")
        
        # Write the formalized CIF
        with open(output_file, 'w') as f:
            f.writelines(cleaned_lines)
            f.write("_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_charge\n")
            f.writelines(formatted_atoms)


    def quick_formalize(self, output_file=None, atoms=(25, 420), col=(0, 2, -1)):
        """
        Quick formalization of a CIF file without a header.
        
        Parameters:
            output_file (str): Optional. Path to the output file. Defaults to 'formalized_<原始文件名>.cif'.
            atoms (tuple): (start_line, atom_count), start from `start_line`, read `atom_count` lines.
            col (tuple): (element_col, coordinate_col, charge_col) for data extraction.
        """
        # 默认输出文件名
        if output_file is None:
            base_name = os.path.basename(self.filepath)
            output_file = f"formalized_{base_name}"

        start_line, atom_count = atoms
        element_col, coordinate_col, charge_col = col

        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        atoms = []
        for line in lines[start_line - 1:start_line - 1 + atom_count]:
            parts = line.split()
            if len(parts) >= max(coordinate_col + 3, charge_col + 1):
                element = parts[element_col]
                x, y, z = [float(parts[coordinate_col + i]) % 1 for i in range(3)]
                charge = parts[charge_col]
                atoms.append((element, x, y, z, charge))

        # 逆序排序并去重
        atoms.sort(key=lambda atom: (atom[1], atom[2], atom[3]), reverse=True)
        seen_particles = set()
        formatted_atoms = []
        for atom in atoms:
            key = (atom[1], atom[2], atom[3], atom[4])
            if key not in seen_particles:
                seen_particles.add(key)
                formatted_atoms.append(f"{atom[0]}   {atom[1]:.5f}  {atom[2]:.5f}  {atom[3]:.5f}   {atom[4]}\n")

        # 写入目标文件
        with open(output_file, 'w') as f:
            f.writelines(formatted_atoms)

        print(f"Formalization complete. Output saved to '{output_file}'")


"""
Mingrui.Zuo, 2024/12/12
Parse point charges and coordinates from charged CIF
The CIF is generated from car2cif.py
Using nearest neighbor interpolating and Gaussian smoothing.
Plot a 2-D heatmap to display the point charge distribution.

### use asaf-env on this laptop

2025/01/15
- modified parse_cif_charge, add axis='all'
- added a differential point charge distribution plot.

2025/02/12
- modified parse_cif_charge, add x_col and q_col

2025/02/20
- added vmin, vmax to scale charge distribution plot
"""



@staticmethod
def charge_from_xlsx(excel_filepath, q_col='q5', supercell=(1, 1, 1), element_list='All'):
    df = pd.read_excel(excel_filepath)[['element', 'x', 'y', 'z', q_col]]
    df = df if element_list == 'All' else df[df['element'].isin(element_list)]
    return [(row['x'] + dx, row['y'] + dy, row['z'] + dz, row[q_col])
            for dx in range(supercell[0]) for dy in range(supercell[1]) for dz in range(supercell[2]) for _, row in df.iterrows()]


@staticmethod
def generate_heatmap(point_charges, colormap='viridis', x=3, y=3, a=1, b=1, axis='Z', ax=None,  
                     title='Charge distribution', colorbar_label='Point Charge value', vmin=None, vmax=None, figsize=(10, 8), sigma=5):
    """
    Generates a heatmap for the distribution of point charges and returns the figure object.

    Parameters:
    point_charges (list): List of tuples containing (x, y, z, charge).
    colormap (str): The colormap to be used for the heatmap.
    x (int): The size of the grid along the x-axis.
    y (int): The size of the grid along the y-axis.
    a (float): Lattice parameter for the x dimension.
    b (float): Lattice parameter for the y dimension.
    axis (str): The axis to project the data onto ('X', 'Y', or 'Z').
    title (str): The title of the heatmap.
    ax (matplotlib.axes.Axes, optional): The axes object to plot on. If None, a new figure is created.
    vmin, vmax: by default is None

    Returns:
    fig: The figure object containing the heatmap.
    ax: The axes object for the heatmap.
    """
    if ax is None:  # If no axes object is provided, create a new figure and axes
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure  # Use the provided figure

    # Projection logic based on axis
    if axis == 'Z':
        # Projection onto the XY plane (ignore Z-coordinate)
        point_charges_2d = [(x, y, q) for x, y, _, q in point_charges]
    elif axis == 'X':
        # Projection onto the YZ plane (ignore X-coordinate)
        point_charges_2d = [(y, z, q) for _, y, z, q in point_charges]
    elif axis == 'Y':
        # Projection onto the XZ plane (ignore Y-coordinate)
        point_charges_2d = [(x, z, q) for x, _, z, q in point_charges]
    else:
        raise ValueError("Invalid value for 'axis'. Choose from 'X', 'Y', or 'Z'.")

    # Extract coordinates and charges for heatmap generation
    x_coords = [x for x, y, q in point_charges_2d]
    y_coords = [y for x, y, q in point_charges_2d]
    charges = [q for x, y, q in point_charges_2d]

    # Create grid for interpolation
    x_min, x_max = 0, x
    y_min, y_max = 0, y

    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 1000),
        np.linspace(y_min, y_max, 1000)
    )

    # Interpolate charge data onto the grid and apply Gaussian smoothing
    grid_q = griddata((x_coords, y_coords), charges, (grid_x, grid_y), method='nearest')
    smoothed_density_gaussian = gaussian_filter(grid_q, sigma=sigma)

    # Plot the heatmap
    if vmin is None and vmax is None:
        im = ax.imshow(smoothed_density_gaussian, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap=colormap, aspect=b/a)  # aspect='auto'
    else:
        im = ax.imshow(smoothed_density_gaussian, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap=colormap, aspect=b/a, vmin=vmin, vmax=vmax)  # aspect='auto'
    fig.colorbar(im, ax=ax, label=colorbar_label)
    ax.set_title(title)

    # Return the figure and axis objects
    return fig, ax
