#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
"""Unit conversion and manipulation module for scientific calculations.

This module provides a framework for handling physical units and dimensions,
built on top of SymPy's physics.units system. It includes:
- Custom unit definitions for atomic and molecular physics
- Unit conversion utilities
- Physical constants in various unit systems
- Support for parsing unit strings

The module defines several unit systems:
- SI: Standard International units
- atomic: Atomic units (ħ, e, a₀, mₑ)
- hartAng: Mixed Hartree-Angstrom units
- kcalAfs: Mixed kcal/mol-Angstrom-femtosecond units
- kcalAamu: Mixed kcal/mol-Angstrom-amu units
'''


from __future__ import print_function, division, absolute_import
from scipy import constants as sc
from sympy import pi
import math
from typing import Dict, Union, Tuple, Optional
import sympy.physics.units as u
from sympy.physics.units import Quantity, Dimension

# All systems here are dimensionally consistent with SI
_SI = u.systems.SI

# Define extra quantities missing in SymPy
aa = angstrom = angstroms = Quantity("angstrom", abbrev="AA")
_SI.set_quantity_dimension(angstrom, u.length)
_SI.set_quantity_scale_factor(angstrom, u.meter / 10**10)

me = electron_mass = Quantity("electron_mass", abbrev="me")
_SI.set_quantity_dimension(me, u.mass)
_SI.set_quantity_scale_factor(me, sc.m_e * u.kg)

mp = proton_mass = Quantity("proton_mass", abbrev="mp")
_SI.set_quantity_dimension(mp, u.mass)
_SI.set_quantity_scale_factor(mp, sc.m_p * u.kg) 

alpha = fine_structure_constant = sc.fine_structure

a0 = bohr_radius = bohr_radii = Quantity("bohr_radius", abbrev="a0")
_SI.set_quantity_dimension(a0, u.length)
_SI.set_quantity_scale_factor(a0, sc.value(u'Bohr radius')*u.meter)

Eh = hartree = hartrees = Quantity("hartree", abbrev="Eh")
_SI.set_quantity_dimension(hartree, u.energy)
_SI.set_quantity_scale_factor(hartree, sc.value(u'Hartree energy')*u.J)

kcal_mol = Quantity("kcal_mol")
_SI.set_quantity_dimension(kcal_mol, u.energy)
_SI.set_quantity_scale_factor(kcal_mol, sc.kilo*sc.calorie/sc.N_A * u.J)

fs = femtosecond = femtoseconds = Quantity("femtosecond", abbrev="fs")
_SI.set_quantity_dimension(fs, u.time)
_SI.set_quantity_scale_factor(fs, sc.femto*u.second)

coulomb_constant = Quantity("coulomb_constant", abbrev="ke")
_SI.set_quantity_dimension(coulomb_constant, u.length/u.capacitance)
_SI.set_quantity_scale_factor(coulomb_constant, 1/(4*pi*u.vacuum_permittivity))

wn = wavenumber = wavenumbers = Quantity("wavenumber", abbrev="wn")
_SI.set_quantity_dimension(wn, u.energy)
_SI.set_quantity_scale_factor(wn, u.planck*u.c/u.cm)

# A map from "extra" quantities to SymPy Quantity objects
AVAILABLE_UNITS: Dict[str, Quantity] = {
    'angstrom': angstrom,
    'aa': aa,
    'me': me,
    'electron_mass': electron_mass,
    'mp': mp,
    'proton_mass': proton_mass,
    'a0': a0,
    'bohr_radius': bohr_radius,
    'Eh': Eh,
    'hartree': hartree,
    'kcal_mol': kcal_mol,
    'fs': fs,
    'femtosecond': femtosecond,
    'coulomb_constant': coulomb_constant,
    'wn': wn,
    'wavenumber': wavenumber
}

# A map from dimension names to SymPy Dimension objects
dimensions: Dict[str, Dimension] = {
    "energy" : u.energy,
    "length" : u.length,
    "time" : u.time,
    "mass" : u.mass,
    "charge" : u.charge,
    "luminous_intensity" : u.luminous_intensity,
    "amount" : u.amount,
    "current" : u.current,
    "temperature" : u.temperature,
    "action" : u.action,
    "angular_momentum" : u.action,
    "vacuum_permittivity" : u.capacitance / u.length,
}


def _convert_to(value: Union[Quantity, Dimension], 
                base_units: Union[Quantity, Tuple[Quantity, ...]]) -> Quantity:
    """Convert a quantity or dimension to specified base units.

    Args:
        value: Either a Quantity with magnitude and units, or a pure Dimension
        base_units: Single Quantity or tuple of Quantity objects to convert to

    Returns:
        Quantity: The converted quantity in the target base units

    Raises:
        TypeError: If inputs are not of correct type
        ValueError: If conversion is not possible
    """
    # Handle single Quantity case by converting to tuple
    if isinstance(base_units, Quantity):
        base_units = (base_units,)
    elif not isinstance(base_units, tuple) or not all(isinstance(u, Quantity) for u in base_units):
        raise TypeError("base_units must be either a Quantity or a tuple of Quantity objects")
    # Treat the case that value is a Quantity
    try:
        dimension = value.dimension
        scale = 1.0 * value  # Preserve original scale
    except AttributeError:
        # value is a pure Dimension
        if not isinstance(value, Dimension):
            raise TypeError("quantity must be either a Quantity or Dimension")
        dimension = value
        scale = None

    # Create derived quantity for conversion
    ans = Quantity("derived")
    _SI.set_quantity_dimension(ans, dimension)

    # Convert original quantity to base units
    if scale is not None:
        _SI.set_quantity_scale_factor(ans, scale)
        return u.convert_to(ans, base_units).n()
    
    # Deduce the correct combination of base units to represent the Dimension
    # and return 1.0*[combination of base units]
    ans = u.convert_to(1.0 * ans, base_units).n()
    return 1.0 * ans.as_two_terms()[1]


class SI(object):
    """Base unit system using SI (International System) units.

    This class provides methods for converting between different unit systems
    and parsing unit strings. It serves as the base for specific unit system
    implementations.

    Attributes:
        base_units (Tuple[Quantity, ...]): Base units defining the SI system:
            - Length: meter (m)
            - Mass: kilogram (kg)
            - Time: second (s)
            - Current: ampere (A)
            - Amount: mole (mol)
            - Luminous intensity: candela (cd)
            - Temperature: kelvin (K)
    """
    base_units: Tuple[Quantity, ...] = (u.meter, u.kilogram, u.second, 
                                       u.ampere, u.mol, u.cd, u.K)

    def in_base(self, quantity: Union[Quantity, Dimension]) -> Quantity:
        """Convert a quantity to the current unit system's base units.

        Args:
            quantity: Quantity or Dimension to convert

        Returns:
            Quantity: The converted quantity in the current system's base units

        Example:
            >>> si = SI()
            >>> si.in_base(u.meter)  # Returns 1.0 meter in SI base units
        """
        return _convert_to(quantity, self.base_units)

    def in_SI(self, quantity: Union[Quantity, Dimension]) -> Quantity:
        """Convert a quantity to SI base units.

        Args:
            quantity: Quantity or Dimension to convert

        Returns:
            Quantity: The converted quantity in SI base units

        Example:
            >>> atomic_sys = atomic()
            >>> atomic_sys.in_SI(u.bohr)  # Returns bohr radius in meters
        """
        return _convert_to(quantity, SI.base_units)

    def str2valunit(self, string: str) -> Tuple[float, Optional[Quantity]]:
        """Parse a string containing a value and optional unit into separate components.

        This method takes a string input in the format 'value' or 'value unit' and returns
        a tuple containing the numerical value and unit object. If no unit is specified,
        the unit component will be None.

        Args:
            string (str): Input string in format 'value' or 'value unit'
                         Example: '1.23' or '1.23 meter'

        Returns:
            tuple: (float, Unit or None)
                - First element is the parsed numerical value as float
                - Second element is either a sympy.physics.units.Quantity object or None

        Raises:
            ValueError: If string format is invalid or value cannot be converted to float
            AttributeError: If specified unit is not found in sympy.physics.units or local module
        """
        # Split input string into components
        string_list = string.split()

        # Handle case with only value, no unit
        if len(string_list) == 1:
            value = string_list[0]
            uobj = None
        # Handle case with both value and unit
        elif len(string_list) == 2:
            value, unit = string_list
            # Try to find unit first in sympy.physics.units
            try:
                uobj = getattr(u, unit)
            except AttributeError:
                # If not in sympy, try to find in local AVAILABLE_UNITS
                try:
                    uobj = AVAILABLE_UNITS[unit]
                except KeyError as e:
                    raise AttributeError(f'Unknown unit "{unit}"') from e
            # Verify that found object is actually a unit
            if not isinstance(uobj, Quantity):
                raise ValueError(f'"{unit}" is not a valid unit')
        else:
            raise ValueError("The input for string conversion must be of the form 'value' or 'value unit'")

        # Convert string value to float
        try:
            valnum = float(value)
        except ValueError:
            raise ValueError(f"Unable to convert '{value}' to float. Value must be a numeric string.")
        
        return valnum, uobj
    
    def str2base(self, string: Union[str, float]) -> float:
        """Convert string or numeric value to base units of current system.

        Args:
            string: Either a string in format 'value unit' or a numeric value

        Returns:
            float: Converted value in current system's base units

        Raises:
            ValueError: If string format is invalid
            AttributeError: If specified unit is not found

        Example:
            >>> si = SI()
            >>> si.str2base("1.23 meter")  # Returns 1.23
            >>> si.str2base(1.23)  # Returns 1.23
        """
        if isinstance(string, str):
            value, unit = self.str2valunit(string)
            if unit is None:
                return value
            return value * float(self.in_base(unit).as_two_terms()[0])
        return string
        
    def str2SI(self, string: Union[str, float]) -> float:
        """Convert string or numeric value to SI base units.

        Args:
            string: Either a string in format 'value unit' or a numeric value

        Returns:
            float: Converted value in SI base units

        Raises:
            ValueError: If string format is invalid
            AttributeError: If specified unit is not found

        Example:
            >>> atomic_sys = atomic()
            >>> atomic_sys.str2SI("1.0 bohr")  # Returns value in meters
        """
        if type(string) is str:
            value, unit = self.str2valunit(string)
            if unit is None:
                return value
            else:
                return value * float(self.in_SI(unit).as_two_terms()[0])
        else:
            return string

    def __getattr__(self, attr: str) -> float:
        """Dynamic accessor for dimension scale factors.

        Provides access to scale factors for different physical dimensions
        by looking up the dimension in the global dimensions dictionary.

        Args:
            attr: Name of the dimension (e.g., "energy", "length", etc.)

        Returns:
            float: Scale factor for the requested dimension

        Raises:
            AttributeError: If dimension name is not found in dimensions dict

        Example:
            >>> si = SI()
            >>> si.length  # Returns length scale factor (1.0 for SI)
        """
        try:
            dim = dimensions[attr]
        except KeyError:
            raise AttributeError(f"Unknown dimension '{attr}'")
        else:
            SI_units = self.in_SI(dim).as_two_terms()[1]
            return float(
                u.convert_to(self.in_base(dim), SI_units).n().as_two_terms()[0])

    # Physical constants
    @property
    def hbar(self) -> float:
        """Get reduced Planck constant in current unit system.
        
        Returns:
            float: Value of ħ in base units
        """
        return float(self.in_base(u.hbar).as_two_terms()[0])
    
    @property
    def e(self) -> float:
        """Get elementary charge in current unit system.
        
        Returns:
            float: Value of e in base units
        """
        return float(self.in_base(u.elementary_charge).as_two_terms()[0])

    @property
    def kb(self) -> float:
        """Get Boltzmann constant in current unit system.
        
        Returns:
            float: Value of kB in base units
        """
        return float(self.in_base(u.boltzmann_constant).as_two_terms()[0])

    @property
    def amu(self) -> float:
        """Get atomic mass unit in current unit system.
        
        Returns:
            float: Value of atomic mass unit in base units
        """
        return float(self.in_base(u.amu).as_two_terms()[0])

    @property
    def m_e(self) -> float:
        """Get electron mass in current unit system.
        
        Returns:
            float: Value of electron mass in base units
        """
        return float(self.in_base(me).as_two_terms()[0])

    @property
    def me(self) -> float:
        """Get electron mass in current unit system (alias for m_e).
        
        Returns:
            float: Value of electron mass in base units
        """
        return self.m_e

    @property
    def c(self) -> float:
        """Get speed of light in current unit system.
        
        Returns:
            float: Value of c in base units
        """
        return float(self.in_base(u.c).as_two_terms()[0])

    # Temperature conversion
    def betaTemp(self, beta: float) -> float:
        """Convert inverse temperature (beta) to temperature.

        Args:
            beta: Inverse temperature (1/kT)

        Returns:
            float: Temperature in current unit system

        Example:
            >>> si = SI()
            >>> si.betaTemp(1.0)  # Returns T = 1/(kB*beta)
        """
        return 1.0/(self.kb*beta)

    # Wavenumber conversion
    def energy2wn(self, E: float) -> float:
        """Convert energy to wavenumber (cm⁻¹).

        Args:
            E: Energy value in current unit system

        Returns:
            float: Wavenumber in cm⁻¹

        Example:
            >>> si = SI()
            >>> si.energy2wn(1.23e-19)  # Returns wavenumber in cm⁻¹
        """
        # Factor of 200π comes from conversion between angular frequency and wavenumber
        return E * self.energy*self.time / (
            200*math.pi*self.c*self.hbar * self.length*self.action)

    def wn2energy(self, wn: float) -> float:
        """Convert wavenumber (cm⁻¹) to energy.

        Args:
            wn: Wavenumber in cm⁻¹

        Returns:
            float: Energy in current unit system

        Example:
            >>> si = SI()
            >>> si.wn2energy(1000)  # Returns energy for 1000 cm⁻¹
        """
        return 200*math.pi*self.c*self.hbar*wn * self.length*self.action / (
            self.energy*self.time)

    def omega2wn(self, w: float) -> float:
        """Convert angular frequency to wavenumber (cm⁻¹).

        Args:
            w: Angular frequency in current unit system

        Returns:
            float: Wavenumber in cm⁻¹

        Example:
            >>> si = SI()
            >>> si.omega2wn(1e14)  # Returns corresponding wavenumber
        """
        return w / (200*math.pi * (self.c * self.length))

    def wn2omega(self, wn: float) -> float:
        """Convert wavenumber (cm⁻¹) to angular frequency.

        Args:
            wn: Wavenumber in cm⁻¹

        Returns:
            float: Angular frequency in current unit system

        Example:
            >>> si = SI()
            >>> si.wn2omega(1000)  # Returns angular frequency for 1000 cm⁻¹
        """
        return wn * (200*math.pi * (self.c * self.length))


class atomic(SI):
    """Atomic unit system.

    Attributes:
        base_units (Tuple[Quantity, ...]): Base units defining atomic units:
            - Action: ħ (reduced Planck constant)
            - Charge: e (elementary charge)
            - Length: a₀ (Bohr radius)
            - Mass: mₑ (electron mass)
            - Amount: mole (mol)
            - Luminous intensity: candela (cd)
            - Temperature: kelvin (K)
    """
    base_units = (u.hbar, u.elementary_charge, a0, me, u.mol, u.cd, u.K)


class hartAng(SI):
    """Mixed Hartree-Angstrom unit system.

    Attributes:
        base_units (Tuple[Quantity, ...]): Base units:
            - Action: ħ (reduced Planck constant)
            - Charge: e (elementary charge)
            - Length: Å (Angstrom)
            - Energy: Eₕ (Hartree)
            - Amount: mole (mol)
            - Luminous intensity: candela (cd)
            - Temperature: kelvin (K)
    """
    base_units = (u.hbar, u.elementary_charge, aa, Eh, u.mol, u.cd, u.K)


class kcalAfs(SI):
    """Mixed kcal/mol-Angstrom-femtosecond unit system.

    Attributes:
        base_units (Tuple[Quantity, ...]): Base units:
            - Energy: kcal/mol
            - Length: Å (Angstrom)
            - Time: fs (femtosecond)
            - Electrostatic: kₑ (Coulomb constant)
            - Amount: mole (mol)
            - Luminous intensity: candela (cd)
            - Temperature: kelvin (K)
    """
    base_units = (kcal_mol, aa, fs, coulomb_constant, u.mol, u.cd, u.K)


class kcalAamu(SI):
    """Mixed kcal/mol-Angstrom-amu unit system.

    Attributes:
        base_units (Tuple[Quantity, ...]): Base units:
            - Energy: kcal/mol
            - Length: Å (Angstrom)
            - Mass: u (unified atomic mass unit)
            - Electrostatic: kₑ (Coulomb constant)
            - Amount: mole (mol)
            - Luminous intensity: candela (cd)
            - Temperature: kelvin (K)
    """
    base_units = (kcal_mol, aa, u.amu, coulomb_constant, u.mol, u.cd, u.K)


class eVAamu(SI):
    """Mixed eV-Angstrom-amu unit system.

    Attributes:
        base_units (Tuple[Quantity, ...]): Base units:
            - Energy: eV (electronvolt)
            - Length: Å (Angstrom)
            - Mass: u (unified atomic mass unit)
            - Charge: C (Coulomb)
            - Amount: mole (mol)
            - Luminous intensity: candela (cd)
            - Temperature: kelvin (K)
    """
    base_units = (u.electronvolt, aa, u.amu, u.coulomb, u.mol, u.cd, u.K)


class cmbohramu(SI):
    """Mixed wavenumber-Bohr-amu unit system.

    A specialized unit system combining spectroscopic wavenumbers with
    atomic units of length and mass.

    Attributes:
        base_units (Tuple[Quantity, ...]): Base units:
            - Energy: cm⁻¹ (wavenumber)
            - Length: a₀ (Bohr radius)
            - Mass: u (unified atomic mass unit)
            - Charge: C (Coulomb)
            - Amount: mole (mol)
            - Luminous intensity: candela (cd)
            - Temperature: kelvin (K)
    """
    base_units = (wn, a0, u.amu, u.coulomb, u.mol, u.cd, u.K)