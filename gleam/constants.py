r"""
Contains a list of constants and user defined units

"""
__author__ = "Andra Stroe"
__version__ = "0.1"

import yaml
from dataclasses import asdict, is_dataclass, field, dataclass as dat
from typing import Dict, Optional, List, Union, Literal
import functools
import os
import copy # Import the 'copy' module for deepcopy

from pydantic.dataclasses import dataclass
from astropy import units as u
from astropy.table import QTable
from astropy.cosmology import FlatLambdaCDM
from colorama import Fore

AllLines = Literal["all"]
CenterConstraint = Literal["free", "constrained", "fixed"]


class Quantity(u.SpecificTypeQuantity):
    """
    Validation of the types of unit for each parameter, to ensure the right type
    is being given.
    """

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return cls(v)

@dat(frozen=True, eq=True, unsafe_hash=True)
class SourceIdentifier:
    sample: str
    setup: str
    pointing: str
    source: int

    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        sample, setup, pointing, source = v.split('.')
        return cls(sample, setup, pointing, int(source))

class Length(Quantity):
    _equivalent_unit = u.m


class Frequency(Quantity):
    _equivalent_unit = (1 / u.s).unit


class Temperature(Quantity):
    _equivalent_unit = u.K


class Velocity(Quantity):
    _equivalent_unit = u.m / u.s


class ColumnDensity(Quantity):
    _equivalent_unit = 1 / u.cm ** 2


class Flux(Quantity):
    _equivalent_unit = u.erg / u.s / u.cm ** 2


class Luminosity(Quantity):
    _equivalent_unit = u.erg / u.s


class Angle(Quantity):
    _equivalent_unit = u.rad


@dataclass(config=dict(arbitrary_types_allowed=True))
class Config:
    line_list: str
    sky_lines: Optional[str]

    @dataclass
    class Instrument:
        name: str
        rest_spectral_resolution: Velocity

    @dataclass
    class Fitting:
        continuum: bool
        cont_width: Length
        center_constraint: CenterConstraint
        constraints: bool
        mask_sky: bool
        tolerance: Length

    instrument: Instrument
    fitting: Fitting
    cosmology: FlatLambdaCDM


def deep_update(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = value
    return source


ConfigOverrides = dict


@dataclass(config=dict(arbitrary_types_allowed=True))
class Constants:
    cosmology: FlatLambdaCDM = field(default_factory=lambda: FlatLambdaCDM(H0=70, Om0=0.3))
    globals: ConfigOverrides = field(default_factory=ConfigOverrides)
    setups: Dict[str, ConfigOverrides] = field(default_factory=dict)
    sources: Dict[SourceIdentifier, ConfigOverrides] = field(default_factory=dict)

    def __call__(
        self, sample: str, setup_name: str, pointing: str, source_number: int,
    ) -> Config:
        """
        Constructs the final configuration for a source by layering overrides.
        Order of precedence (lowest to highest): Globals -> Setup -> Source.
        """
        # --- FIX: More robust and explicit config merging logic ---
        if not self.globals:
            raise ValueError("Your gleamconfig.yaml must have a 'globals' section containing base parameters like 'line_list', 'instrument', etc.")
        
        # Start with a deep copy of the global settings.
        final_config = copy.deepcopy(self.globals)

        # Diagnostic print to see what's being loaded
        print(f"DEBUG: Globals loaded for source {source_number}: {final_config}")

        # Layer setup-specific overrides
        setup_overrides = self.setups.get(setup_name)
        if setup_overrides:
            deep_update(final_config, setup_overrides)

        # Layer source-specific overrides
        source_overrides = self.sources.get(SourceIdentifier(sample, setup_name, pointing, source_number))
        if source_overrides:
            deep_update(final_config, source_overrides)
            
        # Add the cosmology, which is defined at the top level of the config or defaulted.
        final_config["cosmology"] = self.cosmology

        return Config(**final_config)


def read_config(config_file) -> Constants:
    """
    Read YAML configuration file into a class. Not all parameters have to be 
    set. It not set, a parameter will be set to the default value. The class has 
    defaults that the config file will override. The unit types will also be
    checked for correctness.
    """
    if not isinstance(config_file, (str, os.PathLike)):
        raise TypeError(
            f"Configuration file path must be a string, but got {type(config_file)}"
        )
    
    with open(config_file, 'r') as f:
        config_data = f.read()

    config = yaml.safe_load(config_data)
    # If the file is empty or malformed, config could be None
    if config is None:
        config = {}
        
    return Constants(**config)