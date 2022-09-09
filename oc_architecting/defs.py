"""
The MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Copyright: (c) 2022, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de
         mahmoud.fouda@dlr.de
"""

from typing import *
from dataclasses import dataclass

__all__ = ["ArchElement", "ArchSubSystem", "WEIGHT_OUTPUT", "DURATION_INPUT", "FLTCOND_RHO_INPUT", "FLTCOND_TAS_INPUT"]

DURATION_INPUT = "duration"
FLTCOND_RHO_INPUT = "fltcond|rho"
FLTCOND_TAS_INPUT = "fltcond|Utrue"

WEIGHT_OUTPUT = "subsystem_weight"


@dataclass(frozen=False)
class ArchElement:
    """Base class for an architecture element."""

    name: str

    def __hash__(self):
        return id(self)


class ArchSubSystem:
    """Base class for a subdivision of the propsulsion system architecture."""

    def get_dv_defs(self) -> List[Tuple[str, List[str], str, Any]]:  # (key, paths, unit, default_val)
        raise NotImplementedError
