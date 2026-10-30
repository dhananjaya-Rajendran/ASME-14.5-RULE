from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class GDTCharacteristic(str, Enum):
    FLATNESS = "flatness"
    STRAIGHTNESS = "straightness"
    CIRCULARITY = "circularity"
    CYLINDRICITY = "cylindricity"
    PERPENDICULARITY = "perpendicularity"
    PARALLELISM = "parallelism"
    ANGULARITY = "angularity"
    POSITION = "position"
    PROFILE_SURFACE = "profile_of_a_surface"
    RUNOUT = "circular_runout"


class MaterialModifier(str, Enum):
    NONE = "RFS"
    MMC = "MMC"
    LMC = "LMC"


class VerificationStatus(str, Enum):
    PASS = "PASS"
    VIOLATION = "VIOLATION"
    REQUIRES_CAD_OR_CMM = "REQUIRES_CAD_OR_CMM"


class DatumReference(BaseModel):
    datum_letter: str
    material_condition: MaterialModifier = MaterialModifier.NONE


class FeatureControlFrame(BaseModel):
    id: str
    bbox: List[float]
    characteristic_symbol: str
    tolerance_value: Optional[float] = None
    diameter_zone: bool = False
    material_condition: Optional[MaterialModifier] = None
    datums: List[DatumReference] = Field(default_factory=list)
    associated_feature_type: Literal["hole", "shaft", "planar_surface", "unknown"] = "unknown"
    is_feature_of_size: Optional[bool] = None
    has_size_tolerance_on_feature: Optional[bool] = None


class DimensionItem(BaseModel):
    id: str
    bbox: List[float]
    nominal: float
    upper_tol: float = 0.0
    lower_tol: float = 0.0
    is_basic: bool = False
    has_tolerance: Optional[bool] = None
    is_reference: bool = False
    is_maximum: bool = False
    is_minimum: bool = False
    is_stock_size: bool = False
    covered_by_general_tolerance: Optional[bool] = None


class DrawingExtraction(BaseModel):
    drawing_id: str
    applicable_standard: str = "ASME Y14.5M-1994"
    declared_datums: List[str] = Field(default_factory=list)
    dimensions: List[DimensionItem] = Field(default_factory=list)
    feature_control_frames: List[FeatureControlFrame] = Field(default_factory=list)


class ValidationViolation(BaseModel):
    rule_id: str
    severity: Literal["ERROR", "WARNING", "REQUIRES_CAD_DATA"]
    fcf_id: str
    bbox: List[float]
    description: str
    asme_clause_reference: str
    asme_clause_excerpt: Optional[str] = None
    remediation_action: Optional[str] = None
    knowledge_context: Optional[str] = None
