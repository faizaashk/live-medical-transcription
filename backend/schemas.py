"""Pydantic models describing the structured clinical summary.

Every field has a default so a partially filled LLM answer still validates.
"""
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class Base(BaseModel):
    """LLMs sometimes return numbers (e.g. age: 30). Accept them and store as text."""
    model_config = ConfigDict(coerce_numbers_to_str=True)


class Patient(Base):
    name: Optional[str] = None
    age: Optional[str] = None
    sex: Optional[str] = None
    identifiers: List[str] = Field(default_factory=list)


class HPI(Base):
    onset: Optional[str] = None
    duration: Optional[str] = None
    progression: Optional[str] = None
    aggravating: List[str] = Field(default_factory=list)
    relieving: List[str] = Field(default_factory=list)


class Symptoms(Base):
    positive: List[str] = Field(default_factory=list)
    negative: List[str] = Field(default_factory=list)  # explicitly denied


class PastHistory(Base):
    conditions: List[str] = Field(default_factory=list)
    surgeries: List[str] = Field(default_factory=list)
    hospitalisations: List[str] = Field(default_factory=list)


class Medication(Base):
    name: str = ""
    dose: Optional[str] = None
    adherence: Optional[str] = None


class Medications(Base):
    current: List[Medication] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)


class Observations(Base):
    vitals: List[str] = Field(default_factory=list)
    exam_findings: List[str] = Field(default_factory=list)


class Assessment(Base):
    provisional: Optional[str] = None
    differentials: List[str] = Field(default_factory=list)


class Plan(Base):
    investigations: List[str] = Field(default_factory=list)
    prescriptions: List[str] = Field(default_factory=list)
    advice: List[str] = Field(default_factory=list)
    follow_up: Optional[str] = None


class ClinicalSummary(Base):
    patient: Patient = Field(default_factory=Patient)
    chief_complaint: Optional[str] = None
    hpi: HPI = Field(default_factory=HPI)
    symptoms: Symptoms = Field(default_factory=Symptoms)
    past_medical_history: PastHistory = Field(default_factory=PastHistory)
    medications: Medications = Field(default_factory=Medications)
    observations: Observations = Field(default_factory=Observations)
    assessment: Assessment = Field(default_factory=Assessment)
    plan: Plan = Field(default_factory=Plan)
    clinical_summary: str = ""


class TranscriptIn(Base):
    text: str
