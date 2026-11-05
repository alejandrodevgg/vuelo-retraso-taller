# src/api/schemas.py
# pydantic models for the flight-delay inference API.
#
# The request is ONE flight as it is knowable BEFORE departure (schedule +
# attribution): hour, weekday, airports, optional operator/type. There is no
# field for duration or delay on purpose — those are the leaked columns
# (delay = actual − expected); the API schema cannot even express them.
#
# Airport/operator fields are STRICTER than the raw data (which contains
# non-ICAO codes and lowercase junk): serving a well-formed flight is the
# contract; the messy 2% collapses at training time, not at request time.
# Bad input -> 422 with the pydantic detail (demonstrated in class).
from typing import Optional

from pydantic import BaseModel, Field


class FlightPredictionRequest(BaseModel):
    dep_hour: int = Field(..., ge=0, le=23, description="Scheduled departure hour (UTC, 0-23)")
    dep_weekday: int = Field(..., ge=0, le=6, description="Departure weekday (Mon=0)")
    adep: str = Field(..., pattern="^[A-Z]{4}$", description="Departure ICAO code, 4 uppercase letters (e.g. LEMD)")
    ades: str = Field(..., pattern="^[A-Z]{4}$", description="Destination ICAO code, 4 uppercase letters (e.g. LEBL)")
    icao_operator: Optional[str] = Field(
        None, pattern="^[A-Z0-9]{2,3}$",
        description="Optional 3-letter ICAO operator (e.g. RYR); nulls are a real category (38.8% of flights)",
    )
    typecode: Optional[str] = Field(
        None, pattern="^[A-Z0-9]{2,4}$",
        description="Optional aircraft type (e.g. B744); treated as UNK when absent",
    )


class DelayPredictionResponse(BaseModel):
    probability_delay: float = Field(..., ge=0, le=1, description="P(arrival delay > 15 min)")
    predicted_class: int = Field(..., ge=0, le=1, description="1 when probability_delay >= class_threshold")
    class_threshold: float = Field(0.5, description="Decision threshold baked into training")
    model_version: str
    model_source: str
    prediction_time: str
