"""Shared vocabulary: risk labels and the patient/clinical fields the model can use.

The admin UI renders its forms from CLINICAL_FIELDS (served at /admin/schema), so adding a
field here is enough to start collecting it and feeding it to the model.
"""
from __future__ import annotations

# Ordered by severity. The risk score is the probability-weighted severity (0 / 50 / 100).
LABELS = ["low", "moderate", "high"]
LABEL_SEVERITY = {"low": 0.0, "moderate": 50.0, "high": 100.0}

OVERALL_STATUS = {
    "low": "Healthy Distribution",
    "moderate": "DFU WARNING",
    "high": "CRITICAL ALERT",
}

# kind: number | bool | sex. "required" fields must be present for training and prediction.
CLINICAL_FIELDS = [
    {"key": "age", "label": "Age", "kind": "number", "unit": "years", "required": True, "min": 1, "max": 120,
     "help": "Patient age at the time of recording."},
    {"key": "sex", "label": "Sex", "kind": "sex", "required": True,
     "help": "Male / Female."},
    {"key": "weight_kg", "label": "Weight", "kind": "number", "unit": "kg", "required": True, "min": 20, "max": 250,
     "help": "Body weight. Plantar pressure is normalised by this."},
    {"key": "height_cm", "label": "Height", "kind": "number", "unit": "cm", "required": False, "min": 100, "max": 230,
     "help": "Optional. Enables BMI as a feature."},
    {"key": "diabetes_years", "label": "Diabetes duration", "kind": "number", "unit": "years", "required": False,
     "min": 0, "max": 80, "help": "Years since diabetes diagnosis."},
    {"key": "hba1c", "label": "HbA1c", "kind": "number", "unit": "%", "required": False, "min": 3, "max": 20,
     "help": "Most recent glycated haemoglobin."},
    {"key": "neuropathy", "label": "Peripheral neuropathy (LOPS)", "kind": "bool", "required": False,
     "help": "Loss of protective sensation, e.g. 10 g monofilament test failed."},
    {"key": "pad", "label": "Peripheral artery disease", "kind": "bool", "required": False,
     "help": "PAD diagnosed / abnormal ABI."},
    {"key": "deformity", "label": "Foot deformity", "kind": "bool", "required": False,
     "help": "Hallux valgus, claw/hammer toes, Charcot, prominent metatarsal heads, etc."},
    {"key": "prior_ulcer", "label": "History of foot ulcer", "kind": "bool", "required": False,
     "help": "Any previous diabetic foot ulcer."},
    {"key": "prior_amputation", "label": "History of amputation", "kind": "bool", "required": False,
     "help": "Any previous lower-extremity amputation."},
]
CLINICAL_KEYS = [f["key"] for f in CLINICAL_FIELDS]
REQUIRED_KEYS = [f["key"] for f in CLINICAL_FIELDS if f["required"]]

SENSOR_COLUMNS = ["time", "toe", "met", "heel", "mid", "temp", "rh"]
SENSOR_HEADER = "Time,P1_Toe,P2_Met,P3_Heel,P4_Mid,Temp,Hum"
ZONES = ["toe", "met", "heel", "mid"]

LABEL_GUIDE = {
    "low": "No active ulcer, no pre-ulcerative signs (callus, redness, blister) on this foot during follow-up.",
    "moderate": "Pre-ulcerative signs present (callus, haemorrhage, blister, localized redness/warmth) "
                "or IWGDF risk category 2-3 without an ulcer.",
    "high": "Active ulcer, or an ulcer developed on this foot during the follow-up window.",
}
