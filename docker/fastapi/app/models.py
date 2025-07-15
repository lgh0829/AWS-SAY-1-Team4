from pydantic import BaseModel
from typing import Dict, Any

class ReportData(BaseModel):
    patientInfo: Dict[str, Any]
    reportText: str
    timestamp: str