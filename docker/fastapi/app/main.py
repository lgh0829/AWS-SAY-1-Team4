from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import ReportData
import boto3
import uuid
from datetime import datetime
import os

app = FastAPI(title="Medical Report API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ORIGIN", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경 변수에서 AWS 자격 증명 로드
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE", "MedicalReports")

# DynamoDB 클라이언트 설정
dynamodb = boto3.resource(
    'dynamodb',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

reports_table = dynamodb.Table(DYNAMODB_TABLE)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/reports")
async def create_report(report_data: ReportData):
    try:
        # DynamoDB에 저장할 아이템 구성
        item = {
            'reportId': str(uuid.uuid4()),
            'patientId': report_data.patientInfo.get('id', 'unknown'),
            'patientName': report_data.patientInfo.get('name', 'unknown'),
            'patientAge': report_data.patientInfo.get('age', 'unknown'),
            'patientGender': report_data.patientInfo.get('gender', 'unknown'),
            'reportText': report_data.reportText,
            'timestamp': report_data.timestamp,
            'createdAt': datetime.now().isoformat()
        }
        
        # DynamoDB에 저장
        response = reports_table.put_item(Item=item)
        
        return {
            "status": "success",
            "reportId": item['reportId'],
            "message": "Report saved successfully"
        }
    
    except Exception as e:
        print(f"Error saving report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save report: {str(e)}")

@app.get("/api/reports/{patient_id}")
async def get_patient_reports(patient_id: str):
    try:
        # 특정 환자의 모든 보고서 조회
        response = reports_table.query(
            KeyConditionExpression='patientId = :pid',
            ExpressionAttributeValues={':pid': patient_id}
        )
        
        return {
            "status": "success",
            "reports": response.get('Items', [])
        }
    
    except Exception as e:
        print(f"Error fetching reports: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch reports: {str(e)}")

if __name__ == "__main__":
    # 직접 실행시, Uvicorn 서버 시작
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)