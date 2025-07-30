from pydantic import BaseSettings

class Settings(BaseSettings):
    ORTHANC_URL: str
    ORTHANC_USERNAME: str
    ORTHANC_PASSWORD: str

    S3_BUCKET: str
    S3_PREFIX: str

    SEGMENTATION_ENDPOINT: str
    CLASSIFICATION_ENDPOINT: str

    BEDROCK_API_ENDPOINT: str

    MYSQL_HOST: str
    MYSQL_PORT: int
    MYSQL_USER: str
    MYSQL_PASSWORD: str
    MYSQL_DB: str

    class Config:
        env_file = ".env"

settings = Settings()