import pytest
from fastapi.testclient import TestClient
from app.main import app
import os

client = TestClient(app)


# Test data setup
@pytest.fixture
def sample_resume_file():
    """Creates a sample resume file for testing"""
    content = """
    John Doe
    Software Engineer
    
    Experience:
    - Developed web applications using Python
    - Led team of 5 developers
    """
    with open("test_resume.txt", "w") as f:
        f.write(content)
    yield "test_resume.txt"
    os.remove("test_resume.txt")


@pytest.fixture
def sample_job_description():
    """Sample job description for testing"""
    return """
    Senior Software Engineer position requiring:
    - Python expertise
    - Team leadership experience
    - Web development skills
    """


# Test cases
def test_enhance_resume(sample_resume_file, sample_job_description):
    """Test the resume enhancement endpoint"""
    with open(sample_resume_file, "rb") as f:
        response = client.post(
            "/api/v1/enhancements/enhance-resume",
            files={"resume_file": ("resume.txt", f, "text/plain")},
            data={
                "job_description": sample_job_description,
                "user_id": 1,
                "preferred_job_titles": "Senior Software Engineer",
                "target_industries": "Technology",
                "keywords": "Python,Leadership,Web Development",
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "enhancement_id" in data
    assert "enhanced_resume" in data
    assert "changes_summary" in data


def test_get_enhancement():
    """Test retrieving enhancement details"""
    # First create an enhancement
    # Then retrieve it
    response = client.get("/api/v1/enhancements/1")

    assert response.status_code == 200
    data = response.json()
    assert "enhancement_id" in data
    assert "enhanced_content" in data
    assert "changes_applied" in data
    assert "created_date" in data
