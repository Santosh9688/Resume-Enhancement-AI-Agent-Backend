from docxtpl import DocxTemplate
import io
import os
import tempfile
import shutil
from typing import Dict, Any, Union, List, Optional
from dataclasses import dataclass
from pathlib import Path
from fastapi import BackgroundTasks, HTTPException, status
from fastapi.responses import FileResponse


@dataclass
class ResumeGenerationResult:
    """Represents the result of resume generation."""

    document: io.BytesIO
    filename: str
    content_type: str = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


class ResumeGenerationError(Exception):
    """Custom exception for resume generation errors."""

    pass


class ResumeDocumentGenerator:
    """
    A service class for generating resume documents using docxtpl.

    This class handles the transformation of resume data into a formatted Word document
    using a predefined template. It includes error handling, data validation, and
    proper file handling.
    """

    def __init__(self, template_path: Union[str, Path]):
        """
        Initialize the resume generator with a template path.

        Args:
            template_path: Path to the Word template file (.docx)

        Raises:
            FileNotFoundError: If the template file doesn't exist
            ResumeGenerationError: If there's an error loading the template
        """
        self.template_path = Path(template_path)
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found at: {template_path}")

        try:
            # Verify template can be loaded
            DocxTemplate(self.template_path)
        except Exception as e:
            raise ResumeGenerationError(f"Failed to load template: {str(e)}")

    def _validate_resume_data(self, data: Dict[str, Any]) -> None:
        """
        Validate the required fields in the resume data.

        Args:
            data: Dictionary containing resume data

        Raises:
            ResumeGenerationError: If required fields are missing
        """
        required_fields = {
            "candidate_name",
            "job_title",
            "candidate_email",
            "candidate_phone",
            "summary_text",
        }

        missing_fields = required_fields - set(data.keys())
        if missing_fields:
            raise ResumeGenerationError(
                f"Missing required fields: {', '.join(missing_fields)}"
            )

    def _process_job_responsibilities(
        self, responsibilities: Optional[List[str]]
    ) -> List[str]:
        """
        Process and format job responsibilities.

        Args:
            responsibilities: List of responsibility strings

        Returns:
            Formatted list of responsibilities
        """
        if not responsibilities:
            return []

        return [resp.strip() for resp in responsibilities if resp and resp.strip()]

    def _prepare_template_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare and format the data for template rendering.

        Args:
            data: Original resume data

        Returns:
            Processed and formatted data ready for template
        """
        # Create a copy to avoid modifying the original data
        template_data = data.copy()

        # Process responsibilities for each job if present
        for job_num in range(1, 4):  # Handle up to 3 jobs
            resp_key = f"job{job_num}_responsibilities"
            if resp_key in template_data:
                template_data[resp_key] = self._process_job_responsibilities(
                    template_data[resp_key]
                )

        return template_data

    def generate_resume(
        self, resume_data: Dict[str, Any], output_filename: Optional[str] = None
    ) -> ResumeGenerationResult:
        """
        Generate a resume document from the provided data.

        This method takes a dictionary of resume data, validates it, processes it,
        and generates a Word document using the template. The document is returned
        as a BytesIO object along with metadata.

        Args:
            resume_data: Dictionary containing all resume fields and data
            output_filename: Optional custom filename for the output document

        Returns:
            ResumeGenerationResult containing the document and metadata

        Raises:
            ResumeGenerationError: If there's an error during generation
        """
        try:
            # Validate the input data
            self._validate_resume_data(resume_data)

            # Prepare data for template
            template_data = self._prepare_template_data(resume_data)

            # Load and render template
            doc = DocxTemplate(self.template_path)
            doc.render(template_data)

            # Save to BytesIO buffer
            output_buffer = io.BytesIO()
            doc.save(output_buffer)
            output_buffer.seek(0)

            # Generate filename if not provided
            if not output_filename:
                safe_name = resume_data["candidate_name"].replace(" ", "_")
                output_filename = f"Resume_{safe_name}.docx"

            return ResumeGenerationResult(
                document=output_buffer, filename=output_filename
            )

        except ResumeGenerationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Wrap other errors
            raise ResumeGenerationError(f"Failed to generate resume: {str(e)}")

    async def generate_resume_docx(
        self, resume_data: Dict[str, Any], background_tasks: BackgroundTasks
    ) -> FileResponse:
        """
        Generate a resume document and return it as a FileResponse for FastAPI.

        Args:
            resume_data: Dictionary containing all resume fields
            background_tasks: FastAPI BackgroundTasks for cleanup

        Returns:
            FileResponse containing the generated document
        """
        try:
            temp_dir = tempfile.mkdtemp(prefix="resume_gen_")
            result = self.generate_resume(resume_data)
            temp_file_path = os.path.join(temp_dir, result.filename)

            with open(temp_file_path, "wb") as f:
                f.write(result.document.getvalue())

            async def cleanup_temp_files():
                try:
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Cleanup error: {str(e)}")

            background_tasks.add_task(cleanup_temp_files)

            return FileResponse(
                path=temp_file_path,
                filename=result.filename,
                media_type=result.content_type,
                background=background_tasks,
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate resume: {str(e)}",
            )


# Create global instance of ResumeDocumentGenerator
template_path = Path(__file__).parent.parent / "templates" / "ATS_Resume_Template.docx"
resume_generator = ResumeDocumentGenerator(template_path)

# Example usage:
if __name__ == "__main__":
    # Example resume data matching template fields
    sample_data = {
        # Basic Candidate Info
        "candidate_name": "Santosh Reddy",
        "linkedin_username": "santosh-reddy-k-fullstack-developer",
        "candidate_email": "santoshkanmanthareddy@gmail.com",
        "candidate_phone": "561-308-8704",
        # Target Job Title & Summary
        "job_title": "Senior Full Stack Developer (AWS, .NET, React)",
        "summary_text": (
            "Highly skilled Full Stack Developer with 12+ years of experience "
            "delivering secure, scalable, and high-performing web applications. "
            "Proven expertise in AWS (Lambda, EC2, S3, RDS), .NET Core, and React.js. "
            "Committed to implementing DevOps best practices, mentoring junior developers, "
            "and driving innovation through cloud-native architectures. Passionate about "
            "collaboration, continuous learning, and delivering impactful solutions."
        ),
        # Key Skills Section
        "backend_skills": (
            ".NET Core, C#, ASP.NET, Microservices, GraphQL, " "AWS Lambda (Python)"
        ),
        "frontend_skills": "React.js (Hooks, Redux), TypeScript, JavaScript, Material-UI",
        "database_skills": (
            "SQL Server, PostgreSQL, T-SQL, DynamoDB, MongoDB, "
            "Database Optimization & Migration"
        ),
        "cloud_technologies": (
            "AWS (Lambda, Step Functions, S3, EC2, RDS, DynamoDB, API Gateway), "
            "Terraform, CloudFormation, Pulumi"
        ),
        "devops_skills": (
            "CI/CD Pipelines, Docker, Kubernetes (EKS), Git, "
            "Code Scanning (SonarQube, BlackDuck)"
        ),
        "methodologies": "Agile/Scrum, Test-Driven Development (TDD), DevOps Culture",
        "soft_skills": (
            "Leadership, Mentoring, Communication, Team Collaboration, "
            "Problem-Solving, Adaptability"
        ),
        # Career Highlights (4 bullets)
        "career_highlight_1": (
            "Led the design and deployment of a cloud-native .NET/React application on AWS, "
            "improving scalability by 40%."
        ),
        "career_highlight_2": (
            "Implemented automated CI/CD pipelines using Docker and Kubernetes, "
            "cutting deployment times by 50%."
        ),
        "career_highlight_3": (
            "Integrated SonarQube scans and robust testing frameworks to reduce "
            "production bugs by 30%."
        ),
        "career_highlight_4": (
            "Enhanced frontend responsiveness with React.js and Material-UI, increasing "
            "user engagement by 25%."
        ),
        # ----------------------------
        # Job 1: Blankfactor (TARGET ROLE)
        # ----------------------------
        "job1_company": "Blankfactor",
        "job1_location": "Remote / Miami, FL",
        "job1_title": "Senior Full Stack Developer (AWS, .NET, React)",
        "job1_dates": "Jan 2021 – Present",
        "job1_description": (
            "At Blankfactor, I design, develop, and maintain secure, scalable, "
            "and high-performing web applications in a fast-paced environment. "
            "My role involves leveraging AWS services for serverless architectures, "
            "building microservices in .NET, and creating dynamic React.js frontends."
        ),
        "job1_responsibilities": [
            "Developed robust .NET Core backend services with RESTful APIs, GraphQL, and AWS Lambda integrations.",
            "Built responsive frontends using React.js (Hooks, Redux) to provide a seamless user experience.",
            "Automated infrastructure provisioning with CloudFormation and Terraform, enabling quicker deployments.",
            "Enforced best practices in code quality through SonarQube scans, unit tests, and thorough code reviews.",
            "Mentored junior developers, guiding them on AWS, containerization, and serverless design patterns.",
            "Collaborated with cross-functional teams to plan architectural solutions and roadmap execution.",
        ],
        # ----------------------------
        # Job 2: JP Morgan Chase
        # ----------------------------
        "job2_company": "JP Morgan Chase",
        "job2_location": "Dallas, TX, US",
        "job2_title": ".NET Full Stack Developer",
        "job2_dates": "Mar 2018 – Dec 2020",
        "job2_description": (
            "Served as a .NET Full Stack Developer, contributing to mission-critical "
            "financial applications. Specialized in backend services using ASP.NET and "
            "frontend interfaces using React and Angular."
        ),
        "job2_responsibilities": [
            "Developed and maintained full-stack .NET Core and Angular/React applications for banking portals.",
            "Integrated CI/CD pipelines using Azure DevOps to streamline builds and deployments.",
            "Implemented JWT-based authentication and optimized performance using caching strategies.",
            "Optimized complex SQL queries and stored procedures to handle large-scale financial transactions.",
            "Collaborated in Agile Scrum ceremonies, providing technical insights for sprint planning and estimation.",
        ],
        # ----------------------------
        # Job 3: YuppTV
        # ----------------------------
        "job3_company": "YuppTV",
        "job3_location": "Hyderabad, India",
        "job3_title": "Software Developer (Full Stack)",
        "job3_dates": "Feb 2015 – Feb 2018",
        "job3_description": (
            "Began as an intern and progressed to a Full Stack Developer role, "
            "contributing to high-traffic streaming media platforms. Gained "
            "foundational experience in .NET, JavaScript frameworks, and cloud services."
        ),
        "job3_responsibilities": [
            "Built MVC-based web applications in .NET Framework and C#, delivering optimized video streaming modules.",
            "Integrated RESTful APIs for seamless communication between front-end interfaces and microservices.",
            "Maintained SQL Server databases, handling indexing and performance tuning for large data sets.",
            "Collaborated with UX/UI teams to roll out responsive designs across desktop and mobile platforms.",
            "Participated in code reviews to uphold coding standards and promote continuous learning.",
        ],
        # Education Section
        "education_heading": "EDUCATION & TRAINING",
        "masters_title": "Master's, Computer Science",
        "masters_institution": "Cleveland State University",
        "masters_year": "May 2024",
        "bachelor_title": "Bachelor of Technology, Computers & Science",
        "bachelor_institution": "CVR College of Engineering",
        "bachelor_year": "April 2017",
    }

    # Initialize generator with template path
    template_path = (
        Path(__file__).parent.parent / "templates" / "ATS_Resume_Template.docx"
    )
    resume_generator = ResumeDocumentGenerator(template_path)

    # Generate resume
    result = resume_generator.generate_resume(sample_data)
