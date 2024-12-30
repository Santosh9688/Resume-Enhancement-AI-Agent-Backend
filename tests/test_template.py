import os
import sys
import logging
from pathlib import Path
from docxtpl import DocxTemplate
from app.handlers.template_handler import DocxTemplateHandler

# Add the parent directory to Python path so we can import our app modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="template_test.log",
    filemode="w",
)

logger = logging.getLogger("template_test")


def test_template():
    """
    Test the resume template structure and validate all placeholders.
    This function checks if the template file exists and verifies all required
    placeholders are present.
    """
    try:
        # Get the absolute path to your template
        template_path = os.path.join(
            parent_dir, "app", "templates", "ATS_Resume_Template.docx"
        )
        logger.info(f"Testing template at path: {template_path}")

        # First, verify the template file exists
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found at: {template_path}")
        logger.info("Template file found successfully")

        # Try to load the template directly first
        logger.info("Attempting to load template directly...")
        doc = DocxTemplate(template_path)
        logger.info("Template loaded successfully")

        # Get all placeholders from the template
        placeholders = sorted(doc.undeclared_template_variables)
        logger.info("Found the following placeholders in template:")
        for placeholder in placeholders:
            logger.info(f"  - {placeholder}")

        # Define expected placeholders based on your template structure
        expected_placeholders = {
            "candidate_name",
            "linkedin_username",
            "candidate_email",
            "candidate_phone",
            "job_title",
            "summary_text",
            "key_skills",
            "backend_skills",
            "frontend_skills",
            "database_skills",
            "cloud_technologies",
            "devops_skills",
            "methodologies",
            "soft_skills",
            "career_highlights",
            "career_highlight_1",
            "career_highlight_2",
            "career_highlight_3",
            "career_highlight_4",
            "professional_experience",
            "job1_company",
            "job1_location",
            "job1_title",
            "job1_dates",
            "job1_description",
            "job1_responsibilities",
            "key_responsibilities",
            "job2_company",
            "job2_location",
            "job2_title",
            "job2_dates",
            "job2_description",
            "job2_responsibilities",
            "job3_company",
            "job3_location",
            "job3_title",
            "job3_dates",
            "job3_description",
            "job3_responsibilities",
            "education_heading",
            "masters_title",
            "masters_institution",
            "masters_year",
            "bachelor_title",
            "bachelor_institution",
            "bachelor_year",
        }

        # Check for missing placeholders
        missing_placeholders = expected_placeholders - set(placeholders)
        if missing_placeholders:
            logger.warning("Missing placeholders in template:")
            for placeholder in sorted(missing_placeholders):
                logger.warning(f"  - {placeholder}")
            print(
                "\nWarning: Some expected placeholders are missing from the template."
            )
            print("Missing placeholders:", ", ".join(sorted(missing_placeholders)))

        # Check for unexpected placeholders
        unexpected_placeholders = set(placeholders) - expected_placeholders
        if unexpected_placeholders:
            logger.warning("Unexpected placeholders found in template:")
            for placeholder in sorted(unexpected_placeholders):
                logger.warning(f"  - {placeholder}")
            print("\nWarning: Template contains unexpected placeholders.")
            print(
                "Unexpected placeholders:", ", ".join(sorted(unexpected_placeholders))
            )

        # Now test the template handler
        logger.info("Testing DocxTemplateHandler...")
        handler = DocxTemplateHandler(template_path)
        logger.info("DocxTemplateHandler initialized successfully")

        print("\nTemplate validation completed successfully!")
        print(f"Total placeholders found: {len(placeholders)}")
        print("Check template_test.log for detailed information")

    except Exception as e:
        logger.error(f"Template validation failed: {str(e)}", exc_info=True)
        print(f"\nError: Template validation failed: {str(e)}")
        print("Check template_test.log for detailed error information")
        raise


if __name__ == "__main__":
    print("Starting template validation...")
    test_template()
