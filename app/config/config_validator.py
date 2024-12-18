import os


def validate_environment():
    """Validate all required environment variables are present and properly formatted."""
    required_vars = {
        "DB_SERVER": "Database server name",
        "DB_NAME": "Database name",
        "ANTHROPIC_API_KEY": "Anthropic API key",
    }

    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var} ({description})")

    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
