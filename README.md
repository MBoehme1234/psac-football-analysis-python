# PSAC Football Analysis Python

This repository contains Python code for PSAC Football Analysis, configured for AWS deployment.

## Project Structure

```
.
├── python-app/          # Main application code
├── buildspec.yml       # AWS CodeBuild configuration
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## AWS Deployment

This project is configured for AWS CodeBuild deployment. The build process includes:

1. Installing Python 3.9
2. Installing dependencies from requirements.txt
3. Running tests
4. Creating build artifacts

## Local Development

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run tests:
   ```bash
   python -m pytest
   ```

## AWS CodeBuild Configuration

The `buildspec.yml` file contains the build configuration for AWS CodeBuild. It specifies:
- Python runtime version
- Build commands
- Test execution
- Artifact creation 