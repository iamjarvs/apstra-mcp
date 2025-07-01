# apstra_project_tempalte# Python Streamlit Project Template

A reusable template for Python Streamlit projects with Docker support and GitHub Actions integration.

## Features

- **Modular Structure**: Well-organized code structure for maintainability
- **Streamlit UI**: Pre-configured Streamlit application with multiple pages
- **Docker Support**: Easy containerization with Docker
- **GitHub Actions**: Automated Docker builds
- **Testing Setup**: Basic testing framework with pytest

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Docker (for containerization)
- Git

### Using this Template

1. Click the "Use this template" button on GitHub to create a new repository based on this template.
2. Clone your new repository:

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

3. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. Run the application:

```bash
streamlit run app/ui/streamlit_app.py
```

### Docker Usage

Build and run using Docker:

```bash
# Using the provided script
chmod +x build-docker.sh
./build-docker.sh build
./build-docker.sh run

# Or manually
docker build -t streamlit-app:latest .
docker run -d -p 8501:8501 --name streamlit-app-container streamlit-app:latest
```

Access the application at http://localhost:8501

## Customizing the Template

1. Update the base Docker image in the `Dockerfile`
2. Modify `requirements.txt` to include your project-specific dependencies
3. Update the application name and description in `app/ui/streamlit_app.py`
4. Add your own utility functions in `app/utils/helpers.py`
5. Implement your application logic in the appropriate modules

## GitHub Actions

This template includes a GitHub Actions workflow that:

1. Builds the Docker image on pushes to main/master branches and tags
2. Pushes the image to GitHub Container Registry
3. Uses caching for faster builds

To use this feature:

1. Ensure your repository has access to GitHub Packages
2. The GitHub Actions workflow will run automatically on push

## Project Structure

```
streamlit-template/
├── .github/workflows/       # GitHub Actions workflows
├── app/                     # Application code
│   ├── main.py              # Entry point
│   ├── ui/                  # Streamlit UI code
│   └── utils/               # Utility functions
├── tests/                   # Test files
├── Dockerfile               # Docker configuration
├── requirements.txt         # Python dependencies
└── build-docker.sh          # Script for Docker operations
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.