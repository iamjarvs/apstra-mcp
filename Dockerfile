# Use your base docker image as a starting point
# Replace "your-base-image:tag" with your actual base image
FROM iamjarvs/base-dev-image:latest

LABEL maintainer="Adam J"
LABEL description="Streamlit application Docker image"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false

# Set the working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Create a non-root user to run the application
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Run the application
CMD ["streamlit", "run", "app/ui/streamlit_app.py", "--server.address=0.0.0.0"]

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1