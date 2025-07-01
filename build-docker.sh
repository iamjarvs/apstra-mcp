#!/bin/bash
# Script to build and run the Docker image locally

# Set variables
IMAGE_NAME="streamlit-app"
IMAGE_TAG="latest"
CONTAINER_NAME="streamlit-app-container"
PORT=8501

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to display usage
function show_usage {
    echo -e "${YELLOW}Usage:${NC}"
    echo -e "  $0 [build|run|stop|clean]"
    echo ""
    echo -e "${YELLOW}Commands:${NC}"
    echo -e "  build    Build the Docker image"
    echo -e "  run      Run the Docker container"
    echo -e "  stop     Stop the running container"
    echo -e "  clean    Remove the container and image"
    echo -e "  help     Show this help message"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo -e "  $0 build      # Build the Docker image"
    echo -e "  $0 run        # Run the Docker container"
}

# Function to build the Docker image
function build_image {
    echo -e "${GREEN}Building Docker image ${IMAGE_NAME}:${IMAGE_TAG}...${NC}"
    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Docker image built successfully!${NC}"
    else
        echo -e "${RED}Error building Docker image!${NC}"
        exit 1
    fi
}

# Function to run the Docker container
function run_container {
    echo -e "${GREEN}Running Docker container ${CONTAINER_NAME}...${NC}"
    
    # Check if the container is already running
    if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
        echo -e "${YELLOW}Container ${CONTAINER_NAME} is already running.${NC}"
        echo -e "${YELLOW}To stop it, run: $0 stop${NC}"
        return 1
    fi
    
    # Check if the container exists but is stopped
    if [ "$(docker ps -aq -f status=exited -f name=${CONTAINER_NAME})" ]; then
        echo -e "${YELLOW}Removing stopped container ${CONTAINER_NAME}...${NC}"
        docker rm ${CONTAINER_NAME}
    fi
    
    # Run the container
    docker run -d \
        --name ${CONTAINER_NAME} \
        -p ${PORT}:8501 \
        --restart unless-stopped \
        ${IMAGE_NAME}:${IMAGE_TAG}
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Container started successfully!${NC}"
        echo -e "${GREEN}Streamlit app is now running at http://localhost:${PORT}${NC}"
    else
        echo -e "${RED}Error starting container!${NC}"
        exit 1
    fi
}

# Function to stop the Docker container
function stop_container {
    echo -e "${GREEN}Stopping Docker container ${CONTAINER_NAME}...${NC}"
    
    # Check if the container is running
    if [ ! "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
        echo -e "${YELLOW}Container ${CONTAINER_NAME} is not running.${NC}"
        return 0
    fi
    
    # Stop the container
    docker stop ${CONTAINER_NAME}
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Container stopped successfully!${NC}"
    else
        echo -e "${RED}Error stopping container!${NC}"
        exit 1
    fi
}

# Function to clean up Docker resources
function clean_resources {
    echo -e "${GREEN}Cleaning up Docker resources...${NC}"
    
    # Stop the container if it's running
    if [ "$(docker ps -q -f name=${CONTAINER_NAME})" ]; then
        docker stop ${CONTAINER_NAME}
    fi
    
    # Remove the container if it exists
    if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
        docker rm ${CONTAINER_NAME}
    fi
    
    # Remove the image if it exists
    if [ "$(docker images -q ${IMAGE_NAME}:${IMAGE_TAG})" ]; then
        docker rmi ${IMAGE_NAME}:${IMAGE_TAG}
    fi
    
    echo -e "${GREEN}Cleanup complete!${NC}"
}

# Main function to handle commands
function main {
    case "$1" in
        build)
            build_image
            ;;
        run)
            run_container
            ;;
        stop)
            stop_container
            ;;
        clean)
            clean_resources
            ;;
        help)
            show_usage
            ;;
        *)
            echo -e "${RED}Invalid command: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
}

# Check if command is provided
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

# Execute main function with the provided command
main "$1"