#!/bin/bash

# Docker Build Script for HFT Crypto Bot
# Builds all service images with proper tagging and optimization

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="crypto-bot"
VERSION=${VERSION:-"latest"}
REGISTRY=${REGISTRY:-""}
BUILD_ARGS=${BUILD_ARGS:-""}

echo -e "${BLUE}🐳 Building HFT Crypto Bot Docker Images${NC}"
echo -e "${BLUE}======================================${NC}"

# Function to build a service
build_service() {
    local service=$1
    local dockerfile_path=$2
    local context=${3:-"."}
    
    echo -e "\n${YELLOW}📦 Building ${service}...${NC}"
    
    local image_name="${PROJECT_NAME}-${service}:${VERSION}"
    if [ -n "$REGISTRY" ]; then
        image_name="${REGISTRY}/${image_name}"
    fi
    
    # Build the image
    if docker build \
        --file "${dockerfile_path}" \
        --tag "${image_name}" \
        --tag "${PROJECT_NAME}-${service}:latest" \
        ${BUILD_ARGS} \
        "${context}"; then
        echo -e "${GREEN}✅ Successfully built ${service}${NC}"
        
        # Show image size
        local size=$(docker images --format "table {{.Size}}" "${image_name}" | tail -n 1)
        echo -e "${BLUE}📏 Image size: ${size}${NC}"
    else
        echo -e "${RED}❌ Failed to build ${service}${NC}"
        exit 1
    fi
}

# Function to build all services
build_all_services() {
    echo -e "\n${YELLOW}🏗️  Building all HFT services...${NC}"
    
    # Build each service
    build_service "ingestion" "services/ingestion/Dockerfile"
    build_service "kalman-filter" "services/kalman_filter/Dockerfile"
    build_service "garch-model" "services/garch_model/Dockerfile"
    build_service "execution" "services/execution/Dockerfile"
    build_service "risk-manager" "services/risk_manager/Dockerfile"
    
    # Build Streamlit app
    build_service "streamlit" "Dockerfile.streamlit"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] [SERVICE]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -v, --version VER   Set version tag (default: latest)"
    echo "  -r, --registry REG  Set registry prefix"
    echo "  --no-cache          Build without cache"
    echo "  --parallel          Build services in parallel"
    echo ""
    echo "Services:"
    echo "  ingestion           Build ingestion service only"
    echo "  kalman-filter       Build Kalman filter service only"
    echo "  garch-model         Build GARCH model service only"
    echo "  execution           Build execution service only"
    echo "  risk-manager        Build risk manager service only"
    echo "  streamlit           Build Streamlit app only"
    echo "  all                 Build all services (default)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Build all services"
    echo "  $0 ingestion                          # Build ingestion service only"
    echo "  $0 --version v1.2.3 all              # Build all with version tag"
    echo "  $0 --registry myregistry.com all     # Build with registry prefix"
    echo "  $0 --no-cache execution               # Build execution without cache"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        --no-cache)
            BUILD_ARGS="${BUILD_ARGS} --no-cache"
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        ingestion|kalman-filter|garch-model|execution|risk-manager|streamlit)
            SERVICE="$1"
            shift
            ;;
        all)
            SERVICE="all"
            shift
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Default to building all services
SERVICE=${SERVICE:-"all"}

echo -e "${BLUE}Configuration:${NC}"
echo -e "  Project: ${PROJECT_NAME}"
echo -e "  Version: ${VERSION}"
echo -e "  Registry: ${REGISTRY:-"(none)"}"
echo -e "  Service: ${SERVICE}"
echo -e "  Build Args: ${BUILD_ARGS:-"(none)"}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Build based on service selection
case $SERVICE in
    "all")
        build_all_services
        ;;
    "ingestion")
        build_service "ingestion" "services/ingestion/Dockerfile"
        ;;
    "kalman-filter")
        build_service "kalman-filter" "services/kalman_filter/Dockerfile"
        ;;
    "garch-model")
        build_service "garch-model" "services/garch_model/Dockerfile"
        ;;
    "execution")
        build_service "execution" "services/execution/Dockerfile"
        ;;
    "risk-manager")
        build_service "risk-manager" "services/risk_manager/Dockerfile"
        ;;
    "streamlit")
        build_service "streamlit" "Dockerfile.streamlit"
        ;;
    *)
        echo -e "${RED}❌ Unknown service: ${SERVICE}${NC}"
        show_usage
        exit 1
        ;;
esac

echo -e "\n${GREEN}🎉 Build completed successfully!${NC}"

# Show final image list
echo -e "\n${BLUE}📋 Built images:${NC}"
docker images | grep "${PROJECT_NAME}" | head -10

echo -e "\n${YELLOW}💡 Next steps:${NC}"
echo -e "  • Run services: ${BLUE}docker-compose up -d${NC}"
echo -e "  • View logs: ${BLUE}docker-compose logs -f [service]${NC}"
echo -e "  • Check status: ${BLUE}docker-compose ps${NC}"
