#!/bin/bash

# Deployment Script for HFT Crypto Bot
# Handles development, staging, and production deployments

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="crypto-bot"
ENVIRONMENT=${ENVIRONMENT:-"development"}
COMPOSE_FILES=""
ENV_FILE=""

echo -e "${BLUE}🚀 HFT Crypto Bot Deployment${NC}"
echo -e "${BLUE}=============================${NC}"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] [COMMAND]"
    echo ""
    echo "Options:"
    echo "  -e, --env ENV       Set environment (development|staging|production)"
    echo "  -f, --file FILE     Additional compose file"
    echo "  --env-file FILE     Environment file to use"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Commands:"
    echo "  up                  Start all services"
    echo "  down                Stop all services"
    echo "  restart             Restart all services"
    echo "  build               Build all images"
    echo "  logs [SERVICE]      Show logs for service(s)"
    echo "  status              Show service status"
    echo "  health              Check service health"
    echo "  clean               Clean up containers and volumes"
    echo "  shell SERVICE       Open shell in service container"
    echo ""
    echo "Examples:"
    echo "  $0 up                           # Start development environment"
    echo "  $0 -e production up             # Start production environment"
    echo "  $0 logs execution-service       # Show execution service logs"
    echo "  $0 health                       # Check all service health"
}

# Function to set compose files based on environment
set_compose_files() {
    case $ENVIRONMENT in
        "development"|"dev")
            COMPOSE_FILES="-f docker-compose.yml -f docker-compose.dev.yml"
            ENV_FILE=".env.dev"
            ;;
        "staging"|"stage")
            COMPOSE_FILES="-f docker-compose.yml"
            ENV_FILE=".env.staging"
            ;;
        "production"|"prod")
            COMPOSE_FILES="-f docker-compose.yml -f docker-compose.prod.yml"
            ENV_FILE=".env.prod"
            ;;
        *)
            echo -e "${RED}❌ Unknown environment: ${ENVIRONMENT}${NC}"
            echo -e "Valid environments: development, staging, production"
            exit 1
            ;;
    esac
    
    # Add env file if it exists
    if [ -f "$ENV_FILE" ]; then
        COMPOSE_FILES="--env-file $ENV_FILE $COMPOSE_FILES"
    fi
}

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is not installed${NC}"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}❌ Docker Compose is not installed${NC}"
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}❌ Docker is not running${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Prerequisites check passed${NC}"
}

# Function to create required directories
create_directories() {
    echo -e "${YELLOW}📁 Creating required directories...${NC}"
    
    mkdir -p logs data
    mkdir -p monitoring/{prometheus,grafana,alertmanager}
    
    # Set permissions for data directory
    if [ "$ENVIRONMENT" = "production" ]; then
        chmod 755 data logs
    fi
    
    echo -e "${GREEN}✅ Directories created${NC}"
}

# Function to start services
start_services() {
    echo -e "${YELLOW}🚀 Starting services in ${ENVIRONMENT} mode...${NC}"
    
    # Pull latest images if in production
    if [ "$ENVIRONMENT" = "production" ]; then
        echo -e "${YELLOW}📥 Pulling latest images...${NC}"
        docker-compose $COMPOSE_FILES pull
    fi
    
    # Start services
    docker-compose $COMPOSE_FILES up -d
    
    echo -e "${GREEN}✅ Services started${NC}"
    
    # Show status
    show_status
}

# Function to stop services
stop_services() {
    echo -e "${YELLOW}🛑 Stopping services...${NC}"
    
    docker-compose $COMPOSE_FILES down
    
    echo -e "${GREEN}✅ Services stopped${NC}"
}

# Function to restart services
restart_services() {
    echo -e "${YELLOW}🔄 Restarting services...${NC}"
    
    docker-compose $COMPOSE_FILES restart
    
    echo -e "${GREEN}✅ Services restarted${NC}"
}

# Function to build images
build_images() {
    echo -e "${YELLOW}🏗️  Building images...${NC}"
    
    docker-compose $COMPOSE_FILES build --parallel
    
    echo -e "${GREEN}✅ Images built${NC}"
}

# Function to show logs
show_logs() {
    local service=$1
    
    if [ -n "$service" ]; then
        echo -e "${YELLOW}📋 Showing logs for ${service}...${NC}"
        docker-compose $COMPOSE_FILES logs -f "$service"
    else
        echo -e "${YELLOW}📋 Showing logs for all services...${NC}"
        docker-compose $COMPOSE_FILES logs -f
    fi
}

# Function to show status
show_status() {
    echo -e "${YELLOW}📊 Service Status:${NC}"
    docker-compose $COMPOSE_FILES ps
    
    echo -e "\n${YELLOW}💾 Volume Usage:${NC}"
    docker system df
}

# Function to check health
check_health() {
    echo -e "${YELLOW}🏥 Checking service health...${NC}"
    
    # Get list of services
    local services=$(docker-compose $COMPOSE_FILES ps --services)
    
    for service in $services; do
        local container_name=$(docker-compose $COMPOSE_FILES ps -q "$service")
        if [ -n "$container_name" ]; then
            local health=$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "no-healthcheck")
            local status=$(docker inspect --format='{{.State.Status}}' "$container_name" 2>/dev/null || echo "unknown")
            
            case $health in
                "healthy")
                    echo -e "  ${GREEN}✅ ${service}: ${status} (${health})${NC}"
                    ;;
                "unhealthy")
                    echo -e "  ${RED}❌ ${service}: ${status} (${health})${NC}"
                    ;;
                "starting")
                    echo -e "  ${YELLOW}⏳ ${service}: ${status} (${health})${NC}"
                    ;;
                "no-healthcheck")
                    echo -e "  ${BLUE}ℹ️  ${service}: ${status} (no healthcheck)${NC}"
                    ;;
                *)
                    echo -e "  ${YELLOW}❓ ${service}: ${status}${NC}"
                    ;;
            esac
        else
            echo -e "  ${RED}❌ ${service}: not running${NC}"
        fi
    done
}

# Function to clean up
clean_up() {
    echo -e "${YELLOW}🧹 Cleaning up...${NC}"
    
    # Stop and remove containers
    docker-compose $COMPOSE_FILES down -v --remove-orphans
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes (be careful in production)
    if [ "$ENVIRONMENT" != "production" ]; then
        docker volume prune -f
    fi
    
    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

# Function to open shell in container
open_shell() {
    local service=$1
    
    if [ -z "$service" ]; then
        echo -e "${RED}❌ Please specify a service name${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}🐚 Opening shell in ${service}...${NC}"
    docker-compose $COMPOSE_FILES exec "$service" /bin/bash
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -f|--file)
            COMPOSE_FILES="$COMPOSE_FILES -f $2"
            shift 2
            ;;
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        up|down|restart|build|logs|status|health|clean|shell)
            COMMAND="$1"
            shift
            break
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Default command
COMMAND=${COMMAND:-"up"}

# Set compose files based on environment
set_compose_files

echo -e "${BLUE}Configuration:${NC}"
echo -e "  Environment: ${ENVIRONMENT}"
echo -e "  Compose files: ${COMPOSE_FILES}"
echo -e "  Command: ${COMMAND}"

# Check prerequisites
check_prerequisites

# Create required directories
create_directories

# Execute command
case $COMMAND in
    "up")
        start_services
        ;;
    "down")
        stop_services
        ;;
    "restart")
        restart_services
        ;;
    "build")
        build_images
        ;;
    "logs")
        show_logs "$1"
        ;;
    "status")
        show_status
        ;;
    "health")
        check_health
        ;;
    "clean")
        clean_up
        ;;
    "shell")
        open_shell "$1"
        ;;
    *)
        echo -e "${RED}❌ Unknown command: ${COMMAND}${NC}"
        show_usage
        exit 1
        ;;
esac
