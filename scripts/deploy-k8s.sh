#!/bin/bash

# Kubernetes Deployment Script for HFT Crypto Bot
# Deploys the HFT system to a Kubernetes cluster

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${ENVIRONMENT:-"staging"}
NAMESPACE="hft-${ENVIRONMENT}"
IMAGE_TAG=${IMAGE_TAG:-"latest"}
REGISTRY=${REGISTRY:-"docker.io"}
IMAGE_PREFIX=${IMAGE_PREFIX:-"hft-crypto-bot"}

echo -e "${BLUE}🚀 Deploying HFT Crypto Bot to Kubernetes${NC}"
echo -e "${BLUE}==========================================${NC}"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] [COMMAND]"
    echo ""
    echo "Options:"
    echo "  -e, --env ENV       Environment (staging|production) [default: staging]"
    echo "  -t, --tag TAG       Image tag [default: latest]"
    echo "  -r, --registry REG  Docker registry [default: docker.io]"
    echo "  -n, --namespace NS  Kubernetes namespace [default: hft-ENV]"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Commands:"
    echo "  deploy              Deploy all services (default)"
    echo "  undeploy            Remove all deployments"
    echo "  status              Show deployment status"
    echo "  logs SERVICE        Show logs for specific service"
    echo "  rollback            Rollback to previous version"
    echo "  scale SERVICE N     Scale service to N replicas"
    echo ""
    echo "Examples:"
    echo "  $0 deploy                           # Deploy staging environment"
    echo "  $0 -e production deploy             # Deploy production environment"
    echo "  $0 -t v1.2.3 deploy                # Deploy specific image tag"
    echo "  $0 status                           # Check deployment status"
    echo "  $0 logs orchestrator-service        # Show orchestrator logs"
    echo "  $0 scale execution-service 3        # Scale execution service to 3 replicas"
}

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}❌ kubectl is not installed${NC}"
        exit 1
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        echo -e "${RED}❌ Cannot connect to Kubernetes cluster${NC}"
        echo -e "${YELLOW}💡 Please configure kubectl to connect to your cluster${NC}"
        exit 1
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        echo -e "${YELLOW}⚠️  Namespace ${NAMESPACE} does not exist, will create it${NC}"
    fi
    
    echo -e "${GREEN}✅ Prerequisites check passed${NC}"
}

# Function to update image tags
update_image_tags() {
    echo -e "${YELLOW}🏷️  Updating image tags to ${IMAGE_TAG}...${NC}"
    
    local manifests_dir="k8s/${ENVIRONMENT}"
    
    # Create temporary directory for modified manifests
    local temp_dir=$(mktemp -d)
    cp -r "$manifests_dir"/* "$temp_dir"/
    
    # Update image tags in deployment manifests
    find "$temp_dir" -name "*.yaml" -exec sed -i.bak \
        "s|${REGISTRY}/${IMAGE_PREFIX}-\([^:]*\):latest|${REGISTRY}/${IMAGE_PREFIX}-\1:${IMAGE_TAG}|g" {} \;
    
    # Clean up backup files
    find "$temp_dir" -name "*.bak" -delete
    
    echo "$temp_dir"
}

# Function to deploy services
deploy_services() {
    echo -e "${YELLOW}🚀 Deploying services to ${ENVIRONMENT}...${NC}"
    
    local manifests_dir=$(update_image_tags)
    
    # Apply manifests in order
    echo -e "${BLUE}📋 Applying Kubernetes manifests...${NC}"
    
    # 1. Namespace and basic resources
    kubectl apply -f "$manifests_dir/namespace.yaml"
    kubectl apply -f "$manifests_dir/configmap.yaml"
    kubectl apply -f "$manifests_dir/secrets.yaml"
    
    # 2. Infrastructure services
    kubectl apply -f "$manifests_dir/infrastructure.yaml"
    
    # 3. Core HFT services
    kubectl apply -f "$manifests_dir/services.yaml"
    
    # 4. Monitoring stack
    kubectl apply -f "$manifests_dir/monitoring.yaml"
    
    echo -e "${GREEN}✅ Manifests applied successfully${NC}"
    
    # Wait for deployments to be ready
    echo -e "${YELLOW}⏳ Waiting for deployments to be ready...${NC}"
    
    local services=(
        "redis"
        "kafka"
        "ingestion-service"
        "kalman-filter-service"
        "garch-model-service"
        "execution-service"
        "risk-manager-service"
        "orchestrator-service"
    )
    
    for service in "${services[@]}"; do
        echo -e "${BLUE}⏳ Waiting for ${service}...${NC}"
        kubectl rollout status deployment/"$service" -n "$NAMESPACE" --timeout=600s
    done
    
    echo -e "${GREEN}🎉 All services deployed successfully!${NC}"
    
    # Cleanup temp directory
    rm -rf "$manifests_dir"
    
    # Show final status
    show_status
}

# Function to undeploy services
undeploy_services() {
    echo -e "${YELLOW}🗑️  Undeploying services from ${ENVIRONMENT}...${NC}"
    
    # Delete in reverse order
    kubectl delete -f "k8s/${ENVIRONMENT}/monitoring.yaml" --ignore-not-found=true
    kubectl delete -f "k8s/${ENVIRONMENT}/services.yaml" --ignore-not-found=true
    kubectl delete -f "k8s/${ENVIRONMENT}/infrastructure.yaml" --ignore-not-found=true
    kubectl delete -f "k8s/${ENVIRONMENT}/configmap.yaml" --ignore-not-found=true
    kubectl delete -f "k8s/${ENVIRONMENT}/secrets.yaml" --ignore-not-found=true
    
    # Optionally delete namespace (ask user)
    read -p "Delete namespace ${NAMESPACE}? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
        echo -e "${GREEN}✅ Namespace ${NAMESPACE} deleted${NC}"
    fi
    
    echo -e "${GREEN}✅ Services undeployed successfully${NC}"
}

# Function to show status
show_status() {
    echo -e "${YELLOW}📊 Deployment Status for ${ENVIRONMENT}:${NC}"
    echo ""
    
    # Check namespace
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        echo -e "${GREEN}✅ Namespace: ${NAMESPACE}${NC}"
    else
        echo -e "${RED}❌ Namespace: ${NAMESPACE} (not found)${NC}"
        return 1
    fi
    
    # Check deployments
    echo -e "\n${BLUE}🚀 Deployments:${NC}"
    kubectl get deployments -n "$NAMESPACE" -o wide
    
    # Check services
    echo -e "\n${BLUE}🌐 Services:${NC}"
    kubectl get services -n "$NAMESPACE" -o wide
    
    # Check pods
    echo -e "\n${BLUE}🐳 Pods:${NC}"
    kubectl get pods -n "$NAMESPACE" -o wide
    
    # Check health endpoints
    echo -e "\n${BLUE}🏥 Health Checks:${NC}"
    check_service_health
    
    # Show resource usage
    echo -e "\n${BLUE}📈 Resource Usage:${NC}"
    kubectl top pods -n "$NAMESPACE" 2>/dev/null || echo "Metrics server not available"
}

# Function to check service health
check_service_health() {
    local services=(
        "ingestion-service:8000"
        "kalman-filter-service:8001"
        "garch-model-service:8002"
        "execution-service:8003"
        "risk-manager-service:8004"
        "orchestrator-service:8005"
    )
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r service_name port <<< "$service_info"
        
        # Try to port-forward and check health
        local pod=$(kubectl get pods -n "$NAMESPACE" -l app="$service_name" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
        
        if [ -n "$pod" ]; then
            if kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.status.phase}' | grep -q "Running"; then
                echo -e "  ${GREEN}✅ ${service_name}: Running${NC}"
            else
                echo -e "  ${YELLOW}⏳ ${service_name}: Starting${NC}"
            fi
        else
            echo -e "  ${RED}❌ ${service_name}: Not found${NC}"
        fi
    done
}

# Function to show logs
show_logs() {
    local service=$1
    
    if [ -z "$service" ]; then
        echo -e "${RED}❌ Please specify a service name${NC}"
        echo -e "${YELLOW}Available services: ingestion-service, kalman-filter-service, garch-model-service, execution-service, risk-manager-service, orchestrator-service${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}📋 Showing logs for ${service}...${NC}"
    kubectl logs -f deployment/"$service" -n "$NAMESPACE"
}

# Function to rollback deployment
rollback_deployment() {
    echo -e "${YELLOW}🔄 Rolling back deployment...${NC}"
    
    local services=(
        "ingestion-service"
        "kalman-filter-service"
        "garch-model-service"
        "execution-service"
        "risk-manager-service"
        "orchestrator-service"
    )
    
    for service in "${services[@]}"; do
        echo -e "${BLUE}🔄 Rolling back ${service}...${NC}"
        kubectl rollout undo deployment/"$service" -n "$NAMESPACE"
        kubectl rollout status deployment/"$service" -n "$NAMESPACE" --timeout=300s
    done
    
    echo -e "${GREEN}✅ Rollback completed${NC}"
}

# Function to scale service
scale_service() {
    local service=$1
    local replicas=$2
    
    if [ -z "$service" ] || [ -z "$replicas" ]; then
        echo -e "${RED}❌ Please specify service name and replica count${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}📈 Scaling ${service} to ${replicas} replicas...${NC}"
    kubectl scale deployment/"$service" --replicas="$replicas" -n "$NAMESPACE"
    kubectl rollout status deployment/"$service" -n "$NAMESPACE" --timeout=300s
    
    echo -e "${GREEN}✅ ${service} scaled to ${replicas} replicas${NC}"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            NAMESPACE="hft-${ENVIRONMENT}"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        deploy|undeploy|status|logs|rollback|scale)
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
COMMAND=${COMMAND:-"deploy"}

echo -e "${BLUE}Configuration:${NC}"
echo -e "  Environment: ${ENVIRONMENT}"
echo -e "  Namespace: ${NAMESPACE}"
echo -e "  Image Tag: ${IMAGE_TAG}"
echo -e "  Registry: ${REGISTRY}"
echo -e "  Command: ${COMMAND}"
echo ""

# Check prerequisites
check_prerequisites

# Execute command
case $COMMAND in
    "deploy")
        deploy_services
        ;;
    "undeploy")
        undeploy_services
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs "$1"
        ;;
    "rollback")
        rollback_deployment
        ;;
    "scale")
        scale_service "$1" "$2"
        ;;
    *)
        echo -e "${RED}❌ Unknown command: ${COMMAND}${NC}"
        show_usage
        exit 1
        ;;
esac
