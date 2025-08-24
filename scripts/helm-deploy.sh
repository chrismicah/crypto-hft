#!/bin/bash

# Helm Deployment Script for HFT Crypto Bot
# Handles deployment, testing, and management of the complete Helm-based application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CHART_PATH="infrastructure/helm/hft-crypto-bot"
RELEASE_NAME="hft-crypto-bot"
NAMESPACE="hft-production"
TIMEOUT="600s"
VALUES_FILE=""
DRY_RUN=false
UPGRADE=false
FORCE=false
DEBUG=false

echo -e "${BLUE}🚀 HFT Crypto Bot Helm Deployment${NC}"
echo -e "${BLUE}=================================${NC}"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] [COMMAND]"
    echo ""
    echo "Options:"
    echo "  -n, --namespace NS      Kubernetes namespace [default: hft-production]"
    echo "  -r, --release NAME      Helm release name [default: hft-crypto-bot]"
    echo "  -f, --values FILE       Values file to use"
    echo "  -t, --timeout DURATION  Timeout for deployment [default: 600s]"
    echo "  --dry-run               Perform a dry run"
    echo "  --upgrade               Upgrade existing release"
    echo "  --force                 Force deployment"
    echo "  --debug                 Enable debug mode"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Commands:"
    echo "  install                 Install the Helm chart (default)"
    echo "  upgrade                 Upgrade existing installation"
    echo "  uninstall               Uninstall the release"
    echo "  status                  Show release status"
    echo "  test                    Run Helm tests"
    echo "  lint                    Lint Helm charts"
    echo "  template                Generate templates"
    echo "  rollback VERSION        Rollback to specific version"
    echo "  load-test               Run load test on ingestion service"
    echo ""
    echo "Examples:"
    echo "  $0 install                              # Install with default values"
    echo "  $0 -f values-staging.yaml install      # Install with custom values"
    echo "  $0 --dry-run install                   # Perform dry run"
    echo "  $0 upgrade                              # Upgrade existing release"
    echo "  $0 test                                 # Run Helm tests"
    echo "  $0 load-test                           # Test HPA scaling"
}

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"
    
    # Check Helm
    if ! command -v helm &> /dev/null; then
        echo -e "${RED}❌ Helm is not installed${NC}"
        echo -e "${YELLOW}💡 Install Helm: https://helm.sh/docs/intro/install/${NC}"
        exit 1
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}❌ kubectl is not installed${NC}"
        exit 1
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        echo -e "${RED}❌ Cannot connect to Kubernetes cluster${NC}"
        exit 1
    fi
    
    # Check Helm chart exists
    if [ ! -f "$CHART_PATH/Chart.yaml" ]; then
        echo -e "${RED}❌ Helm chart not found at $CHART_PATH${NC}"
        exit 1
    fi
    
    # Check if metrics server is available (for HPA)
    if ! kubectl get apiservice v1beta1.metrics.k8s.io &> /dev/null; then
        echo -e "${YELLOW}⚠️  Metrics server not found - HPA may not work${NC}"
    fi
    
    echo -e "${GREEN}✅ Prerequisites check passed${NC}"
}

# Function to lint Helm charts
lint_charts() {
    echo -e "${YELLOW}🔍 Linting Helm charts...${NC}"
    
    # Lint main chart
    echo -e "${BLUE}📋 Linting main chart...${NC}"
    if helm lint "$CHART_PATH" ${VALUES_FILE:+--values $VALUES_FILE}; then
        echo -e "${GREEN}✅ Main chart lint passed${NC}"
    else
        echo -e "${RED}❌ Main chart lint failed${NC}"
        return 1
    fi
    
    # Lint individual service charts
    echo -e "${BLUE}📋 Linting service charts...${NC}"
    local charts_dir="infrastructure/helm/charts"
    local failed=0
    
    for chart_dir in "$charts_dir"/*; do
        if [ -d "$chart_dir" ] && [ -f "$chart_dir/Chart.yaml" ]; then
            chart_name=$(basename "$chart_dir")
            echo -e "${BLUE}  🔍 Linting $chart_name...${NC}"
            
            if helm lint "$chart_dir"; then
                echo -e "${GREEN}  ✅ $chart_name lint passed${NC}"
            else
                echo -e "${RED}  ❌ $chart_name lint failed${NC}"
                failed=1
            fi
        fi
    done
    
    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}🎉 All charts passed linting!${NC}"
        return 0
    else
        echo -e "${RED}💥 Some charts failed linting!${NC}"
        return 1
    fi
}

# Function to generate templates
generate_templates() {
    echo -e "${YELLOW}📄 Generating Helm templates...${NC}"
    
    local output_dir="generated-manifests"
    rm -rf "$output_dir"
    mkdir -p "$output_dir"
    
    helm template "$RELEASE_NAME" "$CHART_PATH" \
        --namespace "$NAMESPACE" \
        ${VALUES_FILE:+--values $VALUES_FILE} \
        --output-dir "$output_dir"
    
    echo -e "${GREEN}✅ Templates generated in $output_dir${NC}"
    echo -e "${BLUE}📂 Generated files:${NC}"
    find "$output_dir" -name "*.yaml" | sort
}

# Function to install Helm release
install_release() {
    echo -e "${YELLOW}🚀 Installing Helm release...${NC}"
    
    local cmd_args=(
        "install" "$RELEASE_NAME" "$CHART_PATH"
        "--namespace" "$NAMESPACE"
        "--create-namespace"
        "--timeout" "$TIMEOUT"
        "--wait"
    )
    
    if [ -n "$VALUES_FILE" ]; then
        cmd_args+=("--values" "$VALUES_FILE")
    fi
    
    if [ "$DRY_RUN" = true ]; then
        cmd_args+=("--dry-run")
    fi
    
    if [ "$DEBUG" = true ]; then
        cmd_args+=("--debug")
    fi
    
    echo -e "${BLUE}📋 Running: helm ${cmd_args[*]}${NC}"
    
    if helm "${cmd_args[@]}"; then
        echo -e "${GREEN}✅ Helm release installed successfully!${NC}"
        
        if [ "$DRY_RUN" = false ]; then
            show_deployment_info
        fi
        return 0
    else
        echo -e "${RED}❌ Helm release installation failed!${NC}"
        return 1
    fi
}

# Function to upgrade Helm release
upgrade_release() {
    echo -e "${YELLOW}⬆️  Upgrading Helm release...${NC}"
    
    local cmd_args=(
        "upgrade" "$RELEASE_NAME" "$CHART_PATH"
        "--namespace" "$NAMESPACE"
        "--timeout" "$TIMEOUT"
        "--wait"
    )
    
    if [ -n "$VALUES_FILE" ]; then
        cmd_args+=("--values" "$VALUES_FILE")
    fi
    
    if [ "$DRY_RUN" = true ]; then
        cmd_args+=("--dry-run")
    fi
    
    if [ "$FORCE" = true ]; then
        cmd_args+=("--force")
    fi
    
    if [ "$DEBUG" = true ]; then
        cmd_args+=("--debug")
    fi
    
    echo -e "${BLUE}📋 Running: helm ${cmd_args[*]}${NC}"
    
    if helm "${cmd_args[@]}"; then
        echo -e "${GREEN}✅ Helm release upgraded successfully!${NC}"
        
        if [ "$DRY_RUN" = false ]; then
            show_deployment_info
        fi
        return 0
    else
        echo -e "${RED}❌ Helm release upgrade failed!${NC}"
        return 1
    fi
}

# Function to show deployment info
show_deployment_info() {
    echo -e "\n${YELLOW}📊 Deployment Information:${NC}"
    
    # Show release status
    echo -e "${BLUE}🏷️  Release Status:${NC}"
    helm status "$RELEASE_NAME" --namespace "$NAMESPACE"
    
    # Show pods
    echo -e "\n${BLUE}🐳 Pods:${NC}"
    kubectl get pods -n "$NAMESPACE" -o wide
    
    # Show services
    echo -e "\n${BLUE}🌐 Services:${NC}"
    kubectl get services -n "$NAMESPACE" -o wide
    
    # Show HPAs
    echo -e "\n${BLUE}📈 Horizontal Pod Autoscalers:${NC}"
    kubectl get hpa -n "$NAMESPACE" || echo "No HPAs found"
    
    # Show PVCs
    echo -e "\n${BLUE}💾 Persistent Volume Claims:${NC}"
    kubectl get pvc -n "$NAMESPACE" || echo "No PVCs found"
}

# Function to uninstall release
uninstall_release() {
    echo -e "${YELLOW}🗑️  Uninstalling Helm release...${NC}"
    
    if helm uninstall "$RELEASE_NAME" --namespace "$NAMESPACE" --timeout "$TIMEOUT"; then
        echo -e "${GREEN}✅ Helm release uninstalled successfully!${NC}"
        
        # Optionally delete namespace
        read -p "Delete namespace $NAMESPACE? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kubectl delete namespace "$NAMESPACE" --ignore-not-found=true
            echo -e "${GREEN}✅ Namespace $NAMESPACE deleted${NC}"
        fi
        
        return 0
    else
        echo -e "${RED}❌ Helm release uninstall failed!${NC}"
        return 1
    fi
}

# Function to show release status
show_status() {
    echo -e "${YELLOW}📊 Release Status:${NC}"
    
    if helm status "$RELEASE_NAME" --namespace "$NAMESPACE"; then
        show_deployment_info
    else
        echo -e "${RED}❌ Release not found or not accessible${NC}"
        return 1
    fi
}

# Function to run Helm tests
run_tests() {
    echo -e "${YELLOW}🧪 Running Helm tests...${NC}"
    
    if helm test "$RELEASE_NAME" --namespace "$NAMESPACE" --timeout "$TIMEOUT"; then
        echo -e "${GREEN}✅ Helm tests passed!${NC}"
        return 0
    else
        echo -e "${RED}❌ Helm tests failed!${NC}"
        return 1
    fi
}

# Function to rollback release
rollback_release() {
    local version=$1
    
    if [ -z "$version" ]; then
        echo -e "${RED}❌ Please specify version to rollback to${NC}"
        echo -e "${YELLOW}💡 Available versions:${NC}"
        helm history "$RELEASE_NAME" --namespace "$NAMESPACE"
        return 1
    fi
    
    echo -e "${YELLOW}🔄 Rolling back to version $version...${NC}"
    
    if helm rollback "$RELEASE_NAME" "$version" --namespace "$NAMESPACE" --timeout "$TIMEOUT" --wait; then
        echo -e "${GREEN}✅ Rollback completed successfully!${NC}"
        show_deployment_info
        return 0
    else
        echo -e "${RED}❌ Rollback failed!${NC}"
        return 1
    fi
}

# Function to run load test
run_load_test() {
    echo -e "${YELLOW}🔥 Running load test on ingestion service...${NC}"
    
    # Check if ingestion service is running
    if ! kubectl get deployment ingestion-service -n "$NAMESPACE" &> /dev/null; then
        echo -e "${RED}❌ Ingestion service not found${NC}"
        return 1
    fi
    
    echo -e "${BLUE}📊 Current HPA status:${NC}"
    kubectl get hpa -n "$NAMESPACE" | grep ingestion || echo "No ingestion HPA found"
    
    echo -e "${BLUE}📊 Current pod count:${NC}"
    initial_pods=$(kubectl get pods -n "$NAMESPACE" -l app=ingestion-service --no-headers | wc -l)
    echo "Initial pods: $initial_pods"
    
    # Create load test job
    echo -e "${BLUE}🚀 Creating load test job...${NC}"
    
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ingestion-load-test
  namespace: $NAMESPACE
spec:
  template:
    spec:
      containers:
      - name: load-test
        image: curlimages/curl:latest
        command:
        - /bin/sh
        - -c
        - |
          echo "Starting load test..."
          for i in \$(seq 1 300); do
            curl -s http://ingestion-service:8000/health > /dev/null &
            curl -s http://ingestion-service:8000/metrics > /dev/null &
            if [ \$((i % 10)) -eq 0 ]; then
              echo "Completed \$i requests"
            fi
            sleep 0.1
          done
          wait
          echo "Load test completed"
      restartPolicy: Never
  backoffLimit: 1
EOF
    
    # Monitor the load test and HPA scaling
    echo -e "${BLUE}📈 Monitoring HPA scaling...${NC}"
    
    for i in {1..60}; do
        current_pods=$(kubectl get pods -n "$NAMESPACE" -l app=ingestion-service --no-headers | wc -l)
        hpa_status=$(kubectl get hpa -n "$NAMESPACE" | grep ingestion | awk '{print $3"/"$4" "$5" "$6}' || echo "N/A")
        
        echo "Time: ${i}s | Pods: $current_pods | HPA: $hpa_status"
        
        # Check if scaling occurred
        if [ "$current_pods" -gt "$initial_pods" ]; then
            echo -e "${GREEN}✅ HPA scaling detected! Pods increased from $initial_pods to $current_pods${NC}"
            scaling_detected=true
        fi
        
        sleep 5
    done
    
    # Clean up load test job
    echo -e "${BLUE}🧹 Cleaning up load test job...${NC}"
    kubectl delete job ingestion-load-test -n "$NAMESPACE" --ignore-not-found=true
    
    # Wait for scale down
    echo -e "${BLUE}⬇️  Waiting for scale down (this may take a few minutes)...${NC}"
    sleep 60
    
    final_pods=$(kubectl get pods -n "$NAMESPACE" -l app=ingestion-service --no-headers | wc -l)
    echo "Final pods: $final_pods"
    
    if [ "${scaling_detected:-false}" = true ]; then
        echo -e "${GREEN}🎉 Load test completed successfully! HPA scaling verified.${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  Load test completed but no scaling detected${NC}"
        echo -e "${YELLOW}💡 Check HPA configuration and metrics server${NC}"
        return 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -r|--release)
            RELEASE_NAME="$2"
            shift 2
            ;;
        -f|--values)
            VALUES_FILE="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --upgrade)
            UPGRADE=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        install|upgrade|uninstall|status|test|lint|template|rollback|load-test)
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
COMMAND=${COMMAND:-"install"}

echo -e "${BLUE}Configuration:${NC}"
echo -e "  Release Name: ${RELEASE_NAME}"
echo -e "  Namespace: ${NAMESPACE}"
echo -e "  Chart Path: ${CHART_PATH}"
echo -e "  Timeout: ${TIMEOUT}"
echo -e "  Values File: ${VALUES_FILE:-"default"}"
echo -e "  Command: ${COMMAND}"
echo ""

# Check prerequisites
check_prerequisites

# Execute command
case $COMMAND in
    "install")
        if [ "$UPGRADE" = true ] || helm status "$RELEASE_NAME" --namespace "$NAMESPACE" &> /dev/null; then
            upgrade_release
        else
            install_release
        fi
        ;;
    "upgrade")
        upgrade_release
        ;;
    "uninstall")
        uninstall_release
        ;;
    "status")
        show_status
        ;;
    "test")
        run_tests
        ;;
    "lint")
        lint_charts
        ;;
    "template")
        generate_templates
        ;;
    "rollback")
        rollback_release "$1"
        ;;
    "load-test")
        run_load_test
        ;;
    *)
        echo -e "${RED}❌ Unknown command: ${COMMAND}${NC}"
        show_usage
        exit 1
        ;;
esac
