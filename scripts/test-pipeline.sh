#!/bin/bash

# Pipeline Testing Script
# Simulates GitHub Actions pipeline stages locally for testing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.11"
TEST_COVERAGE_THRESHOLD=80

echo -e "${BLUE}🧪 HFT Crypto Bot Pipeline Testing${NC}"
echo -e "${BLUE}===================================${NC}"

# Function to show stage header
show_stage() {
    echo -e "\n${YELLOW}📋 Stage: $1${NC}"
    echo -e "${YELLOW}$(printf '=%.0s' {1..40})${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    show_stage "Prerequisites Check"
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VER=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        echo -e "${GREEN}✅ Python ${PYTHON_VER} installed${NC}"
    else
        echo -e "${RED}❌ Python 3 not found${NC}"
        exit 1
    fi
    
    # Check pip
    if command -v pip3 &> /dev/null; then
        echo -e "${GREEN}✅ pip3 available${NC}"
    else
        echo -e "${RED}❌ pip3 not found${NC}"
        exit 1
    fi
    
    # Check Docker
    if command -v docker &> /dev/null; then
        echo -e "${GREEN}✅ Docker available${NC}"
    else
        echo -e "${YELLOW}⚠️  Docker not found (build stage will be skipped)${NC}"
    fi
    
    # Check kubectl (optional)
    if command -v kubectl &> /dev/null; then
        echo -e "${GREEN}✅ kubectl available${NC}"
    else
        echo -e "${YELLOW}⚠️  kubectl not found (deploy stage will be skipped)${NC}"
    fi
}

# Function to install dependencies
install_dependencies() {
    show_stage "Install Dependencies"
    
    echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
    pip3 install -r requirements.txt
    
    echo -e "${BLUE}📦 Installing development dependencies...${NC}"
    pip3 install black isort flake8 mypy bandit safety pytest pytest-cov pytest-asyncio pytest-mock
    
    echo -e "${GREEN}✅ Dependencies installed${NC}"
}

# Function to run linting stage
run_linting() {
    show_stage "Code Quality & Linting"
    
    local failed=0
    
    # Black formatting check
    echo -e "${BLUE}🖤 Running Black (code formatting)...${NC}"
    if black --check --diff services/ common/ tests/; then
        echo -e "${GREEN}✅ Black: Code formatting is correct${NC}"
    else
        echo -e "${RED}❌ Black: Code formatting issues found${NC}"
        failed=1
    fi
    
    # isort import sorting check
    echo -e "${BLUE}📋 Running isort (import sorting)...${NC}"
    if isort --check-only --diff services/ common/ tests/; then
        echo -e "${GREEN}✅ isort: Import sorting is correct${NC}"
    else
        echo -e "${RED}❌ isort: Import sorting issues found${NC}"
        failed=1
    fi
    
    # flake8 linting
    echo -e "${BLUE}🔍 Running flake8 (linting)...${NC}"
    if flake8 services/ common/ tests/ --max-line-length=100 --extend-ignore=E203,W503 --statistics; then
        echo -e "${GREEN}✅ flake8: No linting issues found${NC}"
    else
        echo -e "${RED}❌ flake8: Linting issues found${NC}"
        failed=1
    fi
    
    # mypy type checking
    echo -e "${BLUE}🔢 Running mypy (type checking)...${NC}"
    if mypy services/ common/ --ignore-missing-imports --no-strict-optional; then
        echo -e "${GREEN}✅ mypy: Type checking passed${NC}"
    else
        echo -e "${YELLOW}⚠️  mypy: Type checking issues found (non-blocking)${NC}"
    fi
    
    # bandit security scanning
    echo -e "${BLUE}🔒 Running bandit (security scan)...${NC}"
    if bandit -r services/ common/ -f json -o bandit-report.json; then
        echo -e "${GREEN}✅ bandit: No security issues found${NC}"
    else
        echo -e "${YELLOW}⚠️  bandit: Security issues found (check bandit-report.json)${NC}"
    fi
    
    # safety dependency check
    echo -e "${BLUE}🛡️  Running safety (dependency vulnerability check)...${NC}"
    if safety check --json --output safety-report.json; then
        echo -e "${GREEN}✅ safety: No vulnerable dependencies found${NC}"
    else
        echo -e "${YELLOW}⚠️  safety: Vulnerable dependencies found (check safety-report.json)${NC}"
    fi
    
    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}🎉 Linting stage passed!${NC}"
        return 0
    else
        echo -e "${RED}💥 Linting stage failed!${NC}"
        return 1
    fi
}

# Function to run tests
run_tests() {
    show_stage "Unit & Integration Tests"
    
    # Create test environment
    echo -e "${BLUE}🏗️  Setting up test environment...${NC}"
    mkdir -p logs data
    
    # Check if Redis is available for integration tests
    if command -v redis-server &> /dev/null && pgrep redis-server > /dev/null; then
        echo -e "${GREEN}✅ Redis server is running${NC}"
        export REDIS_URL="redis://localhost:6379"
    else
        echo -e "${YELLOW}⚠️  Redis server not running, integration tests may fail${NC}"
        export REDIS_URL="redis://localhost:6379"
    fi
    
    # Run tests with coverage
    echo -e "${BLUE}🧪 Running tests with coverage...${NC}"
    if pytest tests/ -v \
        --cov=services --cov=common \
        --cov-report=xml --cov-report=html \
        --cov-report=term-missing \
        --junitxml=pytest-report.xml \
        --cov-fail-under=${TEST_COVERAGE_THRESHOLD}; then
        echo -e "${GREEN}✅ Tests passed with sufficient coverage!${NC}"
        return 0
    else
        echo -e "${RED}❌ Tests failed or coverage below ${TEST_COVERAGE_THRESHOLD}%${NC}"
        return 1
    fi
}

# Function to validate Docker builds
validate_docker_builds() {
    show_stage "Docker Build Validation"
    
    if ! command -v docker &> /dev/null; then
        echo -e "${YELLOW}⚠️  Docker not available, skipping build validation${NC}"
        return 0
    fi
    
    local services=("ingestion" "kalman_filter" "garch_model" "execution" "risk_manager" "orchestrator")
    local failed=0
    
    for service in "${services[@]}"; do
        echo -e "${BLUE}🐳 Validating Docker build for ${service}...${NC}"
        
        if docker build --dry-run -f "services/${service}/Dockerfile" . > /dev/null 2>&1; then
            echo -e "${GREEN}✅ ${service}: Dockerfile is valid${NC}"
        else
            echo -e "${RED}❌ ${service}: Dockerfile validation failed${NC}"
            failed=1
        fi
    done
    
    # Validate monitoring Dockerfiles
    echo -e "${BLUE}🐳 Validating Streamlit Dockerfile...${NC}"
    if docker build --dry-run -f "Dockerfile.streamlit" . > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Streamlit: Dockerfile is valid${NC}"
    else
        echo -e "${RED}❌ Streamlit: Dockerfile validation failed${NC}"
        failed=1
    fi
    
    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}🎉 Docker build validation passed!${NC}"
        return 0
    else
        echo -e "${RED}💥 Docker build validation failed!${NC}"
        return 1
    fi
}

# Function to validate Kubernetes manifests
validate_k8s_manifests() {
    show_stage "Kubernetes Manifest Validation"
    
    echo -e "${BLUE}📋 Validating Kubernetes manifests...${NC}"
    
    # Validate YAML syntax
    local failed=0
    for env in staging production; do
        echo -e "${BLUE}🔍 Validating ${env} manifests...${NC}"
        
        for file in k8s/${env}/*.yaml; do
            if [ -f "$file" ]; then
                if python3 -c "import yaml; yaml.safe_load(open('$file', 'r'))" 2>/dev/null; then
                    echo -e "${GREEN}✅ $(basename $file): Valid YAML${NC}"
                else
                    echo -e "${RED}❌ $(basename $file): Invalid YAML${NC}"
                    failed=1
                fi
            fi
        done
    done
    
    # If kubectl is available, validate against cluster
    if command -v kubectl &> /dev/null; then
        echo -e "${BLUE}🔍 Validating against Kubernetes API...${NC}"
        
        for file in k8s/staging/*.yaml; do
            if [ -f "$file" ]; then
                if kubectl apply --dry-run=client -f "$file" > /dev/null 2>&1; then
                    echo -e "${GREEN}✅ $(basename $file): Valid Kubernetes manifest${NC}"
                else
                    echo -e "${YELLOW}⚠️  $(basename $file): Kubectl validation warning${NC}"
                fi
            fi
        done
    else
        echo -e "${YELLOW}⚠️  kubectl not available, skipping API validation${NC}"
    fi
    
    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}🎉 Kubernetes manifest validation passed!${NC}"
        return 0
    else
        echo -e "${RED}💥 Kubernetes manifest validation failed!${NC}"
        return 1
    fi
}

# Function to simulate security scanning
simulate_security_scan() {
    show_stage "Security Scanning Simulation"
    
    echo -e "${BLUE}🔒 Simulating container security scanning...${NC}"
    
    # Check if security reports exist
    if [ -f "bandit-report.json" ]; then
        echo -e "${GREEN}✅ Bandit security report found${NC}"
        
        # Parse bandit report for critical issues
        if python3 -c "
import json
try:
    with open('bandit-report.json', 'r') as f:
        data = json.load(f)
        high_severity = [r for r in data.get('results', []) if r.get('issue_severity') == 'HIGH']
        if high_severity:
            print(f'❌ Found {len(high_severity)} high severity security issues')
            exit(1)
        else:
            print('✅ No high severity security issues found')
except Exception as e:
    print(f'⚠️  Could not parse bandit report: {e}')
"; then
            echo -e "${GREEN}✅ Security scan: No critical issues${NC}"
        else
            echo -e "${RED}❌ Security scan: Critical issues found${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠️  No bandit report found${NC}"
    fi
    
    if [ -f "safety-report.json" ]; then
        echo -e "${GREEN}✅ Safety vulnerability report found${NC}"
    else
        echo -e "${YELLOW}⚠️  No safety report found${NC}"
    fi
    
    echo -e "${GREEN}🎉 Security scanning simulation completed!${NC}"
    return 0
}

# Function to show summary
show_summary() {
    show_stage "Pipeline Summary"
    
    echo -e "${BLUE}📊 Test Results:${NC}"
    if [ -f "pytest-report.xml" ]; then
        echo -e "${GREEN}✅ Test report: pytest-report.xml${NC}"
    fi
    
    if [ -f "coverage.xml" ]; then
        echo -e "${GREEN}✅ Coverage report: coverage.xml${NC}"
    fi
    
    if [ -d "htmlcov" ]; then
        echo -e "${GREEN}✅ HTML coverage: htmlcov/index.html${NC}"
    fi
    
    if [ -f "bandit-report.json" ]; then
        echo -e "${GREEN}✅ Security report: bandit-report.json${NC}"
    fi
    
    if [ -f "safety-report.json" ]; then
        echo -e "${GREEN}✅ Vulnerability report: safety-report.json${NC}"
    fi
    
    echo -e "\n${BLUE}📂 Generated Artifacts:${NC}"
    ls -la *.xml *.json htmlcov/ 2>/dev/null || echo "No artifacts generated"
    
    echo -e "\n${GREEN}🎉 Pipeline testing completed!${NC}"
    echo -e "${BLUE}💡 To clean up: rm -rf htmlcov/ *.xml *.json logs/ data/${NC}"
}

# Function to clean up
cleanup() {
    if [ "$1" = "--clean" ]; then
        echo -e "${YELLOW}🧹 Cleaning up artifacts...${NC}"
        rm -rf htmlcov/ *.xml *.json logs/ data/ .coverage
        echo -e "${GREEN}✅ Cleanup completed${NC}"
        exit 0
    fi
}

# Main execution
main() {
    # Check for cleanup flag
    cleanup "$1"
    
    # Run pipeline stages
    local failed=0
    
    check_prerequisites || failed=1
    
    if [ $failed -eq 0 ]; then
        install_dependencies || failed=1
    fi
    
    if [ $failed -eq 0 ]; then
        run_linting || failed=1
    fi
    
    if [ $failed -eq 0 ]; then
        run_tests || failed=1
    fi
    
    if [ $failed -eq 0 ]; then
        validate_docker_builds || failed=1
    fi
    
    if [ $failed -eq 0 ]; then
        validate_k8s_manifests || failed=1
    fi
    
    if [ $failed -eq 0 ]; then
        simulate_security_scan || failed=1
    fi
    
    show_summary
    
    if [ $failed -eq 0 ]; then
        echo -e "\n${GREEN}🎉 All pipeline stages passed!${NC}"
        echo -e "${GREEN}✅ Ready for production deployment${NC}"
        exit 0
    else
        echo -e "\n${RED}💥 Pipeline failed!${NC}"
        echo -e "${RED}❌ Fix issues before deploying${NC}"
        exit 1
    fi
}

# Show usage
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --clean    Clean up generated artifacts"
    echo "  --help     Show this help message"
    echo ""
    echo "This script simulates the GitHub Actions CI/CD pipeline locally."
    echo "It runs linting, testing, and validation stages."
    exit 0
fi

# Run main function
main "$@"
