#!/bin/bash

# Health Check Script for HFT Crypto Bot
# Verifies all services are running and healthy

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TIMEOUT=10
VERBOSE=false
ENVIRONMENT="development"

echo -e "${BLUE}🏥 HFT Crypto Bot Health Check${NC}"
echo -e "${BLUE}==============================${NC}"

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -e, --env ENV       Environment (development|production)"
    echo "  -t, --timeout SEC   Request timeout in seconds (default: 10)"
    echo "  -v, --verbose       Verbose output"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                  # Check development environment"
    echo "  $0 -e production    # Check production environment"
    echo "  $0 -v -t 30         # Verbose output with 30s timeout"
}

# Function to check HTTP endpoint
check_http_endpoint() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}
    
    if $VERBOSE; then
        echo -e "${YELLOW}  Checking ${name} at ${url}...${NC}"
    fi
    
    local response=$(curl -s -w "%{http_code}" -m $TIMEOUT "$url" 2>/dev/null || echo "000")
    local http_code="${response: -3}"
    local body="${response%???}"
    
    if [ "$http_code" = "$expected_status" ]; then
        echo -e "  ${GREEN}✅ ${name}: OK (${http_code})${NC}"
        if $VERBOSE && [ -n "$body" ]; then
            echo -e "    ${BLUE}Response: ${body:0:100}${NC}"
        fi
        return 0
    else
        echo -e "  ${RED}❌ ${name}: FAILED (${http_code})${NC}"
        if $VERBOSE && [ -n "$body" ]; then
            echo -e "    ${RED}Response: ${body:0:100}${NC}"
        fi
        return 1
    fi
}

# Function to check Docker container health
check_container_health() {
    local container_name=$1
    local service_name=$2
    
    if ! docker ps --format "table {{.Names}}" | grep -q "^${container_name}$"; then
        echo -e "  ${RED}❌ ${service_name}: Container not running${NC}"
        return 1
    fi
    
    local health=$(docker inspect --format='{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "no-healthcheck")
    local status=$(docker inspect --format='{{.State.Status}}' "$container_name" 2>/dev/null || echo "unknown")
    
    case $health in
        "healthy")
            echo -e "  ${GREEN}✅ ${service_name}: Container healthy${NC}"
            return 0
            ;;
        "unhealthy")
            echo -e "  ${RED}❌ ${service_name}: Container unhealthy${NC}"
            if $VERBOSE; then
                local last_output=$(docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' "$container_name" 2>/dev/null | tail -1)
                echo -e "    ${RED}Health check output: ${last_output}${NC}"
            fi
            return 1
            ;;
        "starting")
            echo -e "  ${YELLOW}⏳ ${service_name}: Container starting${NC}"
            return 1
            ;;
        "no-healthcheck")
            if [ "$status" = "running" ]; then
                echo -e "  ${BLUE}ℹ️  ${service_name}: Running (no healthcheck)${NC}"
                return 0
            else
                echo -e "  ${RED}❌ ${service_name}: Not running (${status})${NC}"
                return 1
            fi
            ;;
        *)
            echo -e "  ${YELLOW}❓ ${service_name}: Unknown health status (${health})${NC}"
            return 1
            ;;
    esac
}

# Function to check Kafka connectivity
check_kafka() {
    echo -e "\n${YELLOW}📡 Checking Kafka...${NC}"
    
    # Check if Kafka container is running
    if ! check_container_health "crypto-bot-kafka" "Kafka Broker"; then
        return 1
    fi
    
    # Try to list topics (requires kafka tools in container)
    if docker exec crypto-bot-kafka kafka-topics --bootstrap-server localhost:9092 --list >/dev/null 2>&1; then
        echo -e "  ${GREEN}✅ Kafka: Topic listing successful${NC}"
        
        if $VERBOSE; then
            echo -e "  ${BLUE}Available topics:${NC}"
            docker exec crypto-bot-kafka kafka-topics --bootstrap-server localhost:9092 --list 2>/dev/null | sed 's/^/    /'
        fi
    else
        echo -e "  ${YELLOW}⚠️  Kafka: Cannot list topics (may be starting)${NC}"
    fi
}

# Function to check Redis connectivity
check_redis() {
    echo -e "\n${YELLOW}💾 Checking Redis...${NC}"
    
    if ! check_container_health "crypto-bot-redis" "Redis"; then
        return 1
    fi
    
    # Try to ping Redis
    if docker exec crypto-bot-redis redis-cli ping >/dev/null 2>&1; then
        echo -e "  ${GREEN}✅ Redis: Ping successful${NC}"
        
        if $VERBOSE; then
            local info=$(docker exec crypto-bot-redis redis-cli info server 2>/dev/null | grep "redis_version" || echo "version unknown")
            echo -e "  ${BLUE}Redis info: ${info}${NC}"
        fi
    else
        echo -e "  ${RED}❌ Redis: Ping failed${NC}"
        return 1
    fi
}

# Function to check HFT services
check_hft_services() {
    echo -e "\n${YELLOW}🤖 Checking HFT Services...${NC}"
    
    local services=(
        "ingestion-service:crypto-bot-ingestion:8000"
        "kalman-filter-service:crypto-bot-kalman-filter:8001"
        "garch-volatility-service:crypto-bot-garch-volatility:8002"
        "execution-service:crypto-bot-execution:8003"
        "risk-manager-service:crypto-bot-risk-manager:8004"
    )
    
    local failed=0
    
    for service_info in "${services[@]}"; do
        IFS=':' read -r service_name container_name port <<< "$service_info"
        
        # Check container health
        if ! check_container_health "$container_name" "$service_name"; then
            ((failed++))
            continue
        fi
        
        # Check HTTP health endpoint
        if ! check_http_endpoint "$service_name" "http://localhost:${port}/health"; then
            ((failed++))
        fi
        
        # Check metrics endpoint
        if $VERBOSE; then
            if check_http_endpoint "${service_name} metrics" "http://localhost:${port}/metrics" 200; then
                echo -e "    ${BLUE}Metrics endpoint available${NC}"
            fi
        fi
    done
    
    return $failed
}

# Function to check monitoring services
check_monitoring_services() {
    echo -e "\n${YELLOW}📊 Checking Monitoring Services...${NC}"
    
    local failed=0
    
    # Prometheus
    if check_container_health "crypto-bot-prometheus" "Prometheus"; then
        if check_http_endpoint "Prometheus" "http://localhost:9090/-/healthy"; then
            if $VERBOSE; then
                # Check number of targets
                local targets=$(curl -s "http://localhost:9090/api/v1/targets" 2>/dev/null | grep -o '"health":"up"' | wc -l || echo "0")
                echo -e "    ${BLUE}Active targets: ${targets}${NC}"
            fi
        else
            ((failed++))
        fi
    else
        ((failed++))
    fi
    
    # Grafana
    if check_container_health "crypto-bot-grafana" "Grafana"; then
        if ! check_http_endpoint "Grafana" "http://localhost:3000/api/health"; then
            ((failed++))
        fi
    else
        ((failed++))
    fi
    
    # AlertManager
    if check_container_health "crypto-bot-alertmanager" "AlertManager"; then
        if ! check_http_endpoint "AlertManager" "http://localhost:9093/-/healthy"; then
            ((failed++))
        fi
    else
        ((failed++))
    fi
    
    # Streamlit
    if check_container_health "crypto-bot-streamlit" "Streamlit"; then
        if ! check_http_endpoint "Streamlit" "http://localhost:8501/_stcore/health"; then
            ((failed++))
        fi
    else
        ((failed++))
    fi
    
    return $failed
}

# Function to check system resources
check_system_resources() {
    echo -e "\n${YELLOW}💻 Checking System Resources...${NC}"
    
    # Docker system info
    local docker_info=$(docker system df 2>/dev/null || echo "Docker system info unavailable")
    
    # Memory usage
    local memory_usage=$(docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}" 2>/dev/null | grep crypto-bot || echo "No containers found")
    
    if $VERBOSE; then
        echo -e "  ${BLUE}Docker System Usage:${NC}"
        echo "$docker_info" | sed 's/^/    /'
        
        echo -e "  ${BLUE}Container Memory Usage:${NC}"
        echo "$memory_usage" | sed 's/^/    /'
    fi
    
    # Check disk space
    local disk_usage=$(df -h . | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$disk_usage" -gt 90 ]; then
        echo -e "  ${RED}⚠️  Disk usage high: ${disk_usage}%${NC}"
        return 1
    else
        echo -e "  ${GREEN}✅ Disk usage: ${disk_usage}%${NC}"
    fi
    
    return 0
}

# Function to generate summary report
generate_summary() {
    local total_checks=$1
    local failed_checks=$2
    
    echo -e "\n${BLUE}📋 Health Check Summary${NC}"
    echo -e "${BLUE}======================${NC}"
    
    local success_rate=$(( (total_checks - failed_checks) * 100 / total_checks ))
    
    if [ $failed_checks -eq 0 ]; then
        echo -e "${GREEN}🎉 All systems operational! (${success_rate}%)${NC}"
        echo -e "${GREEN}✅ ${total_checks}/${total_checks} checks passed${NC}"
    elif [ $failed_checks -lt 3 ]; then
        echo -e "${YELLOW}⚠️  Minor issues detected (${success_rate}%)${NC}"
        echo -e "${YELLOW}⚠️  $((total_checks - failed_checks))/${total_checks} checks passed${NC}"
        echo -e "${YELLOW}💡 Some services may be starting or have minor issues${NC}"
    else
        echo -e "${RED}❌ Major issues detected (${success_rate}%)${NC}"
        echo -e "${RED}❌ $((total_checks - failed_checks))/${total_checks} checks passed${NC}"
        echo -e "${RED}🚨 System requires attention${NC}"
    fi
    
    return $failed_checks
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

echo -e "${BLUE}Configuration:${NC}"
echo -e "  Environment: ${ENVIRONMENT}"
echo -e "  Timeout: ${TIMEOUT}s"
echo -e "  Verbose: ${VERBOSE}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "\n${RED}❌ Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Initialize counters
total_checks=0
failed_checks=0

# Run health checks
echo -e "\n${YELLOW}🔍 Starting health checks...${NC}"

# Check infrastructure services
check_kafka
if [ $? -ne 0 ]; then ((failed_checks++)); fi
((total_checks++))

check_redis
if [ $? -ne 0 ]; then ((failed_checks++)); fi
((total_checks++))

# Check HFT services
check_hft_services
hft_failures=$?
failed_checks=$((failed_checks + hft_failures))
total_checks=$((total_checks + 5))  # 5 HFT services

# Check monitoring services
check_monitoring_services
monitoring_failures=$?
failed_checks=$((failed_checks + monitoring_failures))
total_checks=$((total_checks + 4))  # 4 monitoring services

# Check system resources
check_system_resources
if [ $? -ne 0 ]; then ((failed_checks++)); fi
((total_checks++))

# Generate summary
generate_summary $total_checks $failed_checks
exit_code=$?

echo -e "\n${BLUE}💡 Next steps:${NC}"
if [ $exit_code -eq 0 ]; then
    echo -e "  • Monitor dashboards: ${BLUE}http://localhost:3000${NC}"
    echo -e "  • Check metrics: ${BLUE}http://localhost:9090${NC}"
    echo -e "  • View logs: ${BLUE}docker-compose logs -f${NC}"
else
    echo -e "  • Check logs: ${BLUE}docker-compose logs -f${NC}"
    echo -e "  • Restart services: ${BLUE}docker-compose restart${NC}"
    echo -e "  • Check configuration: ${BLUE}docker-compose config${NC}"
fi

exit $exit_code
