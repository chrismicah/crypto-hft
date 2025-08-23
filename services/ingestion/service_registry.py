"""Service registry for microservice discovery and health monitoring."""

import json
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import redis.asyncio as redis

from .config import settings
from common.logger import get_logger

logger = get_logger(__name__)


class ServiceRegistry:
    """Redis-based service registry for microservice discovery."""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.connected = False
        self.service_key = f"service:{settings.service_name}"
        self.heartbeat_interval = 30  # seconds
        self.service_ttl = 60  # seconds
        
    async def connect(self) -> bool:
        """Connect to Redis."""
        try:
            self.redis_client = redis.from_url(settings.redis_url)
            
            # Test connection
            await self.redis_client.ping()
            self.connected = True
            
            logger.info("Connected to Redis service registry", url=settings.redis_url)
            return True
            
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis_client:
            try:
                # Unregister service
                await self.unregister_service()
                await self.redis_client.close()
                self.connected = False
                logger.info("Disconnected from Redis service registry")
            except Exception as e:
                logger.error("Error disconnecting from Redis", error=str(e))
    
    async def register_service(self, health_data: Dict[str, Any]) -> bool:
        """Register this service in the registry."""
        if not self.connected or not self.redis_client:
            return False
        
        try:
            service_info = {
                "service_name": settings.service_name,
                "service_version": settings.service_version,
                "health_check_port": settings.health_check_port,
                "prometheus_port": settings.prometheus_port,
                "symbols": settings.symbols_list,
                "health_data": health_data,
                "last_heartbeat": datetime.utcnow().isoformat(),
                "registered_at": datetime.utcnow().isoformat()
            }
            
            # Store service info with TTL
            await self.redis_client.setex(
                self.service_key,
                self.service_ttl,
                json.dumps(service_info)
            )
            
            # Add to services set
            await self.redis_client.sadd("services", settings.service_name)
            
            logger.debug("Service registered", service_name=settings.service_name)
            return True
            
        except Exception as e:
            logger.error("Failed to register service", error=str(e))
            return False
    
    async def unregister_service(self) -> bool:
        """Unregister this service from the registry."""
        if not self.connected or not self.redis_client:
            return False
        
        try:
            # Remove service info
            await self.redis_client.delete(self.service_key)
            
            # Remove from services set
            await self.redis_client.srem("services", settings.service_name)
            
            logger.info("Service unregistered", service_name=settings.service_name)
            return True
            
        except Exception as e:
            logger.error("Failed to unregister service", error=str(e))
            return False
    
    async def heartbeat(self, health_data: Dict[str, Any]) -> bool:
        """Send heartbeat to maintain service registration."""
        if not self.connected or not self.redis_client:
            return False
        
        try:
            # Get current service info
            current_info = await self.redis_client.get(self.service_key)
            
            if current_info:
                service_info = json.loads(current_info)
                service_info["health_data"] = health_data
                service_info["last_heartbeat"] = datetime.utcnow().isoformat()
                
                # Update with new TTL
                await self.redis_client.setex(
                    self.service_key,
                    self.service_ttl,
                    json.dumps(service_info)
                )
                
                logger.debug("Heartbeat sent", service_name=settings.service_name)
                return True
            else:
                # Service not registered, register it
                return await self.register_service(health_data)
                
        except Exception as e:
            logger.error("Failed to send heartbeat", error=str(e))
            return False
    
    async def get_service(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get service information by name."""
        if not self.connected or not self.redis_client:
            return None
        
        try:
            service_info = await self.redis_client.get(f"service:{service_name}")
            
            if service_info:
                return json.loads(service_info)
            return None
            
        except Exception as e:
            logger.error("Failed to get service info", service_name=service_name, error=str(e))
            return None
    
    async def get_all_services(self) -> List[Dict[str, Any]]:
        """Get all registered services."""
        if not self.connected or not self.redis_client:
            return []
        
        try:
            service_names = await self.redis_client.smembers("services")
            services = []
            
            for service_name in service_names:
                service_info = await self.get_service(service_name.decode())
                if service_info:
                    services.append(service_info)
            
            return services
            
        except Exception as e:
            logger.error("Failed to get all services", error=str(e))
            return []
    
    async def get_healthy_services(self) -> List[Dict[str, Any]]:
        """Get only healthy services (recent heartbeat)."""
        all_services = await self.get_all_services()
        healthy_services = []
        
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.service_ttl)
        
        for service in all_services:
            try:
                last_heartbeat = datetime.fromisoformat(service["last_heartbeat"])
                if last_heartbeat > cutoff_time:
                    healthy_services.append(service)
            except (KeyError, ValueError):
                # Skip services with invalid heartbeat timestamp
                continue
        
        return healthy_services
    
    async def start_heartbeat_task(self, get_health_data_callback):
        """Start background heartbeat task."""
        logger.info("Starting heartbeat task", interval=self.heartbeat_interval)
        
        while self.connected:
            try:
                health_data = get_health_data_callback()
                await self.heartbeat(health_data)
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                logger.info("Heartbeat task cancelled")
                break
            except Exception as e:
                logger.error("Error in heartbeat task", error=str(e))
                await asyncio.sleep(5)  # Short delay before retry


class HealthStatus:
    """Health status model for microservices."""
    
    def __init__(self):
        self.websocket_connected = False
        self.kafka_connected = False
        self.redis_connected = False
        self.message_count = 0
        self.error_count = 0
        self.uptime_seconds = 0
        self.last_message_time: Optional[datetime] = None
        self.start_time = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        self.uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "websocket_connected": self.websocket_connected,
            "kafka_connected": self.kafka_connected,
            "redis_connected": self.redis_connected,
            "message_count": self.message_count,
            "error_count": self.error_count,
            "uptime_seconds": self.uptime_seconds,
            "last_message_time": self.last_message_time.isoformat() if self.last_message_time else None,
            "status": self.get_overall_status()
        }
    
    def get_overall_status(self) -> str:
        """Get overall health status."""
        if not self.websocket_connected or not self.kafka_connected:
            return "unhealthy"
        
        # Check if we've received messages recently (within last 2 minutes)
        if self.last_message_time:
            time_since_last = datetime.utcnow() - self.last_message_time
            if time_since_last > timedelta(minutes=2):
                return "degraded"
        
        return "healthy"
