"""
Main LLM Analyst service for anomaly diagnosis.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import signal
import sys

from .models.anomaly_models import (
    DiagnosisRequest, DiagnosisResult, AnomalyDetection,
    DataSource, TimeSeriesData, AnomalyType, AnomalySeverity
)
from .collectors.data_collector import DataCollectorOrchestrator
from .agents.llm_analyst import LLMAnalyst
from .analysis.anomaly_detector import AnomalyDetectionOrchestrator
from ..common.db.client import DatabaseClient
from ..common.logger import get_logger, configure_logging
from ..common.metrics import ServiceMetrics


class LLMAnalystService:
    """Main LLM Analyst service for automated anomaly diagnosis."""
    
    def __init__(
        self,
        db_client: DatabaseClient,
        openai_api_key: Optional[str] = None,
        detection_interval: int = 300,  # 5 minutes
        analysis_lookback_hours: int = 24
    ):
        self.db_client = db_client
        self.detection_interval = detection_interval
        self.analysis_lookback_hours = analysis_lookback_hours
        self.logger = get_logger("llm_analyst_service")
        
        # Initialize components
        self.data_collector = DataCollectorOrchestrator(db_client)
        self.anomaly_detector = AnomalyDetectionOrchestrator()
        self.llm_analyst = LLMAnalyst(api_key=openai_api_key)
        
        # Service state
        self.is_running = False
        self.current_diagnoses = {}
        self.diagnosis_history = []
        
        # Metrics
        self.metrics = ServiceMetrics("llm_analyst")
        
        # Performance tracking
        self.performance_stats = {
            'total_detections': 0,
            'total_diagnoses': 0,
            'avg_diagnosis_time': 0.0,
            'successful_diagnoses': 0,
            'failed_diagnoses': 0
        }
    
    async def start(self):
        """Start the LLM Analyst service."""
        self.logger.info("Starting LLM Analyst Service")
        self.is_running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Start main detection and diagnosis loop
            await self._main_loop()
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the LLM Analyst service."""
        self.logger.info("Stopping LLM Analyst Service")
        self.is_running = False
        
        # Save current state
        await self._save_state()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.is_running = False
    
    async def _main_loop(self):
        """Main detection and diagnosis loop."""
        self.logger.info("Starting main detection loop")
        
        while self.is_running:
            try:
                # Run detection cycle
                await self._run_detection_cycle()
                
                # Wait for next cycle
                await asyncio.sleep(self.detection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in detection cycle: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def _run_detection_cycle(self):
        """Run a complete detection and diagnosis cycle."""
        cycle_start = datetime.now()
        self.logger.info(f"Starting detection cycle at {cycle_start}")
        
        try:
            # Collect data for analysis
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=self.analysis_lookback_hours)
            
            data_sources = [
                DataSource.PNL_DATA,
                DataSource.MARKET_DATA,
                DataSource.SYSTEM_LOGS,
                DataSource.FUNDING_RATES
            ]
            
            collected_data = await self.data_collector.collect_all_data(
                start_time, end_time, data_sources
            )
            
            if not collected_data:
                self.logger.warning("No data collected, skipping cycle")
                return
            
            # Convert collected data to time series format
            time_series_data = self._convert_to_time_series(collected_data)
            
            # Run anomaly detection
            anomalies = await self.anomaly_detector.run_detection_cycle(
                time_series_data, end_time
            )
            
            self.performance_stats['total_detections'] += len(anomalies)
            
            # Process each anomaly
            for anomaly in anomalies:
                await self._process_anomaly(anomaly, collected_data)
            
            # Update metrics
            self.metrics.record_request("detection_cycle")
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            self.logger.info(f"Detection cycle completed in {cycle_duration:.2f}s, found {len(anomalies)} anomalies")
            
        except Exception as e:
            self.logger.error(f"Error in detection cycle: {e}")
            self.metrics.record_error("detection_cycle_error")
    
    async def _process_anomaly(
        self,
        anomaly: AnomalyDetection,
        collected_data: Dict[str, Any]
    ):
        """Process a detected anomaly with LLM diagnosis."""
        try:
            self.logger.info(f"Processing anomaly {anomaly.id} ({anomaly.anomaly_type.value})")
            
            # Skip if already being processed
            if anomaly.id in self.current_diagnoses:
                return
            
            # Create diagnosis request
            request = DiagnosisRequest(
                anomaly=anomaly,
                pnl_data=collected_data.get('pnl_data'),
                market_data=collected_data.get('price_data', {}),
                market_context=collected_data.get('market_context'),
                system_metrics=collected_data.get('system_metrics', []),
                trading_metrics=collected_data.get('trading_metrics', []),
                lookback_hours=self.analysis_lookback_hours,
                include_external_factors=True
            )
            
            # Mark as being processed
            self.current_diagnoses[anomaly.id] = {
                'start_time': datetime.now(),
                'anomaly': anomaly
            }
            
            # Perform LLM diagnosis
            diagnosis_result = await self.llm_analyst.diagnose_anomaly(
                request, collected_data
            )
            
            # Store diagnosis result
            await self._store_diagnosis_result(diagnosis_result)
            
            # Send alerts if necessary
            await self._send_alerts(anomaly, diagnosis_result)
            
            # Update performance stats
            self.performance_stats['total_diagnoses'] += 1
            self.performance_stats['successful_diagnoses'] += 1
            self.performance_stats['avg_diagnosis_time'] = (
                (self.performance_stats['avg_diagnosis_time'] * (self.performance_stats['total_diagnoses'] - 1) +
                 diagnosis_result.analysis_duration) / self.performance_stats['total_diagnoses']
            )
            
            self.logger.info(f"Completed diagnosis for anomaly {anomaly.id}")
            
        except Exception as e:
            self.logger.error(f"Error processing anomaly {anomaly.id}: {e}")
            self.performance_stats['failed_diagnoses'] += 1
            self.metrics.record_error("diagnosis_error")
        finally:
            # Remove from current processing
            if anomaly.id in self.current_diagnoses:
                del self.current_diagnoses[anomaly.id]
    
    def _convert_to_time_series(
        self,
        collected_data: Dict[str, Any]
    ) -> Dict[str, TimeSeriesData]:
        """Convert collected data to time series format for anomaly detection."""
        time_series = {}
        
        # PnL data
        if 'pnl_data' in collected_data:
            time_series['pnl'] = collected_data['pnl_data']
        
        # Price data
        if 'price_data' in collected_data:
            for symbol, price_data in collected_data['price_data'].items():
                time_series[f'price_{symbol.replace("/", "_")}'] = price_data
        
        # System metrics
        if 'system_metrics' in collected_data:
            metrics = collected_data['system_metrics']
            if metrics:
                # Convert system metrics to time series
                timestamps = [m.timestamp for m in metrics]
                
                # CPU usage
                cpu_values = [m.cpu_usage for m in metrics]
                time_series['cpu_usage'] = TimeSeriesData(
                    timestamps=timestamps,
                    values=cpu_values,
                    labels=[f"CPU: {v:.1f}%" for v in cpu_values]
                )
                
                # Memory usage
                memory_values = [m.memory_usage for m in metrics]
                time_series['memory_usage'] = TimeSeriesData(
                    timestamps=timestamps,
                    values=memory_values,
                    labels=[f"Memory: {v:.1f}%" for v in memory_values]
                )
                
                # Network latency
                latency_values = [m.network_latency for m in metrics]
                time_series['network_latency'] = TimeSeriesData(
                    timestamps=timestamps,
                    values=latency_values,
                    labels=[f"Latency: {v:.1f}ms" for v in latency_values]
                )
                
                # Error rate
                error_values = [m.error_rate for m in metrics]
                time_series['error_rate'] = TimeSeriesData(
                    timestamps=timestamps,
                    values=error_values,
                    labels=[f"Errors: {v:.3f}" for v in error_values]
                )
        
        # Trading metrics
        if 'trading_metrics' in collected_data:
            metrics = collected_data['trading_metrics']
            if metrics:
                timestamps = [m.timestamp for m in metrics]
                
                # Sharpe ratio
                sharpe_values = [m.sharpe_ratio for m in metrics]
                time_series['sharpe_ratio'] = TimeSeriesData(
                    timestamps=timestamps,
                    values=sharpe_values,
                    labels=[f"Sharpe: {v:.2f}" for v in sharpe_values]
                )
                
                # Win rate
                win_rate_values = [m.win_rate for m in metrics]
                time_series['win_rate'] = TimeSeriesData(
                    timestamps=timestamps,
                    values=win_rate_values,
                    labels=[f"Win Rate: {v:.1%}" for v in win_rate_values]
                )
        
        # Market context (if available)
        if 'market_context' in collected_data:
            context = collected_data['market_context']
            
            # Create single-point time series for current market state
            time_series['market_volatility'] = TimeSeriesData(
                timestamps=[context.timestamp],
                values=[context.market_volatility],
                labels=[f"Volatility: {context.market_volatility:.2f}%"]
            )
            
            # Funding rates
            for symbol, rate in context.funding_rates.items():
                time_series[f'funding_rate_{symbol}'] = TimeSeriesData(
                    timestamps=[context.timestamp],
                    values=[rate * 10000],  # Convert to basis points
                    labels=[f"{symbol} Funding: {rate*10000:.1f}bp"]
                )
        
        return time_series
    
    async def _store_diagnosis_result(self, result: DiagnosisResult):
        """Store diagnosis result in database."""
        try:
            # Convert to JSON for storage
            result_json = json.dumps(result.to_dict(), default=str)
            
            # Store in database (would need to create diagnosis_results table)
            # For now, store in diagnosis history
            self.diagnosis_history.append(result)
            
            # Limit history size
            if len(self.diagnosis_history) > 1000:
                self.diagnosis_history = self.diagnosis_history[-1000:]
            
            self.logger.info(f"Stored diagnosis result for anomaly {result.anomaly_id}")
            
        except Exception as e:
            self.logger.error(f"Error storing diagnosis result: {e}")
    
    async def _send_alerts(
        self,
        anomaly: AnomalyDetection,
        diagnosis: DiagnosisResult
    ):
        """Send alerts for critical anomalies."""
        try:
            # Only alert for high severity anomalies
            if anomaly.severity in [AnomalySeverity.HIGH, AnomalySeverity.CRITICAL]:
                alert_message = self._create_alert_message(anomaly, diagnosis)
                
                # Log alert (in production, would send to Slack, email, etc.)
                self.logger.warning(f"ALERT: {alert_message}")
                
                # Could integrate with alerting systems here
                # await self._send_slack_alert(alert_message)
                # await self._send_email_alert(alert_message)
                
        except Exception as e:
            self.logger.error(f"Error sending alerts: {e}")
    
    def _create_alert_message(
        self,
        anomaly: AnomalyDetection,
        diagnosis: DiagnosisResult
    ) -> str:
        """Create alert message for anomaly."""
        return f"""
🚨 ANOMALY DETECTED 🚨

Type: {anomaly.anomaly_type.value}
Severity: {anomaly.severity.value}
Confidence: {anomaly.confidence:.2f}
Time: {anomaly.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Primary Cause: {diagnosis.primary_hypothesis.title}
Confidence: {diagnosis.primary_hypothesis.confidence:.2f}

Executive Summary:
{diagnosis.executive_summary}

Immediate Actions:
{chr(10).join(['• ' + action for action in diagnosis.immediate_actions])}

Analysis Duration: {diagnosis.analysis_duration:.1f}s
Model: {diagnosis.llm_model_used}
        """.strip()
    
    async def _save_state(self):
        """Save current service state."""
        try:
            state = {
                'performance_stats': self.performance_stats,
                'diagnosis_history_count': len(self.diagnosis_history),
                'current_diagnoses_count': len(self.current_diagnoses),
                'llm_performance': self.llm_analyst.get_performance_metrics(),
                'detection_stats': self.anomaly_detector.get_detection_statistics()
            }
            
            # Save to file (in production, might save to database)
            state_file = "llm_analyst_state.json"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            self.logger.info(f"Saved service state to {state_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
    
    # API methods for external interaction
    
    async def diagnose_anomaly_on_demand(
        self,
        anomaly_type: AnomalyType,
        start_time: datetime,
        end_time: datetime,
        affected_metrics: List[str] = None
    ) -> DiagnosisResult:
        """Perform on-demand anomaly diagnosis."""
        try:
            self.logger.info(f"On-demand diagnosis requested for {anomaly_type.value}")
            
            # Collect data
            data_sources = [DataSource.PNL_DATA, DataSource.MARKET_DATA, DataSource.SYSTEM_LOGS]
            collected_data = await self.data_collector.collect_all_data(
                start_time, end_time, data_sources
            )
            
            # Create synthetic anomaly for analysis
            anomaly = AnomalyDetection(
                id=f"on_demand_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=end_time,
                anomaly_type=anomaly_type,
                severity=AnomalySeverity.MEDIUM,
                confidence=0.8,
                z_score=2.0,
                p_value=0.05,
                threshold_exceeded=1.0,
                affected_metrics=affected_metrics or [],
                data_sources=data_sources
            )
            
            # Create diagnosis request
            request = DiagnosisRequest(
                anomaly=anomaly,
                pnl_data=collected_data.get('pnl_data'),
                market_data=collected_data.get('price_data', {}),
                market_context=collected_data.get('market_context'),
                system_metrics=collected_data.get('system_metrics', []),
                trading_metrics=collected_data.get('trading_metrics', []),
                lookback_hours=int((end_time - start_time).total_seconds() / 3600)
            )
            
            # Perform diagnosis
            result = await self.llm_analyst.diagnose_anomaly(request, collected_data)
            
            # Store result
            await self._store_diagnosis_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in on-demand diagnosis: {e}")
            raise
    
    def get_recent_diagnoses(self, limit: int = 10) -> List[DiagnosisResult]:
        """Get recent diagnosis results."""
        return self.diagnosis_history[-limit:] if self.diagnosis_history else []
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status."""
        return {
            'is_running': self.is_running,
            'detection_interval': self.detection_interval,
            'analysis_lookback_hours': self.analysis_lookback_hours,
            'current_diagnoses': len(self.current_diagnoses),
            'diagnosis_history_count': len(self.diagnosis_history),
            'performance_stats': self.performance_stats,
            'llm_performance': self.llm_analyst.get_performance_metrics(),
            'detection_stats': self.anomaly_detector.get_detection_statistics()
        }
    
    def add_feedback(
        self,
        anomaly_id: str,
        is_accurate: bool,
        feedback_notes: str = ""
    ):
        """Add feedback about diagnosis accuracy."""
        # Add feedback to anomaly detector
        self.anomaly_detector.add_feedback(anomaly_id, is_accurate, feedback_notes)
        
        # Log feedback
        self.logger.info(f"Received feedback for {anomaly_id}: {'Accurate' if is_accurate else 'Inaccurate'}")


async def main():
    """Main entry point for the LLM Analyst service."""
    # Configure logging
    configure_logging()
    logger = get_logger("main")
    
    try:
        # Initialize database client
        db_client = DatabaseClient()
        await db_client.initialize_db()
        
        # Get OpenAI API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            sys.exit(1)
        
        # Create and start service
        service = LLMAnalystService(
            db_client=db_client,
            openai_api_key=openai_api_key,
            detection_interval=300,  # 5 minutes
            analysis_lookback_hours=24
        )
        
        await service.start()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
