"""State persistence manager for Kalman filter."""

import os
import joblib
import structlog
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time
from dataclasses import asdict

from .filter import KalmanState, DynamicHedgeRatioKalman, PairTradingKalmanFilter

logger = structlog.get_logger(__name__)


class StateManager:
    """
    Manages saving and loading of Kalman filter states to/from disk.
    
    Provides both manual and automatic periodic persistence.
    """
    
    def __init__(
        self,
        state_dir: str = "data/kalman_states",
        auto_save_interval: int = 300,  # 5 minutes
        backup_retention_days: int = 7
    ):
        """
        Initialize the state manager.
        
        Args:
            state_dir: Directory to store state files
            auto_save_interval: Automatic save interval in seconds (0 to disable)
            backup_retention_days: How many days to keep backup files
        """
        self.state_dir = Path(state_dir)
        self.auto_save_interval = auto_save_interval
        self.backup_retention_days = backup_retention_days
        
        # Create state directory if it doesn't exist
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-save thread
        self._auto_save_thread = None
        self._stop_auto_save = threading.Event()
        self._filters_to_save: Dict[str, DynamicHedgeRatioKalman] = {}
        self._pair_filters_to_save: Dict[str, PairTradingKalmanFilter] = {}
        
        logger.info(
            "State manager initialized",
            state_dir=str(self.state_dir),
            auto_save_interval=auto_save_interval,
            backup_retention_days=backup_retention_days
        )
    
    def save_state(
        self,
        filter_obj: DynamicHedgeRatioKalman,
        identifier: str,
        create_backup: bool = True
    ) -> bool:
        """
        Save a single Kalman filter state to disk.
        
        Args:
            filter_obj: The Kalman filter to save
            identifier: Unique identifier for the filter
            create_backup: Whether to create a backup of existing state
            
        Returns:
            True if successful, False otherwise
        """
        try:
            state_file = self.state_dir / f"{identifier}.joblib"
            
            # Create backup if requested and file exists
            if create_backup and state_file.exists():
                self._create_backup(state_file)
            
            # Get current state
            state = filter_obj.get_current_state()
            
            # Prepare data for serialization
            state_data = {
                'state': {
                    'state_mean': state.state_mean,
                    'state_covariance': state.state_covariance,
                    'timestamp': state.timestamp,
                    'n_observations': state.n_observations
                },
                'filter_params': {
                    'process_variance': filter_obj.process_variance,
                    'observation_variance': filter_obj.observation_variance,
                    'transition_matrix': filter_obj.transition_matrix,
                    'observation_matrix': filter_obj.observation_matrix,
                    'process_covariance': filter_obj.process_covariance,
                    'observation_covariance': filter_obj.observation_covariance
                },
                'metadata': {
                    'saved_at': datetime.utcnow(),
                    'identifier': identifier,
                    'version': '1.0'
                }
            }
            
            # Save to disk
            joblib.dump(state_data, state_file, compress=3)
            
            logger.info(
                "Filter state saved",
                identifier=identifier,
                file=str(state_file),
                n_observations=state.n_observations
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to save filter state",
                identifier=identifier,
                error=str(e),
                exc_info=True
            )
            return False
    
    def load_state(
        self,
        identifier: str,
        filter_obj: Optional[DynamicHedgeRatioKalman] = None
    ) -> Optional[DynamicHedgeRatioKalman]:
        """
        Load a Kalman filter state from disk.
        
        Args:
            identifier: Unique identifier for the filter
            filter_obj: Existing filter to load state into (optional)
            
        Returns:
            Loaded filter object or None if failed
        """
        try:
            state_file = self.state_dir / f"{identifier}.joblib"
            
            if not state_file.exists():
                logger.warning("State file not found", identifier=identifier, file=str(state_file))
                return None
            
            # Load data from disk
            state_data = joblib.load(state_file)
            
            # Validate data structure
            if not self._validate_state_data(state_data):
                logger.error("Invalid state data format", identifier=identifier)
                return None
            
            # Create filter if not provided
            if filter_obj is None:
                params = state_data['filter_params']
                filter_obj = DynamicHedgeRatioKalman(
                    process_variance=params['process_variance'],
                    observation_variance=params['observation_variance']
                )
            
            # Restore state
            state_info = state_data['state']
            state = KalmanState(
                state_mean=state_info['state_mean'],
                state_covariance=state_info['state_covariance'],
                timestamp=state_info['timestamp'],
                n_observations=state_info['n_observations']
            )
            
            filter_obj.set_state(state)
            
            logger.info(
                "Filter state loaded",
                identifier=identifier,
                file=str(state_file),
                n_observations=state.n_observations,
                saved_at=state_data['metadata']['saved_at'].isoformat()
            )
            
            return filter_obj
            
        except Exception as e:
            logger.error(
                "Failed to load filter state",
                identifier=identifier,
                error=str(e),
                exc_info=True
            )
            return None
    
    def save_pair_filters(
        self,
        pair_filter: PairTradingKalmanFilter,
        identifier: str = "pair_filters"
    ) -> bool:
        """
        Save all filters from a PairTradingKalmanFilter.
        
        Args:
            pair_filter: The pair trading filter manager
            identifier: Base identifier for the saved files
            
        Returns:
            True if all saves successful, False otherwise
        """
        try:
            success_count = 0
            total_count = len(pair_filter.filters)
            
            for pair_id, filter_obj in pair_filter.filters.items():
                filter_identifier = f"{identifier}_{pair_id}"
                if self.save_state(filter_obj, filter_identifier):
                    success_count += 1
            
            # Save pair configurations
            config_file = self.state_dir / f"{identifier}_configs.joblib"
            config_data = {
                'pair_configs': pair_filter.pair_configs,
                'metadata': {
                    'saved_at': datetime.utcnow(),
                    'identifier': identifier,
                    'version': '1.0'
                }
            }
            joblib.dump(config_data, config_file, compress=3)
            
            logger.info(
                "Pair filters saved",
                identifier=identifier,
                success_count=success_count,
                total_count=total_count
            )
            
            return success_count == total_count
            
        except Exception as e:
            logger.error(
                "Failed to save pair filters",
                identifier=identifier,
                error=str(e),
                exc_info=True
            )
            return False
    
    def load_pair_filters(
        self,
        identifier: str = "pair_filters"
    ) -> Optional[PairTradingKalmanFilter]:
        """
        Load all filters into a PairTradingKalmanFilter.
        
        Args:
            identifier: Base identifier for the saved files
            
        Returns:
            Loaded PairTradingKalmanFilter or None if failed
        """
        try:
            # Load pair configurations
            config_file = self.state_dir / f"{identifier}_configs.joblib"
            if not config_file.exists():
                logger.warning("Pair configs file not found", file=str(config_file))
                return None
            
            config_data = joblib.load(config_file)
            pair_configs = config_data['pair_configs']
            
            # Create new pair filter
            pair_filter = PairTradingKalmanFilter()
            
            # Load each pair
            loaded_count = 0
            for pair_id, config in pair_configs.items():
                # Add pair to manager
                pair_filter.add_pair(
                    pair_id=pair_id,
                    asset1=config['asset1'],
                    asset2=config['asset2'],
                    initial_hedge_ratio=config['initial_hedge_ratio'],
                    process_variance=config['process_variance'],
                    observation_variance=config['observation_variance']
                )
                
                # Load state
                filter_identifier = f"{identifier}_{pair_id}"
                loaded_filter = self.load_state(filter_identifier, pair_filter.filters[pair_id])
                
                if loaded_filter:
                    loaded_count += 1
                else:
                    logger.warning("Failed to load pair state", pair_id=pair_id)
            
            logger.info(
                "Pair filters loaded",
                identifier=identifier,
                loaded_count=loaded_count,
                total_count=len(pair_configs)
            )
            
            return pair_filter if loaded_count > 0 else None
            
        except Exception as e:
            logger.error(
                "Failed to load pair filters",
                identifier=identifier,
                error=str(e),
                exc_info=True
            )
            return None
    
    def start_auto_save(self) -> None:
        """Start automatic periodic saving."""
        if self.auto_save_interval <= 0:
            logger.info("Auto-save disabled (interval <= 0)")
            return
        
        if self._auto_save_thread and self._auto_save_thread.is_alive():
            logger.warning("Auto-save thread already running")
            return
        
        self._stop_auto_save.clear()
        self._auto_save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self._auto_save_thread.start()
        
        logger.info("Auto-save started", interval=self.auto_save_interval)
    
    def stop_auto_save(self) -> None:
        """Stop automatic periodic saving."""
        if self._auto_save_thread and self._auto_save_thread.is_alive():
            self._stop_auto_save.set()
            self._auto_save_thread.join(timeout=5)
            logger.info("Auto-save stopped")
    
    def register_for_auto_save(
        self,
        identifier: str,
        filter_obj: DynamicHedgeRatioKalman
    ) -> None:
        """Register a filter for automatic saving."""
        self._filters_to_save[identifier] = filter_obj
        logger.debug("Filter registered for auto-save", identifier=identifier)
    
    def register_pair_filter_for_auto_save(
        self,
        identifier: str,
        pair_filter: PairTradingKalmanFilter
    ) -> None:
        """Register a pair filter for automatic saving."""
        self._pair_filters_to_save[identifier] = pair_filter
        logger.debug("Pair filter registered for auto-save", identifier=identifier)
    
    def _auto_save_loop(self) -> None:
        """Auto-save loop running in background thread."""
        while not self._stop_auto_save.wait(self.auto_save_interval):
            try:
                # Save individual filters
                for identifier, filter_obj in self._filters_to_save.items():
                    self.save_state(filter_obj, identifier, create_backup=False)
                
                # Save pair filters
                for identifier, pair_filter in self._pair_filters_to_save.items():
                    self.save_pair_filters(pair_filter, identifier)
                
                # Clean up old backups
                self._cleanup_old_backups()
                
            except Exception as e:
                logger.error("Error in auto-save loop", error=str(e), exc_info=True)
    
    def _create_backup(self, state_file: Path) -> None:
        """Create a backup of an existing state file."""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_file = state_file.parent / f"{state_file.stem}_backup_{timestamp}.joblib"
            
            # Copy the file
            import shutil
            shutil.copy2(state_file, backup_file)
            
            logger.debug("Backup created", original=str(state_file), backup=str(backup_file))
            
        except Exception as e:
            logger.warning("Failed to create backup", file=str(state_file), error=str(e))
    
    def _cleanup_old_backups(self) -> None:
        """Remove backup files older than retention period."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.backup_retention_days)
            
            for backup_file in self.state_dir.glob("*_backup_*.joblib"):
                if backup_file.stat().st_mtime < cutoff_date.timestamp():
                    backup_file.unlink()
                    logger.debug("Old backup removed", file=str(backup_file))
                    
        except Exception as e:
            logger.warning("Error cleaning up old backups", error=str(e))
    
    def _validate_state_data(self, state_data: Dict[str, Any]) -> bool:
        """Validate the structure of loaded state data."""
        try:
            required_keys = ['state', 'filter_params', 'metadata']
            if not all(key in state_data for key in required_keys):
                return False
            
            state_keys = ['state_mean', 'state_covariance', 'timestamp', 'n_observations']
            if not all(key in state_data['state'] for key in state_keys):
                return False
            
            param_keys = ['process_variance', 'observation_variance']
            if not all(key in state_data['filter_params'] for key in param_keys):
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_available_states(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available saved states."""
        states = {}
        
        try:
            for state_file in self.state_dir.glob("*.joblib"):
                if "_backup_" in state_file.name or "_configs" in state_file.name:
                    continue
                
                try:
                    state_data = joblib.load(state_file)
                    identifier = state_file.stem
                    
                    states[identifier] = {
                        'file': str(state_file),
                        'size_bytes': state_file.stat().st_size,
                        'modified': datetime.fromtimestamp(state_file.stat().st_mtime),
                        'n_observations': state_data['state']['n_observations'],
                        'saved_at': state_data['metadata']['saved_at'],
                        'version': state_data['metadata'].get('version', 'unknown')
                    }
                    
                except Exception as e:
                    logger.warning(
                        "Failed to read state file info",
                        file=str(state_file),
                        error=str(e)
                    )
                    
        except Exception as e:
            logger.error("Error getting available states", error=str(e))
        
        return states
    
    def __enter__(self):
        """Context manager entry."""
        self.start_auto_save()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_auto_save()
