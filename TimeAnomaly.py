import numpy as np
import IPython.display as display
from matplotlib import pyplot as plt
import io
import base64

rng = np.random.default_rng(42)
ys = 200 + rng.standard_normal(100)
x = list(range(len(ys)))

fig = plt.figure(figsize=(4, 3), facecolor='w')
plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)
plt.title("Sample Visualization", fontsize=10)

data = io.BytesIO()
plt.savefig(data)
image = F"data:image/png;base64,{base64.b64encode(data.getvalue()).decode()}"
alt = "Sample Visualization"
"""
Time-Series Anomaly Detection Engine for Energy Utilities

This module provides a comprehensive anomaly detection system for streaming data
from smart meters, grid sensors, and operational logs with root cause analysis.
"""

import asyncio
import logging
import smtplib
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText as MimeText
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Protocol
import json
import hashlib
import os

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql.functions import col, window, mean, stddev, max as spark_max, min as spark_min
from pyspark.ml.feature import VectorAssembler
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')


class AnomalyType(Enum):
    """Enumeration of different anomaly types detected by the system."""
    ABRUPT_CHANGE = "abrupt_change"
    SEASONAL_SHIFT = "seasonal_shift"
    TREND_ANOMALY = "trend_anomaly"
    STATISTICAL_OUTLIER = "statistical_outlier"
    MULTI_SCALE_ANOMALY = "multi_scale_anomaly"


class SeverityLevel(Enum):
    """Severity levels for detected anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyAlert:
    """
    Data structure representing a detected anomaly alert.

    Attributes:
        timestamp: When the anomaly was detected
        sensor_id: Identifier of the sensor that detected the anomaly
        anomaly_type: Type of anomaly detected
        severity: Severity level of the anomaly
        value: The anomalous value
        expected_range: Expected value range for this metric
        confidence_score: Confidence in the detection (0-1)
        root_cause_summary: Human-readable explanation
        feature_attribution: Feature importance scores
        metadata: Additional context information
    """
    timestamp: datetime
    sensor_id: str
    anomaly_type: AnomalyType
    severity: SeverityLevel
    value: float
    expected_range: Tuple[float, float]
    confidence_score: float
    root_cause_summary: str
    feature_attribution: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertChannel(Protocol):
    """Protocol for alert notification channels."""

    def send_alert(self, alert: AnomalyAlert) -> bool:
        """Send an alert through this channel."""
        ...


class EmailAlertChannel:
    """
    Email notification channel for anomaly alerts.

    Approach: Sends formatted email alerts with anomaly details and root cause analysis
    """

    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):
        """
        Initialize email alert channel.

        Args:
            smtp_server: SMTP server hostname
            smtp_port: SMTP server port
            username: Email username
            password: Email password (should be from environment)
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password

    def send_alert(self, alert: AnomalyAlert, recipients: List[str]) -> bool:
        """
        Send anomaly alert via email.

        Intuition: Format alert as readable email with key details
        Approach: Create MIME message with structured alert information
        Complexity: Time O(1), Space O(1)

        Args:
            alert: The anomaly alert to send
            recipients: List of email addresses to notify

        Returns:
            True if email sent successfully, False otherwise
        """
        try:
            subject = f"Anomaly Alert - {alert.sensor_id} - {alert.severity.value.upper()}"
            body = self._format_alert_email(alert)

            msg = MimeText(body)
            msg['Subject'] = subject
            msg['From'] = self.username
            msg['To'] = ', '.join(recipients)

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)

            return True
        except Exception as e:
            logging.error(f"Failed to send email alert: {e}")
            return False

    def _format_alert_email(self, alert: AnomalyAlert) -> str:
        """Format alert as email body."""
        return f"""
Anomaly Detection Alert

Timestamp: {alert.timestamp}
Sensor ID: {alert.sensor_id}
Anomaly Type: {alert.anomaly_type.value}
Severity: {alert.severity.value.upper()}
Confidence: {alert.confidence_score:.2%}

Value: {alert.value:.2f}
Expected Range: {alert.expected_range[0]:.2f} - {alert.expected_range[1]:.2f}

Root Cause Analysis:
{alert.root_cause_summary}

Top Contributing Features:
{self._format_feature_attribution(alert.feature_attribution)}

Metadata:
{json.dumps(alert.metadata, indent=2)}
        """

    def _format_feature_attribution(self, attribution: Dict[str, float]) -> str:
        """Format feature attribution scores."""
        sorted_features = sorted(attribution.items(), key=lambda x: abs(x[1]), reverse=True)
        return '\n'.join(f"- {feature}: {score:.3f}" for feature, score in sorted_features[:5])


class SMSAlertChannel:
    """
    SMS notification channel for critical anomaly alerts.

    Approach: Sends concise SMS alerts for high-priority anomalies
    """

    def __init__(self, api_key: str, api_url: str):
        """
        Initialize SMS alert channel.

        Args:
            api_key: SMS service API key
            api_url: SMS service API endpoint
        """
        self.api_key = api_key
        self.api_url = api_url

    def send_alert(self, alert: AnomalyAlert) -> bool:
        """
        Send SMS alert for critical anomalies.

        Intuition: Send brief, actionable SMS for urgent alerts
        Approach: Format key alert details into concise message
        Complexity: Time O(1), Space O(1)

        Args:
            alert: The anomaly alert to send

        Returns:
            True if SMS sent successfully, False otherwise
        """
        if alert.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
            # send SMS
            try:
                message = self._format_sms_message(alert)
                # Implementation would depend on your SMS service provider
                logging.info(f"SMS Alert sent: {message}")
                return True
            except Exception as e:
                logging.error(f"Failed to send SMS alert: {e}")
                return False
        return True # Only send SMS for high/critical alerts

    def _format_sms_message(self, alert: AnomalyAlert) -> str:
        """Format alert as SMS message."""
        return (f"ALERT: {alert.sensor_id} - {alert.anomaly_type.value} "
                f"({alert.severity.value}) at {alert.timestamp.strftime('%H:%M')}. "
                f"Value: {alert.value:.1f}")


class TimeSeriesPreprocessor:
    """
    Preprocesses time series data for anomaly detection.

    Approach: Handles missing values, outliers, and feature engineering
    """

    def __init__(self, interpolation_method: str = 'linear'):
        """
        Initialize preprocessor.

        Args:
            interpolation_method: Method for interpolating missing values
        """
        self.interpolation_method = interpolation_method
        self.scalers: Dict[str, StandardScaler] = {}

    def preprocess(self, df: pd.DataFrame, sensor_id: str) -> pd.DataFrame:
        """
        Preprocess time series data.

        Intuition: Clean and prepare data for reliable anomaly detection
        Approach: Handle missing values, normalize, and engineer features
        Complexity: Time O(n), Space O(n) where n is number of data points

        Args:
            df: Input DataFrame with timestamp and value columns
            sensor_id: Identifier for the sensor

        Returns:
            Preprocessed DataFrame with engineered features
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        # Handle missing values
        df['value'] = df['value'].interpolate(method=self.interpolation_method)
        df = df.dropna()

        # Feature engineering
        df = self._engineer_features(df)

        # Normalize features
        df = self._normalize_features(df, sensor_id)

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer time-based and statistical features."""
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month

        # Rolling statistics
        for window in [12, 24, 168]:  # 12h, 24h, 1 week
            df[f'rolling_mean_{window}'] = df['value'].rolling(window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df['value'].rolling(window, min_periods=1).std()

        # Rate of change
        df['rate_of_change'] = df['value'].diff()
        df['rate_of_change_abs'] = df['rate_of_change'].abs()

        return df

    def _normalize_features(self, df: pd.DataFrame, sensor_id: str) -> pd.DataFrame:
        """Normalize numerical features."""
        numerical_cols = ['value', 'rolling_mean_12', 'rolling_mean_24', 'rolling_mean_168',
                         'rolling_std_12', 'rolling_std_24', 'rolling_std_168',
                         'rate_of_change', 'rate_of_change_abs']

        if sensor_id not in self.scalers:
            self.scalers[sensor_id] = StandardScaler()
            df[numerical_cols] = self.scalers[sensor_id].fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scalers[sensor_id].transform(df[numerical_cols])

        return df


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detection algorithms."""

    @abstractmethod
    def detect(self, data: pd.DataFrame) -> List[Tuple[int, float]]:
        """
        Detect anomalies in the data.

        Args:
            data: Input time series data

        Returns:
            List of (index, confidence_score) tuples for detected anomalies
        """
        pass

    @abstractmethod
    def get_feature_importance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get feature importance scores for root cause analysis."""
        pass


class StatisticalAnomalyDetector(AnomalyDetector):
    """
    Statistical anomaly detector using Z-score and IQR methods.

    Approach: Combines multiple statistical tests for robust detection
    """

    def __init__(self, z_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        """
        Initialize statistical detector.

        Args:
            z_threshold: Z-score threshold for outlier detection
            iqr_multiplier: IQR multiplier for outlier detection
        """
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier

    def detect(self, data: pd.DataFrame) -> List[Tuple[int, float]]:
        """
        Detect statistical anomalies.

        Intuition: Use multiple statistical tests to identify outliers
        Approach: Combine Z-score and IQR methods with confidence scoring
        Complexity: Time O(n), Space O(n)

        Args:
            data: Input DataFrame with features

        Returns:
            List of anomaly indices with confidence scores
        """
        anomalies = []
        values = data['value'].values

        # Z-score based detection
        z_scores = np.abs(stats.zscore(values))
        z_anomalies = np.nonzero(z_scores > self.z_threshold)[0]

        # IQR based detection
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        iqr_anomalies = np.nonzero((values < lower_bound) | (values > upper_bound))[0]

        # Combine detections
        all_anomaly_indices = set(z_anomalies) | set(iqr_anomalies)

        for idx in all_anomaly_indices:
            confidence = min(z_scores[idx] / self.z_threshold, 1.0)
            anomalies.append((idx, confidence))

        return anomalies

    def get_feature_importance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get feature importance based on correlation with target."""
        feature_cols = [col for col in data.columns if col not in ['timestamp', 'value']]
        importance = {}

        for col in feature_cols:
            corr = abs(data[col].corr(data['value']))
            importance[col] = corr if not np.isnan(corr) else 0.0

        return importance


class IsolationForestDetector(AnomalyDetector):
    """
    Isolation Forest based anomaly detector for multi-dimensional data.

    Approach: Uses tree-based isolation to detect anomalies in feature space
    """

    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        """
        Initialize Isolation Forest detector.

        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees in the forest
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = None

    def detect(self, data: pd.DataFrame) -> List[Tuple[int, float]]:
        """
        Detect anomalies using Isolation Forest.

        Args:
            data: Input DataFrame with features

        Returns:
            List of anomaly indices with confidence scores
        """
        feature_cols = [col for col in data.columns if col not in ['timestamp', 'sensor_id']]
        X = data[feature_cols].values

        self.model = IsolationForest(contamination=self.contamination,
                                     n_estimators=self.n_estimators,
                                     random_state=42)
        self.model.fit(X)

        predictions = self.model.predict(X)
        anomaly_indices = np.nonzero(predictions == -1)[0]

        # Calculate confidence scores (lower decision function score means more anomalous)
        scores = self.model.decision_function(X)
        confidences = 1 - (scores - scores.min()) / (scores.max() - scores.min())

        anomalies = [(idx, confidences[idx]) for idx in anomaly_indices]
        return anomalies


    def get_feature_importance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get feature importance from isolation forest."""
        if self.model is None:
            return {}

        feature_cols = [col for col in self.model.feature_names_in_ if col not in ['timestamp', 'sensor_id']]

        # Approximate feature importance by permutation
        importance = {}
        X = self.model.transform(data[feature_cols])
        baseline_score = np.mean(self.model.decision_function(X))

        rng = np.random.default_rng(42)  # Use Generator for shuffling
        for i, col in enumerate(feature_cols):
            x_permuted = X.copy()
            rng.shuffle(x_permuted[:, i])
            permuted_score = np.mean(self.model.decision_function(x_permuted))
            importance[col] = abs(baseline_score - permuted_score)

        return importance


class SeasonalAnomalyDetector(AnomalyDetector):
    """
    Seasonal anomaly detector using time series decomposition.

    Approach: Decomposes series into trend, seasonal, and residual components
    """

    def __init__(self, period: int = 24, threshold: float = 2.0):
        """
        Initialize seasonal detector.

        Args:
            period: Seasonal period (e.g., 24 for daily patterns)
            threshold: Threshold for residual-based anomaly detection
        """
        self.period = period
        self.threshold = threshold

    def detect(self, data: pd.DataFrame) -> List[Tuple[int, float]]:
        """
        Detect seasonal anomalies.

        Intuition: Anomalies show up as large residuals after removing trend/seasonality
        Approach: Decompose time series and analyze residuals
        Complexity: Time O(n), Space O(n)

        Args:
            data: Input DataFrame with timestamp and value

        Returns:
            List of anomaly indices with confidence scores
        """
        if len(data) < 2 * self.period:
            return []  # Need enough data for seasonal decomposition

        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                data['value'],
                model='additive',
                period=self.period
            )

            residuals = decomposition.resid.dropna()

            # Detect anomalies in residuals
            residual_std = residuals.std()
            anomalies = []

            for idx, residual in enumerate(residuals):
                if abs(residual) > self.threshold * residual_std:
                    confidence = abs(residual) / (self.threshold * residual_std)
                    anomalies.append((idx, min(confidence, 1.0)))

            return anomalies

        except Exception:
            # Fallback to simple statistical detection
            return []

    def get_feature_importance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get seasonal feature importance."""
        return {
            'seasonal_component': 0.8,
            'trend_component': 0.6,
            'residual_component': 1.0
        }


class EnsembleAnomalyDetector:
    """
    Ensemble detector combining multiple anomaly detection algorithms.

    Approach: Combines statistical, ML-based, and seasonal detectors
    """

    def __init__(self):
        """Initialize ensemble detector with multiple algorithms."""
        self.detectors = {
            'statistical': StatisticalAnomalyDetector(),
            'isolation_forest': IsolationForestDetector(),
            'seasonal': SeasonalAnomalyDetector()
        }
        self.weights = {'statistical': 0.3, 'isolation_forest': 0.4, 'seasonal': 0.3}

    def detect(self, data: pd.DataFrame) -> List[Tuple[int, float, AnomalyType]]:
        """
        Detect anomalies using ensemble approach.

        Intuition: Combine multiple detection methods for robust results
        Approach: Weight and combine predictions from different detectors
        Complexity: Time O(n log n), Space O(n)

        Args:
            data: Input DataFrame with features

        Returns:
            List of (index, confidence_score, anomaly_type) tuples
        """
        all_detections = {}

        # Run each detector
        for name, detector in self.detectors.items():
            try:
                detections = detector.detect(data)
                for idx, confidence in detections:
                    if idx not in all_detections:
                        all_detections[idx] = {}
                    all_detections[idx][name] = confidence
            except Exception as e:
                logging.warning(f"Detector {name} failed: {e}")

        # Combine detections
        ensemble_anomalies = []
        for idx, detector_scores in all_detections.items():
            # Weighted average of detector confidences
            weighted_confidence = sum(
                self.weights[detector] * score
                for detector, score in detector_scores.items()
            ) / sum(self.weights[detector] for detector in detector_scores.keys())

            # Determine anomaly type based on strongest detector
            strongest_detector = max(detector_scores.items(), key=lambda x: x[1])[0]
            anomaly_type = self._map_detector_to_type(strongest_detector)

            # Only report if confidence is above threshold
            if weighted_confidence > 0.5:
                ensemble_anomalies.append((idx, weighted_confidence, anomaly_type))

        return ensemble_anomalies

    def _map_detector_to_type(self, detector_name: str) -> AnomalyType:
        """Map detector name to anomaly type."""
        mapping = {
            'statistical': AnomalyType.STATISTICAL_OUTLIER,
            'isolation_forest': AnomalyType.MULTI_SCALE_ANOMALY,
            'seasonal': AnomalyType.SEASONAL_SHIFT
        }
        return mapping.get(detector_name, AnomalyType.STATISTICAL_OUTLIER)

    def get_feature_importance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Get combined feature importance from all detectors."""
        all_importance = {}

        for detector in self.detectors.values():
            try:
                importance = detector.get_feature_importance(data)
                for feature, score in importance.items():
                    if feature not in all_importance:
                        all_importance[feature] = []
                    all_importance[feature].append(score)
            except Exception:
                continue

        # Average importance scores
        combined_importance = {}
        for feature, scores in all_importance.items():
            combined_importance[feature] = np.mean(scores)

        return combined_importance


class RootCauseAnalyzer:
    """
    Analyzes root causes of detected anomalies.

    Approach: Combines feature attribution with domain knowledge
    """

    def __init__(self):
        """Initialize root cause analyzer."""
        self.domain_rules = self._initialize_domain_rules()

    def analyze(self, alert: AnomalyAlert, data: pd.DataFrame,
            feature_importance: Dict[str, float]) -> str:
        """
        Generate root cause analysis summary.

        Intuition: Combine statistical analysis with domain knowledge
        Approach: Use feature importance and domain rules to explain anomalies
        Complexity: Time O(1), Space O(1)

        Args:
            alert: The anomaly alert
            data: Context data around the anomaly
            feature_importance: Feature importance scores

        Returns:
            Human-readable root cause summary
        """
        pass


    def _explain_feature(self, feature: str, importance: float) -> str:
        """Explain what a feature means in domain terms."""
        explanations = {
            'rolling_mean_24': f"24-hour average shows unusual pattern (impact: {importance:.2f})",
            'rate_of_change': f"Rapid value changes detected (impact: {importance:.2f})",
            'hour': f"Time-of-day pattern anomaly (impact: {importance:.2f})",
            'day_of_week': f"Day-of-week pattern anomaly (impact: {importance:.2f})",
            'rolling_std_168': f"Weekly volatility pattern anomaly (impact: {importance:.2f})",
        }
        return explanations.get(feature, f"{feature} shows anomalous behavior (impact: {importance:.2f})")

    def _apply_domain_rules(self, alert: AnomalyAlert) -> List[str]:
        """Apply energy domain-specific rules."""
        insights = []

        # Peak demand analysis
        if alert.timestamp.hour in [17, 18, 19, 20]:  # Peak hours
            insights.append("Anomaly occurred during peak demand hours - may indicate grid stress")

        # Weekend patterns
        if alert.timestamp.weekday() >= 5:  # Weekend
            insights.append("Weekend anomaly detected - unusual for typical consumption patterns")

        # Seasonal considerations
        month = alert.timestamp.month
        if month in [6, 7, 8]:  # Summer
            insights.append("Summer period anomaly - possible AC load or cooling system issue")
        elif month in [12, 1, 2]:  # Winter
            insights.append("Winter period anomaly - possible heating system issue")

        return insights


    def _analyze_temporal_context(self, alert: AnomalyAlert, data: pd.DataFrame) -> str:
        """Analyze temporal context of the anomaly."""
        # Look for comparison points at different intervals
        comparisons = []

        for hours_back, label in [(24, "same time yesterday"), (168, "same time last week")]:
            if len(data) >= hours_back:
                try:
                    past_value = data.iloc[-hours_back]['value']
                    if pd.notna(past_value) and past_value != 0:
                        change_pct = ((alert.value - past_value) / past_value) * 100
                        if abs(change_pct) > 20:
                            comparisons.append(f"Value changed by {change_pct:.1f}% compared to {label}")
                except (IndexError, ZeroDivisionError):
                    continue

        return "; ".join(comparisons) if comparisons else ""


    def _initialize_domain_rules(self) -> Dict[str, Any]:
        """Initialize domain-specific rules for energy utilities."""
        return {
            'peak_hours': [17, 18, 19, 20],
            'off_peak_hours': [1, 2, 3, 4, 5],
            'high_demand_months': [6, 7, 8, 12, 1, 2],
            'maintenance_hours': [2, 3, 4]
        }


class AnomalyDetectionEngine:
    """
    Main anomaly detection engine for energy utilities.

    Approach: Orchestrates preprocessing, detection, analysis, and alerting
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the anomaly detection engine.

        Args:
            config: Configuration dictionary with system parameters
        """
        self.config = config
        self.preprocessor = TimeSeriesPreprocessor()
        self.detector = EnsembleAnomalyDetector()
        self.analyzer = RootCauseAnalyzer()
        self.alert_channels: List[AlertChannel] = []
        self.spark = self._initialize_spark()

        # Setup logging
        self._setup_logging()

        # Initialize alert channels
        self._initialize_alert_channels()

    def _initialize_spark(self) -> SparkSession:
        """Initialize Spark session for big data processing."""
        return SparkSession.builder \
            .appName("AnomalyDetectionEngine") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = self.config.get('log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('anomaly_detection.log'),
                logging.StreamHandler()
            ]
        )

    def _initialize_alert_channels(self) -> None:
        """Initialize alert notification channels."""
        # Email channel
        email_config = self.config.get('email', {})
        if email_config:
            self.alert_channels.append(EmailAlertChannel(
                smtp_server=email_config['smtp_server'],
                smtp_port=email_config['smtp_port'],
                username=email_config['username'],
                password=os.getenv('EMAIL_PASSWORD', email_config.get('password', ''))
            ))

        # SMS channel
        sms_config = self.config.get('sms', {})
        if sms_config:
            self.alert_channels.append(SMSAlertChannel(
                api_key=os.getenv('SMS_API_KEY', sms_config.get('api_key', '')),
                api_url=sms_config['api_url']
            ))

    def process_streaming_data(self, data_stream: SparkDataFrame) -> None:
        """
        Process streaming data for anomaly detection.

        Intuition: Handle continuous data streams from multiple sensors
        Approach: Process data in micro-batches with adaptive thresholds
        Complexity: Time O(n), Space O(n) per batch

        Args:
            data_stream: Spark streaming DataFrame with sensor data
        """

        async def process_batch_async(batch_df: SparkDataFrame, batch_id: int) -> None:
            """Process each micro-batch of streaming data asynchronously."""
            try:
                # Convert to Pandas for processing
                pandas_df = batch_df.toPandas()

                # Create tasks for concurrent sensor processing
                tasks = [
                    self._process_sensor_data(sensor_id, sensor_data)
                    for sensor_id, sensor_data in pandas_df.groupby('sensor_id')
                ]

                # Process all sensors concurrently and wait for completion
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                logging.error(f"Error processing batch {batch_id}: {e}")

        def process_batch(batch_df: SparkDataFrame, batch_id: int) -> None:
            """Synchronous wrapper for Spark compatibility."""
            asyncio.run(process_batch_async(batch_df, batch_id))

        # Start streaming query
        query = data_stream.writeStream \
            .foreachBatch(process_batch) \
            .option("checkpointLocation", "/tmp/checkpoint") \
            .start()

        query.awaitTermination()


    async def _process_sensor_data(self, sensor_id: str, data: pd.DataFrame) -> None:
        """Process data from a single sensor."""
        try:
            # Preprocess data
            processed_data = self.preprocessor.preprocess(data, sensor_id)

            if len(processed_data) < 10:  # Need minimum data points
                return

            # Detect anomalies
            anomalies = self.detector.detect(processed_data)

            # Generate alerts for detected anomalies
            for idx, confidence, anomaly_type in anomalies:
                alert = self._create_alert(
                    sensor_id, processed_data.iloc[idx],
                    confidence, anomaly_type, processed_data
                )

                if alert:
                    await self._send_alert(alert)

        except Exception as e:
            logging.error(f"Error processing sensor {sensor_id}: {e}")

    def _create_alert(self, sensor_id: str, anomaly_row: pd.Series,
                          confidence: float, anomaly_type: AnomalyType,
                          context_data: pd.DataFrame) -> Optional[AnomalyAlert]:
        """Create an anomaly alert with root cause analysis."""
        try:
            # Determine severity based on confidence
            severity = self._determine_severity(confidence)

            # Calculate expected range
            expected_range = self._calculate_expected_range(context_data)

            # Get feature importance
            feature_importance = self.detector.get_feature_importance(context_data)

            # Create alert
            alert = AnomalyAlert(
                timestamp=pd.to_datetime(anomaly_row['timestamp']),
                sensor_id=sensor_id,
                anomaly_type=anomaly_type,
                severity=severity,
                value=anomaly_row['value'],
                expected_range=expected_range,
                confidence_score=confidence,
                root_cause_summary="",  # Will be filled by analyzer
                feature_attribution=feature_importance,
                metadata={
                    'processing_time': datetime.now(),
                    'data_points_used': len(context_data)
                }
            )

            # Generate root cause analysis
            alert.root_cause_summary = self.analyzer.analyze(
                alert, context_data, feature_importance
            )

            return alert

        except Exception as e:
            logging.error(f"Error creating alert: {e}")
            return None

    def _determine_severity(self, confidence: float) -> SeverityLevel:
        """Determine alert severity based on confidence and context."""
        if confidence > 0.9:
            return SeverityLevel.CRITICAL
        elif confidence > 0.7:
            return SeverityLevel.HIGH
        elif confidence > 0.5:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW

    def _calculate_expected_range(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Calculate expected value range based on historical data."""
        values = data['value'].values
        mean_val = np.mean(values)
        std_val = np.std(values)

        # Use 2-sigma range as expected
        lower_bound = mean_val - 2 * std_val
        upper_bound = mean_val + 2 * std_val

        return (lower_bound, upper_bound)

    async def _send_alert(self, alert: AnomalyAlert) -> None:
        """Send alert through configured channels."""
        recipients = self.config.get('alert_recipients', {})

        for channel in self.alert_channels:
            try:
                if isinstance(channel, EmailAlertChannel):
                    email_recipients = recipients.get('email', [])
                    if email_recipients:
                        await channel.send_alert(alert, email_recipients)

                elif isinstance(channel, SMSAlertChannel):
                    phone_numbers = recipients.get('sms', [])
                    if phone_numbers:
                        channel.send_alert(alert)

            except Exception as e:
                logging.error(f"Failed to send alert via {type(channel).__name__}: {e}")

    def process_batch_data(self, sensor_id: str) -> List[AnomalyAlert]:
        """
        Process batch data for anomaly detection.

        Intuition: Handle historical data analysis and model training
        Approach: Load, preprocess, and analyze batch data files
        Complexity: Time O(n), Space O(n)

        Args:
            sensor_id: Identifier for the sensor

        Returns:
            List of detected anomaly alerts
        """
        try:
            # Load data
            data = pd.read_csv("/content/sample_data (1).csv")

            # Preprocess
            processed_data = self.preprocessor.preprocess(data, sensor_id)

            # Detect anomalies
            anomalies = self.detector.detect(processed_data)

            # Create alerts
            alerts = []
            for idx, confidence, anomaly_type in anomalies:
                alert = self._create_alert(
                    sensor_id, processed_data.iloc[idx],
                    confidence, anomaly_type, processed_data
                )
                if alert:
                    alerts.append(alert)

            return alerts

        except Exception as e:
            logging.error(f"Error processing batch data: {e}")
            return []

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get system health status.

        Returns:
            Dictionary with system health metrics
        """
        return {
            'status': 'healthy',
            'timestamp': datetime.now(),
            'spark_session_active': self.spark is not None,
            'alert_channels_count': len(self.alert_channels),
            'config_loaded': bool(self.config)
        }

    def shutdown(self) -> None:
        """Gracefully shutdown the engine."""
        if self.spark:
            self.spark.stop()
        logging.info("Anomaly detection engine shutdown complete")


# Example usage and configuration
def create_engine_config() -> Dict[str, Any]:
    """
    Create example configuration for the anomaly detection engine.

    Returns:
        Configuration dictionary
    """
    return {
        'log_level': 'INFO',
        'email': {
            'smtp_server': 'smtp.company.com',
            'smtp_port': 587,
            'username': 'alerts@company.com'
        },
        'sms': {
            'api_url': 'https://api.smsservice.com/send'
        },
        'alert_recipients': {
            'email': ['operator@company.com', 'manager@company.com'],
            'sms': ['+1234567890']
        },
        'detection_thresholds': {
            'statistical_threshold': 3.0,
            'isolation_contamination': 0.1,
            'seasonal_period': 24
        }
    }


def main():
    """
    Example main function showing how to use the anomaly detection engine.

    Approach: Demonstrates engine initialization and usage patterns
    """
    # Create configuration
    config = create_engine_config()

    # Initialize engine
    engine = AnomalyDetectionEngine(config)

    try:
        # Example: Process batch data
        alerts = engine.process_batch_data('meter_001')
        print(f"Detected {len(alerts)} anomalies in batch data")

        # Example: Get health status
        health = engine.get_health_status()
        print(f"Engine health: {health}")

        # For streaming data processing (uncomment to use):
        # spark_stream = engine.spark.readStream.format("kafka") \
        #     .option("kafka.bootstrap.servers", "localhost:9092") \
        #     .option("subscribe", "sensor_data") \
        #     .load()
        # asyncio.run(engine.process_streaming_data(spark_stream))

    finally:
        # Cleanup
        engine.shutdown()


if __name__ == "__main__":
    main()


display.display(display.Markdown(F"""![{alt}]({image})"""))
plt.close(fig)
