"""
LLM-based analyst agent for anomaly diagnosis.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import asdict

# LangChain imports
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback

# OpenAI direct import for advanced features
import openai

from ..models.anomaly_models import (
    AnomalyDetection, DiagnosisRequest, DiagnosisResult, Hypothesis,
    AnalysisTemplate, AnalysisContext, AnomalyType, AnomalySeverity,
    format_data_for_llm
)
from ...common.logger import get_logger


class ChainOfThoughtPrompts:
    """Chain-of-thought prompting templates for different analysis types."""
    
    SYSTEM_PROMPT = """You are an expert quantitative analyst specializing in high-frequency trading systems and cryptocurrency markets. Your role is to diagnose performance anomalies by analyzing trading data, market conditions, and system metrics.

Key capabilities:
- Deep understanding of crypto market microstructure
- Statistical analysis and anomaly detection
- System performance diagnosis
- Risk management and trading strategy analysis

Analysis approach:
1. Examine the data systematically
2. Identify patterns and correlations
3. Form hypotheses based on evidence
4. Consider multiple explanations
5. Provide actionable recommendations

Always think step-by-step and explain your reasoning clearly."""

    PERFORMANCE_DROP_ANALYSIS = """Analyze this performance drop anomaly using chain-of-thought reasoning:

**Step 1: Data Examination**
Review the provided data:
- PnL timeline and magnitude of drop
- Market conditions during the period
- System metrics and potential technical issues
- Trading activity and execution quality

**Step 2: Pattern Recognition**
Look for patterns:
- Timing correlations with market events
- Gradual vs sudden performance changes
- Correlation with specific trading pairs or strategies
- System resource utilization patterns

**Step 3: Hypothesis Formation**
Consider potential causes:
- Market regime changes (volatility, correlation breakdown)
- Technical issues (latency, connectivity, system overload)
- Strategy-specific problems (parameter drift, model degradation)
- External factors (funding rates, news events, regulatory changes)

**Step 4: Evidence Evaluation**
For each hypothesis, evaluate:
- Supporting evidence from the data
- Contradicting evidence
- Statistical significance of correlations
- Precedent from historical similar events

**Step 5: Root Cause Determination**
Determine the most likely root cause based on:
- Strength of evidence
- Timing alignment
- Magnitude of impact
- Logical causation chain

**Step 6: Recommendations**
Provide specific, actionable recommendations:
- Immediate actions to mitigate ongoing issues
- Preventive measures for future occurrences
- Monitoring improvements
- Strategy adjustments if needed

Data to analyze:
{data_summary}

Anomaly details:
{anomaly_details}

Please provide your analysis following this chain-of-thought approach."""

    CORRELATION_BREAK_ANALYSIS = """Analyze this correlation breakdown anomaly:

**Step 1: Correlation Analysis**
- Examine historical correlation patterns
- Identify when and how correlations changed
- Measure the magnitude of the breakdown
- Check if it's pair-specific or market-wide

**Step 2: Market Context Assessment**
- Review concurrent market events
- Check for regime changes or volatility spikes
- Analyze funding rate changes
- Look for news or regulatory developments

**Step 3: Strategy Impact Evaluation**
- Assess impact on cointegration-based strategies
- Evaluate hedge ratio effectiveness
- Check for increased basis risk
- Analyze position sizing implications

**Step 4: Causation Analysis**
Consider potential causes:
- Market structure changes
- Liquidity shifts
- Institutional flow changes
- Technical factors (exchange issues, API problems)

**Step 5: Recovery Assessment**
- Determine if correlation is recovering
- Estimate timeline for normalization
- Assess whether this is a permanent shift

**Step 6: Strategic Response**
- Recommend position adjustments
- Suggest risk management changes
- Propose monitoring enhancements
- Consider strategy parameter updates

Data: {data_summary}
Anomaly: {anomaly_details}"""

    FUNDING_RATE_ANALYSIS = """Analyze this funding rate anomaly:

**Step 1: Funding Rate Pattern Analysis**
- Examine funding rate levels and changes
- Compare to historical norms
- Identify affected trading pairs
- Assess rate change velocity and magnitude

**Step 2: Market Sentiment Assessment**
- Determine if rates indicate bullish/bearish sentiment
- Check for futures-spot basis changes
- Analyze open interest trends
- Review liquidation data

**Step 3: Strategy Impact Analysis**
- Evaluate impact on carry strategies
- Assess hedging cost changes
- Check for basis trading opportunities
- Analyze position financing implications

**Step 4: Causation Investigation**
- Look for market catalysts
- Check for large institutional flows
- Examine regulatory or news impacts
- Assess technical market factors

**Step 5: Persistence Evaluation**
- Determine if this is temporary or structural
- Assess mean reversion likelihood
- Consider seasonal or cyclical factors

**Step 6: Trading Adjustments**
- Recommend position size adjustments
- Suggest hedging strategy changes
- Propose new opportunity identification
- Advise on risk management updates

Data: {data_summary}
Anomaly: {anomaly_details}"""

    SYSTEM_ERROR_ANALYSIS = """Analyze this system error anomaly:

**Step 1: Error Pattern Analysis**
- Categorize error types and frequency
- Identify affected system components
- Analyze error timing and clustering
- Assess error severity and impact

**Step 2: System Resource Review**
- Check CPU, memory, and network utilization
- Analyze database performance metrics
- Review message queue health
- Examine API response times

**Step 3: Dependency Analysis**
- Check external service availability
- Review exchange connectivity
- Analyze third-party API status
- Assess network infrastructure

**Step 4: Code and Configuration Review**
- Look for recent deployments or changes
- Check configuration drift
- Analyze log patterns for code issues
- Review resource allocation settings

**Step 5: Impact Assessment**
- Quantify trading impact
- Assess data quality degradation
- Measure performance implications
- Evaluate risk exposure changes

**Step 6: Resolution Strategy**
- Provide immediate mitigation steps
- Suggest system improvements
- Recommend monitoring enhancements
- Propose preventive measures

Data: {data_summary}
Anomaly: {anomaly_details}"""


class LLMAnalyst:
    """LLM-powered analyst for anomaly diagnosis."""
    
    def __init__(
        self,
        model_name: str = "gpt-4-1106-preview",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = get_logger("llm_analyst")
        
        # Initialize OpenAI
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=openai.api_key
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Analysis templates
        self.templates = {
            AnomalyType.PERFORMANCE_DROP: ChainOfThoughtPrompts.PERFORMANCE_DROP_ANALYSIS,
            AnomalyType.CORRELATION_BREAK: ChainOfThoughtPrompts.CORRELATION_BREAK_ANALYSIS,
            AnomalyType.FUNDING_RATE_ANOMALY: ChainOfThoughtPrompts.FUNDING_RATE_ANALYSIS,
            AnomalyType.SYSTEM_ERROR: ChainOfThoughtPrompts.SYSTEM_ERROR_ANALYSIS
        }
        
        # Performance tracking
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.analysis_count = 0
    
    async def diagnose_anomaly(
        self,
        request: DiagnosisRequest,
        context_data: Dict[str, Any]
    ) -> DiagnosisResult:
        """Perform comprehensive anomaly diagnosis."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting diagnosis for anomaly {request.anomaly.id}")
            
            # Prepare analysis context
            analysis_context = await self._prepare_analysis_context(request, context_data)
            
            # Select appropriate analysis template
            template = self._select_analysis_template(request.anomaly.anomaly_type)
            
            # Perform chain-of-thought analysis
            primary_analysis = await self._perform_cot_analysis(
                template, analysis_context
            )
            
            # Generate alternative hypotheses
            alternative_hypotheses = await self._generate_alternative_hypotheses(
                analysis_context, primary_analysis
            )
            
            # Create executive summary
            executive_summary = await self._create_executive_summary(
                primary_analysis, alternative_hypotheses
            )
            
            # Extract actionable recommendations
            recommendations = self._extract_recommendations(primary_analysis)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                primary_analysis, analysis_context
            )
            
            # Create result
            result = DiagnosisResult(
                anomaly_id=request.anomaly.id,
                timestamp=datetime.now(),
                primary_hypothesis=self._parse_primary_hypothesis(primary_analysis),
                alternative_hypotheses=alternative_hypotheses,
                executive_summary=executive_summary,
                detailed_analysis=primary_analysis,
                immediate_actions=recommendations.get('immediate', []),
                preventive_measures=recommendations.get('preventive', []),
                monitoring_recommendations=recommendations.get('monitoring', []),
                analysis_duration=time.time() - start_time,
                llm_model_used=self.model_name,
                confidence_score=confidence_score,
                key_correlations=analysis_context.correlations,
                statistical_tests=analysis_context.statistical_tests
            )
            
            self.analysis_count += 1
            self.logger.info(f"Diagnosis completed for anomaly {request.anomaly.id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in anomaly diagnosis: {e}")
            
            # Return basic result with error information
            return DiagnosisResult(
                anomaly_id=request.anomaly.id,
                timestamp=datetime.now(),
                primary_hypothesis=Hypothesis(
                    title="Analysis Error",
                    description=f"Failed to complete analysis: {str(e)}",
                    confidence=0.0,
                    supporting_evidence=[],
                    recommended_actions=["Review system logs", "Retry analysis"]
                ),
                executive_summary=f"Analysis failed due to: {str(e)}",
                analysis_duration=time.time() - start_time,
                llm_model_used=self.model_name,
                confidence_score=0.0
            )
    
    async def _prepare_analysis_context(
        self,
        request: DiagnosisRequest,
        context_data: Dict[str, Any]
    ) -> AnalysisContext:
        """Prepare analysis context with processed data."""
        
        # Create analysis template
        template = AnalysisTemplate(
            name=f"{request.anomaly.anomaly_type.value}_analysis",
            description=f"Analysis template for {request.anomaly.anomaly_type.value}",
            data_requirements=request.anomaly.data_sources,
            analysis_steps=[
                "Data examination",
                "Pattern recognition", 
                "Hypothesis formation",
                "Evidence evaluation",
                "Root cause determination",
                "Recommendations"
            ],
            system_prompt=ChainOfThoughtPrompts.SYSTEM_PROMPT,
            analysis_prompt=self.templates.get(
                request.anomaly.anomaly_type,
                ChainOfThoughtPrompts.PERFORMANCE_DROP_ANALYSIS
            )
        )
        
        # Create context
        context = AnalysisContext(
            template=template,
            request=request
        )
        
        # Process collected data
        context.processed_data = await self._process_collected_data(context_data)
        
        # Calculate correlations
        context.correlations = self._calculate_correlations(context_data)
        
        # Perform statistical tests
        context.statistical_tests = self._perform_statistical_tests(context_data, request.anomaly)
        
        return context
    
    async def _process_collected_data(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process collected data for LLM consumption."""
        processed = {}
        
        # Process PnL data
        if 'pnl_data' in context_data:
            pnl_data = context_data['pnl_data']
            processed['pnl_summary'] = {
                'total_points': len(pnl_data.values),
                'statistics': pnl_data.get_statistics(),
                'recent_values': pnl_data.values[-10:] if pnl_data.values else [],
                'trend': 'declining' if len(pnl_data.values) > 1 and pnl_data.values[-1] < pnl_data.values[0] else 'stable'
            }
        
        # Process market context
        if 'market_context' in context_data:
            market_context = context_data['market_context']
            processed['market_summary'] = market_context.to_dict()
        
        # Process system metrics
        if 'system_metrics' in context_data:
            metrics = context_data['system_metrics']
            if metrics:
                processed['system_summary'] = {
                    'avg_cpu': np.mean([m.cpu_usage for m in metrics]),
                    'avg_memory': np.mean([m.memory_usage for m in metrics]),
                    'avg_latency': np.mean([m.network_latency for m in metrics]),
                    'total_errors': sum([m.error_rate for m in metrics]),
                    'peak_response_time': max([m.response_time_p95 for m in metrics])
                }
        
        # Process trading metrics
        if 'trading_metrics' in context_data:
            trading_metrics = context_data['trading_metrics']
            if trading_metrics:
                processed['trading_summary'] = {
                    'total_pnl': sum([tm.pnl for tm in trading_metrics]),
                    'avg_sharpe': np.mean([tm.sharpe_ratio for tm in trading_metrics]),
                    'max_drawdown': max([tm.max_drawdown for tm in trading_metrics]),
                    'avg_win_rate': np.mean([tm.win_rate for tm in trading_metrics]),
                    'total_trades': sum([tm.total_trades for tm in trading_metrics])
                }
        
        return processed
    
    def _calculate_correlations(self, context_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate key correlations in the data."""
        correlations = {}
        
        try:
            # Correlation between PnL and market volatility
            if 'pnl_data' in context_data and 'market_context' in context_data:
                pnl_values = context_data['pnl_data'].values
                if len(pnl_values) > 1:
                    market_vol = context_data['market_context'].market_volatility
                    # Simplified correlation (would need aligned time series)
                    correlations['pnl_market_volatility'] = 0.0  # Placeholder
            
            # System performance correlations
            if 'system_metrics' in context_data and 'trading_metrics' in context_data:
                sys_metrics = context_data['system_metrics']
                trading_metrics = context_data['trading_metrics']
                
                if sys_metrics and trading_metrics:
                    # Correlation between system latency and trading performance
                    latencies = [m.network_latency for m in sys_metrics]
                    pnls = [tm.pnl for tm in trading_metrics[:len(latencies)]]
                    
                    if len(latencies) > 1 and len(pnls) > 1:
                        corr = np.corrcoef(latencies, pnls)[0, 1]
                        if not np.isnan(corr):
                            correlations['latency_pnl'] = float(corr)
            
        except Exception as e:
            self.logger.warning(f"Error calculating correlations: {e}")
        
        return correlations
    
    def _perform_statistical_tests(
        self,
        context_data: Dict[str, Any],
        anomaly: AnomalyDetection
    ) -> Dict[str, Any]:
        """Perform statistical tests on the data."""
        tests = {}
        
        try:
            from scipy import stats
            
            # Test for PnL distribution changes
            if 'pnl_data' in context_data:
                pnl_values = context_data['pnl_data'].values
                if len(pnl_values) > 10:
                    # Split into before/after anomaly
                    split_point = len(pnl_values) // 2
                    before = pnl_values[:split_point]
                    after = pnl_values[split_point:]
                    
                    # T-test for mean difference
                    if len(before) > 1 and len(after) > 1:
                        t_stat, p_value = stats.ttest_ind(before, after)
                        tests['pnl_mean_change'] = {
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        }
                    
                    # Normality test
                    shapiro_stat, shapiro_p = stats.shapiro(pnl_values[-min(50, len(pnl_values)):])
                    tests['pnl_normality'] = {
                        'statistic': float(shapiro_stat),
                        'p_value': float(shapiro_p),
                        'is_normal': shapiro_p > 0.05
                    }
            
        except Exception as e:
            self.logger.warning(f"Error performing statistical tests: {e}")
        
        return tests
    
    async def _perform_cot_analysis(
        self,
        template: str,
        context: AnalysisContext
    ) -> str:
        """Perform chain-of-thought analysis using LLM."""
        
        # Prepare data summary
        data_summary = self._create_data_summary(context)
        anomaly_details = self._create_anomaly_summary(context.request.anomaly)
        
        # Format the prompt
        formatted_prompt = template.format(
            data_summary=data_summary,
            anomaly_details=anomaly_details
        )
        
        # Create messages
        messages = [
            SystemMessage(content=ChainOfThoughtPrompts.SYSTEM_PROMPT),
            HumanMessage(content=formatted_prompt)
        ]
        
        # Track token usage
        with get_openai_callback() as cb:
            try:
                response = await self.llm.agenerate([messages])
                analysis = response.generations[0][0].text
                
                # Update usage tracking
                self.total_tokens_used += cb.total_tokens
                self.total_cost += cb.total_cost
                
                context.add_message("assistant", analysis)
                context.token_usage = {
                    'prompt_tokens': cb.prompt_tokens,
                    'completion_tokens': cb.completion_tokens,
                    'total_tokens': cb.total_tokens,
                    'total_cost': cb.total_cost
                }
                
                return analysis
                
            except Exception as e:
                self.logger.error(f"Error in LLM analysis: {e}")
                return f"Analysis failed: {str(e)}"
    
    async def _generate_alternative_hypotheses(
        self,
        context: AnalysisContext,
        primary_analysis: str
    ) -> List[Hypothesis]:
        """Generate alternative hypotheses."""
        
        prompt = f"""Based on the primary analysis below, generate 2-3 alternative hypotheses that could explain the anomaly. For each hypothesis, provide:
1. Title (brief description)
2. Detailed explanation
3. Confidence level (0.0-1.0)
4. Supporting evidence
5. Recommended actions

Primary analysis:
{primary_analysis[:1000]}...

Please format your response as JSON with the following structure:
{{
  "hypotheses": [
    {{
      "title": "Alternative Hypothesis 1",
      "description": "Detailed explanation...",
      "confidence": 0.7,
      "supporting_evidence": ["Evidence 1", "Evidence 2"],
      "recommended_actions": ["Action 1", "Action 2"]
    }}
  ]
}}"""
        
        try:
            messages = [
                SystemMessage(content="You are an expert analyst generating alternative explanations."),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.agenerate([messages])
            response_text = response.generations[0][0].text
            
            # Parse JSON response
            try:
                parsed = json.loads(response_text)
                hypotheses = []
                
                for hyp_data in parsed.get('hypotheses', []):
                    hypothesis = Hypothesis(
                        title=hyp_data.get('title', 'Unknown'),
                        description=hyp_data.get('description', ''),
                        confidence=float(hyp_data.get('confidence', 0.5)),
                        supporting_evidence=hyp_data.get('supporting_evidence', []),
                        recommended_actions=hyp_data.get('recommended_actions', [])
                    )
                    hypotheses.append(hypothesis)
                
                return hypotheses
                
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse alternative hypotheses JSON")
                return []
                
        except Exception as e:
            self.logger.error(f"Error generating alternative hypotheses: {e}")
            return []
    
    async def _create_executive_summary(
        self,
        primary_analysis: str,
        alternative_hypotheses: List[Hypothesis]
    ) -> str:
        """Create executive summary."""
        
        prompt = f"""Create a concise executive summary (2-3 paragraphs) of the anomaly analysis for senior management. Include:
1. What happened (the anomaly)
2. Most likely cause
3. Impact assessment
4. Key recommended actions

Primary analysis:
{primary_analysis[:1500]}...

Alternative hypotheses:
{[h.title for h in alternative_hypotheses]}

Keep it business-focused and actionable."""
        
        try:
            messages = [
                SystemMessage(content="You are creating an executive summary for senior management."),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.agenerate([messages])
            return response.generations[0][0].text
            
        except Exception as e:
            self.logger.error(f"Error creating executive summary: {e}")
            return "Executive summary generation failed."
    
    def _select_analysis_template(self, anomaly_type: AnomalyType) -> str:
        """Select appropriate analysis template."""
        return self.templates.get(
            anomaly_type,
            ChainOfThoughtPrompts.PERFORMANCE_DROP_ANALYSIS
        )
    
    def _create_data_summary(self, context: AnalysisContext) -> str:
        """Create formatted data summary for LLM."""
        summary_parts = []
        
        for key, value in context.processed_data.items():
            formatted_value = format_data_for_llm(value, max_length=500)
            summary_parts.append(f"{key}:\n{formatted_value}\n")
        
        if context.correlations:
            corr_summary = "\n".join([f"{k}: {v:.3f}" for k, v in context.correlations.items()])
            summary_parts.append(f"Key Correlations:\n{corr_summary}\n")
        
        if context.statistical_tests:
            test_summary = format_data_for_llm(context.statistical_tests, max_length=300)
            summary_parts.append(f"Statistical Tests:\n{test_summary}\n")
        
        return "\n".join(summary_parts)
    
    def _create_anomaly_summary(self, anomaly: AnomalyDetection) -> str:
        """Create formatted anomaly summary."""
        return f"""Anomaly ID: {anomaly.id}
Type: {anomaly.anomaly_type.value}
Severity: {anomaly.severity.value}
Confidence: {anomaly.confidence:.2f}
Z-Score: {anomaly.z_score:.2f}
P-Value: {anomaly.p_value:.4f}
Affected Metrics: {', '.join(anomaly.affected_metrics)}
Detection Time: {anomaly.timestamp.isoformat()}"""
    
    def _parse_primary_hypothesis(self, analysis: str) -> Hypothesis:
        """Parse primary hypothesis from analysis text."""
        # This is a simplified parser - in production, would use more sophisticated NLP
        lines = analysis.split('\n')
        
        title = "Primary Hypothesis"
        description = analysis[:500] + "..." if len(analysis) > 500 else analysis
        confidence = 0.8  # Default confidence
        
        # Try to extract specific sections
        supporting_evidence = []
        recommended_actions = []
        
        current_section = None
        for line in lines:
            line = line.strip()
            if 'evidence' in line.lower() or 'support' in line.lower():
                current_section = 'evidence'
            elif 'recommend' in line.lower() or 'action' in line.lower():
                current_section = 'actions'
            elif line.startswith('-') or line.startswith('•'):
                if current_section == 'evidence':
                    supporting_evidence.append(line[1:].strip())
                elif current_section == 'actions':
                    recommended_actions.append(line[1:].strip())
        
        return Hypothesis(
            title=title,
            description=description,
            confidence=confidence,
            supporting_evidence=supporting_evidence,
            recommended_actions=recommended_actions
        )
    
    def _extract_recommendations(self, analysis: str) -> Dict[str, List[str]]:
        """Extract recommendations from analysis."""
        recommendations = {
            'immediate': [],
            'preventive': [],
            'monitoring': []
        }
        
        lines = analysis.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if 'immediate' in line.lower():
                current_section = 'immediate'
            elif 'preventive' in line.lower() or 'prevent' in line.lower():
                current_section = 'preventive'
            elif 'monitor' in line.lower():
                current_section = 'monitoring'
            elif line.startswith('-') or line.startswith('•'):
                if current_section:
                    recommendations[current_section].append(line[1:].strip())
        
        return recommendations
    
    def _calculate_confidence_score(
        self,
        analysis: str,
        context: AnalysisContext
    ) -> float:
        """Calculate overall confidence score for the analysis."""
        base_confidence = 0.7
        
        # Adjust based on data quality
        if context.processed_data:
            data_quality = len(context.processed_data) / 5.0  # Normalize
            base_confidence += min(0.2, data_quality * 0.1)
        
        # Adjust based on statistical significance
        if context.statistical_tests:
            significant_tests = sum(
                1 for test in context.statistical_tests.values()
                if isinstance(test, dict) and test.get('significant', False)
            )
            if significant_tests > 0:
                base_confidence += 0.1
        
        # Adjust based on correlation strength
        if context.correlations:
            strong_correlations = sum(
                1 for corr in context.correlations.values()
                if abs(corr) > 0.5
            )
            if strong_correlations > 0:
                base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the analyst."""
        return {
            'total_analyses': self.analysis_count,
            'total_tokens_used': self.total_tokens_used,
            'total_cost_usd': self.total_cost,
            'avg_tokens_per_analysis': self.total_tokens_used / max(1, self.analysis_count),
            'avg_cost_per_analysis': self.total_cost / max(1, self.analysis_count)
        }
