"""
ESG Scoring Engine
Based on MSCI ESG Rating Methodology
Calculates Environmental, Social, and Governance scores
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ESGCalculator:
    """
    Calculate ESG scores using industry-standard methodology
    Based on MSCI ESG Rating framework
    """
    
    # Industry-specific weights (based on materiality)
    INDUSTRY_WEIGHTS = {
        'Technology': {'E': 0.40, 'S': 0.30, 'G': 0.30},
        'Energy': {'E': 0.50, 'S': 0.25, 'G': 0.25},
        'Healthcare': {'E': 0.20, 'S': 0.50, 'G': 0.30},
        'Financials': {'E': 0.15, 'S': 0.35, 'G': 0.50},
        'Consumer Discretionary': {'E': 0.30, 'S': 0.40, 'G': 0.30},
        'Consumer Staples': {'E': 0.35, 'S': 0.35, 'G': 0.30},
        'Industrials': {'E': 0.40, 'S': 0.35, 'G': 0.25},
        'Materials': {'E': 0.50, 'S': 0.30, 'G': 0.20},
        'Utilities': {'E': 0.55, 'S': 0.25, 'G': 0.20},
        'Real Estate': {'E': 0.45, 'S': 0.30, 'G': 0.25},
        'Communication Services': {'E': 0.25, 'S': 0.45, 'G': 0.30},
    }
    
    # Default weights if industry not specified
    DEFAULT_WEIGHTS = {'E': 0.33, 'S': 0.33, 'G': 0.34}
    
    def __init__(self):
        self.rating_scale = {
            'AAA': (85, 100),
            'AA': (70, 84),
            'A': (60, 69),
            'BBB': (50, 59),
            'BB': (40, 49),
            'B': (30, 39),
            'CCC': (0, 29)
        }
    
    def calculate_environmental_score(self, company_data: Dict) -> float:
        """
        Calculate Environmental Score (0-100)
        
        Factors:
        - Carbon emissions intensity
        - Renewable energy usage
        - Water usage efficiency
        - Waste management
        - Environmental innovations
        """
        
        # Extract environmental metrics
        carbon_intensity = company_data.get('carbon_intensity', 0)
        renewable_pct = company_data.get('renewable_energy_pct', 0)
        water_usage = company_data.get('water_usage', 0)
        waste_recycling = company_data.get('waste_recycling_pct', 0)
        innovations = company_data.get('environmental_innovations', 0)
        
        # Normalize and score each factor (0-100)
        
        # Carbon intensity (lower is better)
        # Assume industry average is 100 tons CO2/revenue
        carbon_score = max(0, 100 - (carbon_intensity / 100) * 100)
        
        # Renewable energy (higher is better)
        renewable_score = renewable_pct
        
        # Water efficiency (normalized)
        water_score = self._normalize_metric(water_usage, 0, 1000, inverse=True)
        
        # Waste recycling (higher is better)
        waste_score = waste_recycling
        
        # Innovations (more is better)
        innovation_score = min(100, innovations * 10)
        
        # Weighted average
        e_score = (
            carbon_score * 0.35 +
            renewable_score * 0.25 +
            water_score * 0.15 +
            waste_score * 0.15 +
            innovation_score * 0.10
        )
        
        return round(e_score, 2)
    
    def calculate_social_score(self, company_data: Dict) -> float:
        """
        Calculate Social Score (0-100)
        
        Factors:
        - Employee satisfaction
        - Diversity and inclusion
        - Labor practices
        - Human rights
        - Community investment
        """
        
        employee_satisfaction = company_data.get('employee_satisfaction', 50)
        diversity_score = company_data.get('diversity_score', 50)
        female_pct = company_data.get('female_employees_pct', 30)
        turnover_rate = company_data.get('employee_turnover_rate', 15)
        training_hours = company_data.get('training_hours_per_employee', 20)
        community_investment = company_data.get('community_investment', 0)
        labor_practices = company_data.get('labor_practices_score', 50)
        human_rights = company_data.get('human_rights_score', 50)
        
        # Score each factor
        
        # Employee satisfaction (already 0-100)
        satisfaction_score = employee_satisfaction
        
        # Diversity (combination of diversity score and gender balance)
        diversity_total = (diversity_score * 0.7 + self._normalize_metric(female_pct, 0, 50) * 0.3)
        
        # Employee retention (lower turnover is better)
        retention_score = max(0, 100 - (turnover_rate * 3))
        
        # Training investment
        training_score = min(100, training_hours * 2)
        
        # Community investment (normalized)
        community_score = self._normalize_metric(community_investment, 0, 10000000)
        
        # Labor and human rights (already 0-100)
        labor_score = labor_practices
        rights_score = human_rights
        
        # Weighted average
        s_score = (
            satisfaction_score * 0.20 +
            diversity_total * 0.20 +
            retention_score * 0.15 +
            training_score * 0.10 +
            community_score * 0.10 +
            labor_score * 0.15 +
            rights_score * 0.10
        )
        
        return round(s_score, 2)
    
    def calculate_governance_score(self, company_data: Dict) -> float:
        """
        Calculate Governance Score (0-100)
        
        Factors:
        - Board independence
        - Board diversity
        - Executive compensation fairness
        - Shareholder rights
        - Anti-corruption measures
        - Tax transparency
        """
        
        board_independence = company_data.get('board_independence', 50)
        board_diversity = company_data.get('board_diversity', 30)
        female_board = company_data.get('female_board_members', 20)
        exec_comp_ratio = company_data.get('executive_compensation_ratio', 200)
        shareholder_rights = company_data.get('shareholder_rights_score', 50)
        anti_corruption = company_data.get('anti_corruption_score', 50)
        tax_transparency = company_data.get('tax_transparency_score', 50)
        
        # Score each factor
        
        # Board independence (higher is better, target 75%+)
        independence_score = min(100, (board_independence / 75) * 100)
        
        # Board diversity
        diversity_score = (board_diversity * 0.6 + self._normalize_metric(female_board, 0, 50) * 0.4)
        
        # Executive compensation (lower ratio is better, penalize above 100:1)
        if exec_comp_ratio <= 100:
            comp_score = 100
        else:
            comp_score = max(0, 100 - ((exec_comp_ratio - 100) / 10))
        
        # Other factors (already 0-100)
        shareholder_score = shareholder_rights
        corruption_score = anti_corruption
        tax_score = tax_transparency
        
        # Weighted average
        g_score = (
            independence_score * 0.20 +
            diversity_score * 0.15 +
            comp_score * 0.15 +
            shareholder_score * 0.20 +
            corruption_score * 0.20 +
            tax_score * 0.10
        )
        
        return round(g_score, 2)
    
    def calculate_esg_score(self, company_data: Dict, 
                           sector: Optional[str] = None) -> Dict:
        """
        Calculate comprehensive ESG score
        
        Args:
            company_data: Dict with company metrics
            sector: Industry sector for weighted scoring
        
        Returns:
            Dict with E, S, G scores and overall rating
        """
        
        logger.info(f"Calculating ESG score for company in {sector or 'Unknown'} sector")
        
        # Calculate individual pillar scores
        e_score = self.calculate_environmental_score(company_data)
        s_score = self.calculate_social_score(company_data)
        g_score = self.calculate_governance_score(company_data)
        
        # Get industry weights
        weights = self.INDUSTRY_WEIGHTS.get(sector, self.DEFAULT_WEIGHTS)
        
        # Calculate weighted overall score
        overall_score = (
            e_score * weights['E'] +
            s_score * weights['S'] +
            g_score * weights['G']
        )
        
        # Determine rating
        rating = self._score_to_rating(overall_score)
        
        # Calculate controversy adjustment
        controversies = company_data.get('esg_controversies', 0)
        controversy_penalty = min(20, controversies * 5)  # Max 20 point penalty
        adjusted_score = max(0, overall_score - controversy_penalty)
        adjusted_rating = self._score_to_rating(adjusted_score)
        
        result = {
            'environmental_score': e_score,
            'social_score': s_score,
            'governance_score': g_score,
            'overall_score': round(overall_score, 2),
            'adjusted_score': round(adjusted_score, 2),
            'rating': rating,
            'adjusted_rating': adjusted_rating,
            'sector': sector or 'Unknown',
            'weights': weights,
            'controversies': controversies,
            'controversy_penalty': controversy_penalty,
            'calculation_date': datetime.now().isoformat()
        }
        
        logger.info(f"ESG Score calculated: {adjusted_rating} ({adjusted_score:.1f}/100)")
        
        return result
    
    def calculate_portfolio_esg(self, holdings: list) -> Dict:
        """
        Calculate weighted ESG score for entire portfolio
        
        Args:
            holdings: List of dicts with 'ticker', 'value', 'esg_data'
        
        Returns:
            Portfolio-level ESG metrics
        """
        
        total_value = sum(h['value'] for h in holdings)
        
        if total_value == 0:
            return {
                'portfolio_esg_score': 0,
                'portfolio_rating': 'N/A',
                'carbon_intensity': 0,
                'holdings_count': 0
            }
        
        # Calculate weighted scores
        weighted_e = 0
        weighted_s = 0
        weighted_g = 0
        total_carbon = 0
        
        for holding in holdings:
            weight = holding['value'] / total_value
            esg_data = holding.get('esg_data', {})
            
            weighted_e += esg_data.get('environmental_score', 50) * weight
            weighted_s += esg_data.get('social_score', 50) * weight
            weighted_g += esg_data.get('governance_score', 50) * weight
            
            # Carbon footprint
            carbon = esg_data.get('carbon_emissions', 0)
            total_carbon += carbon * weight
        
        # Overall portfolio score
        portfolio_score = (weighted_e + weighted_s + weighted_g) / 3
        portfolio_rating = self._score_to_rating(portfolio_score)
        
        # Carbon intensity (tons CO2 per $1M invested)
        carbon_intensity = (total_carbon / total_value) * 1000000 if total_value > 0 else 0
        
        return {
            'portfolio_esg_score': round(portfolio_score, 2),
            'portfolio_rating': portfolio_rating,
            'environmental_score': round(weighted_e, 2),
            'social_score': round(weighted_s, 2),
            'governance_score': round(weighted_g, 2),
            'carbon_intensity': round(carbon_intensity, 2),
            'carbon_footprint': round(total_carbon, 2),
            'holdings_count': len(holdings),
            'total_value': total_value
        }
    
    def calculate_esg_risk(self, esg_score: float, controversies: int) -> Dict:
        """
        Calculate ESG-related risks
        
        Returns:
            Risk assessment based on ESG performance
        """
        
        # Base risk (inverse of score)
        base_risk = 100 - esg_score
        
        # Controversy risk
        controversy_risk = min(50, controversies * 10)
        
        # Regulatory risk (low ESG = higher regulatory risk)
        if esg_score < 40:
            regulatory_risk = 80
        elif esg_score < 60:
            regulatory_risk = 50
        else:
            regulatory_risk = 20
        
        # Reputation risk
        reputation_risk = (base_risk * 0.6 + controversy_risk * 0.4)
        
        # Stranded assets risk (environmental transition risk)
        stranded_assets_risk = max(0, 100 - esg_score * 1.5)
        
        # Overall ESG risk score
        overall_risk = (
            base_risk * 0.30 +
            controversy_risk * 0.25 +
            regulatory_risk * 0.20 +
            reputation_risk * 0.15 +
            stranded_assets_risk * 0.10
        )
        
        return {
            'esg_risk_score': round(overall_risk, 2),
            'environmental_risk': round(stranded_assets_risk, 2),
            'social_risk': round(reputation_risk, 2),
            'governance_risk': round(regulatory_risk, 2),
            'controversy_risk': round(controversy_risk, 2),
            'risk_level': self._risk_to_level(overall_risk)
        }
    
    def _normalize_metric(self, value: float, min_val: float, max_val: float, 
                         inverse: bool = False) -> float:
        """Normalize a metric to 0-100 scale"""
        if max_val == min_val:
            return 50
        
        normalized = ((value - min_val) / (max_val - min_val)) * 100
        normalized = max(0, min(100, normalized))
        
        if inverse:
            normalized = 100 - normalized
        
        return normalized
    
    def _score_to_rating(self, score: float) -> str:
        """Convert numerical score to letter rating"""
        for rating, (min_score, max_score) in self.rating_scale.items():
            if min_score <= score <= max_score:
                return rating
        return 'CCC'
    
    def _risk_to_level(self, risk: float) -> str:
        """Convert risk score to risk level"""
        if risk >= 75:
            return 'Very High'
        elif risk >= 60:
            return 'High'
        elif risk >= 40:
            return 'Medium'
        elif risk >= 20:
            return 'Low'
        else:
            return 'Very Low'


# Example usage
if __name__ == "__main__":
    # Sample company data
    company_data = {
        'carbon_intensity': 50,
        'renewable_energy_pct': 60,
        'water_usage': 500,
        'waste_recycling_pct': 70,
        'environmental_innovations': 5,
        'employee_satisfaction': 75,
        'diversity_score': 65,
        'female_employees_pct': 45,
        'employee_turnover_rate': 10,
        'training_hours_per_employee': 40,
        'community_investment': 5000000,
        'labor_practices_score': 80,
        'human_rights_score': 85,
        'board_independence': 70,
        'board_diversity': 60,
        'female_board_members': 35,
        'executive_compensation_ratio': 120,
        'shareholder_rights_score': 75,
        'anti_corruption_score': 80,
        'tax_transparency_score': 70,
        'esg_controversies': 1
    }
    
    calculator = ESGCalculator()
    result = calculator.calculate_esg_score(company_data, sector='Technology')
    
    print("\n=== ESG SCORE ANALYSIS ===")
    print(f"Environmental Score: {result['environmental_score']}/100")
    print(f"Social Score: {result['social_score']}/100")
    print(f"Governance Score: {result['governance_score']}/100")
    print(f"\nOverall Score: {result['adjusted_score']}/100")
    print(f"ESG Rating: {result['adjusted_rating']}")
    print(f"Sector: {result['sector']}")
    print(f"Controversies: {result['controversies']} (Penalty: -{result['controversy_penalty']} pts)")