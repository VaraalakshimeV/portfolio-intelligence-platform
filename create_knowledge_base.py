"""
Create Financial Knowledge Base for RAG
Stores financial concepts, formulas, and company data in Pinecone
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

print("=" * 70)
print("CREATING FINANCIAL KNOWLEDGE BASE")
print("=" * 70)

# Initialize Pinecone
print("\n1. Connecting to Pinecone...")
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index('fintech-rag')
print(f"   ✓ Connected to index: fintech-rag")

# Initialize embedding model
print("\n2. Loading embedding model...")
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print(f"   ✓ Model loaded (384 dimensions)")

# Financial knowledge documents
print("\n3. Preparing financial knowledge documents...")

knowledge_base = [
    {
        'id': 'var_definition',
        'text': """Value at Risk (VaR) is a statistical measure of the potential loss in portfolio value over a specific time period at a given confidence level. 
        For example, a daily VaR of $10,000 at 95% confidence means there's a 5% chance of losing more than $10,000 in one day. 
        VaR is calculated using three methods: Historical (using past returns), Parametric (assuming normal distribution), 
        and Monte Carlo (simulating future scenarios). Financial institutions use VaR for risk management and regulatory compliance.""",
        'category': 'risk_metrics'
    },
    {
        'id': 'sharpe_ratio',
        'text': """The Sharpe Ratio measures risk-adjusted returns by comparing excess returns to volatility. 
        Formula: (Portfolio Return - Risk-Free Rate) / Portfolio Standard Deviation. 
        A Sharpe Ratio above 1.0 is considered good, above 2.0 is very good, and above 3.0 is excellent. 
        It helps investors understand if returns justify the risk taken. Higher Sharpe ratios indicate better risk-adjusted performance.""",
        'category': 'risk_metrics'
    },
    {
        'id': 'esg_overview',
        'text': """ESG stands for Environmental, Social, and Governance - three pillars for measuring corporate sustainability. 
        Environmental criteria examine a company's carbon footprint, renewable energy use, and waste management. 
        Social factors include employee treatment, diversity, and community impact. 
        Governance assesses board independence, executive compensation, and shareholder rights. 
        ESG scores range from 0-100, with ratings from CCC (worst) to AAA (best) based on MSCI methodology.""",
        'category': 'esg'
    },
    {
        'id': 'esg_scoring',
        'text': """ESG scoring uses industry-specific weights because materiality varies by sector. 
        Technology companies emphasize environmental factors (40% weight) due to data center energy use. 
        Energy companies face higher environmental scrutiny (50% weight) for emissions. 
        Financial firms prioritize governance (50% weight) for regulatory compliance. 
        Scores aggregate multiple data points: SEC filings, EPA data, employee reviews, and proxy statements.""",
        'category': 'esg'
    },
    {
        'id': 'monte_carlo',
        'text': """Monte Carlo simulation forecasts portfolio risk by running thousands of random scenarios. 
        The method generates random returns based on historical mean and volatility, simulating potential future paths. 
        With 10,000 simulations, we can estimate the distribution of possible outcomes and calculate probabilistic risk measures. 
        It's more sophisticated than historical VaR because it accounts for various market conditions and non-normal distributions.""",
        'category': 'risk_metrics'
    },
    {
        'id': 'max_drawdown',
        'text': """Maximum Drawdown measures the largest peak-to-trough decline in portfolio value. 
        It represents the worst possible loss an investor would have experienced. For example, a 20% max drawdown means 
        the portfolio fell 20% from its highest point. This metric is crucial for understanding downside risk and 
        setting appropriate position sizes. Recovery time from drawdowns is also important - large drawdowns require 
        disproportionately large gains to recover.""",
        'category': 'risk_metrics'
    },
    {
        'id': 'diversification',
        'text': """Portfolio diversification reduces risk by holding uncorrelated assets. 
        A diversification ratio above 1.5 indicates good diversification. Sector concentration above 30% in any single sector 
        increases risk. Geographic diversification and asset class diversity (stocks, bonds, alternatives) further reduce volatility. 
        The correlation matrix shows how assets move together - lower correlation means better diversification.""",
        'category': 'portfolio'
    },
    {
        'id': 'carbon_footprint',
        'text': """Carbon footprint measures greenhouse gas emissions attributed to investments. 
        It's reported as tons of CO2 equivalent per million dollars invested (carbon intensity). 
        Investors can reduce portfolio carbon by avoiding fossil fuel companies, choosing renewable energy firms, 
        and investing in companies with strong climate commitments. Carbon intensity below 100 tons/$1M is considered low.""",
        'category': 'esg'
    },
    {
        'id': 'beta',
        'text': """Beta measures a portfolio's volatility relative to the market (usually S&P 500). 
        Beta = 1.0 means the portfolio moves with the market. Beta > 1.0 indicates higher volatility (aggressive). 
        Beta < 1.0 means less volatile (defensive). Beta is calculated using covariance between portfolio and market returns 
        divided by market variance. It's key for understanding systematic risk exposure.""",
        'category': 'risk_metrics'
    },
    {
        'id': 'alpha',
        'text': """Alpha measures excess returns above what would be expected given the portfolio's beta. 
        Positive alpha indicates outperformance - the portfolio generated returns beyond market exposure. 
        Alpha = Portfolio Return - [Risk-Free Rate + Beta × (Market Return - Risk-Free Rate)]. 
        Generating consistent positive alpha is difficult and indicates skillful active management.""",
        'category': 'risk_metrics'
    }
]

print(f"   ✓ Prepared {len(knowledge_base)} knowledge documents")

# Embed and upload to Pinecone
print("\n4. Embedding documents and uploading to Pinecone...")

vectors = []
for doc in knowledge_base:
    # Create embedding
    embedding = embedder.encode(doc['text']).tolist()
    
    # Prepare vector
    vectors.append({
        'id': doc['id'],
        'values': embedding,
        'metadata': {
            'text': doc['text'],
            'category': doc['category']
        }
    })
    
    print(f"   ✓ Embedded: {doc['id']}")

# Upload to Pinecone
print("\n5. Uploading vectors to Pinecone...")
index.upsert(vectors=vectors)

print(f"   ✓ Uploaded {len(vectors)} vectors")

# Verify
stats = index.describe_index_stats()
print(f"\n6. Verification:")
print(f"   ✓ Total vectors in index: {stats['total_vector_count']}")

print("\n" + "=" * 70)
print("✅ KNOWLEDGE BASE CREATED!")
print("=" * 70)

print("\n💡 What's in the knowledge base:")
print("   • Risk metrics definitions (VaR, Sharpe, Beta, Alpha)")
print("   • ESG scoring methodology")
print("   • Portfolio concepts")
print("   • Carbon footprint calculations")

print("\n🚀 Now the chatbot can answer questions using this knowledge!")