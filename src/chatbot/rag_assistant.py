"""
Complete RAG Financial Assistant
Combines: Custom ML + Pinecone RAG + Gemini LLM
"""
import os
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
from src.database.database import SessionLocal
from src.database.models import Portfolio, RiskMetrics, Holding, CompanyInfo

load_dotenv()

# Configure Gemini
genai.configure(api_key=st.secrets.get('GOOGLE_API_KEY', os.getenv('GOOGLE_API_KEY', '')))

class RAGFinancialAssistant:
    """
    Complete AI assistant with:
    1. Custom ML models (risk prediction)
    2. RAG (Pinecone knowledge base)
    3. LLM (Gemini for natural language)
    """
    
    def __init__(self):
        print("Initializing RAG Assistant...")
        
        # Pinecone for RAG
        pc = Pinecone(api_key=st.secrets.get('PINECONE_API_KEY', os.getenv('PINECONE_API_KEY', '')))
        self.index = pc.Index('fintech-rag')
        
        # Embedding model — upgraded to 768-dim for better semantic recall
        self.embedder = SentenceTransformer('all-mpnet-base-v2')
        
        # Gemini LLM
        self.llm = genai.GenerativeModel('gemini-2.5-flash')
        
        print("✓ RAG Assistant ready!")
    
    RETRIEVAL_MIN_SCORE = 0.30

    CATEGORY_KEYWORDS = {
        'risk_metrics': ['var', 'sharpe', 'volatility', 'drawdown', 'risk',
                         'sortino', 'beta', 'alpha', 'monte carlo', 'cvar'],
        'esg':          ['esg', 'environmental', 'social', 'governance',
                         'carbon', 'sustainability', 'green', 'emissions'],
        'portfolio':    ['holdings', 'allocation', 'weight', 'position',
                         'portfolio', 'aum', 'diversif', 'concentration'],
    }

    def _detect_categories(self, query: str) -> list:
        q = query.lower()
        matched = [cat for cat, kws in self.CATEGORY_KEYWORDS.items()
                   if any(kw in q for kw in kws)]
        return matched if matched else list(self.CATEGORY_KEYWORDS.keys())

    def retrieve_knowledge(self, query: str, top_k: int = 3) -> list:
        """
        Retrieve from Pinecone with metadata category filter + cosine post-filter.
        Only searches within relevant categories — improves context precision.
        """
        query_embedding = self.embedder.encode(query)
        categories      = self._detect_categories(query)

        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter={"category": {"$in": categories}}
        )

        docs = []
        for match in results['matches']:
            doc_text = match['metadata']['text']
            doc_emb = self.embedder.encode(doc_text)
            # cosine similarity
            sim = float(np.dot(query_embedding, doc_emb) /
                        (np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)))
            if sim >= self.RETRIEVAL_MIN_SCORE:
                docs.append({
                    'text': doc_text,
                    'category': match['metadata']['category'],
                    'score': match['score'],
                    'cosine_sim': round(sim, 3),
                })

        return docs
    
    def query(self, user_question: str) -> dict:
        """
        Answer user question using hybrid approach
        
        Flow:
        1. Retrieve relevant knowledge from Pinecone (RAG)
        2. Get data from database (your calculations)
        3. Combine with Gemini for natural response
        """
        
        print(f"\n{'='*70}")
        print(f"Processing: {user_question}")
        print(f"{'='*70}")
        
        # Step 1: Retrieve relevant knowledge (RAG)
        print("\n1. Searching knowledge base...")
        knowledge_docs = self.retrieve_knowledge(user_question, top_k=2)
        
        if knowledge_docs:
            print(f"   ✓ Found {len(knowledge_docs)} relevant documents")
            for i, doc in enumerate(knowledge_docs):
                print(f"   • Doc {i+1}: {doc['category']} (score: {doc['score']:.3f})")
        
        # Step 2: Get real data from database
        print("\n2. Fetching portfolio data...")
        db = SessionLocal()
        try:
            portfolio = db.query(Portfolio).first()
            risk = db.query(RiskMetrics).order_by(RiskMetrics.calculation_date.desc()).first()
            holdings = db.query(Holding).filter(Holding.portfolio_id == portfolio.id).all()
            
            # Build context from YOUR data
            portfolio_context = f"""
Portfolio: {portfolio.name}
Value: ${portfolio.total_value:,.0f}
ESG Rating: {portfolio.esg_rating} ({portfolio.esg_score_overall:.1f}/100)
Holdings: {len(holdings)} stocks ({', '.join([h.ticker for h in holdings])})
"""
            
            if risk:
                portfolio_context += f"""
Risk Metrics (Custom Monte Carlo Calculation):
- Daily VaR: ${risk.var_95_daily * portfolio.total_value:,.0f}
- Sharpe Ratio: {risk.sharpe_ratio:.2f}
- Volatility: {risk.volatility*100:.1f}%
- Max Drawdown: {risk.max_drawdown*100:.1f}%
"""
            
            print(f"   ✓ Loaded portfolio data")
            
        finally:
            db.close()
        
        # Step 3: Build context from RAG
        knowledge_context = "\n\n".join([
            f"Reference {i+1}: {doc['text'][:300]}..."
            for i, doc in enumerate(knowledge_docs)
        ])
        
        # Step 4: Generate response with Gemini
        print("\n3. Generating response with Gemini...")
        
        prompt = f"""You are a concise financial advisor assistant. Answer in 3-4 sentences maximum.
Use ONLY the data and knowledge provided below — do not add external information.

PORTFOLIO DATA (custom calculations):
{portfolio_context}

FINANCIAL KNOWLEDGE (knowledge base):
{knowledge_context}

QUESTION: {user_question}

Answer directly in 3-4 sentences. Reference specific numbers from the portfolio data when relevant.
ANSWER:"""

        response = self.llm.generate_content(prompt)
        
        print(f"   ✓ Response generated")
        
        return {
            'answer': response.text,
            'knowledge_used': len(knowledge_docs),
            'rag_docs': knowledge_docs,
            'source': 'hybrid_rag_custom_ml'
        }


# Test the complete system
if __name__ == "__main__":
    print("=" * 70)
    print("TESTING COMPLETE RAG CHATBOT")
    print("=" * 70)
    
    assistant = RAGFinancialAssistant()
    
    # Test questions
    questions = [
        "What's my portfolio risk and what does VaR mean?",
        "Explain my ESG score",
        "What is the Sharpe ratio and what's mine?",
        "How does my portfolio carbon footprint compare?"
    ]
    
    for q in questions:
        response = assistant.query(q)
        
        print(f"\n💬 ANSWER:")
        print(response['answer'])
        print(f"\n📊 Knowledge docs used: {response['knowledge_used']}")
        print(f"🔧 Source: {response['source']}")
        print("\n" + "=" * 70)
    
    print("\n✅ RAG CHATBOT TEST COMPLETE!")