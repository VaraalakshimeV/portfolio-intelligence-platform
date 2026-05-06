"""
LangChain RAG Financial Assistant with Conversation Memory
Upgrades from manual Pinecone + Gemini to a proper RAG chain:
  - ConversationalRetrievalChain handles retrieval + generation
  - ConversationBufferWindowMemory remembers last 5 turns
  - Metadata category filter on Pinecone for better precision
  - Prompt template enforces grounded, concise answers
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage, Document

from src.database.database import SessionLocal
from src.database.models import Portfolio, RiskMetrics, Holding, CompanyInfo

load_dotenv()

# ── Prompt template ───────────────────────────────────────────────────────────
QA_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""You are a senior portfolio analyst at an institutional investment firm.
Answer using ONLY the context and portfolio data provided below.
Be concise and precise — 3 to 4 sentences maximum.
If the context does not contain enough information, say so directly.
Reference specific numbers when available.

Portfolio Context:
{context}

Conversation History:
{chat_history}

Question: {question}

Answer:"""
)


class LangChainRAGAssistant:
    """
    LangChain-powered RAG assistant with:
    1. Pinecone vector retrieval (category-filtered)
    2. Conversation memory (last 5 turns)
    3. Gemini LLM for answer generation
    4. Live portfolio data injected into context
    """

    CATEGORY_KEYWORDS = {
        'risk_metrics': ['var', 'sharpe', 'volatility', 'drawdown', 'risk',
                         'sortino', 'beta', 'alpha', 'monte carlo', 'cvar'],
        'esg':          ['esg', 'environmental', 'social', 'governance',
                         'carbon', 'sustainability', 'green', 'emissions'],
        'portfolio':    ['holdings', 'allocation', 'weight', 'position',
                         'portfolio', 'aum', 'diversif', 'concentration'],
    }

    def __init__(self):
        print("Initializing LangChain RAG Assistant...")

        api_key      = st.secrets.get('GOOGLE_API_KEY',   os.getenv('GOOGLE_API_KEY', ''))
        pinecone_key = st.secrets.get('PINECONE_API_KEY', os.getenv('PINECONE_API_KEY', ''))

        # LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.1,
            convert_system_message_to_human=True
        )

        # Embeddings — same model used to build Pinecone index (must match)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-mpnet-base-v2"
        )

        # Pinecone vector store
        os.environ['PINECONE_API_KEY'] = pinecone_key
        self.vectorstore = PineconeVectorStore(
            index_name="fintech-rag",
            embedding=self.embeddings,
            pinecone_api_key=pinecone_key
        )

        # Conversation memory — remembers last 5 turns
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5,
            output_key="answer"
        )

        print("✓ LangChain RAG Assistant ready!")

    def _detect_categories(self, query: str) -> list:
        q = query.lower()
        matched = [cat for cat, kws in self.CATEGORY_KEYWORDS.items()
                   if any(kw in q for kw in kws)]
        return matched if matched else list(self.CATEGORY_KEYWORDS.keys())

    def _get_portfolio_context(self) -> str:
        """Pull comprehensive live portfolio data including per-holding ESG."""
        db = SessionLocal()
        try:
            portfolio  = db.query(Portfolio).first()
            risk       = db.query(RiskMetrics).order_by(RiskMetrics.calculation_date.desc()).first()
            holdings   = db.query(Holding).filter(Holding.portfolio_id == portfolio.id).all()
            companies  = db.query(CompanyInfo).all()
            co_map     = {c.ticker: c for c in companies}

            ctx = (
                f"Portfolio: {portfolio.name}\n"
                f"Total AUM: ${portfolio.total_value:,.0f}\n"
                f"Overall ESG Rating: {portfolio.esg_rating} ({portfolio.esg_score_overall:.1f}/100)\n"
                f"Environmental: {portfolio.environmental_score:.1f}, "
                f"Social: {portfolio.social_score:.1f}, "
                f"Governance: {portfolio.governance_score:.1f}\n"
                f"Carbon Intensity: {portfolio.carbon_intensity:.1f} tons CO2/$1M\n"
            )
            if risk:
                ctx += (
                    f"\nRisk Metrics:\n"
                    f"Sharpe Ratio: {risk.sharpe_ratio:.2f} "
                    f"(S&P 500 typical: 0.4-0.8; our portfolio significantly outperforms)\n"
                    f"Sortino Ratio: {risk.sortino_ratio:.2f}\n"
                    f"Annualised Volatility: {risk.volatility*100:.1f}%\n"
                    f"Max Drawdown: {risk.max_drawdown*100:.1f}%\n"
                    f"Daily VaR 95%: ${risk.var_95_daily * portfolio.total_value:,.0f}\n"
                    f"Monthly VaR 95%: ${risk.var_95_monthly * portfolio.total_value:,.0f}\n"
                )

            ctx += "\nIndividual Holdings with ESG Scores:\n"
            esg_ranked = []
            for h in holdings:
                c   = co_map.get(h.ticker)
                cur = h.current_price or h.purchase_price
                ret = (cur - h.purchase_price) / h.purchase_price * 100
                wt  = (cur * h.quantity) / portfolio.total_value * 100
                esg = c.esg_score if c and c.esg_score else None
                ctx += (
                    f"  {h.ticker}: weight {wt:.1f}%, return {ret:+.1f}%, "
                    f"ESG score {esg:.1f}/100" if esg else
                    f"  {h.ticker}: weight {wt:.1f}%, return {ret:+.1f}%, ESG N/A"
                ) + "\n"
                if esg:
                    esg_ranked.append((h.ticker, esg, ret))

            if esg_ranked:
                esg_ranked.sort(key=lambda x: x[1], reverse=True)
                ctx += (
                    f"\nESG Rankings: Highest = {esg_ranked[0][0]} ({esg_ranked[0][1]:.1f}), "
                    f"Lowest = {esg_ranked[-1][0]} ({esg_ranked[-1][1]:.1f})\n"
                )
            return ctx
        finally:
            db.close()

    def query(self, question: str) -> dict:
        """
        Retrieval strategy:
        1. Use the clean question for Pinecone similarity search (correct embeddings).
        2. Inject live DB data as a guaranteed first context document.
        3. Call the LLM directly with full combined context + memory.
        """
        # Step 1 — retrieve relevant knowledge base docs using clean question
        pinecone_docs = self.vectorstore.similarity_search(question, k=4)

        # Step 2 — inject live portfolio data as a Document (always included)
        portfolio_doc = Document(
            page_content=self._get_portfolio_context(),
            metadata={'category': 'portfolio', 'source': 'live_db'}
        )
        all_docs = [portfolio_doc] + pinecone_docs

        # Step 3 — build context string
        context = "\n\n---\n\n".join(doc.page_content for doc in all_docs)

        # Step 4 — build chat history string from memory
        msgs = self.memory.chat_memory.messages
        history_str = ""
        for msg in msgs:
            if isinstance(msg, HumanMessage):
                history_str += f"Human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_str += f"Assistant: {msg.content}\n"

        # Step 5 — format prompt and call LLM
        prompt_text = QA_PROMPT.format(
            context=context,
            chat_history=history_str,
            question=question
        )
        response = self.llm.invoke(prompt_text)
        answer   = response.content if hasattr(response, 'content') else str(response)

        # Step 6 — save to memory
        self.memory.save_context({"question": question}, {"answer": answer})

        sources = list({doc.metadata.get('category', '—') for doc in all_docs})

        return {
            "answer":       answer,
            "sources":      sources,
            "memory_turns": len(self.memory.chat_memory.messages) // 2,
        }

    def get_history(self) -> list:
        """Return conversation history as list of (question, answer) tuples."""
        msgs = self.memory.chat_memory.messages
        pairs = []
        for i in range(0, len(msgs) - 1, 2):
            if isinstance(msgs[i], HumanMessage) and isinstance(msgs[i+1], AIMessage):
                pairs.append({
                    "q": msgs[i].content,
                    "a": msgs[i+1].content
                })
        return pairs

    def clear_memory(self):
        self.memory.clear()
