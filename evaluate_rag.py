"""
RAG Evaluation Suite
Metrics: Context Precision, Context Recall, Answer Relevancy, Faithfulness
Uses sentence-transformers for embedding-based metrics and Gemini as LLM judge.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

genai.configure(api_key=os.getenv('GOOGLE_API_KEY', ''))

# ---------------------------------------------------------------------------
# Evaluation dataset: question, ground_truth, expected_doc_ids
# ---------------------------------------------------------------------------
EVAL_DATASET = [
    # ── VaR ──────────────────────────────────────────────────────────────
    {
        "question": "What is Value at Risk (VaR) and how is it calculated?",
        "ground_truth": "Value at Risk is a statistical measure of potential loss over a time period at a given confidence level. It is calculated using Historical, Parametric, or Monte Carlo methods.",
        "expected_doc_ids": ["var_definition", "monte_carlo"],
    },
    {
        "question": "What does a daily VaR of 10000 dollars at 95 percent confidence mean?",
        "ground_truth": "A daily VaR of $10,000 at 95% confidence means there is a 5% chance of losing more than $10,000 in one day.",
        "expected_doc_ids": ["var_definition"],
    },
    {
        "question": "What are the three methods for calculating VaR?",
        "ground_truth": "The three methods are Historical (using past returns), Parametric (assuming normal distribution), and Monte Carlo (simulating future scenarios).",
        "expected_doc_ids": ["var_definition", "monte_carlo"],
    },
    {
        "question": "How do financial institutions use VaR?",
        "ground_truth": "Financial institutions use VaR for risk management and regulatory compliance.",
        "expected_doc_ids": ["var_definition"],
    },
    {
        "question": "What is CVaR and how does it differ from VaR?",
        "ground_truth": "CVaR, or Conditional VaR, is the average of losses beyond the VaR threshold. It represents the expected loss in the worst cases that exceed VaR.",
        "expected_doc_ids": ["var_definition"],
    },
    {
        "question": "What confidence level is typically used for VaR calculations?",
        "ground_truth": "VaR is typically calculated at a 95% confidence level, meaning there is a 5% chance of exceeding the loss.",
        "expected_doc_ids": ["var_definition"],
    },
    # ── Sharpe Ratio ─────────────────────────────────────────────────────
    {
        "question": "What is the Sharpe Ratio and what is a good value?",
        "ground_truth": "The Sharpe Ratio measures risk-adjusted returns by comparing excess returns to volatility. Above 1.0 is good, above 2.0 is very good, and above 3.0 is excellent.",
        "expected_doc_ids": ["sharpe_ratio"],
    },
    {
        "question": "What is the formula for the Sharpe Ratio?",
        "ground_truth": "Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Standard Deviation.",
        "expected_doc_ids": ["sharpe_ratio"],
    },
    {
        "question": "What Sharpe Ratio is considered excellent performance?",
        "ground_truth": "A Sharpe Ratio above 3.0 is considered excellent performance.",
        "expected_doc_ids": ["sharpe_ratio"],
    },
    {
        "question": "Why do investors use the Sharpe Ratio?",
        "ground_truth": "Investors use the Sharpe Ratio to understand whether returns justify the risk taken. Higher Sharpe ratios indicate better risk-adjusted performance.",
        "expected_doc_ids": ["sharpe_ratio"],
    },
    {
        "question": "How does the Sharpe Ratio help compare two portfolios?",
        "ground_truth": "The Sharpe Ratio allows comparison of portfolios by normalizing returns against their volatility, so a portfolio with higher Sharpe has better risk-adjusted returns regardless of absolute return size.",
        "expected_doc_ids": ["sharpe_ratio"],
    },
    # ── ESG Overview ─────────────────────────────────────────────────────
    {
        "question": "What does ESG stand for and how are scores calculated?",
        "ground_truth": "ESG stands for Environmental, Social, and Governance. Scores range from 0-100 with ratings from CCC to AAA using industry-specific weights based on MSCI methodology.",
        "expected_doc_ids": ["esg_overview", "esg_scoring"],
    },
    {
        "question": "What do environmental criteria in ESG measure?",
        "ground_truth": "Environmental criteria examine a company's carbon footprint, renewable energy use, and waste management.",
        "expected_doc_ids": ["esg_overview"],
    },
    {
        "question": "What is the best possible ESG rating?",
        "ground_truth": "The best possible ESG rating is AAA, and the worst is CCC, based on the MSCI methodology rating scale.",
        "expected_doc_ids": ["esg_overview"],
    },
    {
        "question": "What does the governance pillar assess in ESG?",
        "ground_truth": "Governance assesses board independence, executive compensation, and shareholder rights.",
        "expected_doc_ids": ["esg_overview"],
    },
    {
        "question": "What score range does ESG use?",
        "ground_truth": "ESG scores range from 0 to 100.",
        "expected_doc_ids": ["esg_overview"],
    },
    # ── ESG Scoring ──────────────────────────────────────────────────────
    {
        "question": "Why are ESG weights industry-specific?",
        "ground_truth": "ESG scoring uses industry-specific weights because materiality varies by sector — different environmental or governance risks matter more depending on the industry.",
        "expected_doc_ids": ["esg_scoring"],
    },
    {
        "question": "What environmental weight do technology companies get in ESG scoring?",
        "ground_truth": "Technology companies get a 40% environmental weight due to data center energy use.",
        "expected_doc_ids": ["esg_scoring"],
    },
    {
        "question": "What data sources are used for ESG scoring?",
        "ground_truth": "ESG scores aggregate data from SEC filings, EPA data, employee reviews, and proxy statements.",
        "expected_doc_ids": ["esg_scoring"],
    },
    {
        "question": "Why do financial firms prioritize governance in ESG?",
        "ground_truth": "Financial firms prioritize governance with a 50% weight for regulatory compliance.",
        "expected_doc_ids": ["esg_scoring"],
    },
    {
        "question": "Why do energy companies have a higher environmental weight in ESG?",
        "ground_truth": "Energy companies face higher environmental scrutiny with a 50% environmental weight due to their emissions.",
        "expected_doc_ids": ["esg_scoring"],
    },
    # ── Monte Carlo ──────────────────────────────────────────────────────
    {
        "question": "What is Monte Carlo simulation and why is it used for risk?",
        "ground_truth": "Monte Carlo simulation forecasts portfolio risk by running thousands of random scenarios based on historical mean and volatility. It is more sophisticated than historical VaR because it accounts for non-normal distributions.",
        "expected_doc_ids": ["monte_carlo", "var_definition"],
    },
    {
        "question": "How many simulations does Monte Carlo typically run?",
        "ground_truth": "Monte Carlo typically runs 10,000 simulations to estimate the distribution of possible outcomes.",
        "expected_doc_ids": ["monte_carlo"],
    },
    {
        "question": "Why is Monte Carlo better than historical VaR for non-normal distributions?",
        "ground_truth": "Monte Carlo accounts for various market conditions and non-normal distributions, whereas historical VaR is limited to patterns already observed in past data.",
        "expected_doc_ids": ["monte_carlo"],
    },
    {
        "question": "What inputs does Monte Carlo simulation use to generate scenarios?",
        "ground_truth": "Monte Carlo generates random returns based on historical mean and volatility to simulate potential future paths.",
        "expected_doc_ids": ["monte_carlo"],
    },
    {
        "question": "How does Monte Carlo simulation estimate probabilistic risk measures?",
        "ground_truth": "By running thousands of simulations, Monte Carlo builds a distribution of possible outcomes from which probabilistic risk measures like VaR can be calculated.",
        "expected_doc_ids": ["monte_carlo"],
    },
    # ── Max Drawdown ─────────────────────────────────────────────────────
    {
        "question": "What is maximum drawdown and why does it matter?",
        "ground_truth": "Maximum drawdown is the largest peak-to-trough decline in portfolio value, representing the worst possible loss an investor would have experienced.",
        "expected_doc_ids": ["max_drawdown"],
    },
    {
        "question": "What does a 20 percent maximum drawdown mean?",
        "ground_truth": "A 20% max drawdown means the portfolio fell 20% from its highest point.",
        "expected_doc_ids": ["max_drawdown"],
    },
    {
        "question": "Why is recovery time from drawdowns important?",
        "ground_truth": "Recovery time is important because large drawdowns require disproportionately large gains to recover — a 50% loss requires a 100% gain to break even.",
        "expected_doc_ids": ["max_drawdown"],
    },
    {
        "question": "How is maximum drawdown used to set position sizes?",
        "ground_truth": "Maximum drawdown is crucial for setting appropriate position sizes because it reveals the worst downside risk the portfolio has historically experienced.",
        "expected_doc_ids": ["max_drawdown"],
    },
    {
        "question": "What makes large drawdowns particularly dangerous for investors?",
        "ground_truth": "Large drawdowns are dangerous because they require disproportionately large gains to recover — the larger the drawdown, the harder it is to return to previous peak value.",
        "expected_doc_ids": ["max_drawdown"],
    },
    # ── Diversification ──────────────────────────────────────────────────
    {
        "question": "How does portfolio diversification reduce risk?",
        "ground_truth": "Diversification reduces risk by holding uncorrelated assets. A diversification ratio above 1.5 indicates good diversification. Lower correlation between assets means better diversification.",
        "expected_doc_ids": ["diversification"],
    },
    {
        "question": "What diversification ratio indicates good diversification?",
        "ground_truth": "A diversification ratio above 1.5 indicates good diversification.",
        "expected_doc_ids": ["diversification"],
    },
    {
        "question": "What sector concentration level increases portfolio risk?",
        "ground_truth": "Sector concentration above 30% in any single sector increases risk.",
        "expected_doc_ids": ["diversification"],
    },
    {
        "question": "How does the correlation matrix help with portfolio diversification?",
        "ground_truth": "The correlation matrix shows how assets move together. Lower correlation between assets means better diversification and lower overall portfolio risk.",
        "expected_doc_ids": ["diversification"],
    },
    {
        "question": "What types of diversification reduce portfolio volatility?",
        "ground_truth": "Geographic diversification and asset class diversity across stocks, bonds, and alternatives further reduce volatility beyond sector diversification.",
        "expected_doc_ids": ["diversification"],
    },
    # ── Carbon Footprint ─────────────────────────────────────────────────
    {
        "question": "What is carbon footprint in investing and what is a low value?",
        "ground_truth": "Carbon footprint in investing is measured as tons of CO2 per million dollars invested. Carbon intensity below 100 tons per million dollars is considered low.",
        "expected_doc_ids": ["carbon_footprint", "esg_overview"],
    },
    {
        "question": "How is carbon intensity measured for a portfolio?",
        "ground_truth": "Carbon intensity is reported as tons of CO2 equivalent per million dollars invested.",
        "expected_doc_ids": ["carbon_footprint"],
    },
    {
        "question": "How can investors reduce their portfolio carbon footprint?",
        "ground_truth": "Investors can reduce portfolio carbon by avoiding fossil fuel companies, choosing renewable energy firms, and investing in companies with strong climate commitments.",
        "expected_doc_ids": ["carbon_footprint"],
    },
    {
        "question": "What threshold is considered low carbon intensity for a portfolio?",
        "ground_truth": "Carbon intensity below 100 tons of CO2 per million dollars invested is considered low.",
        "expected_doc_ids": ["carbon_footprint"],
    },
    {
        "question": "How does investing in renewable energy companies affect carbon footprint?",
        "ground_truth": "Choosing renewable energy firms helps reduce portfolio carbon footprint as part of a strategy to lower overall CO2 intensity.",
        "expected_doc_ids": ["carbon_footprint"],
    },
    # ── Beta ─────────────────────────────────────────────────────────────
    {
        "question": "What is Beta and how is it interpreted?",
        "ground_truth": "Beta measures portfolio volatility relative to the market. Beta of 1.0 moves with the market, above 1.0 is aggressive (more volatile), below 1.0 is defensive (less volatile).",
        "expected_doc_ids": ["beta"],
    },
    {
        "question": "What does a beta of 1.0 mean for a portfolio?",
        "ground_truth": "A beta of 1.0 means the portfolio moves with the market — it has the same volatility as the benchmark.",
        "expected_doc_ids": ["beta"],
    },
    {
        "question": "What does a beta above 1 indicate about a portfolio?",
        "ground_truth": "A beta above 1.0 indicates higher volatility than the market, making it an aggressive portfolio.",
        "expected_doc_ids": ["beta"],
    },
    {
        "question": "How is beta calculated mathematically?",
        "ground_truth": "Beta is calculated using the covariance between portfolio and market returns divided by the market variance.",
        "expected_doc_ids": ["beta"],
    },
    {
        "question": "What does a defensive portfolio beta look like?",
        "ground_truth": "A defensive portfolio has a beta below 1.0, meaning it is less volatile than the market.",
        "expected_doc_ids": ["beta"],
    },
    {
        "question": "What type of risk does beta measure?",
        "ground_truth": "Beta measures systematic risk — the exposure of a portfolio to broad market movements that cannot be diversified away.",
        "expected_doc_ids": ["beta"],
    },
    # ── Alpha ────────────────────────────────────────────────────────────
    {
        "question": "What is alpha in portfolio management?",
        "ground_truth": "Alpha measures excess returns above what would be expected given the portfolio's beta. Positive alpha indicates the portfolio outperformed market expectations.",
        "expected_doc_ids": ["alpha"],
    },
    {
        "question": "What does positive alpha mean for a portfolio?",
        "ground_truth": "Positive alpha indicates outperformance — the portfolio generated returns beyond what market exposure alone would explain.",
        "expected_doc_ids": ["alpha"],
    },
    {
        "question": "What is the formula for calculating alpha?",
        "ground_truth": "Alpha = Portfolio Return - [Risk-Free Rate + Beta × (Market Return - Risk-Free Rate)].",
        "expected_doc_ids": ["alpha"],
    },
    {
        "question": "Why is it difficult to generate consistent positive alpha?",
        "ground_truth": "Generating consistent positive alpha is difficult and indicates skillful active management, as markets are competitive and most returns can be explained by market exposure.",
        "expected_doc_ids": ["alpha"],
    },
    {
        "question": "How does alpha relate to beta in measuring portfolio performance?",
        "ground_truth": "Beta measures market exposure (systematic risk) while alpha measures returns above and beyond what that beta exposure would predict — together they explain total portfolio performance.",
        "expected_doc_ids": ["alpha", "beta"],
    },
    {
        "question": "What does it mean to outperform the market according to alpha?",
        "ground_truth": "Outperforming the market means generating positive alpha — returns that exceed what the portfolio's level of market risk would be expected to produce.",
        "expected_doc_ids": ["alpha"],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RAGEvaluator:
    RELEVANCE_THRESHOLD = 0.30  # cosine sim above which a doc is "relevant" (also used as retrieval filter)

    def __init__(self):
        print("Initializing RAG Evaluator...")
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY', ''))
        self.index = pc.Index('fintech-rag')
        self.embedder = SentenceTransformer('all-mpnet-base-v2')
        self.llm = genai.GenerativeModel('gemini-2.5-flash')
        print("✓ Evaluator ready\n")

    # ------------------------------------------------------------------ #
    # Retrieval                                                            #
    # ------------------------------------------------------------------ #

    def retrieve(self, question: str, top_k: int = 3) -> list[dict]:
        q_emb = self.embedder.encode(question)
        results = self.index.query(vector=q_emb.tolist(), top_k=top_k, include_metadata=True)
        docs = []
        for m in results["matches"]:
            doc_emb = self.embedder.encode(m["metadata"]["text"])
            sim = cosine_similarity(q_emb, doc_emb)
            if sim >= self.RELEVANCE_THRESHOLD:
                docs.append({
                    "id": m["id"],
                    "text": m["metadata"]["text"],
                    "category": m["metadata"]["category"],
                    "pinecone_score": m["score"],
                    "cosine_sim": round(sim, 3),
                })
        return docs

    # ------------------------------------------------------------------ #
    # Answer generation (RAG-only, no DB dependency)                      #
    # ------------------------------------------------------------------ #

    def generate_answer(self, question: str, context_docs: list[dict]) -> str:
        context = "\n\n".join(
            f"[Doc {i+1}] {doc['text'][:400]}" for i, doc in enumerate(context_docs)
        )
        prompt = f"""You are a concise financial advisor. Answer in 3-4 sentences using ONLY the context below.

CONTEXT:
{context}

QUESTION: {question}

Answer directly in 3-4 sentences. No preamble.
ANSWER:"""
        return self.llm.generate_content(prompt).text.strip()

    # ------------------------------------------------------------------ #
    # Metric 1 — Context Precision                                        #
    # Fraction of retrieved docs that are semantically relevant to query  #
    # ------------------------------------------------------------------ #

    def context_precision(self, question: str, retrieved_docs: list[dict]) -> float:
        if not retrieved_docs:
            return 0.0
        q_emb = self.embedder.encode(question)
        relevant = sum(
            1
            for doc in retrieved_docs
            if cosine_similarity(q_emb, self.embedder.encode(doc["text"])) >= self.RELEVANCE_THRESHOLD
        )
        return relevant / len(retrieved_docs)

    # ------------------------------------------------------------------ #
    # Metric 2 — Context Recall                                           #
    # How well the best retrieved doc covers the ground truth             #
    # ------------------------------------------------------------------ #

    def context_recall(self, ground_truth: str, retrieved_docs: list[dict]) -> float:
        if not retrieved_docs:
            return 0.0
        gt_emb = self.embedder.encode(ground_truth)
        scores = [
            cosine_similarity(gt_emb, self.embedder.encode(doc["text"]))
            for doc in retrieved_docs
        ]
        return max(scores)

    # ------------------------------------------------------------------ #
    # Metric 3 — Answer Relevancy                                         #
    # Cosine similarity between question embedding and answer embedding   #
    # ------------------------------------------------------------------ #

    def answer_relevancy(self, question: str, answer: str) -> float:
        q_emb = self.embedder.encode(question)
        a_emb = self.embedder.encode(answer)
        return cosine_similarity(q_emb, a_emb)

    # ------------------------------------------------------------------ #
    # Metric 4 — Faithfulness (Gemini-as-judge)                          #
    # Are the claims in the answer grounded in the retrieved context?     #
    # ------------------------------------------------------------------ #

    def faithfulness(self, answer: str, context_docs: list[dict]) -> float:
        context = "\n\n".join(doc["text"][:400] for doc in context_docs)
        prompt = f"""You are evaluating whether an AI answer is faithful to its source context.

CONTEXT:
{context}

ANSWER:
{answer}

Task: Identify each factual claim in the ANSWER. For each claim, decide if it is
SUPPORTED or NOT SUPPORTED by the CONTEXT.

Respond in JSON only:
{{"supported": <int>, "not_supported": <int>, "faithfulness_score": <float 0-1>}}"""
        try:
            response = self.llm.generate_content(prompt)
            raw = response.text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw.strip())
            return float(data.get("faithfulness_score", 0.0))
        except Exception:
            # Fallback: parse manually
            return 0.0

    # ------------------------------------------------------------------ #
    # Run full evaluation                                                  #
    # ------------------------------------------------------------------ #

    def evaluate(self, top_k: int = 3) -> list[dict]:
        results = []

        print("=" * 72)
        print("RAG EVALUATION RESULTS")
        print("=" * 72)

        for i, sample in enumerate(EVAL_DATASET, 1):
            q = sample["question"]
            gt = sample["ground_truth"]
            expected = sample["expected_doc_ids"]

            print(f"\n[{i}/{len(EVAL_DATASET)}] {q[:65]}...")

            # Retrieve
            retrieved = self.retrieve(q, top_k=top_k)
            retrieved_ids = [d["id"] for d in retrieved]

            # Generate answer
            answer = self.generate_answer(q, retrieved)

            # Compute metrics
            cp = self.context_precision(q, retrieved)
            cr = self.context_recall(gt, retrieved)
            ar = self.answer_relevancy(q, answer)
            f  = self.faithfulness(answer, retrieved)

            # Hit rate: did we retrieve the expected docs?
            hits = len(set(retrieved_ids) & set(expected))
            hit_rate = hits / len(expected) if expected else 0.0

            row = {
                "question": q,
                "retrieved_ids": retrieved_ids,
                "expected_ids": expected,
                "hit_rate": round(hit_rate, 3),
                "context_precision": round(cp, 3),
                "context_recall": round(cr, 3),
                "answer_relevancy": round(ar, 3),
                "faithfulness": round(f, 3),
                "answer": answer,
            }
            results.append(row)

            print(f"  Retrieved : {retrieved_ids}")
            print(f"  Expected  : {expected}")
            print(f"  Hit Rate  : {hit_rate:.2f}  |  CP: {cp:.2f}  |  CR: {cr:.2f}  |  AR: {ar:.2f}  |  Faith: {f:.2f}")

        # Summary
        print("\n" + "=" * 72)
        print("AGGREGATE METRICS")
        print("=" * 72)
        metrics = ["hit_rate", "context_precision", "context_recall", "answer_relevancy", "faithfulness"]
        for m in metrics:
            avg = np.mean([r[m] for r in results])
            print(f"  {m:<22} : {avg:.3f}")
        print("=" * 72)

        # Save results
        out_path = Path(__file__).parent / "rag_eval_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Full results saved to {out_path.name}")

        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    evaluator = RAGEvaluator()
    evaluator.evaluate(top_k=3)
