# 🤖 Decision Debater AI: Multi-Agent Decision Intelligence System

## 🎯 System Overview
**Decision Debater AI** is a production-ready Multi-Agent Decision Intelligence System. It goes beyond a standard chatbot by utilizing a structured debate format between autonomous agents, grounded in factual knowledge via Retrieval-Augmented Generation (RAG) and real-time Web Search. 

The system orchestrates a Router, RAG, Web Search, Job Agent, Startup Agent, Critic Agent, and Judge Agent to deliver a structured, logically sound, and heavily analyzed final decision.

---

## ⚙️ Core Architecture

1. **Router Agent:** Analyzes the query and routes context retrieval to RAG, Web Search, or Both.
2. **RAG System:** Utilizes ChromaDB to fetch relevant embeddings from pre-loaded career and startup PDFs.
3. **Web Search Agent:** Uses Tavily/SerpAPI (represented via DuckDuckGo in this snippet for easy execution) to fetch real-time market data.
4. **Job Agent:** Argues the pro-job perspective (stability, security, benefits).
5. **Startup Agent:** Argues the pro-startup perspective (growth, equity, independence).
6. **Critic Agent:** Analyzes both arguments for logical flaws, bias, and missing risks.
7. **Judge Agent:** Synthesizes the debate, critic feedback, and retrieved context to produce a final, structured verdict.

---

## 💻 Python Implementation

Save the following code as `app.py`.

```python
import os
import json
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
# Ensure your OPENAI_API_KEY is set in your environment variables
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.2)
search_tool = DuckDuckGoSearchRun()

# ---------------------------------------------------------
# 2. RAG SYSTEM SETUP (ChromaDB)
# ---------------------------------------------------------
def setup_rag():
    """
    Initializes ChromaDB. In production, this loads your PDFs 
    (career advice, startup risk datasets, job market trends).
    """
    # Dummy setup for demonstration. Replace with actual document loaders and splitters.
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
    return vectorstore.as_retriever(search_kwargs={"k": 3})

rag_retriever = setup_rag()

# ---------------------------------------------------------
# 3. AGENT DEFINITIONS
# ---------------------------------------------------------

# --- A. Router Agent ---
router_prompt = PromptTemplate.from_template(
    "Analyze the following user decision query: '{query}'. "
    "Determine if answering this requires factual historical data (RAG), "
    "real-time/latest statistics (WEB), or both (BOTH). "
    "Output strictly one word: RAG, WEB, or BOTH."
)
router_chain = router_prompt | llm | StrOutputParser()

# --- B. Job Agent ---
job_prompt = PromptTemplate.from_template(
    "You are the Job Agent. Your goal is to argue in favor of taking a traditional job. "
    "Focus on stability, safety, long-term security, and guaranteed income. "
    "Use the following context to ground your arguments: {context}. "
    "Provide structured arguments in bullet points."
)
job_chain = job_prompt | llm | StrOutputParser()

# --- C. Startup Agent ---
startup_prompt = PromptTemplate.from_template(
    "You are the Startup Agent. Your goal is to argue in favor of starting a company or joining an early-stage startup. "
    "Focus on exponential growth, risk-taking, independence, and equity. "
    "Use the following context to ground your arguments: {context}. "
    "Provide structured arguments in bullet points."
)
startup_chain = startup_prompt | llm | StrOutputParser()

# --- D. Critic Agent ---
critic_prompt = PromptTemplate.from_template(
    "You are the Critic Agent. Review the Job Argument and the Startup Argument below. "
    "Find logical flaws, missing risks, and biases in BOTH sides. "
    "Job Argument: {job_arg}\n"
    "Startup Argument: {startup_arg}\n"
    "Output your critique clearly, pointing out weaknesses in both."
)
critic_chain = critic_prompt | llm | StrOutputParser()

# --- E. Judge Agent ---
judge_prompt = PromptTemplate.from_template(
    "You are the Judge Agent. Based on the Job Argument, Startup Argument, and Critic Analysis, "
    "make a final decision. Output strictly in the following JSON format: "
    "{{\n"
    "  \"Decision\": \"Job | Startup | Balanced\",\n"
    "  \"Job_Score\": \"<score out of 10>\",\n"
    "  \"Startup_Score\": \"<score out of 10>\",\n"
    "  \"Reasoning_Summary\": \"<short explanation>\"\n"
    "}}\n\n"
    "Job Argument: {job_arg}\n"
    "Startup Argument: {startup_arg}\n"
    "Critic Analysis: {critic_arg}"
)
judge_chain = judge_prompt | llm | StrOutputParser()

# ---------------------------------------------------------
# 4. ORCHESTRATION PIPELINE
# ---------------------------------------------------------
def run_debate_pipeline(query):
    # 1. Routing
    route = router_chain.invoke({"query": query}).strip().upper()
    
    # 2. Context Retrieval
    context = ""
    if route in ["RAG", "BOTH"]:
        try:
            docs = rag_retriever.invoke(query)
            context += "RAG Context: " + " ".join([d.page_content for d in docs]) + "\n"
        except Exception:
            context += "RAG Context: (Database empty/not initialized)\n"
            
    if route in ["WEB", "BOTH"]:
        web_results = search_tool.invoke(query + " statistics real-world data")
        context += "Web Context: " + web_results + "\n"
        
    # 3. Parallel Debate Generation
    job_argument = job_chain.invoke({"context": context})
    startup_argument = startup_chain.invoke({"context": context})
    
    # 4. Critic Evaluation
    critic_analysis = critic_chain.invoke({
        "job_arg": job_argument, 
        "startup_arg": startup_argument
    })
    
    # 5. Judge Verdict
    judge_output = judge_chain.invoke({
        "job_arg": job_argument,
        "startup_arg": startup_argument,
        "critic_arg": critic_analysis
    })
    
    try:
        final_verdict = json.loads(judge_output)
    except json.JSONDecodeError:
        final_verdict = {
            "Decision": "Balanced",
            "Job_Score": "N/A",
            "Startup_Score": "N/A",
            "Reasoning_Summary": "Failed to parse Judge JSON output."
        }
        
    return job_argument, startup_argument, critic_analysis, final_verdict

# ---------------------------------------------------------
# 5. GRADIO FRONTEND INTERFACE
# ---------------------------------------------------------
def gradio_interface(query):
    job, startup, critic, final = run_debate_pipeline(query)
    
    final_decision_str = final.get("Decision", "N/A")
    score_table = f"**Job:** {final.get('Job_Score', '0')}/10 | **Startup:** {final.get('Startup_Score', '0')}/10"
    reasoning = final.get("Reasoning_Summary", "N/A")
    
    return job, startup, critic, final_decision_str, score_table, reasoning

with gr.Blocks(theme=gr.themes.Soft(), title="Decision Debater AI") as demo:
    gr.Markdown("# ⚖️ Decision Debater AI")
    gr.Markdown("Enter your career or business dilemma (e.g., 'Job vs Startup?'). The system will route to RAG/Web, debate the options, critique them, and render a final verdict.")
    
    with gr.Row():
        user_input = gr.Textbox(label="User Question", placeholder="e.g., Should I quit my job to build an AI startup?", scale=4)
        analyze_btn = gr.Button("Analyze ⚡", variant="primary", scale=1)
        
    with gr.Row():
        with gr.Column():
            job_out = gr.Markdown(label="🏢 Job Agent Argument")
        with gr.Column():
            startup_out = gr.Markdown(label="🚀 Startup Agent Argument")
            
    with gr.Row():
        critic_out = gr.Markdown(label="🔍 Critic Analysis")
        
    with gr.Row():
        with gr.Group():
            gr.Markdown("### 🏆 Final Decision")
            decision_out = gr.Textbox(label="Verdict")
            score_out = gr.Markdown(label="Score Table")
            reasoning_out = gr.Textbox(label="Reasoning Summary", lines=3)

    analyze_btn.click(
        fn=gradio_interface,
        inputs=user_input,
        outputs=[job_out, startup_out, critic_out, decision_out, score_out, reasoning_out]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)