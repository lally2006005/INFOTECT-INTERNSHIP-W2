# INFOTECT-INTERNSHIP-W2
# Infotect internship week 2
# The Retrieval Engine

# Prerequisites
# You will need the previous dependencies plus fastapi and uvicorn:

pip install fastapi uvicorn langchain-openai langchain-pinecone langchain

# The Retrieval Engine Implementation

import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ==========================================
# 1. SETUP & INITIALIZATION
# ==========================================
app = FastAPI(title="DocuMind Enterprise - Retrieval Engine")

INDEX_NAME = "documind-enterprise-index"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o" # Production grade LLM

# Initialize Vector Store connection
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vector_store = PineconeVectorStore(
    index=pc.Index(INDEX_NAME), 
    embedding=embeddings, 
    text_key="text"
)

# Initialize LLM
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0)

# ==========================================
# 2. PROMPT INJECTION & CHAIN SETUP
# ==========================================

# Step A: Contextualize History-Aware Question
# This prompt converts follow-up questions into standalone queries based on chat history.
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

retriever = vector_store.as_retriever(search_kwargs={"k": 4})
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Step B: Strict System Prompt Injection (Compliance & Hallucination Guardrails)
# Forces context-only answers and precise refusal language when context lacks the answer.
system_prompt = (
    "You are 'DocuMind Enterprise', an advanced context-aware corporate brain. "
    "Use the following pieces of retrieved context to answer the question. "
    "CRITICAL RULES:\n"
    "1. Strict Context-Only: Answer the question strictly using ONLY the provided context.\n"
    "2. Hallucination Guardrail: If the information is not explicitly present in the provided context, "
    "you MUST issue a clear, standardized refusal exactly as: 'I don't know' or 'This is outside my scope'. "
    "Do NOT attempt to invent, extrapolate, or use external knowledge.\n"
    "3. No External Assumptions: If the user asks about external or high-profile topics not explicitly covered "
    "in the corporate documents (e.g., world politics, external figures), refuse immediately.\n\n"
    "Context:\n{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Step C: Combine into the Final Retrieval Chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# ==========================================
# 3. REQUEST / RESPONSE SCHEMAS
# ==========================================
class ChatMessageDTO(BaseModel):
    role: str  # "human" or "ai"
    content: str

class ChatRequest(BaseModel):
    input: str
    chat_history: List[ChatMessageDTO] = []

class ChatResponse(BaseModel):
    answer: str

def parse_chat_history(dto_list: List[ChatMessageDTO]) -> List[BaseMessage]:
    """Helper to convert API payload history to LangChain message objects."""
    history = []
    for msg in dto_list:
        if msg.role == "human":
            history.append(HumanMessage(content=msg.content))
        elif msg.role == "ai":
            history.append(AIMessage(content=msg.content))
    return history

# ==========================================
# 4. API ENDPOINT
# ==========================================
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        formatted_history = parse_chat_history(request.chat_history)
        
        # Execute the RAG chain
        response = rag_chain.invoke({
            "input": request.input,
            "chat_history": formatted_history
        })
        
        return ChatResponse(answer=response["answer"])
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API with: uvicorn main:app --reload

# Verification and Testing Focus
# Per the Week 2 testing blueprint, you need to verify Hallucination Refusal on external topics and History-Aware Retrieval for follow-ups. Run this test suite against the engine

import requests

API_URL = "http://127.0.0.1:8000/chat"

# -------------------------------------------------------------------
# Test 1: Hallucination Guardrail Test (High-profile external topic)
# -------------------------------------------------------------------
payload_external = {
    "input": "Who is the President of the USA?",
    "chat_history": []
}

response_1 = requests.post(API_URL, json=payload_external).json()
print("--- Test 1: Hallucination Guardrail ---")
print(f"Query: {payload_external['input']}")
print(f"Bot Output: {response_1['answer']}") 
# EXPECTED OUTPUT: "I don't know" or "This is outside my scope"

# -------------------------------------------------------------------
# Test 2: History-Aware Retrieval (Follow-up Question handling)
# -------------------------------------------------------------------
payload_followup = {
    "input": "How long does it take?",
    "chat_history": [
        {"role": "human", "content": "What is the policy for processing a standard corporate refund request?"},
        {"role": "ai", "content": "According to the corporate policy, standard refund requests are processed within 5 business days."}
    ]
}

response_2 = requests.post(API_URL, json=payload_followup).json()
print("\n--- Test 2: History-Aware Context Resolution ---")
print(f"Follow-up Query: {payload_followup['input']}")
print(f"Bot Output: {response_2['answer']}")
# EXPECTED OUTPUT: Resolves "it" to "processing a standard corporate refund request" and pulls from the context cleanly.
