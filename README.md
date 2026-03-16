# Personalized Finance AI Assistant

This project implements a **personalized AI assistant for financial guidance** using LLM-based reasoning and structured user profiling.

The system maintains user context across interactions, allowing the assistant to generate recommendations based on a user’s financial situation, preferences, and goals.

---

## Features

- Personalized financial guidance
- Persistent user profiling
- Structured financial signal computation
- Hybrid memory architecture
- LLM-based reasoning and responses
- FastAPI backend for scalable interaction
- Voice and text interaction support

---

## System Architecture

User Query  
→ Session Context Retrieval  
→ User Profile + Financial Signals  
→ Prompt Construction  
→ LLM Response Generation  
→ Updated Context Memory

---

## Personalization Mechanism

The assistant builds a **persistent user profile** containing financial attributes such as:

- income
- expenses
- savings
- financial goals
- risk tolerance

These signals are used to generate responses tailored to the user's financial context.

---

## Tech Stack

- Python
- FastAPI
- OpenAI API
- LLM Systems
- Machine Learning
- Vector Retrieval

---

## Purpose

The goal of this project is to explore **personalized AI assistants** that can maintain user context and generate tailored financial insights over time.
