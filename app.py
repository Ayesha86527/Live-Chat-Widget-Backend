
import logging
import os
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import models
from database import Base, SessionLocal, engine


# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in the environment.")

groq_client = Groq(api_key=GROQ_API_KEY)


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="Revotic AI Chatbot API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Database Setup
# --------------------------------------------------

Base.metadata.create_all(bind=engine)


# --------------------------------------------------
# System Prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a helpful assistant who works for Revotic AI.

Your job is to answer user queries concisely related to Revotic AI
in a polite and professional tone.

Revotic AI is a software company that builds:
- Intelligent automation tools
- Custom AI/ML solutions
- Generative AI solutions
- Web and app development solutions

Their mission is to help startups, enterprises, and businesses
unlock their true potential with future-ready technology.

Core services offered by Revotic AI:
- AI Automation
- AI SaaS Solutions
- Full-Stack Development
- AI Dashboards
- UI/UX Design

Revotic AI has worked with brands like Huda Beauty, TOMS,
Lush, and many others.

Their website is:
https://revoticai.com/

Email:
contact@revoticai.com

Contact page:
https://revoticai.com/contact/

You should only respond to queries regarding Revotic AI.

For irrelevant queries, respond exactly:
"I am sorry but I cannot help you with that, however,
I will be happy to answer any query regarding Revotic AI."
"""


# --------------------------------------------------
# Chat Completion Function
# --------------------------------------------------

def chat_completion(user_query: str) -> str:
    """
    Send a user query to Groq and return the AI response.
    """

    completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_query,
            },
        ],
        temperature=1,
        max_completion_tokens=2000,
        top_p=1,
        reasoning_effort="low",
        stream=False,
    )

    response = completion.choices[0].message.content

    if not response:
        raise RuntimeError("The AI returned an empty response.")

    return response


# --------------------------------------------------
# Pydantic Schemas
# --------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )


class QueryResponse(BaseModel):
    response: str
    status: str = "success"


class ChatHistoryBase(BaseModel):
    message: str
    response: str


class ChatHistoryCreate(ChatHistoryBase):
    pass


class ChatHistoryResponse(ChatHistoryBase):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True


class ChatHistoryQuery(BaseModel):
    limit: Optional[int] = Field(default=50, ge=1, le=100)
    offset: Optional[int] = Field(default=0, ge=0)


# --------------------------------------------------
# Database Dependency
# --------------------------------------------------

def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------
# Save Chat to Database
# --------------------------------------------------

def save_chat_to_db(
    db: Session,
    user_message: str,
    ai_response: str,
):
    try:
        chat_entry = models.ChatHistory(
            message=user_message,
            response=ai_response,
        )

        db.add(chat_entry)
        db.commit()
        db.refresh(chat_entry)

        logger.info(
            "Chat saved to database with ID: %s",
            chat_entry.id,
        )

        return chat_entry

    except Exception:
        db.rollback()

        logger.exception("Error saving chat to database")

        raise


# --------------------------------------------------
# Health Check Endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Revotic AI Chatbot API is running",
        "status": "success",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }


# --------------------------------------------------
# Chatbot Endpoint
# --------------------------------------------------

@app.post("/ask", response_model=QueryResponse)
def ask_endpoint(
    request: QueryRequest,
    db: Session = Depends(get_db),
):
    user_query = request.query.strip()

    if not user_query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    try:
        logger.info(
            "Processing query of length: %s",
            len(user_query),
        )

        # Get AI response
        ai_response = chat_completion(user_query)

        # Save conversation to database
        save_chat_to_db(
            db=db,
            user_message=user_query,
            ai_response=ai_response,
        )

        logger.info("Query processed successfully")

        return QueryResponse(
            response=ai_response,
            status="success",
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception("Error processing chat request")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request.",
        )


# --------------------------------------------------
# Chat History Endpoint
# --------------------------------------------------

@app.get(
    "/chat/history",
    response_model=List[ChatHistoryResponse],
)
def get_chat_history(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """
    Get chat history with pagination.
    """

    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 100.",
        )

    if offset < 0:
        raise HTTPException(
            status_code=400,
            detail="Offset cannot be negative.",
        )

    try:

        chats = (
            db.query(models.ChatHistory)
            .order_by(models.ChatHistory.timestamp.asc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        return chats

    except Exception:
        logger.exception("Error fetching chat history")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while fetching chat history.",
        )
