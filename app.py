
import logging
import os
from datetime import datetime
from typing import Generator, List

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

import models
from database import Base, SessionLocal, engine


# ==================================================
# CONFIGURATION
# ==================================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Please add it to your .env file."
    )

groq_client = Groq(api_key=GROQ_API_KEY)


# ==================================================
# FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Revotic AI Chatbot API",
    description="AI chatbot API for Revotic AI",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# DATABASE SETUP
# ==================================================

try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully.")

except Exception:
    logger.exception("Failed to create database tables.")
    raise


# ==================================================
# SYSTEM PROMPT
# ==================================================

SYSTEM_PROMPT = """
You are a helpful assistant who works for Revotic AI.

Your job is to answer user queries concisely related to Revotic AI
in a polite and professional tone.

Revotic AI is a software company that builds intelligent
automation tools, custom AI/ML solutions, generative AI solutions,
and next-level web and app development.

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

Website:
https://revoticai.com/

Email:
contact@revoticai.com

Contact page:
https://revoticai.com/contact/

Only respond to queries regarding Revotic AI.

For irrelevant queries, respond exactly:

"I am sorry but I cannot help you with that, however,
I will be happy to answer any query regarding Revotic AI."
"""


# ==================================================
# CHAT COMPLETION
# ==================================================

def chat_completion(user_query: str) -> str:
    """
    Send a user query to Groq and return the AI response.
    """

    try:
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

        if not completion.choices:
            raise RuntimeError(
                "The AI returned no completion choices."
            )

        response = completion.choices[0].message.content

        if not response or not response.strip():
            raise RuntimeError(
                "The AI returned an empty response."
            )

        return response.strip()

    except Exception:
        logger.exception("Groq API request failed.")
        raise


# ==================================================
# PYDANTIC SCHEMAS
# ==================================================

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User's question",
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

    model_config = ConfigDict(from_attributes=True)


class ChatHistoryQuery(BaseModel):
    limit: int = Field(
        default=50,
        ge=1,
        le=100,
    )

    offset: int = Field(
        default=0,
        ge=0,
    )


# ==================================================
# DATABASE DEPENDENCY
# ==================================================

def get_db() -> Generator[Session, None, None]:
    """
    Create and provide a database session.
    """

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


# ==================================================
# SAVE CHAT TO DATABASE
# ==================================================

def save_chat_to_db(
    db: Session,
    user_message: str,
    ai_response: str,
) -> models.ChatHistory:
    """
    Save a chat message and AI response to the database.
    """

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

        logger.exception(
            "Error saving chat to database."
        )

        raise


# ==================================================
# ROOT ENDPOINT
# ==================================================

@app.get("/")
def root():
    return {
        "message": "Revotic AI Chatbot API is running",
        "status": "success",
    }


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }


# ==================================================
# CHATBOT ENDPOINT
# ==================================================

@app.post(
    "/ask",
    response_model=QueryResponse,
)
def ask_endpoint(
    request: QueryRequest,
    db: Session = Depends(get_db),
):
    """
    Receive a user query, generate an AI response,
    and save the conversation to the database.
    """

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

        # Generate AI response
        ai_response = chat_completion(user_query)

        # Save conversation
        save_chat_to_db(
            db=db,
            user_message=user_query,
            ai_response=ai_response,
        )

        logger.info(
            "Query processed and saved successfully."
        )

        return QueryResponse(
            response=ai_response,
            status="success",
        )

    except HTTPException:
        raise

    except Exception:
        logger.exception(
            "Error processing chat request."
        )

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request.",
        )


# ==================================================
# CHAT HISTORY ENDPOINT
# ==================================================

@app.get(
    "/chat/history",
    response_model=List[ChatHistoryResponse],
)
def get_chat_history(
    limit: int = Query(
        default=50,
        ge=1,
        le=100,
        description="Number of chats to return",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of chats to skip",
    ),
    db: Session = Depends(get_db),
):
    """
    Get chat history with pagination.

    Returns newest chats first.
    """

    try:
        chats = (
            db.query(models.ChatHistory)
            .order_by(
                models.ChatHistory.timestamp.desc(),
                models.ChatHistory.id.desc(),
            )
            .offset(offset)
            .limit(limit)
            .all()
        )

        return chats

    except Exception:
        logger.exception(
            "Error fetching chat history."
        )

        raise HTTPException(
            status_code=500,
            detail="An error occurred while fetching chat history.",
        )
