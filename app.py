from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import logging
from dotenv import load_dotenv
import os
from typing import Optional, Annotated, List
import models
from database import engine, SessionLocal, Base
from sqlalchemy.orm import Session
from datetime import datetime
import uuid


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create Database Tables
Base.metadata.create_all(bind=engine)

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

Groq_client = Groq(api_key=groq_api_key)

# Chat Completion Function

SYSTEM_PROMPT="""You are a helpful assistant who works for Revotic AI. Your job is to answer user queries concisely related to Revotic AI in a polite and professional
        tone. Revotic AI a software company which build intelligent automation tools, custom AI/ML, GenAI solutions, and next-level Web & App development.
        Their mission is to help startups, enterprises, and businesses unlock their true potential with future-ready technology.
        The Core Serices offered by Revotic AI are:
        - AI Automation
        - AI SaaS Solutions
        - Full-Stack Development
        - AI Dashboards
        - UI/UX Design
        Revotic AI has worked with top brands like hudabeauty, Toms, Lush and many others achieving 99% customer satisfaction.
        Their website to know more about them which is https://revoticai.com/ and email address is contact@revoticai.com and this web page 
        https://revoticai.com/contact/ for contact.
        You will only respond to queries regarding Revotic AI, in case of any irrelevant query just say
        I am sorry but I can cannot help you with that, however, I will be happy to answer any query regarding Revotic AI."""

def chat_completion(user_query):
    completion = Groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_query
        }],
        temperature=1,
        max_completion_tokens=2000,
        top_p=1,
        reasoning_effort="low",
        stream=False,
        stop=None
    )
    
    response = ""  
    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:  
            response += content
    
    return response


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request model
class QueryRequest(BaseModel):
    query: str = Field(max_length=1000)

# Response model
class QueryResponse(BaseModel):
    response: str
    status: str = "success"

# Models for Chat History
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
    limit: Optional[int] = 50
    offset: Optional[int] = 0

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Save Messages into the Database
def save_chat_to_db(db: Session, user_message: str, ai_response: str):
    try:
        chat_entry = models.ChatHistory(
            message=user_message,
            response=ai_response
        )
        db.add(chat_entry)
        db.commit()
        db.refresh(chat_entry)
        logger.info(f"Chat saved to database with ID: {chat_entry.id}")
        return chat_entry
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving chat to database: {str(e)}")
        raise e
    

# Chatbot Endpoint
@app.post("/ask", response_model=QueryResponse)
def ask_endpoint(request: QueryRequest, db: Session = Depends(get_db)):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        logger.info(f"Processing query of length: {len(request.query)}")
        
        # Get AI response
        ai_response = chat_completion(request.query)
        
        # Save to database
        save_chat_to_db(db, request.query, ai_response)
        
        logger.info("Query processed and saved successfully")
        return QueryResponse(response=ai_response)
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
# Chat History Endpoint
@app.get("/chat/history", response_model=List[ChatHistoryResponse])
def get_chat_history(limit: int = 50, offset: int = 0, db: Session = Depends(get_db)):
    """Get chat history with pagination"""
    try:
        chats = db.query(models.ChatHistory)\
                 .order_by(models.ChatHistory.timestamp.desc())\
                 .offset(offset)\
                 .limit(limit)\
                 .all()
        return chats
    except Exception as e:
        logger.error(f"Error fetching chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))











