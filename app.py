import os
import json
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI  # Added standard client for Groq routing

import vector_store
import booking_tools

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Saumya Jain AI Persona Chatbot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Detect LLM provider based on available API keys
openai_api_key = os.environ.get("OPENAI_API_KEY")
gemini_api_key = os.environ.get("GEMINI_API_KEY")
groq_api_key = os.environ.get("GROQ_API_KEY")

# Initialize unified voice client & model
voice_client = None
voice_model = None

# Primary fallback logic including Groq for low-latency voice
if groq_api_key:
    llm_provider = "groq"
    voice_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_api_key)
    voice_model = "llama-3.3-70b-versatile"
    logger.info("Voice LLM Provider: Groq (llama-3.3-70b-versatile)")
elif openai_api_key:
    llm_provider = "openai"
    voice_client = OpenAI(api_key=openai_api_key)
    voice_model = "gpt-4o-mini"
    logger.info("Voice LLM Provider: OpenAI (gpt-4o-mini)")
elif gemini_api_key:
    llm_provider = "gemini"
    voice_client = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=gemini_api_key
    )
    voice_model = "gemini-2.0-flash"
    logger.info("Voice LLM Provider: Google Gemini via OpenAI-compat (gemini-2.0-flash)")
else:
    llm_provider = None
    logger.warning("No LLM API keys found.")

# Keep groq_client alias to prevent name errors elsewhere
groq_client = voice_client

# Tools for meeting booking
def check_available_slots() -> str:
    logger.info("Tool check_available_slots invoked.")
    slots = booking_tools.get_available_slots()
    if not slots:
        return "No available slots found in the next 7 days."
    return "Available slots (first 3):\n" + "\n".join(slots)

def create_booking(start_time: str, name: str, email: str) -> str:
    logger.info(f"Tool create_booking invoked for {name} ({email}) at {start_time}.")
    result = booking_tools.book_meeting(start_time, name, email)
    if "error" in result:
        return f"Error booking meeting: {result['error']}"
    data = result.get("data", {})
    uid = data.get("uid", "")
    meeting_url = data.get("meetingUrl") or data.get("location") or f"https://cal.com/booking/{uid}"
    title = data.get("title", "Meeting")
    return f"Success! Meeting '{title}' scheduled. Booking ID: {uid}. Link: {meeting_url}"

# OpenAI tools schema
openai_tools = [
    {
        "type": "function",
        "function": {
            "name": "check_available_slots",
            "description": "Check the available meeting slots on Cal.com for the next 7 days.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_booking",
            "description": "Book a meeting on Cal.com for a specific slot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "ISO 8601 timestamp format."},
                    "name": {"type": "string"},
                    "email": {"type": "string"}
                },
                "required": ["start_time", "name", "email"]
            }
        }
    }
]

openai_client = None
_stream_gemini = None

if llm_provider == "openai":
    from openai import AsyncOpenAI
    openai_client = AsyncOpenAI(api_key=openai_api_key)
elif llm_provider == "gemini":
    import google.generativeai as genai
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel("gemini-2.0-flash", tools=[check_available_slots, create_booking])

    async def _stream_gemini_impl(chat, message):
        stream = await chat.send_message_async(message)
        function_calls = []
        async for chunk in stream:
            if hasattr(chunk, "candidates") and chunk.candidates:
                for candidate in chunk.candidates:
                    if hasattr(candidate, "content") and candidate.content and hasattr(candidate.content, "parts"):
                        for part in candidate.content.parts:
                            if hasattr(part, "function_call") and part.function_call:
                                function_calls.append(part.function_call)
            try:
                if chunk.text:
                    yield f"data: {json.dumps({'content': chunk.text})}\n\n"
            except ValueError:
                pass
        
        if function_calls:
            for fn in function_calls:
                fn_name = fn.name
                fn_args = dict(fn.args) if fn.args else {}
                if fn_name in ("check_available_slots", "get_available_slots"):
                    res = check_available_slots()
                elif fn_name in ("create_booking", "book_meeting"):
                    res = create_booking(fn_args.get("start_time"), fn_args.get("name"), fn_args.get("email"))
                else:
                    res = "Unrecognized tool"
                
                from google.ai import generativelanguage as glm
                response_part = glm.Part(
                    function_response=glm.FunctionResponse(
                        name=fn_name,
                        response={"result": res}
                    )
                )
                follow_up_stream = await chat.send_message_async(response_part)
                async for chunk in follow_up_stream:
                    try:
                        if chunk.text:
                            yield f"data: {json.dumps({'content': chunk.text})}\n\n"
                    except ValueError:
                        pass
        yield "data: [DONE]\n\n"
    _stream_gemini = _stream_gemini_impl

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

def _build_system_prompt(chunks: List[Dict[str, Any]]) -> str:
    # 1. Base Persona (Saumya's details from resume)
    base_persona = (
        "You are Saumya Jain,based in Bengaluru a highly skilled Data Science and AI/ML Engineer. "
        "Speak in the first person ('I', 'my', 'me'). Be conversational, professional, warm, and direct. "
        "Since this is a voice assistant, keep your responses concise, friendly, and natural (1 to 3 sentences max).\n\n"
        "Here is your background and biography:\n"
        "- Education: Scaler School of Technology (2024-2028, Bachelor's + Master's in CS) and BITS Pilani (2024-2027, Bachelor's in CS).\n"
        "- Technical Skills: Python, SQL, FastAPI, ChromaDB, SBERT, LangChain, PostgreSQL, Scikit-learn, XGBoost, Random Forest, Collaborative Filtering, Pandas, NumPy.\n"
        "- Key Projects:\n"
        "  1. papertrail: A PDF-constrained conversational agent using PyMuPDF, LangChain, SBERT, ChromaDB, Llama 3.1 via Groq. Features zero hallucination, hinge/multilingual parsing, and 0.25 similarity refusal threshold.\n"
        "  2. Semantic Resume Screening System: Custom Named Entity Recognition (NER) extractor and SBERT embeddings on a FastAPI backend. Achieved 85%+ accuracy.\n"
        "  3. Customer Churn Prediction System: Built automated SQL/ETL feature pipeline on PostgreSQL and compared XGBoost, Random Forest, and Logistic Regression models.\n"
        "  4. Time-Aware Movie Recommendation System: Temporal collaborative filtering with time-decay weighting to outperform matrix-factorization baselines.\n"
        "  5. Smart Attendance Dashboard (AcademiQ): FastAPI + React full-stack platform with ML flagging at-risk students (80%+ recall).\n"
        "- Achievements: Ranked top 10 out of 400 students for ML project; Andrew Ng ML Course and IBM Data Science Professional certifications.\n"
        "- Contact Info: work.saumyaajain@gmail.com, +91-9915001452, linkedin.com/in/saumyajain, github.com/saumyajain03."
    )
    
    # 2. Append RAG context if available
    context_parts = []
    for i, chunk in enumerate(chunks):
        source = chunk["metadata"].get("source", "unknown")
        if source == "github":
            repo_name = chunk["metadata"].get("name", "unknown")
            context_parts.append(f"[Reference {i+1}] GitHub Repo ({repo_name}):\n{chunk['content']}")
        else:
            context_parts.append(f"[Reference {i+1}] Resume Document:\n{chunk['content']}")
            
    if context_parts:
        context = "\n\n---\n\n".join(context_parts)
        rag_prompt = (
            "\n\nUse this additional reference context from your documents and repository metadata to answer specific questions accurately:\n"
            f"{context}\n"
            "If a user asks about technical details of your code or project files, use this reference data. "
            "If the reference context does not contain the answer, answer naturally using your base persona."
        )
        return f"{base_persona}{rag_prompt}"
    else:
        return base_persona

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        store = vector_store.get_default_store()
        chunks = store.query(request.message, top_k=5)
        system_prompt = _build_system_prompt(chunks)
    except Exception as e:
        logger.error(f"Vector store query failed: {e}")
        system_prompt = "You are Saumya Jain, a AI/ML engineer. Be clear and concise."
        
    if llm_provider == "gemini":
        gemini_history = []
        for msg in request.history:
            if msg.role == "system":
                continue
            role = "user" if msg.role == "user" else "model"
            gemini_history.append({
                "role": role,
                "parts": [msg.content]
            })
            
        import google.generativeai as genai
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            tools=[check_available_slots, create_booking],
            system_instruction=system_prompt
        )
        
        chat = model.start_chat(history=gemini_history)
        
        import sys
        stream_func = sys.modules[__name__]._stream_gemini
        
        return StreamingResponse(stream_func(chat, request.message), media_type="text/event-stream")
        
    elif llm_provider == "openai" or llm_provider == "groq":
        openai_messages = []
        openai_messages.append({"role": "system", "content": system_prompt})
        for msg in request.history:
            openai_messages.append({"role": msg.role, "content": msg.content})
        openai_messages.append({"role": "user", "content": request.message})
        
        import sys
        module = sys.modules[__name__]
        client_to_use = module.openai_client or voice_client
        
        async def openai_sse_generator():
            try:
                import inspect
                res_or_coro = client_to_use.chat.completions.create(
                    model="gpt-4o-mini" if llm_provider == "openai" else voice_model,
                    messages=openai_messages,
                    tools=openai_tools,
                    stream=True
                )
                if inspect.isawaitable(res_or_coro):
                    response = await res_or_coro
                else:
                    response = res_or_coro
                
                tool_calls = []
                async for chunk in response:
                    choice = chunk.choices[0] if chunk.choices else None
                    if not choice:
                        continue
                    
                    delta_content = choice.delta.content
                    if delta_content:
                        yield f"data: {json.dumps({'content': delta_content})}\n\n"
                        
                    delta_tool_calls = choice.delta.tool_calls
                    if delta_tool_calls:
                        for tc in delta_tool_calls:
                            if len(tool_calls) <= tc.index:
                                tool_calls.append({
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""}
                                })
                            curr = tool_calls[tc.index]
                            if tc.id:
                                curr["id"] = tc.id
                            if tc.function:
                                if tc.function.name:
                                    curr["function"]["name"] += tc.function.name
                                if tc.function.arguments:
                                    curr["function"]["arguments"] += tc.function.arguments
                
                if tool_calls:
                    assistant_msg = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls
                    }
                    new_messages = list(openai_messages)
                    new_messages.append(assistant_msg)
                    
                    for tc in tool_calls:
                        fn_name = tc["function"]["name"]
                        try:
                            fn_args = json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {}
                        except Exception:
                            fn_args = {}
                            
                        if fn_name in ("check_available_slots", "get_available_slots"):
                            res = check_available_slots()
                        elif fn_name in ("create_booking", "book_meeting"):
                            res = create_booking(fn_args.get("start_time"), fn_args.get("name"), fn_args.get("email"))
                        else:
                            res = "Unrecognized tool"
                            
                        tool_response_msg = {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": fn_name,
                            "content": res
                        }
                        new_messages.append(tool_response_msg)
                        
                    res_or_coro2 = client_to_use.chat.completions.create(
                        model="gpt-4o-mini" if llm_provider == "openai" else voice_model,
                        messages=new_messages,
                        stream=True
                    )
                    if inspect.isawaitable(res_or_coro2):
                        second_response = await res_or_coro2
                    else:
                        second_response = res_or_coro2
                    
                    async for chunk in second_response:
                        choice = chunk.choices[0] if chunk.choices else None
                        if choice and choice.delta.content:
                            yield f"data: {json.dumps({'content': choice.delta.content})}\n\n"
                
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error in OpenAI streaming: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                
        return StreamingResponse(openai_sse_generator(), media_type="text/event-stream")
    else:
        raise HTTPException(status_code=500, detail="No LLM provider configured.")

# ------------------------------------------------------------------
# NEW WORKER: Dedicated Endpoint for Vapi Voice Calls 🎙️
# ------------------------------------------------------------------
@app.post("/chat/completions")
async def vapi_chat_completions(request: Request):
    """
    OpenAI-compatible endpoint directly handling Vapi requests.
    """
    try:
        payload = await request.json()
        logger.info(f"Incoming payload keys: {list(payload.keys())}")
        incoming_messages = payload.get("messages", [])
        
        # 1. Safely extract latest user query
        latest_query = ""
        for m in reversed(incoming_messages):
            if m.get("role") == "user":
                latest_query = m.get("content", "")
                break
        
        # 2. Grab Context (with fallback if vector store fails)
        try:
            store = vector_store.get_default_store()
            chunks = store.query(latest_query, top_k=3)
            system_prompt = _build_system_prompt(chunks)
        except Exception as ve:
            logger.error(f"Vector store search failed, using fallback prompt: {ve}")
            system_prompt = "You are Saumya Jain, a backend engineer. Be clear and concise."
        
        # 3. Reassemble history cleanly and merge system messages to avoid duplicates/conflicts
        compiled_messages = []
        has_system_msg = False
        for m in incoming_messages:
            if m.get("role") == "system":
                # Merge the context-rich RAG prompt with Vapi's dashboard prompt
                vapi_system_prompt = m.get("content", "")
                merged_content = f"{system_prompt}\n\nAdditional Instructions:\n{vapi_system_prompt}" if vapi_system_prompt else system_prompt
                compiled_messages.append({"role": "system", "content": merged_content})
                has_system_msg = True
            elif "role" in m and "content" in m:
                compiled_messages.append({"role": m["role"], "content": m["content"]})
        
        # If no system prompt was present in the incoming history, prepend the RAG prompt
        if not has_system_msg:
            compiled_messages.insert(0, {"role": "system", "content": system_prompt})
        
        # 4. Stream generator with exact chunk parsing
        async def sse_stream_generator():
            try:
                if not voice_client or not voice_model:
                    raise ValueError("No LLM API keys configured. Set GROQ_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY.")
                
                response = voice_client.chat.completions.create(
                    model=voice_model,
                    messages=compiled_messages,
                    tools=payload.get("tools", openai_tools) if payload.get("tools") else None,
                    stream=True
                )
                for chunk in response:
                    # Convert object safely to dict first, then to json string
                    chunk_dict = chunk.model_dump()
                    yield f"data: {json.dumps(chunk_dict)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as stream_err:
                logger.error(f"Error inside voice stream loop: {stream_err}")
                yield f"data: {json.dumps({'error': str(stream_err)})}\n\n"

        return StreamingResponse(sse_stream_generator(), media_type="text/event-stream")

    except Exception as top_err:
        logger.error(f"Critical 500 error entry point: {top_err}")
        raise HTTPException(status_code=500, detail=str(top_err))
# ------------------------------------------------------------------
# KEEP EXISTING WEB UTILITIES (Webhook/Health)
# ------------------------------------------------------------------
@app.post("/vapi-webhook")
async def vapi_webhook(payload: dict):
    """Fallback handler for tracking standard or asynchronous tool calls."""
    logger.info(f"Received Webhook: {json.dumps(payload)}")
    message = payload.get("message", {})
    if message.get("type") != "tool-calls":
        return {"status": "success"}
        
    results = []
    for tc in message.get("toolCalls", []):
        tool_id = tc.get("id")
        func_name = tc.get("function", {}).get("name")
        arguments = tc.get("function", {}).get("arguments", {})
        
        if func_name == "check_available_slots" or func_name == "get_available_slots":
            res = check_available_slots()
        elif func_name == "create_booking" or func_name == "book_meeting":
            res = create_booking(arguments.get("start_time"), arguments.get("name"), arguments.get("email"))
        else:
            res = "Unrecognized tool"
        results.append({"toolCallId": tool_id, "result": res})
    return {"results": results}

@app.get("/health")
def health_check():
    return {"status": "ok", "llm_provider": llm_provider}

class BookingRequest(BaseModel):
    start_time: str
    name: str
    email: str

@app.get("/slots")
def get_slots_endpoint():
    try:
        slots = booking_tools.get_available_slots()
        return {"slots": slots}
    except Exception as e:
        logger.error(f"Error fetching slots: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bookings")
def create_booking_endpoint(req: BookingRequest):
    try:
        result = booking_tools.book_meeting(req.start_time, req.name, req.email)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error creating booking: {e}")
        raise HTTPException(status_code=500, detail=str(e))