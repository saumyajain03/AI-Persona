import os
import json
import unittest
from unittest.mock import MagicMock, AsyncMock, patch

# Set mock env var before importing app
os.environ["GEMINI_API_KEY"] = "mock-gemini-key"

# We need to patch genai before app imports it
with patch.dict(os.environ, {"GEMINI_API_KEY": "mock-gemini-key"}):
    with patch('google.generativeai.configure'):
        with patch('google.generativeai.GenerativeModel') as mock_gm:
            mock_gm.return_value = MagicMock()
            from fastapi.testclient import TestClient
            from app import app

class AsyncIteratorMock:
    def __init__(self, items):
        self.items = items
        self.index = 0
        
    def __aiter__(self):
        return self
        
    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item

class TestFastAPIChatApp(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        """Tests that the health check endpoint returns 200 OK."""
        with patch('vector_store.RAGVectorStore') as mock_store_class:
            mock_store = MagicMock()
            mock_store.db_type = "chroma"
            mock_store_class.return_value = mock_store
            
            response = self.client.get("/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "ok")

    def test_cors_headers(self):
        """Tests that CORS middleware is active and returns the echoed origin when allowed."""
        response = self.client.options(
            "/chat",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            }
        )
        self.assertEqual(response.status_code, 200)
        # When allow_credentials=True, Starlette echoes back the origin
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:3000")

    @patch('app._stream_gemini')
    @patch('vector_store.get_default_store')
    def test_chat_streaming_endpoint(self, mock_get_store, mock_stream_gemini):
        """Tests the /chat endpoint with mock retrieval and LLM streaming."""
        # 1. Setup mock database response
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {
                "content": "Saumya Jain has experience in full-stack development and Python.",
                "metadata": {"source": "resume", "filename": "resume.pdf"}
            }
        ]
        mock_get_store.return_value = mock_store

        # 2. Setup mock streaming response
        async def mock_stream(*args, **kwargs):
            yield 'data: {"content": "I "}\n\n'
            yield 'data: {"content": "am Saumya."}\n\n'
            yield 'data: [DONE]\n\n'

        mock_stream_gemini.return_value = mock_stream()

        # 3. Request parameters
        payload = {
            "message": "Who are you?",
            "history": []
        }

        # 4. Trigger request
        response = self.client.post("/chat", json=payload)
        
        # Verify Response properties
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("content-type"), "text/event-stream; charset=utf-8")
        
        # Split stream events and parse data
        lines = response.text.split("\n\n")
        non_empty_lines = [line for line in lines if line.strip()]
        
        # We expect: data: {"content": "I "}, data: {"content": "am Saumya."}, data: [DONE]
        self.assertTrue(len(non_empty_lines) >= 3)
        self.assertTrue(non_empty_lines[0].startswith("data: "))
        self.assertTrue(non_empty_lines[1].startswith("data: "))
        self.assertEqual(non_empty_lines[2].strip(), "data: [DONE]")
        
        chunk_1_data = json.loads(non_empty_lines[0][6:])
        chunk_2_data = json.loads(non_empty_lines[1][6:])
        
        self.assertEqual(chunk_1_data["content"], "I ")
        self.assertEqual(chunk_2_data["content"], "am Saumya.")

        # Ensure database query was called
        mock_store.query.assert_called_once_with("Who are you?", top_k=5)
        
        print("[Test Log] FastAPI endpoint streaming test passed successfully.")

    @patch('booking_tools.get_available_slots')
    @patch('google.generativeai.GenerativeModel.start_chat')
    @patch('vector_store.get_default_store')
    def test_chat_streaming_endpoint_with_slots_tool(self, mock_get_store, mock_start_chat, mock_slots):
        """Tests the /chat endpoint using Gemini to invoke check_available_slots tool."""
        # Mock vector store
        mock_store = MagicMock()
        mock_store.query.return_value = []
        mock_get_store.return_value = mock_store
        
        # Mock slots
        mock_slots.return_value = [
            "2026-06-10T10:00:00Z",
            "2026-06-10T11:00:00Z"
        ]
        
        # Mock first response from Gemini with function_call
        mock_fn_call = MagicMock()
        mock_fn_call.name = "check_available_slots"
        mock_fn_call.args = {}
        
        mock_part = MagicMock()
        mock_part.function_call = mock_fn_call
        
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        
        mock_chunk_1 = MagicMock()
        mock_chunk_1.candidates = [mock_candidate]
        mock_chunk_1.text = ""
        
        # Mock second response from Gemini (after tool execution)
        mock_chunk_2 = MagicMock()
        mock_chunk_2.candidates = []
        mock_chunk_2.text = "Here are the available slots: June 10 at 10:00 AM."
        
        # Setup AsyncMock with side effect returning AsyncIteratorMock
        mock_chat = MagicMock()
        mock_chat.send_message_async = AsyncMock()
        mock_chat.send_message_async.side_effect = [
            AsyncIteratorMock([mock_chunk_1]),
            AsyncIteratorMock([mock_chunk_2])
        ]
        mock_start_chat.return_value = mock_chat
        
        # Call chat
        payload = {"message": "Can we schedule a meeting?", "history": []}
        response = self.client.post("/chat", json=payload)
        
        self.assertEqual(response.status_code, 200)
        lines = response.text.split("\n\n")
        non_empty_lines = [line for line in lines if line.strip()]
        
        self.assertTrue(len(non_empty_lines) >= 2)
        chunk_data = json.loads(non_empty_lines[0][6:])
        self.assertIn("Here are the available slots", chunk_data["content"])
        
        # Verify tools were invoked
        mock_slots.assert_called_once()
        print("[Test Log] check_available_slots tool integration test passed successfully.")

    @patch('booking_tools.book_meeting')
    @patch('google.generativeai.GenerativeModel.start_chat')
    @patch('vector_store.get_default_store')
    def test_chat_streaming_endpoint_with_booking_tool(self, mock_get_store, mock_start_chat, mock_booking):
        """Tests the /chat endpoint using Gemini to invoke create_booking tool."""
        # Mock vector store
        mock_store = MagicMock()
        mock_store.query.return_value = []
        mock_get_store.return_value = mock_store
        
        # Mock booking creation response
        mock_booking.return_value = {
            "status": "success",
            "data": {
                "uid": "test-uid-789",
                "title": "Meeting",
                "meetingUrl": "https://cal.com/booking/test-uid-789"
            }
        }
        
        # Mock first response from Gemini with function_call
        mock_fn_call = MagicMock()
        mock_fn_call.name = "create_booking"
        mock_fn_call.args = {
            "start_time": "2026-06-10T10:00:00Z",
            "name": "Jane Doe",
            "email": "jane@example.com"
        }
        
        mock_part = MagicMock()
        mock_part.function_call = mock_fn_call
        
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        
        mock_chunk_1 = MagicMock()
        mock_chunk_1.candidates = [mock_candidate]
        mock_chunk_1.text = ""
        
        # Mock second response from Gemini
        mock_chunk_2 = MagicMock()
        mock_chunk_2.candidates = []
        mock_chunk_2.text = "Meeting booked successfully: ID is test-uid-789."
        
        # Setup AsyncMock with side effect returning AsyncIteratorMock
        mock_chat = MagicMock()
        mock_chat.send_message_async = AsyncMock()
        mock_chat.send_message_async.side_effect = [
            AsyncIteratorMock([mock_chunk_1]),
            AsyncIteratorMock([mock_chunk_2])
        ]
        mock_start_chat.return_value = mock_chat
        
        # Call chat
        payload = {
            "message": "Book the slot for June 10 at 10 AM. Name is Jane Doe, email is jane@example.com",
            "history": []
        }
        response = self.client.post("/chat", json=payload)
        
        self.assertEqual(response.status_code, 200)
        lines = response.text.split("\n\n")
        non_empty_lines = [line for line in lines if line.strip()]
        
        self.assertTrue(len(non_empty_lines) >= 2)
        chunk_data = json.loads(non_empty_lines[0][6:])
        self.assertIn("booked successfully", chunk_data["content"])
        
        # Verify tools were invoked
        mock_booking.assert_called_once_with("2026-06-10T10:00:00Z", "Jane Doe", "jane@example.com")
        print("[Test Log] create_booking tool integration test passed successfully.")

    @patch('app.openai_client')
    @patch('app.llm_provider', 'openai')
    @patch('booking_tools.get_available_slots')
    @patch('vector_store.get_default_store')
    def test_chat_streaming_endpoint_with_openai_slots_tool(self, mock_get_store, mock_slots, mock_openai):
        """Tests the /chat endpoint using OpenAI provider with check_available_slots tool."""
        # Mock vector store
        mock_store = MagicMock()
        mock_store.query.return_value = []
        mock_get_store.return_value = mock_store
        
        # Mock slots tool response
        mock_slots.return_value = [
            "2026-06-10T10:00:00Z"
        ]
        
        # First completion stream from OpenAI yielding a tool call chunk
        mock_tc = MagicMock()
        mock_tc.index = 0
        mock_tc.id = "call_123"
        mock_tc.function.name = "check_available_slots"
        mock_tc.function.arguments = "{}"
        
        mock_choice_1 = MagicMock()
        mock_choice_1.delta.content = ""
        mock_choice_1.delta.tool_calls = [mock_tc]
        
        mock_chunk_1 = MagicMock()
        mock_chunk_1.choices = [mock_choice_1]
        
        # Second chunk to close the stream
        mock_choice_2 = MagicMock()
        mock_choice_2.delta.content = ""
        mock_choice_2.delta.tool_calls = []
        mock_chunk_2 = MagicMock()
        mock_chunk_2.choices = [mock_choice_2]
        
        # Final answer completion stream from OpenAI after tool execution
        mock_choice_3 = MagicMock()
        mock_choice_3.delta.content = "Here are the slots: June 10 at 10 AM."
        mock_choice_3.delta.tool_calls = []
        mock_chunk_3 = MagicMock()
        mock_chunk_3.choices = [mock_choice_3]
        
        # Setup async generator mocks
        async def mock_first_create(*args, **kwargs):
            yield mock_chunk_1
            yield mock_chunk_2
            
        async def mock_second_create(*args, **kwargs):
            yield mock_chunk_3
            
        mock_openai.chat.completions.create.side_effect = [
            mock_first_create(),
            mock_second_create()
        ]
        
        # Call chat
        payload = {"message": "Can we schedule a meeting?", "history": []}
        response = self.client.post("/chat", json=payload)
        
        self.assertEqual(response.status_code, 200)
        lines = response.text.split("\n\n")
        non_empty_lines = [line for line in lines if line.strip()]
        
        self.assertTrue(len(non_empty_lines) >= 2)
        chunk_data = json.loads(non_empty_lines[0][6:])
        self.assertIn("Here are the slots", chunk_data["content"])
        
        mock_slots.assert_called_once()
        print("[Test Log] OpenAI check_available_slots tool integration test passed successfully.")

if __name__ == '__main__':
    unittest.main()
