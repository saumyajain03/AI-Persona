import unittest
from unittest.mock import patch, MagicMock
import os

# Set mock env vars
os.environ["CAL_API_KEY"] = "mock-key"
os.environ["CAL_EVENT_TYPE_ID"] = "12345"

import booking_tools

class TestBookingTools(unittest.TestCase):
    
    @patch('booking_tools.httpx.get')
    def test_get_available_slots_success(self, mock_get):
        # Mock successful Cal.com response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "slots": {
                    "2026-06-10": [
                        {"time": "2026-06-10T10:00:00.000Z"},
                        {"time": "2026-06-10T11:00:00.000Z"}
                    ],
                    "2026-06-11": [
                        {"time": "2026-06-11T14:00:00.000Z"},
                        {"time": "2026-06-11T15:00:00.000Z"}
                    ]
                }
            }
        }
        mock_get.return_value = mock_response
        
        slots = booking_tools.get_available_slots()
        
        # Verify
        self.assertEqual(len(slots), 3)
        self.assertEqual(slots[0], "2026-06-10T10:00:00.000Z")
        self.assertEqual(slots[1], "2026-06-10T11:00:00.000Z")
        self.assertEqual(slots[2], "2026-06-11T14:00:00.000Z")
        mock_get.assert_called_once()
        print("[Test Log] get_available_slots unit test passed successfully.")

    @patch('booking_tools.httpx.post')
    def test_book_meeting_success(self, mock_post):
        # Mock successful Cal.com booking response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "data": {
                "id": 555,
                "uid": "mock-booking-uid",
                "title": "Meeting with Saumya Jain",
                "start": "2026-06-10T10:00:00.000Z",
                "meetingUrl": "https://cal.com/video/mock-meeting"
            }
        }
        mock_post.return_value = mock_response
        
        res = booking_tools.book_meeting(
            start_time="2026-06-10T10:00:00.000Z",
            name="John Doe",
            email="john@example.com"
        )
        
        # Verify
        self.assertEqual(res.get("status"), "success")
        self.assertEqual(res["data"]["uid"], "mock-booking-uid")
        mock_post.assert_called_once()
        print("[Test Log] book_meeting unit test passed successfully.")

if __name__ == '__main__':
    unittest.main()
