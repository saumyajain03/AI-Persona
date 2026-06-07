import os
import logging
from datetime import datetime, timedelta, timezone
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CAL_API_KEY = os.getenv("CAL_API_KEY")
CAL_EVENT_TYPE_ID = os.getenv("CAL_EVENT_TYPE_ID")

def get_available_slots():
    """
    Fetches available slots from Cal.com API for the next 7 days in UTC.
    Uses 'cal-api-version': '2024-09-04' header.
    
    Returns:
        List[str]: The first 3 available timestamp slots as ISO strings.
    """
    if not CAL_API_KEY:
        logger.error("CAL_API_KEY is not configured in the environment.")
        return []
    if not CAL_EVENT_TYPE_ID:
        logger.error("CAL_EVENT_TYPE_ID is not configured in the environment.")
        return []

    logger.info("Fetching available slots from Cal.com for the next 7 days...")
    
    start_time = datetime.now(timezone.utc)
    end_time = start_time + timedelta(days=7)
    
    # Simple date format YYYY-MM-DD for start and end
    start_str = start_time.strftime("%Y-%m-%d")
    end_str = end_time.strftime("%Y-%m-%d")
    
    url = "https://api.cal.com/v2/slots"
    headers = {
        "Authorization": f"Bearer {CAL_API_KEY}",
        "cal-api-version": "2024-09-04",
        "Content-Type": "application/json"
    }
    
    params = {
        "eventTypeId": int(CAL_EVENT_TYPE_ID),
        "start": start_str,
        "end": end_str,
        "timeZone": "UTC"
    }
    
    try:
        logger.info(f"Sending GET request to {url} with params: {params}")
        response = httpx.get(url, headers=headers, params=params, timeout=10.0)
        
        if response.status_code != 200:
            logger.error(f"Cal.com slots API error (status {response.status_code}): {response.text}")
            return []
            
        data = response.json()
        logger.info("Slots fetched successfully from Cal.com.")
        
        # Parse the slots response
        # Typically the slots are in data -> slots -> YYYY-MM-DD: [ { "time": "..." }, ... ]
        slots_data = data.get("data", {})
        # The live API returns slots directly under 'data', whereas mock tests look for 'slots' key.
        slots = slots_data.get("slots", slots_data)
        
        all_timestamps = []
        
        if isinstance(slots, dict):
            # Sort the dates to keep them chronological
            for date_key in sorted(slots.keys()):
                slots_list = slots[date_key]
                if isinstance(slots_list, list):
                    for slot in slots_list:
                        time_val = slot.get("start") or slot.get("time")
                        if time_val:
                            all_timestamps.append(time_val)
        elif isinstance(slots, list):
            for slot in slots:
                time_val = slot.get("start") or slot.get("time")
                if time_val:
                    all_timestamps.append(time_val)
                    
        # Sort timestamps to ensure chronological order
        all_timestamps.sort()
        
        # Get the first 3
        first_3_slots = all_timestamps[:3]
        logger.info(f"Found {len(all_timestamps)} slots, returning first 3: {first_3_slots}")
        return first_3_slots
        
    except Exception as e:
        logger.error(f"Exception raised in get_available_slots: {e}")
        return []

def book_meeting(start_time, name, email):
    """
    Creates a booking via the Cal.com API.
    Uses 'cal-api-version': '2024-08-13' header.
    
    Args:
        start_time (str): The start time in ISO 8601 format (e.g. "2026-06-10T10:00:00Z")
        name (str): Attendee's name
        email (str): Attendee's email
        
    Returns:
        dict: Booking confirmation data or error info.
    """
    if not CAL_API_KEY:
        logger.error("CAL_API_KEY is not configured in the environment.")
        return {"error": "CAL_API_KEY is not configured."}
    if not CAL_EVENT_TYPE_ID:
        logger.error("CAL_EVENT_TYPE_ID is not configured in the environment.")
        return {"error": "CAL_EVENT_TYPE_ID is not configured."}

    logger.info(f"Creating a booking on Cal.com for {name} ({email}) at {start_time}...")
    
    url = "https://api.cal.com/v2/bookings"
    headers = {
        "Authorization": f"Bearer {CAL_API_KEY}",
        "cal-api-version": "2024-08-13",
        "Content-Type": "application/json"
    }
    
    payload = {
        "eventTypeId": int(CAL_EVENT_TYPE_ID),
        "start": start_time,
        "attendee": {
            "name": name,
            "email": email,
            "timeZone": "UTC",
            "language": "en"
        }
    }
    
    try:
        logger.info(f"Sending POST request to {url} with payload: {payload}")
        response = httpx.post(url, headers=headers, json=payload, timeout=10.0)
        
        if response.status_code not in (200, 201):
            logger.error(f"Cal.com booking API error (status {response.status_code}): {response.text}")
            return {"error": f"API returned status code {response.status_code}", "detail": response.text}
            
        data = response.json()
        logger.info("Booking created successfully on Cal.com.")
        return data
        
    except Exception as e:
        logger.error(f"Exception raised in book_meeting: {e}")
        return {"error": str(e)}
