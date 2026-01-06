"""
Build with AI: Creating AI Agents with GPT-5
All examples use Python and the OpenAI client.

Prereqs:
  pip install openai
  pip install python-dotenv
  export API_KEY = os.environ[...]
"""
# # ---------------------------------------------------------------------------
# # LESSON 5 (Take your GPT-5 agent live)
# # ---------------------------------------------------------------------------

from fastapi import FastAPI, Request
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, function_tool, ModelSettings, SQLiteSession
from dataclasses import dataclass
from datetime import datetime
import requests
import os
import asyncio

# Load environment variables
_ = load_dotenv(find_dotenv())

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

app = FastAPI()
session = SQLiteSession("travel_assistant")

# ------------------------------
# Define tool schema
# ------------------------------
@dataclass
class WeatherInfo:
    city: str
    country: str
    temp_f: float
    condition: str

@function_tool
def get_weather_forecast(city: str):
    """Fetch weather info using the Weather API"""
    API_KEY = os.environ['WEATHER_API_KEY']
    WEATHER_BASE_URL = 'https://api.weatherapi.com/v1/current.json'

    try:
        today = datetime.today().strftime('%Y-%m-%d')
        params = {"q": city, "aqi": "no", "key": API_KEY}
        response = requests.get(WEATHER_BASE_URL, params=params)
        response.raise_for_status()

        data = response.json()
        if "location" not in data or "current" not in data:
            return f"Could not retrieve weather for '{city}'. Try a more specific place name."

        weather = WeatherInfo(
            city=data["location"]["name"],
            country=data["location"]["country"],
            temp_f=float(data["current"]["temp_f"]),
            condition=data["current"]["condition"]["text"]
        )

        return (
            f"Real-time weather report for {today}:\n"
            f"   - City: {weather.city}\n"
            f"   - Country: {weather.country}\n"
            f"   - Temperature: {weather.temp_f:.1f} °F\n"
            f"   - Weather Conditions: {weather.condition}"
        )
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"

# ------------------------------
# Define the agent
# ------------------------------
trip_agent = Agent(
    name="Trip Coach",
    instructions=(
        "You help travelers plan activities to do while on vacation but you check the weather first in real-time. "
        "When asked about weather, call the get_weather_forecast tool. "
        "Make sure to pick activities that solo travelers will enjoy. Use web search if necessary."
    ),
    model="gpt-5",
    tools=[get_weather_forecast],
    model_settings=ModelSettings(
        reasoning={"effort": "medium"},
        extra_body={"text": {"verbosity": "medium"}}
    )
)

# ------------------------------
# Define request model
# ------------------------------
class UserPrompt(BaseModel):
    prompt: str

# ------------------------------
# *******TO RUN THE FASTAPI, FOLLOW THE STEPS BELOW*************
# 
# cd Lesson 5 folder
# make the Port public
# run the command: uvicorn agent:app --reload
# test in Postman or cURL
# POST to a similar endpoint (replace with your endpoint) - https://ideal-space-barnacle-pwqqgv6gq45396x9-8000.app.github.dev/ask
# Send in a JSON request body:
#   {
#      "prompt": "I'm heading to Atlanta this weekend. What's the weather like, and what should I pack?"
#   }
# ------------------------------
@app.post("/ask")
async def ask_agent(request: UserPrompt):
    result = await Runner.run(
        trip_agent,
        request.prompt,
        session=session
    )
    return {"response": result.final_output}
