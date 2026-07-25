import requests
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

@tool
def predict_precipitation(temperature_2m_max: float, temperature_2m_min: float,
                           windspeed_10m_max: float, relative_humidity_2m_mean: float,
                           surface_pressure_mean: float, day_of_year: int, is_monsoon: int,
                           precip_lag1: float, precip_lag7: float, precip_roll7: float) -> str:
    """Predict tomorrow's precipitation (mm) for Mumbai given weather conditions.
    Use this tool whenever the user asks for a precipitation prediction or forecast."""
    payload = {
        "temperature_2m_max": temperature_2m_max, "temperature_2m_min": temperature_2m_min,
        "windspeed_10m_max": windspeed_10m_max, "relative_humidity_2m_mean": relative_humidity_2m_mean,
        "surface_pressure_mean": surface_pressure_mean, "day_of_year": day_of_year,
        "is_monsoon": is_monsoon, "precip_lag1": precip_lag1,
        "precip_lag7": precip_lag7, "precip_roll7": precip_roll7
    }
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    result = response.json()
    return f"Predicted precipitation: {result['predicted_precipitation_mm']} mm"

llm = ChatOllama(model="qwen2.5:3b")
agent = create_react_agent(llm, tools=[predict_precipitation])

if __name__ == "__main__":
    query = ("Predict the precipitation for a day with max temp 31, min temp 25, "
              "wind speed 18, humidity 82%, pressure 1004 hPa, day of year 200, "
              "during monsoon, with 8mm rain yesterday, 6mm a week ago, and 7mm average over the last week.")
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})


for msg in result["messages"]:
    print(f"[{msg.type}] {msg.content}")