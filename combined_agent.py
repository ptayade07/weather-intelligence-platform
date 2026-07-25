from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import requests

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

@tool
def search_knowledge_base(query: str) -> str:
    """Search the weather/climate knowledge base and this project's own findings
    for background information, definitions, or explanations. Use this for
    conceptual or research questions (e.g. about the monsoon, IMD methods,
    or this project's model results) - NOT for making new predictions."""
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)

@tool
def predict_precipitation(temperature_2m_max: float, temperature_2m_min: float,
                           windspeed_10m_max: float, relative_humidity_2m_mean: float,
                           surface_pressure_mean: float, day_of_year: int, is_monsoon: int,
                           precip_lag1: float, precip_lag7: float, precip_roll7: float) -> str:
    """Predict tomorrow's precipitation (mm) for Mumbai given specific weather
    conditions. Use this tool whenever the user gives numeric weather inputs
    and asks for a forecast or prediction."""
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

@tool
def get_current_weather(city: str) -> str:
    """Get the current live weather for any city in the world by name
    (e.g. 'Nashik', 'Mumbai', 'London'). Use this whenever the user asks
    about current/live/today's weather in a specific place - NOT for
    precipitation predictions, which use predict_precipitation instead."""
    geo_resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1}
    )
    geo_data = geo_resp.json()
    if "results" not in geo_data or not geo_data["results"]:
        return f"Could not find a location matching '{city}'."

    loc = geo_data["results"][0]
    lat, lon = loc["latitude"], loc["longitude"]
    resolved_name = f"{loc['name']}, {loc.get('country', '')}"

    weather_resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,"
                        "windspeed_10m,surface_pressure,cloudcover",
            "timezone": "auto"
        }
    )
    current = weather_resp.json()["current"]

    return (
        f"Current weather in {resolved_name}: "
        f"{current['temperature_2m']}\u00b0C, "
        f"humidity {current['relative_humidity_2m']}%, "
        f"precipitation {current['precipitation']}mm, "
        f"wind {current['windspeed_10m']}km/h, "
        f"pressure {current['surface_pressure']}hPa, "
        f"cloud cover {current['cloudcover']}%."
    )

llm = ChatOllama(model="qwen2.5:3b")
memory = MemorySaver()
agent = create_react_agent(llm, tools=[search_knowledge_base, predict_precipitation, get_current_weather], checkpointer=memory)

config = {"configurable": {"thread_id": "session-1"}}

def ask(question):
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config=config)
    for msg in result["messages"]:
        print(f"  [{msg.type}] {str(msg.content)[:150]}")
    return result["messages"][-1].content

if __name__ == "__main__":
    q1 = "What did the SHAP analysis reveal about which features matter most for precipitation?"
    print(f"Q1: {q1}\nA1: {ask(q1)}\n")

    q2 = ("Now predict precipitation for max temp 30, min temp 24, wind speed 15, "
          "humidity 80%, pressure 1006 hPa, day of year 210, monsoon season, "
          "with 5mm rain yesterday, 4mm a week ago, 5mm average last week.")
    print(f"Q2: {q2}\nA2: {ask(q2)}\n")

    q3 = "Between those two answers, which one used a tool call and which used retrieval?"
    print(f"Q3: {q3}\nA3: {ask(q3)}")
