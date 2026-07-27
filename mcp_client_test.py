import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(command="python", args=["mcp_server.py"])

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Available MCP tools:", [t.name for t in tools.tools])

            result = await session.call_tool("predict_precipitation", arguments={
                "temperature_max": 30.5,
                "temperature_min": 24.0,
                "windspeed_max": 15.2,
                "relative_humidity": 82.0,
                "surface_pressure": 1005.0,
                "precip_lag1": 12.0,
                "precip_lag7": 5.0,
                "precip_roll7": 8.0,
            })
            print("Tool result:", result.content[0].text)

asyncio.run(main())