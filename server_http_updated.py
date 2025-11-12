from typing import Any, List, Dict, Optional
import asyncio
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
import json
from dotenv import load_dotenv
import os
from fastmcp import FastMCP
from fastapi import FastAPI

app = FastAPI()
# ... your routes and logic ...

# Load environment variables from .env file
load_dotenv()

# MongoDB configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "occhubweather_dev")
COLLECTION_METAR = os.getenv("COLLECTION_METAR", "weatherData")

# Create MCP server for HTTP transport
mcp = FastMCP(name="metar-weather-http")

# Global MongoDB client
client = None
db = None

async def get_mongodb_client():
    """Get MongoDB client connection."""
    global client, db
    if client is None:
        client = AsyncIOMotorClient(MONGODB_URL)
        db = client[DATABASE_NAME]
    return client, db

def format_metar_data(metar_doc: Dict) -> str:
    """Format METAR data into a readable string."""
    station = metar_doc.get('stationICAO', 'Unknown')
    iata = metar_doc.get('stationIATA', 'N/A')
    
    result = f"🛩️  Station: {station}"
    if iata:
        result += f" ({iata})"
    result += "\n"
    
    if metar_doc.get('hasMetarData') and 'metar' in metar_doc:
        metar = metar_doc['metar']
        updated_time = metar.get('updatedTime', 'Unknown')
        result += f"📅 Last Updated: {updated_time}\n"
        
        raw_data = metar.get('rawData', 'N/A')
        result += f"📊 Raw METAR: {raw_data}\n"
        
        if 'decodedData' in metar and 'observation' in metar['decodedData']:
            obs = metar['decodedData']['observation']
            result += f"\n🌤️  Weather Conditions:\n"
            result += f"   🌡️  Temperature: {obs.get('airTemperature', 'N/A')}\n"
            result += f"   💧 Dewpoint: {obs.get('dewpointTemperature', 'N/A')}\n"
            result += f"   💨 Wind: {obs.get('windSpeed', 'N/A')} from {obs.get('windDirection', 'N/A')}\n"
            result += f"   👁️  Visibility: {obs.get('horizontalVisibility', 'N/A')}\n"
            result += f"   🌊 Pressure: {obs.get('observedQNH', 'N/A')}\n"
            
            if obs.get('cloudLayers'):
                result += f"   ☁️  Clouds: {', '.join(obs['cloudLayers'])}\n"
            
            if obs.get('weatherConditions'):
                result += f"   🌦️  Weather: {obs['weatherConditions']}\n"
    
    if metar_doc.get('hasTaforData') and 'tafor' in metar_doc:
        tafor = metar_doc['tafor']
        raw_taf = tafor.get('rawData', 'N/A')
        result += f"\n📈 TAF Forecast: {raw_taf}\n"
    
    return result

@mcp.tool()
async def search_metar_data(
    station_icao: str = None,
    station_iata: str = None,
    limit: int = 5
) -> str:
    """
    Search for METAR weather data by airport code.
    Args:
        station_icao: ICAO airport code (e.g., 'VIDP' for Delhi, 'VOBB' for Bangalore)
        station_iata: IATA airport code (e.g., 'DEL' for Delhi, 'BLR' for Bangalore)  
        limit: Maximum number of results to return (default: 5, max: 10)
    Returns:
        str: Formatted weather data for the specified airport(s)
    """
    try:
        _, db = await get_mongodb_client()
        query = {}
        if station_icao:
            query["stationICAO"] = station_icao.upper()
        if station_iata:
            query["stationIATA"] = station_iata.upper()
        limit = min(limit, 10)
        cursor = db[COLLECTION_METAR].find(query).sort("metar.updatedTime", -1).limit(limit)
        results = await cursor.to_list(length=limit)
        if not results:
            filters = []
            if station_icao: filters.append(f"ICAO: {station_icao}")
            if station_iata: filters.append(f"IATA: {station_iata}")
            return f"❌ No METAR data found for: {', '.join(filters)}"
        result = f"🔍 METAR Weather Data ({len(results)} reports found):\n"
        result += "=" * 60 + "\n\n"
        for i, doc in enumerate(results, 1):
            result += f"--- Report {i} ---\n"
            result += format_metar_data(doc)
            result += "\n"
        return result
    except Exception as e:
        return f"❌ Error searching weather data: {str(e)}"

@mcp.tool()
async def list_available_stations() -> str:
    """
    List all available weather stations with their ICAO and IATA codes.
    Returns:
        str: List of all available weather stations
    """
    try:
        _, db = await get_mongodb_client()
        icao_codes = await db[COLLECTION_METAR].distinct("stationICAO")
        icao_codes.sort()
        iata_codes = await db[COLLECTION_METAR].distinct("stationIATA")
        iata_codes = [code for code in iata_codes if code is not None]
        iata_codes.sort()
        total_stations = await db[COLLECTION_METAR].count_documents({})
        result = f"📡 Available Weather Stations ({total_stations:,} total reports)\n"
        result += "=" * 50 + "\n\n"
        result += f"🛩️  ICAO Codes ({len(icao_codes)} stations):\n"
        for i, code in enumerate(icao_codes[:20], 1):
            result += f"   {i:2d}. {code}"
            if i % 5 == 0:
                result += "\n"
            else:
                result += "  "
        if len(icao_codes) > 20:
            result += f"\n   ... and {len(icao_codes) - 20} more\n"
        result += f"\n🏢 IATA Codes ({len(iata_codes)} stations):\n"
        for i, code in enumerate(iata_codes[:20], 1):
            result += f"   {i:2d}. {code}"
            if i % 5 == 0:
                result += "\n"
            else:
                result += "  "
        if len(iata_codes) > 20:
            result += f"\n   ... and {len(iata_codes) - 20} more\n"
        result += f"\n💡 Use search_metar_data(station_icao='VIDP') or search_metar_data(station_iata='DEL') to get weather data!"
        return result
    except Exception as e:
        return f"❌ Error retrieving station list: {str(e)}"

@mcp.tool()
async def get_weather_summary() -> str:
    """
    Get a summary of the weather database statistics.
    Returns:
        str: Database statistics and overview
    """
    try:
        _, db = await get_mongodb_client()
        total_metar = await db[COLLECTION_METAR].count_documents({})
        unique_icao = len(await db[COLLECTION_METAR].distinct("stationICAO"))
        unique_iata = len([code for code in await db[COLLECTION_METAR].distinct("stationIATA") if code is not None])
        with_metar = await db[COLLECTION_METAR].count_documents({"hasMetarData": True})
        with_taf = await db[COLLECTION_METAR].count_documents({"hasTaforData": True})
        result = f"📊 Weather Database Summary\n"
        result += "=" * 30 + "\n\n"
        result += f"📈 Total Reports: {total_metar:,}\n"
        result += f"🛩️  ICAO Stations: {unique_icao}\n"
        result += f"🏢 IATA Stations: {unique_iata}\n"
        result += f"✅ With METAR: {with_metar:,} ({with_metar/total_metar*100:.1f}%)\n"
        result += f"📈 With TAF: {with_taf:,} ({with_taf/total_metar*100:.1f}%)\n"
        return result
    except Exception as e:
        return f"❌ Error retrieving statistics: {str(e)}"

@mcp.tool()
async def ping() -> str:
    """
    Test tool to check if the MCP server is working.
    Returns:
        str: Simple pong response
    """
    return "🏓 Pong! MCP Weather Server is working correctly!"

if __name__ == "__main__":
    # Run with HTTP transport
    mcp.run(transport="http", host="0.0.0.0", port=8080)
