import json
from openai import AzureOpenAI
import tiktoken
from toon import encode, decode

# ---------- tokenizer (close to GPT-4o) ----------
# enc = tiktoken.get_encoding("o200k_base")
# def toklen(s: str) -> int:
#     return len(enc.encode(s))

# def toklen_messages(messages) -> int:
#     # Rough but consistent: count tokens of role + content only
#     total = 0
#     for m in messages:
#         total += toklen(m.get("role",""))
#         content = m.get("content","")
#         if isinstance(content, list):
#             for part in content:
#                 total += toklen(json.dumps(part, separators=(",",":")))
#         else:
#             total += toklen(str(content))
#     return total

# ---------- your TOON payload (paste) ----------

# ---------- (A) Token count for the raw TOON payload ----------
# print("TOON chars:", len(toon_payload))
# print("TOON tokens:", toklen(toon_payload))

# ---------- (B) Token count for original JSON ----------
# Option 1: load from file
# JSON_FILE = None  # e.g. "filtered_airports_by_icao_converted.json"
# if JSON_FILE:
#     with open(JSON_FILE, "r", encoding="utf-8") as f:
#         hello = json.load(f)
# else:
#     # Option 2: if you already have the dict in memory, assign it here:
#     # hello = {...}  # your big dict
#     hello = None

# if hello is not None:
#     json_str = json.dumps(hello, separators=(",", ":"))
#     print("JSON chars:", len(json_str))
#     print("JSON tokens:", toklen(json_str))

#     delta = toklen(json_str) - toklen(toon_payload)
#     print("Δ tokens (JSON − TOON):", delta)
# else:
#     print("JSON tokens: skipped (set JSON_FILE or hello dict)")


toon_payload = r"""legend:{i:icao,c:city,a:airport_name,t:iata}
airports[139]{i,c,a,t}:
VAAH,Ahmedabad,Sardar Vallabhbhai Patel International Airport,AMD
VAAM,Amravati,Amravati Airport,
VAAU,Chhatrapati Sambhajinagar,Aurangabad Airport,IXU
VABB,Mumbai,Chhatrapati Shivaji Maharaj International Airport,BOM
VABJ,Bhuj,Bhuj Airport,BHJ
VABO,Vadodara,Vadodara Airport,BDQ
VABP,Bhopal,Raja Bhoj Airport,BHO
VABV,Bhavnagar,Bhavnagar Airport,BHU
VADU,Diu,Diu Airport,DIU
VAGD,Gondia,Gondia Airport,GDB
VAHS,Rajkot,Rajkot International Airport,HSR
VAID,Indore,Devi Ahilyabai Holkar International Airport,IDR
VAJB,Jabalpur,Jabalpur Airport,JLR
VAJJ,Mumbai,Juhu Aerodrome,
VAJL,Jalgaon,Jalgaon Airport,
VAJM,Jamnagar,Jamnagar Airport,JGA
VAKE,Kandla,Kandla Airport,IXY
VAKP,Kolhapur,Chhatrapati Rajaram Maharaj Airport,KLH
VANM,Navi Mumbai,Dinkar Balu Patil International Airport,NMI
VANP,Nagpur,Dr. Babasaheb Ambedkar International Airport,NAG
VAOZ,Nashik,Nashik International Airport,ISK
VAPO,Pune,Pune Airport,PNQ
VAPR,Porbandar,Porbandar Airport,PBD
VARK,Rajkot,Rajkot Airport,RAJ
VASD,Shirdi,Shirdi Airport,SAG
VASL,Solapur,Solapur Airport,
VASU,Surat,Surat International Airport,
VAUD,Udaipur,Maharana Pratap Airport,UDR
VEAB,Prayagraj,Prayagraj Airport,IXD
VEAH,Azamgarh,Azamgarh Airport (Manduri),
VEAN,Aalo (Along),Aalo/Along Airport,
VEAP,Ambikapur,Ambikapur Airport,
VEAT,Agartala,Maharaja Bir Bikram Airport,IXA
VEAY,Ayodhya,Maharishi Valmiki International Airport,AYJ
VEAZ,Aizawl,Turial Airport,
VEBD,Siliguri,Bagdogra International Airport,IXB
VEBI,Shillong,Shillong Airport,SHL
VEBN,Varanasi,Lal Bahadur Shastri International Airport,VNS
VEBS,Bhubaneswar,Biju Patnaik International Airport,BBI
VEBU,Bilaspur,Bilasa Devi Kevat Airport,
VECC,Kolkata,Netaji Subhas Chandra Bose International Airport,CCU
VECO,Cooch Behar,Cooch Behar Airport,
VECX,Kanpur,Kanpur (Chakeri) AFS – Base Aerodrome,CNN
VEDG,Durgapur,Kazi Nazrul Islam Airport,RDP
VEDO,Deoghar,Deoghar Airport,DGH
VEDZ,Daporijo,Daporijo Airport,
VEGK,Gorakhpur,Gorakhpur Airport,GOP
VEGT,Guwahati,Lokpriya Gopinath Bordoloi International Airport,GAU
VEGY,Gaya,Gaya Airport,GAY
VEHO,Itanagar,Donyi Polo Airport,HGI
VEIM,Imphal,Bir Tikendrajit International Airport,IMF
VEJH,Jharsuguda,Veer Surendra Sai Airport,JRG
VEJP,Jeypore,Jeypore Airport,
VEJR,Jagdalpur,Jagdalpur Airport,JGB
VEJS,Jamshedpur,Sonari Airport,IXW
VEJT,Jorhat,Jorhat Airport,JRH
VEKI,Kushinagar,Kushinagar International Airport,
VEKO,Khajuraho,Khajuraho Airport,HJR
VEKU,Silchar,Silchar Airport,IXS
VELP,Aizawl,Lengpui Airport,AJL
VELR,North Lakhimpur,Lilabari Airport,IXI
VEMN,Dibrugarh,Dibrugarh Airport,DIB
VEMR,Dimapur,Dimapur Airport,DMU
VEPG,Pasighat,Pasighat Airport,
VEPT,Patna,Jay Prakash Narayan Airport,PAT
VEPY,Gangtok,Pakyong Airport,
VERB,Amethi,Fursatganj Airfield,
VERC,Ranchi,Birsa Munda Airport,IXR
VERK,Rourkela,Rourkela Airport,
VERP,Raipur,Swami Vivekananda Airport,RPR
VERU,Dhubri,Rupsi Airport,
VERW,Rewa,Rewa/Chorhata Airport,
VESL,Sultanpur,Sultanpur Amhat Airstrip,
VEST,Satna,Satna Airport,
VETJ,Tezu,Tezu Airport,TEI
VETZ,Tezpur,Tezpur Airport,TEZ
VEUK,Utkela,Utkela Airport,
VIAG,Agra,Agra (Kheria) Airport,AGR
VIAR,Amritsar,Sri Guru Ram Dass Jee International Airport,ATQ
VIBR,Kullu–Manali,Kullu–Manali Airport,KUU
VICG,Mohali,Shaheed Bhagat Singh International Airport,IXC
VIDD,Delhi,Safdarjung Airport,
VIDN,Dehradun,Jolly Grant Airport,DED
VIDP,Delhi,Indira Gandhi International Airport,DEL
VIGG,Kangra,Kangra Airport,DHM
VIGR,Gwalior,Rajmata Vijaya Raje Scindia Airport,GWL
VIJO,Jodhpur,Jodhpur Airport,JDH
VIJP,Jaipur,Jaipur International Airport,JAI
VIJR,Jaisalmer,Jaisalmer Airport,JSA
VIJU,Jammu,Jammu Airport,IXJ
VIKO,Kota,Kota Airport,KTU
VILD,Ludhiana,Ludhiana Airport,LUH
VILH,Leh,Kushok Bakula Rimpochee Airport,IXL
VILK,Lucknow,Chaudhary Charan Singh International Airport,LKO
VIPK,Pathankot,Pathankot Airport,IXP
VIPT,Pantnagar,Pantnagar Airport,PGH
VIRB,Fursatganj (Amethi/Raebareli),Fursatganj Airfield,
VISM,Shimla,Shimla Airport,SLV
VISR,Srinagar,Srinagar Airport,SXR
VOAR,Arakkonam,INS Rajali (Arakkonam Naval Air Station),
VOAT,Agatti Island,Agatti Airport,AGX
VOBG,Bengaluru (HAL),HAL Airport (Hindustan Aeronautics Limited),
VOBL,Bengaluru,Kempegowda International Airport,BLR
VOBM,Belagavi,Belagavi Airport,IXG
VOBX,Campbell Bay (Great Nicobar),INS Baaz (Campbell Bay Naval Air Station),
VOBZ,Vijayawada,Vijayawada International Airport,VGA
VOCB,Coimbatore,Coimbatore International Airport,CJB
VOCC,Kochi (Naval),INS Garuda (Willingdon Island Naval Air Station),
VOCI,Thrissur,Cochin International Airport,COK
VOCL,Malappuram,Kozhikode International Airport,CCJ
VOCP,Kadapa,Kadapa Airport,CDP
VOCX,Car Nicobar,Car Nicobar Air Force Station,CBD
VODX,Shibpur (Diglipur\, A&N Islands),INS Kohassa (Shibpur Airstrip),
VOGA,Mopa,Manohar International Airport,GOX
VOGB,Kalaburagi,Kalaburagi Airport,
VOGO,Goa (Dabolim),Goa International Airport (Dabolim),GOI
VOHB,Hubli,Hubli Airport,HBX
VOHS,Hyderabad,Rajiv Gandhi International Airport,HYD
VOHY,Hyderabad,Begumpet Airport,BPM
VOJV,Toranagallu (Vijayanagar/Ballari),Jindal Vijayanagar Airport,VDY
VOKN,Kannur,Kannur International Airport,CNN
VOKU,Kurnool,Uyyalawada Narasimha Reddy Airport,KJB
VOLT,Latur,Latur Airport,
VOMD,Madurai,Madurai International Airport,IXM
VOML,Mangaluru,Mangaluru International Airport,IXE
VOMM,Chennai,Chennai International Airport,MAA
VOMY,Mysuru,Mysuru Airport,MYQ
VOPB,Port Blair,Veer Savarkar International Airport,IXZ
VOPC,Puducherry,Pondicherry Airport,PNY
VORM,Ramanathapuram (Uchipuli),INS Parundu (Ramnad Naval Air Station),
VORY,Rajahmundry,Rajahmundry Airport,RJA
VOSH,Shivamogga,Rashtrakavi Kuvempu Airport,
VOSM,Salem,Salem Airport,SXV
VOSR,Sindhudurg,Sindhudurg Airport,
VOTK,Thoothukkudi,Tuticorin Airport,TCR
VOTP,Tirupati,Tirupati International Airport,TIR
VOTR,Tiruchirappalli,Tiruchirappalli International Airport,TRZ
VOTV,Thiruvananthapuram,Thiruvananthapuram International Airport,TRV
VOVZ,Visakhapatnam,Visakhapatnam International Airport,VTZ
"""


schema = {
 "_id": "ObjectId",
 "stationICAO": "String",
 "stationIATA": "String",
 "hasMetarData": "Boolean",
 "hasTaforData": "Boolean",
 "metar": {
 "updatedTime": "DateTime (ISO 8601)",
 "firRegion": "String",
 "rawData": "String",
 "decodedData": {
 "observation": {
 "observationTimeUTC": "DateTime (ISO 8601)",
 "observationTimeIST": "DateTime (ISO 8601)",
 "windSpeed": "String",
 "windDirection": "String",
 "horizontalVisibility": "String",
 "weatherConditions": "Null",
 "cloudLayers": ["String"],
 "airTemperature": "String",
 "dewpointTemperature": "String",
 "observedQNH": "String",
 "runwayVisualRange": "Null",
 "windShear": "Null",
 "runwayConditions": "Null"
 },
 "additionalInformation": {
 "weatherTrend": "Null",
 "forecastWeather": "Null"
 },
 "tempoSection": {
 "type": "Null",
 "timePeriod": "Null",
 "windSpeed": "Null",
 "windDirection": "Null",
 "visibility": "Null",
 "weatherConditions": "Null"
 }
 }
 },
 "tafor": {
 "rawData": "String",
 "updatedTime": "Null",
 "timestamp": "DateTime (ISO 8601)"
 }
 }
# schema_str = json.dumps(schema, separators=(",", ":")) 
schema_toon = encode(schema)
# print(schema_toon)
sys_msg ="""
 You are an intelligent agent capable of orchestrating multiple tools to assist users. Below is a list of available tools, each with a name, description of what it does, and the input it requires.

 Guardrails:

 - You may only provide answers that are directly related to the database of airports, city details, or weather data.

 - For Casual greetings or simple pleasantries (e.g., "Hello", "Namaskar","How are you?"), you may respond conversationally(e.g.,"Hi! How can I Assist you today?").

 - For Casual conversation like (e.g., "ok","Thankyou","amazing") you may respond conversationally(e.g.,"Thank You anything else you want me to assist with you").

 - Do not provide answers or guesses about anything outside this scope.

 - If the user's request is outside this scope, respond politely:

 - You will receive airports data encoded in TOON (header+rows).
 - Use legend:===> i:icao,c:city,a:airport_name,t:iata.
 - Parse the table and answer queries precisely.

 "I'm sorry, I can only provide information about airports, city details, or weather. Can I help you with that?"

 Instructions:

 1. Identify which tools can be used to fulfill their request.

 2. Call one or more tools as needed.

 3. Explain how these tools will be used.

 4. Ask for any additional details if required.

 5. Do not give any additional explanation, context, or interpretation. Do not hesitate or ask follow-up questions unless the user explicitly asks for explanation or interpretation of Metar Data.

 6. If duplicate Mongo DB results are present, return only one. If there are differences, return all the unique values.

 7. If the user specifically asks for Metar data, just provide the Raw Metar Data Value.

 8. If asked for Hours Back data and no results come back from query running then specify the latest timestamp that is present in MongoDB
"""

user_prompt = "give me a mongodb query to find all the stations with missing IATA codes and a mongoDB query to tell the latest weatherConditions for indra gandhi airport" 

# ---------- (C) Token count for the full messages you send ----------
messages = [
    {"role": "system", "content": sys_msg},
    {"role": "system", "content": toon_payload},
    {"role": "system", "content": "Schema (JSON):\n" + schema_toon},
    {"role": "user", "content": user_prompt},
]


# print("Messages tokens:", toklen_messages(messages))

# ---------- (D) (Optional) actually call Azure OpenAI ----------
client = AzureOpenAI(
    api_key="",
    api_version="2024-12-01-preview",
    azure_endpoint="/",
)
resp = client.chat.completions.create(
    model="gpt-4o",  # your deployment name
    messages=messages,
    temperature=0.1,
)
print(resp.choices[0].message.content)
print(resp.usage)

