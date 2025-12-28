import pandas as pd
import ta
import logging
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from tqdm import tqdm
import plotly.express as px
import time
import requests
import io
import random
import spacy
from pytrends.request import TrendReq
import numpy as np
import itertools
from arch import arch_model
import warnings
import sqlite3
from diskcache import Cache
from SmartApi import SmartConnect
import pyotp
import os
from dotenv import load_dotenv
from streamlit import cache_data

load_dotenv()

@st.cache_data(ttl=86400)
def load_symbol_token_map():
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return {entry["symbol"]: entry["token"] for entry in data if "symbol" in entry and "token" in entry}
    except Exception as e:
        st.warning(f"⚠️ Failed to load instrument list: {str(e)}")
        return {}

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

CLIENT_ID = os.getenv("CLIENT_ID")
PASSWORD = os.getenv("PASSWORD")
TOTP_SECRET = os.getenv("TOTP_SECRET")
API_KEYS = {
    "Historical": "c3C0tMGn",
    "Trading": os.getenv("TRADING_API_KEY"),
    "Market": os.getenv("MARKET_API_KEY")
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/124.0.2478.80 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 14; SM-S928B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 OPR/110.0.0.0",
    "Mozilla/5.0 (Linux; Android 14; SM-S928B) AppleWebKit/537.36 (KHTML, like Gecko) SamsungBrowser/23.0 Chrome/115.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Brave/124.0.0.0"
]

cache = Cache("stock_data_cache")

TOOLTIPS = {
    "RSI": "Relative Strength Index (30=Oversold, 70=Overbought)",
    "ATR": "Average True Range - Measures market volatility",
    "MACD": "Moving Average Convergence Divergence - Trend following",
    "ADX": "Average Directional Index (25+ = Strong Trend)",
    "Bollinger": "Price volatility bands around moving average",
    "Stop Loss": "Risk management price level based on ATR",
    "VWAP": "Volume Weighted Average Price - Intraday trend indicator",
    "Parabolic_SAR": "Parabolic Stop and Reverse - Trend reversal indicator",
    "Fib_Retracements": "Fibonacci Retracements - Support and resistance levels",
    "Ichimoku": "Ichimoku Cloud - Comprehensive trend indicator",
    "CMF": "Chaikin Money Flow - Buying/selling pressure",
    "Donchian": "Donchian Channels - Breakout detection",
    "Keltner": "Keltner Channels - Volatility bands based on EMA and ATR",
    "TRIX": "Triple Exponential Average - Momentum oscillator with triple smoothing",
    "Ultimate_Osc": "Ultimate Oscillator - Combines short, medium, and long-term momentum",
    "CMO": "Chande Momentum Oscillator - Measures raw momentum (-100 to 100)",
    "VPT": "Volume Price Trend - Tracks trend strength with price and volume",
    "Score": "Measured by RSI, MACD, Ichimoku Cloud, and ATR volatility. Low score = weak signal, high score = strong signal."
}

SECTORS = {
    "Bank": [
        "HDFCBANK-EQ", "ICICIBANK-EQ", "SBIN-EQ", "KOTAKBANK-EQ", "AXISBANK-EQ",
        "INDUSINDBK-EQ", "PNB-EQ", "BANKBARODA-EQ", "CANBK-EQ", "UNIONBANK-EQ",
        "IDFCFIRSTB-EQ", "FEDERALBNK-EQ", "RBLBANK-EQ", "BANDHANBNK-EQ", "INDIANB-EQ",
        "BANKINDIA-EQ", "KARURVYSYA-EQ", "CUB-EQ", "J&KBANK-EQ", "DCBBANK-EQ",
        "AUBANK-EQ", "YESBANK-EQ", "IDBI-EQ", "SOUTHBANK-EQ", "CSBBANK-EQ",
        "TMB-EQ", "KTKBANK-EQ", "EQUITASBNK-EQ", "UJJIVANSFB-EQ","CENTURYPLY-EQ"
    ],
    "IT": [
        "TCS-EQ", "INFY-EQ", "HCLTECH-EQ", "WIPRO-EQ", "TECHM-EQ", "LTIM-EQ",
        "MPHASIS-EQ", "FSL-EQ", "BSOFT-EQ", "NEWGEN-EQ", "ZENSARTECH-EQ",
        "RATEGAIN-EQ", "TANLA-EQ", "COFORGE-EQ", "PERSISTENT-EQ", "CYIENT-EQ",
        "SONATSOFTW-EQ", "KPITTECH-EQ", "TATAELXSI-EQ",
        "INTELLECT-EQ", "HAPPSTMNDS-EQ", "MASTEK-EQ", "ECLERX-EQ", "NIITLTD-EQ",
        "RSYSTEMS-EQ", "OFSS-EQ", "AURIONPRO-EQ", "DATAMATICS-EQ",
        "QUICKHEAL-EQ", "CIGNITITEC-EQ", "SAGILITY-EQ", "ALLDIGI-EQ","BLS-EQ"
    ],
    "Finance": [
        "HDFCBANK-EQ", "ICICIBANK-EQ", "SBIN-EQ", "KOTAKBANK-EQ", "BAJFINANCE-EQ",
        "AXISBANK-EQ", "BAJAJFINSV-EQ", "INDUSINDBK-EQ", "SHRIRAMFIN-EQ", "CHOLAFIN-EQ",
        "SBICARD-EQ", "M&MFIN-EQ", "MUTHOOTFIN-EQ", "LICHSGFIN-EQ", "IDFCFIRSTB-EQ",
        "AUBANK-EQ", "POONAWALLA-EQ", "SUNDARMFIN-EQ", "IIFL-EQ", "ABCAPITAL-EQ",
        "LTF-EQ", "CREDITACC-EQ", "MANAPPURAM-EQ", "DHANI-EQ", "JMFINANCIL-EQ",
        "EDELWEISS-EQ", "INDIASHLTR-EQ", "MOTILALOFS-EQ", "CDSL-EQ", "BSE-EQ",
        "MCX-EQ", "ANGELONE-EQ", "KARURVYSYA-EQ", "RBLBANK-EQ", "PNB-EQ",
        "CANBK-EQ", "UNIONBANK-EQ", "IOB-EQ", "YESBANK-EQ", "UCOBANK-EQ",
        "BANKINDIA-EQ", "CENTRALBK-EQ", "IDBI-EQ", "J&KBANK-EQ", "DCBBANK-EQ",
        "FEDERALBNK-EQ", "SOUTHBANK-EQ", "CSBBANK-EQ", "TMB-EQ", "KTKBANK-EQ",
        "EQUITASBNK-EQ", "UJJIVANSFB-EQ", "BANDHANBNK-EQ", "SURYODAY-EQ", "PSB-EQ",
        "PFS-EQ", "HDFCAMC-EQ", "UTIAMC-EQ", "ABSLAMC-EQ",
        "360ONE-EQ", "ANANDRATHI-EQ", "PNBHOUSING-EQ", "HOMEFIRST-EQ", "AAVAS-EQ",
        "APTUS-EQ", "RECLTD-EQ", "PFC-EQ", "IREDA-EQ", "SMCGLOBAL-EQ", "CHOICEIN-EQ",
        "KFINTECH-EQ", "MASFIN-EQ", "TRIDENT-EQ", "SBFC-EQ",
        "UGROCAP-EQ", "FUSION-EQ", "PAISALO-EQ", "CAPITALSFB-EQ", "NSIL-EQ",
        "SATIN-EQ"
    ],
    "Auto": [
        "MARUTI-EQ","BELRISE-EQ", "TATAMOTORS-EQ", "M&M-EQ", "BAJAJ-AUTO-EQ", "HEROMOTOCO-EQ",
        "EICHERMOT-EQ", "TVSMOTOR-EQ", "ASHOKLEY-EQ", "MRF-EQ", "BALKRISIND-EQ",
        "APOLLOTYRE-EQ", "CEATLTD-EQ", "JKTYRE-EQ", "MOTHERSON-EQ", "BHARATFORG-EQ",
        "SUNDRMFAST-EQ", "EXIDEIND-EQ", "BOSCHLTD-EQ", "ENDURANCE-EQ",
        "UNOMINDA-EQ", "ZFCVINDIA-EQ", "GABRIEL-EQ", "SUPRAJIT-EQ", "LUMAXTECH-EQ",
        "FIEMIND-EQ", "SUBROS-EQ", "JAMNAAUTO-EQ", "SHRIRAMFIN-EQ", "ESCORTS-EQ",
        "ATULAUTO-EQ", "OLECTRA-EQ", "GREAVESCOT-EQ", "SMLISUZU-EQ", "VSTTILLERS-EQ",
        "HINDMOTORS-EQ", "MAHSCOOTER-EQ"
    ],
    "Pharma": [
        "SUNPHARMA-EQ", "CIPLA-EQ", "DRREDDY-EQ", "APOLLOHOSP-EQ", "LUPIN-EQ",
        "DIVISLAB-EQ", "AUROPHARMA-EQ", "ALKEM-EQ", "TORNTPHARM-EQ", "ZYDUSLIFE-EQ",
        "IPCALAB-EQ", "GLENMARK-EQ", "BIOCON-EQ", "ABBOTINDIA-EQ", "SANOFI-EQ",
        "PFIZER-EQ", "GLAXO-EQ", "NATCOPHARM-EQ", "AJANTPHARM-EQ", "GRANULES-EQ",
        "LAURUSLABS-EQ", "STAR-EQ", "JUBLPHARMA-EQ", "ASTRAZEN-EQ", "WOCKPHARMA-EQ",
        "FORTIS-EQ", "MAXHEALTH-EQ", "METROPOLIS-EQ", "THYROCARE-EQ", "POLYMED-EQ",
        "KIMS-EQ", "LALPATHLAB-EQ", "MEDPLUS-EQ", "ERIS-EQ", "INDOCO-EQ",
        "CAPLIPOINT-EQ", "NEULANDLAB-EQ", "SHILPAMED-EQ", "SUVEN-EQ", "AARTIDRUGS-EQ",
        "PGHL-EQ", "SYNGENE-EQ", "VINATIORGA-EQ", "GLAND-EQ", "JBCHEPHARM-EQ",
        "HCG-EQ", "RAINBOW-EQ", "ASTERDM-EQ", "KRSNAA-EQ", "VIJAYA-EQ", "MEDANTA-EQ",
        "BLISSGVS-EQ", "MOREPENLAB-EQ", "RPGLIFE-EQ"
    ],
    "Metals": [
        "TATASTEEL-EQ", "JSWSTEEL-EQ", "HINDALCO-EQ", "VEDL-EQ", "SAIL-EQ",
        "NMDC-EQ", "HINDZINC-EQ", "NATIONALUM-EQ", "JINDALSTEL-EQ", "MOIL-EQ",
        "APLAPOLLO-EQ", "RATNAMANI-EQ", "JSL-EQ", "WELCORP-EQ",
        "SHYAMMETL-EQ", "MIDHANI-EQ", "GRAVITA-EQ", "SARDAEN-EQ", "ASHAPURMIN-EQ",
        "JTLIND-EQ", "RAMASTEEL-EQ", "MAITHANALL-EQ", "KIOCL-EQ", "IMFA-EQ",
        "GMDCLTD-EQ", "VISHNU-EQ", "SANDUMA-EQ", "VRAJ-EQ", "COALINDIA-EQ"
    ],
    "FMCG": [
        "HINDUNILVR-EQ", "ITC-EQ", "NESTLEIND-EQ", "BRITANNIA-EQ",
        "GODREJCP-EQ", "DABUR-EQ", "COLPAL-EQ", "MARICO-EQ", "PGHH-EQ",
        "EMAMILTD-EQ", "GILLETTE-EQ", "HATSUN-EQ", "JYOTHYLAB-EQ", "BAJAJCON-EQ",
        "RADICO-EQ", "TATACONSUM-EQ", "UNITDSPR-EQ", "CCL-EQ", "AVANTIFEED-EQ",
        "BIKAJI-EQ", "VBL-EQ", "ETERNAL-EQ", "DOMS-EQ",
        "GODREJAGRO-EQ", "SAPPHIRE-EQ", "VENKEYS-EQ", "BECTORFOOD-EQ", "KRBL-EQ"
    ],
    "Power": [
        "NTPC-EQ", "POWERGRID-EQ", "ADANIPOWER-EQ", "TATAPOWER-EQ", "JSWENERGY-EQ",
        "NHPC-EQ", "SJVN-EQ", "TORNTPOWER-EQ", "CESC-EQ", "ADANIENSOL-EQ",
        "INDIGRID-EQ", "POWERMECH-EQ", "KEC-EQ", "INOXWIND-EQ", "KPIL-EQ",
        "SUZLON-EQ", "BHEL-EQ", "THERMAX-EQ", "GVPIL-EQ", "VOLTAMP-EQ",
        "TARIL-EQ", "TDPOWERSYS-EQ", "JYOTISTRUC-EQ", "IWEL-EQ", "ACMESOLAR-EQ"
    ],
    "Capital Goods": [
        "LT-EQ", "SIEMENS-EQ", "ABB-EQ", "BEL-EQ", "BHEL-EQ", "HAL-EQ",
        "CUMMINSIND-EQ", "THERMAX-EQ", "AIAENG-EQ", "SKFINDIA-EQ", "GRINDWELL-EQ",
        "TIMKEN-EQ", "KSB-EQ", "ELGIEQUIP-EQ", "LMW-EQ", "KIRLOSENG-EQ",
        "GREAVESCOT-EQ", "TRITURBINE-EQ", "VOLTAS-EQ", "BLUESTARCO-EQ", "HAVELLS-EQ",
        "DIXON-EQ", "KAYNES-EQ", "SYRMA-EQ", "AMBER-EQ", "SUZLON-EQ", "CGPOWER-EQ",
        "APARINDS-EQ", "HBLENGINE-EQ", "KEI-EQ", "POLYCAB-EQ", "RRKABEL-EQ",
        "SCHNEIDER-EQ", "TDPOWERSYS-EQ", "KIRLOSBROS-EQ", "JYOTICNC-EQ", "DATAPATTNS-EQ",
        "INOXWIND-EQ", "KALPATPOWR-EQ", "MAZDOCK-EQ", "COCHINSHIP-EQ", "GRSE-EQ",
        "POWERMECH-EQ", "ISGEC-EQ", "HPL-EQ", "VTL-EQ", "DYNAMATECH-EQ", "JASH-EQ",
        "GMMPFAUDLR-EQ", "ESABINDIA-EQ", "CENTEXT-EQ", "SALASAR-EQ", "TITAGARH-EQ",
        "VGUARD-EQ", "WABAG-EQ", "AZAD-EQ"
    ],
    "Oil & Gas": [
        "RELIANCE-EQ", "ONGC-EQ", "IOC-EQ", "BPCL-EQ", "HINDPETRO-EQ", "GAIL-EQ",
        "PETRONET-EQ", "OIL-EQ", "IGL-EQ", "MGL-EQ", "GUJGASLTD-EQ", "GSPL-EQ",
        "AEGISLOG-EQ", "CHENNPETRO-EQ", "MRPL-EQ", "FLUOROCHEM-EQ", "CASTROLIND-EQ",
        "SOTL-EQ", "PANAMAPET-EQ", "GOCLCORP-EQ"
    ],
    "Chemicals": [
        "PIDILITIND-EQ", "SRF-EQ", "DEEPAKNTR-EQ", "ATUL-EQ", "AARTIIND-EQ",
        "NAVINFLUOR-EQ", "VINATIORGA-EQ", "FINEORG-EQ", "ALKYLAMINE-EQ", "BALAMINES-EQ",
        "GUJFLUORO-EQ", "CLEAN-EQ", "JUBLINGREA-EQ", "GALAXYSURF-EQ", "PCBL-EQ",
        "NOCIL-EQ", "BASF-EQ", "SUDARSCHEM-EQ", "NEOGEN-EQ", "PRIVISCL-EQ",
        "ROSSARI-EQ", "LXCHEM-EQ", "ANURAS-EQ", "CHEMCON-EQ",
        "DMCC-EQ", "TATACHEM-EQ", "COROMANDEL-EQ", "UPL-EQ", "BAYERCROP-EQ",
        "SUMICHEM-EQ", "PIIND-EQ", "EIDPARRY-EQ", "CHEMPLASTS-EQ",
        "IGPL-EQ", "TIRUMALCHM-EQ", "RALLIS-EQ"
    ],
    "Telecom": [
        "BHARTIARTL-EQ", "IDEA-EQ", "INDUSTOWER-EQ", "TATACOMM-EQ",
        "HFCL-EQ", "TEJASNET-EQ", "STLTECH-EQ", "ITI-EQ", "ASTEC-EQ"
    ],
    "Infrastructure": [
        "LT-EQ", "GMRAIRPORT-EQ", "IRB-EQ", "NBCC-EQ", "RVNL-EQ", "KEC-EQ",
        "PNCINFRA-EQ", "KNRCON-EQ", "GRINFRA-EQ", "NCC-EQ", "HGINFRA-EQ",
        "ASHOKA-EQ", "SADBHAV-EQ", "JWL-EQ", "PATELENG-EQ", "KALPATPOWR-EQ",
        "IRCON-EQ", "ENGINERSIN-EQ", "AHLUWALIA-EQ", "PSPPROJECTS-EQ", "CAPACITE-EQ",
        "WELSPUNIND-EQ", "HCC-EQ", "MANINFRA-EQ", "RIIL-EQ",
        "JAYBARMARU-EQ"
    ],
    "Insurance": [
        "SBILIFE-EQ", "HDFCLIFE-EQ", "ICICIGI-EQ", "ICICIPRULI-EQ", "LICI-EQ",
        "GICRE-EQ", "NIACL-EQ", "STARHEALTH-EQ", "MAXFIN-EQ"
    ],
    "Diversified": [
        "ADANIENT-EQ", "GRASIM-EQ",
        "DCMSHRIRAM-EQ", "3MINDIA-EQ", "CENTURYPLY-EQ", "KFINTECH-EQ", "BALMERLAWRI-EQ",
        "GODREJIND-EQ", "BIRLACORPN-EQ"
    ],
    "Cement": [
        "ULTRACEMCO-EQ", "SHREECEM-EQ", "AMBUJACEM-EQ", "ACC-EQ", "JKCEMENT-EQ",
        "DALBHARAT-EQ", "RAMCOCEM-EQ", "NUVOCO-EQ", "JKLAKSHMI-EQ",
        "HEIDELBERG-EQ", "INDIACEM-EQ", "PRISMJOHNS-EQ", "STARCEMENT-EQ", "SAGCEM-EQ",
        "DECCANCE-EQ", "KCP-EQ", "ORIENTCEM-EQ", "HIL-EQ", "EVERESTIND-EQ",
        "VISAKAIND-EQ", "BIGBLOC-EQ"
    ],
    "Realty": [
        "DLF-EQ", "GODREJPROP-EQ", "OBEROIRLTY-EQ", "PHOENIXLTD-EQ", "PRESTIGE-EQ",
        "BRIGADE-EQ", "SOBHA-EQ", "SUNTECK-EQ", "MAHLIFE-EQ", "ANANTRAJ-EQ",
        "KOLTEPATIL-EQ", "PURVA-EQ", "ARVSMART-EQ", "RUSTOMJEE-EQ", "DBREALTY-EQ",
        "IBREALEST-EQ", "OMAXE-EQ", "ASHIANA-EQ", "ELDEHSG-EQ", "TARC-EQ"
    ],
    "Aviation": [
        "INDIGO-EQ", "SPICEJET-EQ", "GMRINFRA-EQ"
    ],
    "Retail": [
        "DMART-EQ", "TRENT-EQ", "ABFRL-EQ", "VMART-EQ", "SHOPERSTOP-EQ",
        "BATAINDIA-EQ", "METROBRAND-EQ", "ARVINDFASN-EQ", "CANTABIL-EQ", "ZOMATO-EQ",
        "NYKAA-EQ", "MANYAVAR-EQ", "LANDMARK-EQ", "V2RETAIL-EQ",
        "THANGAMAYL-EQ", "KALYANKJIL-EQ", "TITAN-EQ"
    ],
    "Media": [
        "ZEEL-EQ", "SUNTV-EQ", "TVTODAY-EQ", "DISHTV-EQ", "HATHWAY-EQ",
        "PVR-EQ", "INOXLEISUR-EQ", "SAREGAMA-EQ", "TIPS-EQ"
    ],
    "Consumer Durables": [
        "WHIRLPOOL-EQ", "DIXON-EQ", "AMBER-EQ", "VOLTAS-EQ", "BLUESTARCO-EQ",
        "HAVELLS-EQ", "CROMPTON-EQ", "VGUARD-EQ", "ORIENTELEC-EQ", "KIRIINDUS-EQ",
        "RAJESHEXPO-EQ", "SYMPHONY-EQ", "TITAN-EQ", "KALPATPOWR-EQ", "RELAXO-EQ",
        "TTKHLTCARE-EQ", "VAIBHAVGBL-EQ", "BAJAJELEC-EQ", "FINEORG-EQ", "CHOLAHLDNG-EQ",
        "BSLIMITED-EQ", "SUPRAJIT-EQ", "NIITLTD-EQ", "APARINDS-EQ"
    ],
    "Consumer Services": [
        "ZOMATO-EQ", "NYKAA-EQ", "ADANIPORTS-EQ", "IRCTC-EQ", "PAYTM-EQ",
        "JUBLFOOD-EQ", "DEVYANI-EQ", "WESTLIFE-EQ", "SAPPHIRE-EQ", "BIKAJI-EQ",
        "EASEMYTRIP-EQ", "IXIGO-EQ", "TEAMLEASE-EQ", "QUESS-EQ", "FIRSTSOURCE-EQ",
        "MINDSPACE-EQ", "MAHINDCIE-EQ", "TATAMTRDVR-EQ", "VMART-EQ", "SHOPERSTOP-EQ",
        "TRENT-EQ", "DMART-EQ", "ABFRL-EQ", "MANYAVAR-EQ", "V2RETAIL-EQ"
    ]
}

@st.cache_resource(ttl=86400)
def get_smartapi_session():
    """
    Initializes and caches the SmartAPI session to avoid repeated logins.
    Logins are limited to 1 per second, but we should only need one per day/session.
    """
    try:
        smart_api = SmartConnect(api_key=API_KEYS["Historical"])
        totp = pyotp.TOTP(TOTP_SECRET)
        data = smart_api.generateSession(CLIENT_ID, PASSWORD, totp.now())
        if data['status']:
            return smart_api
        else:
            st.error(f"⚠️ SmartAPI authentication failed: {data['message']}")
            return None
    except Exception as e:
        st.error(f"⚠️ Error initializing SmartAPI: {str(e)}")
        return None

def tooltip(label, explanation):
    return f"{label} 📌 ({explanation})"

def retry(max_retries=5, delay=5, backoff_factor=2, jitter=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:
                        retries += 1
                        if retries == max_retries:
                            raise e
                        sleep_time = (delay * (backoff_factor ** retries)) + random.uniform(0, jitter)
                        st.warning(f"Rate limit hit. Retrying after {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                    else:
                        raise e
                except (requests.exceptions.RequestException, ConnectionError) as e:
                    retries += 1
                    if retries == max_retries:
                        raise e
                    sleep_time = (delay * (backoff_factor ** retries)) + random.uniform(0, jitter)
                    time.sleep(sleep_time)
            # Fallback if loop ends oddly
            return None 
        return wrapper
    return decorator

@retry(max_retries=5, delay=5)
def fetch_nse_stock_list():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": random.choice(USER_AGENTS)})
        response = session.get(url, timeout=10)
        response.raise_for_status()
        nse_data = pd.read_csv(io.StringIO(response.text))
        stock_list = [f"{symbol}-EQ" for symbol in nse_data['SYMBOL']]
        return stock_list
    except Exception:
        return list(set([stock for sector in SECTORS.values() for stock in sector]))

# SmartAPI Rate Limiter for Candle Data
# Limit: 3 requests per second => 1 request every ~0.34 seconds
last_api_call_time = 0

def enforce_rate_limit(min_interval=0.4):
    global last_api_call_time
    current_time = time.time()
    elapsed = current_time - last_api_call_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    last_api_call_time = time.time()

@retry(max_retries=5, delay=5)
def fetch_stock_data_with_auth(symbol, period="2y", interval="1d"):
    cache_key = f"{symbol}_{period}_{interval}"
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return pd.read_pickle(io.BytesIO(cached_data))

    try:
        if "-EQ" not in symbol:
            symbol = f"{symbol.split('.')[0]}-EQ"

        # Use the cached session instead of creating a new one every time
        smart_api = get_smartapi_session()
        if not smart_api:
            # If session failed, try to re-initialize once (maybe expired)
            SmartConnect.generateSession = get_smartapi_session.func # Hack to access original func if needed, but clearing cache is safer
            st.cache_resource.clear()
            smart_api = get_smartapi_session()
            if not smart_api:
                 raise ValueError("SmartAPI client initialization failed")

        end_date = datetime.now()
        if period == "2y":
            start_date = end_date - timedelta(days=2 * 365)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "1mo":
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=365)

        interval_map = {
            "1d": "ONE_DAY",
            "1h": "ONE_HOUR",
            "5m": "FIVE_MINUTE",
            "15m": "FIFTEEN_MINUTE"
        }
        api_interval = interval_map.get(interval, "ONE_DAY")

        symbol_token_map = load_symbol_token_map()
        symboltoken = symbol_token_map.get(symbol)
        if not symboltoken:
            st.warning(f"⚠️ Token not found for symbol: {symbol}")
            return pd.DataFrame()

        # Enforce rate limit before making the API call
        enforce_rate_limit()

        historical_data = smart_api.getCandleData({
            "exchange": "NSE",
            "symboltoken": symboltoken,
            "interval": api_interval,
            "fromdate": start_date.strftime("%Y-%m-%d %H:%M"),
            "todate": end_date.strftime("%Y-%m-%d %H:%M")
        })

        if historical_data and isinstance(historical_data, dict) and historical_data.get('status') and historical_data.get('data'):
            data = pd.DataFrame(historical_data['data'], columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)

            buffer = io.BytesIO()
            data.to_pickle(buffer)
            cache.set(cache_key, buffer.getvalue(), expire=86400)
            return data
        else:
            # Gentle warning instead of raising huge error if data is just missing
            # print(f"No data for {symbol}: {historical_data}")
            return pd.DataFrame()

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            st.warning(f"⚠️ Rate limit exceeded for {symbol}. Skipping...")
            return pd.DataFrame()
        raise e
    except Exception as e:
        # Check for specific "Rate Limit" string in exception message
        if "exceeding access rate" in str(e):
             st.warning(f"Rate limit hit for {symbol}. Slowing down...")
             time.sleep(2)
             return pd.DataFrame()
        st.warning(f"⚠️ Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

@lru_cache(maxsize=1000)
def fetch_stock_data_cached(symbol, period="2y", interval="1d"):
    return fetch_stock_data_with_auth(symbol, period, interval)

def calculate_advance_decline_ratio(stock_list):
    advances = 0
    declines = 0
    for symbol in stock_list:
        data = fetch_stock_data_cached(symbol)
        if not data.empty:
            if data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                advances += 1
            else:
                declines += 1
    return advances / declines if declines != 0 else 0

def monte_carlo_simulation(data, simulations=1000, days=30):
    returns = data['Close'].pct_change().dropna()
    if len(returns) < 30:
        mean_return = returns.mean()
        std_return = returns.std()
        simulation_results = []
        for _ in range(simulations):
            price_series = [data['Close'].iloc[-1]]
            for _ in range(days):
                price = price_series[-1] * (1 + np.random.normal(mean_return, std_return))
                price_series.append(price)
            simulation_results.append(price_series)
        return simulation_results
    
    model = arch_model(returns, vol='GARCH', p=1, q=1, dist='Normal', rescale=False)
    garch_fit = model.fit(disp='off')
    forecasts = garch_fit.forecast(horizon=days)
    volatility = np.sqrt(forecasts.variance.iloc[-1].values)
    mean_return = returns.mean()
    simulation_results = []
    for _ in range(simulations):
        price_series = [data['Close'].iloc[-1]]
        for i in range(days):
            price = price_series[-1] * (1 + np.random.normal(mean_return, volatility[i]))
            price_series.append(price)
        simulation_results.append(price_series)
    return simulation_results

def extract_entities(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return entities

def get_trending_stocks():
    pytrends = TrendReq(hl='en-US', tz=360)
    trending = pytrends.trending_searches(pn='india')
    return trending

def calculate_confidence_score(data):
    score = 0
    if 'RSI' in data.columns and data['RSI'].iloc[-1] is not None and data['RSI'].iloc[-1] < 30:
        score += 1
    if 'MACD' in data.columns and 'MACD_signal' in data.columns and data['MACD'].iloc[-1] is not None and data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
        score += 1
    if 'Ichimoku_Span_A' in data.columns and data['Close'].iloc[-1] is not None and data['Close'].iloc[-1] > data['Ichimoku_Span_A'].iloc[-1]:
        score += 1
    if 'ATR' in data.columns and data['ATR'].iloc[-1] is not None and data['Close'].iloc[-1] is not None:
        atr_volatility = data['ATR'].iloc[-1] / data['Close'].iloc[-1]
        if atr_volatility < 0.02:
            score += 0.5
        elif atr_volatility > 0.05:
            score -= 0.5
    return min(max(score / 3.5, 0), 1)

def assess_risk(data):
    if 'ATR' in data.columns and data['ATR'].iloc[-1] is not None and data['ATR'].iloc[-1] > data['ATR'].mean():
        return "High Volatility Warning"
    else:
        return "Low Volatility"

def optimize_rsi_window(data, windows=range(5, 15)):
    best_window, best_sharpe = 9, -float('inf')
    returns = data['Close'].pct_change().dropna()
    if len(returns) < 50:
        return best_window
    for window in windows:
        rsi = ta.momentum.RSIIndicator(data['Close'], window=window).rsi()
        signals = (rsi < 30).astype(int) - (rsi > 70).astype(int)
        strategy_returns = signals.shift(1) * returns
        sharpe = strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() != 0 else 0
        if sharpe > best_sharpe:
            best_sharpe, best_window = sharpe, window
    return best_window

def detect_divergence(data):
    rsi = data['RSI']
    price = data['Close']
    recent_highs = price[-5:].idxmax()
    recent_lows = price[-5:].idxmin()
    rsi_highs = rsi[-5:].idxmax()
    rsi_lows = rsi[-5:].idxmin()
    bullish_div = (recent_lows > rsi_lows) and (price[recent_lows] < price[-1]) and (rsi[rsi_lows] < rsi[-1])
    bearish_div = (recent_highs < rsi_highs) and (price[recent_highs] > price[-1]) and (rsi[rsi_highs] > rsi[-1])
    return "Bullish Divergence" if bullish_div else "Bearish Divergence" if bearish_div else "No Divergence"

def calculate_cmo(close, window=14):
    try:
        diff = close.diff()
        up_sum = diff.where(diff > 0, 0).rolling(window=window).sum()
        down_sum = abs(diff.where(diff < 0, 0)).rolling(window=window).sum()
        cmo = 100 * (up_sum - down_sum) / (up_sum + down_sum)
        return cmo
    except Exception as e:
        st.warning(f"⚠️ Failed to compute custom CMO: {str(e)}")
        return None

INDICATOR_MIN_LENGTHS = {
    'RSI': 14,
    'MACD': 26,
    'SMA_50': 50,
    'SMA_200': 200,
    'EMA_20': 20,
    'EMA_50': 50,
    'Bollinger': 20,
    'Stochastic': 14,
    'ATR': 14,
    'ADX': 27,
    'OBV': 1,
    'VWAP': 1,
    'Volume_Spike': 10,
    'Parabolic_SAR': 2,
    'Fibonacci': 1,
    'Divergence': 5,
    'Ichimoku': 52,
    'CMF': 20,
    'Donchian': 20,
    'Keltner': 20,
    'TRIX': 15,
    'Ultimate_Osc': 28,
    'CMO': 14,
    'VPT': 1
}

def can_compute_indicator(data, indicator):
    """
    Checks if sufficient data is available for a specific indicator.
    Returns True if computation is possible, False otherwise.
    """
    required_length = INDICATOR_MIN_LENGTHS.get(indicator, 1)
    return len(data) >= required_length

logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s: %(message)s")

def validate_data(
    data: pd.DataFrame,
    required_columns=None,
    min_length: int = 50,
    max_volume: float | None = 1e10,
    check_positive_prices: bool = True,
) -> bool:
    """
    Comprehensive OHLCV DataFrame validator.
    
    Parameters
    ----------
    data : pd.DataFrame
        Stock price data (must include at least Open/High/Low/Close/Volume columns).
    required_columns : list[str] | None
        Columns that must be present. Defaults to the standard OHLCV set.
    min_length : int
        Minimum number of rows required for the DataFrame.
    max_volume : float | None
        Flag rows with unrealistically large volume figures. Set to None to skip.
    check_positive_prices : bool
        If True, verifies that all price columns are > 0.

    Returns
    -------
    bool
        True if all checks pass; otherwise False (with warnings logged).
    """
    # Default required columns
    if required_columns is None:
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # 1 — basic integrity
    if data is None or data.empty:
        logging.warning("No data provided for validation.")
        return False
    if len(data) < min_length:
        logging.warning("Insufficient data length: %d rows (minimum %d required).",
                        len(data), min_length)
        return False

    # 2 — schema
    missing = [c for c in required_columns if c not in data.columns]
    if missing:
        logging.warning("Missing required columns: %s", ", ".join(missing))
        return False

    # 3 — nulls
    if data[required_columns].isnull().any().any():
        logging.warning("Data contains null values in required columns.")
        return False

    # 4 — positive prices
    price_cols = [c for c in ('Open', 'High', 'Low', 'Close') if c in data.columns]
    if check_positive_prices and (data[price_cols] <= 0).any().any():
        logging.warning("Invalid price values (≤ 0 detected).")
        return False

    # 5 — volume sanity
    if max_volume is not None and 'Volume' in data.columns \
       and data['Volume'].max() > max_volume:
        logging.warning("Abnormal volume values detected (max %.0f > %.0f).",
                        data['Volume'].max(), max_volume)
        return False

    return True

def analyze_stock(data):
    """
    Computes technical indicators for stock data after validation.
    Returns data with indicators or an empty DataFrame on failure.
    """
    if not validate_data(data, min_length=50):
        columns = [
            'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'SMA_50', 'SMA_200', 'EMA_20', 'EMA_50',
            'Upper_Band', 'Middle_Band', 'Lower_Band', 'SlowK', 'SlowD', 'ATR', 'ADX', 'OBV',
            'VWAP', 'Avg_Volume', 'Volume_Spike', 'Parabolic_SAR', 'Fib_23.6', 'Fib_38.2',
            'Fib_50.0', 'Fib_61.8', 'Divergence', 'Ichimoku_Tenkan', 'Ichimoku_Kijun',
            'Ichimoku_Span_A', 'Ichimoku_Span_B', 'Ichimoku_Chikou', 'CMF', 'Donchian_Upper',
            'Donchian_Lower', 'Donchian_Middle', 'Keltner_Upper', 'Keltner_Middle', 'Keltner_Lower',
            'TRIX', 'Ultimate_Osc', 'CMO', 'VPT'
        ]
        for col in columns:
            data[col] = None
        return data

    try:
        if can_compute_indicator(data, 'RSI'):
            rsi_window = optimize_rsi_window(data)
            data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=rsi_window).rsi()
        else:
            data['RSI'] = None
    except Exception as e:
        logging.warning(f"Failed to compute RSI: {str(e)}")
        data['RSI'] = None

    try:
        if can_compute_indicator(data, 'MACD'):
            macd = ta.trend.MACD(data['Close'], window_slow=17, window_fast=8, window_sign=9)
            data['MACD'] = macd.macd()
            data['MACD_signal'] = macd.macd_signal()
            data['MACD_hist'] = macd.macd_diff()
        else:
            data['MACD'] = data['MACD_signal'] = data['MACD_hist'] = None
    except Exception as e:
        logging.warning(f"Failed to compute MACD: {str(e)}")
        data['MACD'] = data['MACD_signal'] = data['MACD_hist'] = None

    try:
        if can_compute_indicator(data, 'SMA_50'):
            data['SMA_50'] = ta.trend.SMAIndicator(data['Close'], window=50).sma_indicator()
        else:
            data['SMA_50'] = None
        if can_compute_indicator(data, 'SMA_200'):
            data['SMA_200'] = ta.trend.SMAIndicator(data['Close'], window=200).sma_indicator()
        else:
            data['SMA_200'] = None
        if can_compute_indicator(data, 'EMA_20'):
            data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
        else:
            data['EMA_20'] = None
        if can_compute_indicator(data, 'EMA_50'):
            data['EMA_50'] = ta.trend.EMAIndicator(data['Close'], window=50).ema_indicator()
        else:
            data['EMA_50'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Moving Averages: {str(e)}")
        data['SMA_50'] = data['SMA_200'] = data['EMA_20'] = data['EMA_50'] = None

    try:
        if can_compute_indicator(data, 'Bollinger'):
            bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
            data['Upper_Band'] = bollinger.bollinger_hband()
            data['Middle_Band'] = bollinger.bollinger_mavg()
            data['Lower_Band'] = bollinger.bollinger_lband()
        else:
            data['Upper_Band'] = data['Middle_Band'] = data['Lower_Band'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Bollinger Bands: {str(e)}")
        data['Upper_Band'] = data['Middle_Band'] = data['Lower_Band'] = None

    try:
        if can_compute_indicator(data, 'Stochastic'):
            stoch = ta.momentum.StochasticOscillator(data['High'], data['Low'], data['Close'], window=14, smooth_window=3)
            data['SlowK'] = stoch.stoch()
            data['SlowD'] = stoch.stoch_signal()
        else:
            data['SlowK'] = data['SlowD'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Stochastic: {str(e)}")
        data['SlowK'] = data['SlowD'] = None

    try:
        if can_compute_indicator(data, 'ATR'):
            data['ATR'] = ta.volatility.AverageTrueRange(data['High'], data['Low'], data['Close'], window=14).average_true_range()
        else:
            data['ATR'] = None
    except Exception as e:
        logging.warning(f"Failed to compute ATR: {str(e)}")
        data['ATR'] = None

    try:
        if can_compute_indicator(data, 'ADX'):
            data['ADX'] = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14).adx()
        else:
            data['ADX'] = None
    except Exception as e:
        logging.warning(f"Failed to compute ADX: {str(e)}")
        data['ADX'] = None

    try:
        if can_compute_indicator(data, 'OBV'):
            data['OBV'] = ta.volume.OnBalanceVolumeIndicator(data['Close'], data['Volume']).on_balance_volume()
        else:
            data['OBV'] = None
    except Exception as e:
        logging.warning(f"Failed to compute OBV: {str(e)}")
        data['OBV'] = None

    try:
        if can_compute_indicator(data, 'VWAP'):
            data['Cumulative_TP'] = ((data['High'] + data['Low'] + data['Close']) / 3) * data['Volume']
            data['Cumulative_Volume'] = data['Volume'].cumsum()
            data['VWAP'] = data['Cumulative_TP'].cumsum() / data['Cumulative_Volume']
        else:
            data['VWAP'] = None
    except Exception as e:
        logging.warning(f"Failed to compute VWAP: {str(e)}")
        data['VWAP'] = None

    try:
        if can_compute_indicator(data, 'Volume_Spike'):
            data['Avg_Volume'] = data['Volume'].rolling(window=10).mean()
            data['Volume_Spike'] = data['Volume'] > (data['Avg_Volume'] * 1.5)
        else:
            data['Avg_Volume'] = data['Volume_Spike'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Volume Spike: {str(e)}")
        data['Avg_Volume'] = data['Volume_Spike'] = None

    try:
        if can_compute_indicator(data, 'Parabolic_SAR'):
            data['Parabolic_SAR'] = ta.trend.PSARIndicator(data['High'], data['Low'], data['Close']).psar()
        else:
            data['Parabolic_SAR'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Parabolic SAR: {str(e)}")
        data['Parabolic_SAR'] = None

    try:
        if can_compute_indicator(data, 'Fibonacci'):
            high = data['High'].max()
            low = data['Low'].min()
            diff = high - low
            data['Fib_23.6'] = high - diff * 0.236
            data['Fib_38.2'] = high - diff * 0.382
            data['Fib_50.0'] = high - diff * 0.5
            data['Fib_61.8'] = high - diff * 0.618
        else:
            data['Fib_23.6'] = data['Fib_38.2'] = data['Fib_50.0'] = data['Fib_61.8'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Fibonacci: {str(e)}")
        data['Fib_23.6'] = data['Fib_38.2'] = data['Fib_50.0'] = data['Fib_61.8'] = None

    try:
        if can_compute_indicator(data, 'Divergence'):
            data['Divergence'] = detect_divergence(data)
        else:
            data['Divergence'] = "No Divergence"
    except Exception as e:
        logging.warning(f"Failed to compute Divergence: {str(e)}")
        data['Divergence'] = "No Divergence"

    try:
        if can_compute_indicator(data, 'Ichimoku'):
            ichimoku = ta.trend.IchimokuIndicator(data['High'], data['Low'], window1=9, window2=26, window3=52)
            data['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
            data['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()
            data['Ichimoku_Span_A'] = ichimoku.ichimoku_a()
            data['Ichimoku_Span_B'] = ichimoku.ichimoku_b()
            data['Ichimoku_Chikou'] = data['Close'].shift(-26)
        else:
            data['Ichimoku_Tenkan'] = data['Ichimoku_Kijun'] = data['Ichimoku_Span_A'] = data['Ichimoku_Span_B'] = data['Ichimoku_Chikou'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Ichimoku: {str(e)}")
        data['Ichimoku_Tenkan'] = data['Ichimoku_Kijun'] = data['Ichimoku_Span_A'] = data['Ichimoku_Span_B'] = data['Ichimoku_Chikou'] = None

    try:
        if can_compute_indicator(data, 'CMF'):
            data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(data['High'], data['Low'], data['Close'], data['Volume'], window=20).chaikin_money_flow()
        else:
            data['CMF'] = None
    except Exception as e:
        logging.warning(f"Failed to compute CMF: {str(e)}")
        data['CMF'] = None

    try:
        if can_compute_indicator(data, 'Donchian'):
            donchian = ta.volatility.DonchianChannel(data['High'], data['Low'], data['Close'], window=20)
            data['Donchian_Upper'] = donchian.donchian_channel_hband()
            data['Donchian_Lower'] = donchian.donchian_channel_lband()
            data['Donchian_Middle'] = donchian.donchian_channel_mband()
        else:
            data['Donchian_Upper'] = data['Donchian_Lower'] = data['Donchian_Middle'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Donchian: {str(e)}")
        data['Donchian_Upper'] = data['Donchian_Lower'] = data['Donchian_Middle'] = None

    try:
        if can_compute_indicator(data, 'Keltner'):
            keltner = ta.volatility.KeltnerChannel(data['High'], data['Low'], data['Close'], window=20, window_atr=10)
            data['Keltner_Upper'] = keltner.keltner_channel_hband()
            data['Keltner_Middle'] = keltner.keltner_channel_mband()
            data['Keltner_Lower'] = keltner.keltner_channel_lband()
        else:
            data['Keltner_Upper'] = data['Keltner_Middle'] = data['Keltner_Lower'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Keltner Channels: {str(e)}")
        data['Keltner_Upper'] = data['Keltner_Middle'] = data['Keltner_Lower'] = None

    try:
        if can_compute_indicator(data, 'TRIX'):
            data['TRIX'] = ta.trend.TRIXIndicator(data['Close'], window=15).trix()
        else:
            data['TRIX'] = None
    except Exception as e:
        logging.warning(f"Failed to compute TRIX: {str(e)}")
        data['TRIX'] = None

    try:
        if can_compute_indicator(data, 'Ultimate_Osc'):
            data['Ultimate_Osc'] = ta.momentum.UltimateOscillator(
                data['High'], data['Low'], data['Close'], window1=7, window2=14, window3=28
            ).ultimate_oscillator()
        else:
            data['Ultimate_Osc'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Ultimate Oscillator: {str(e)}")
        data['Ultimate_Osc'] = None

    try:
        if can_compute_indicator(data, 'CMO'):
            data['CMO'] = calculate_cmo(data['Close'], window=14)
        else:
            data['CMO'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Chande Momentum Oscillator: {str(e)}")
        data['CMO'] = None

    try:
        if can_compute_indicator(data, 'VPT'):
            data['VPT'] = ta.volume.VolumePriceTrendIndicator(data['Close'], data['Volume']).volume_price_trend()
        else:
            data['VPT'] = None
    except Exception as e:
        logging.warning(f"Failed to compute Volume Price Trend: {str(e)}")
        data['VPT'] = None

    return data
    
def calculate_buy_at(data, patience="high"):
    if data.empty or 'RSI' not in data.columns or data['RSI'].iloc[-1] is None:
        st.warning("⚠️ Cannot calculate Buy At due to missing or invalid RSI data.")
        return None
    if 'ATR' in data.columns and pd.notnull(data['ATR'].iloc[-1]):
        current_close = data['Close'].iloc[-1]
        atr = data['ATR'].iloc[-1]
        adx = data['ADX'].iloc[-1] if 'ADX' in data.columns else 0
        upper_band = data['Upper_Band'].iloc[-1] if 'Upper_Band' in data.columns else float('inf')
        
        # 1. Breakout / Strong Momentum (ADX > 25) -> Buy Above Logic
        # Added Volume Confirmation: Volume > 1.2x Avg Volume (User Request)
        vol_confirm = True
        if 'Volume' in data.columns and 'Avg_Volume' in data.columns:
            if data['Volume'].iloc[-1] < data['Avg_Volume'].iloc[-1] * 1.2:
                vol_confirm = False
        
        if adx > 25:
             if vol_confirm:
                 # Strategy: Breakout Entry
                 entry_type = "Breakout"
                 # Buy just above resistance (Upper Band) or strictly above current price if momentum is raging
                 if upper_band != float('inf'):
                     buy_at = max(current_close, upper_band) * 1.001
                 else:
                     buy_at = current_close * 1.002
             else:
                 # Failed Breakout (Low Volume) -> Wait for Pullback instead
                 entry_type = "Pullback"
                 buy_at = current_close - (0.2 * atr)

        
        # 2. Pullback / Trends (ADX 20-30 or patience="low")
        elif adx > 20 or patience == "low":
             entry_type = "Pullback"
             if patience == "low":
                 # Intraday Pullback: Adaptive Depth (0.2 - 0.35 ATR) based on Volatility
                 # Higher Volatility (ATR %) -> Deeper Pullback required
                 atr_pct = atr / current_close
                 pullback_depth = 0.35 if atr_pct > 0.02 else 0.2
                 
                 buy_at = current_close - (pullback_depth * atr)
                 # DISABLE VWAP cap for Intraday to avoid "huge price difference" (User Feedback)
                 # We trust momentum/trend more than mean reversion for day trading
             else:
                 # Daily/Swing Pullback: Deeper discount (0.5 ATR)
                 buy_at = current_close - (0.5 * atr)
                 
                 # Safety for Swing: don't buy above VWAP if trending normally
                 if 'VWAP' in data.columns and pd.notnull(data['VWAP'].iloc[-1]):
                    vwap = data['VWAP'].iloc[-1]
                    buy_at = min(buy_at, vwap * 1.01)

        # 3. Choppy (ADX < 20) -> No Trade
        else:
             entry_type = "Choppy"
             buy_at = None

    else:
        # Fallback if no ATR
        buy_at = data['Close'].iloc[-1] * 0.995 # 0.5% discount
        entry_type = "Standard"
        
    final_price = round(buy_at, 2) if buy_at else None
    return final_price, entry_type

def calculate_stop_loss(data, atr_multiplier=1.5, entry_price=None):
    if data.empty or 'ATR' not in data.columns or data['ATR'].iloc[-1] is None:
        st.warning("⚠️ Cannot calculate Stop Loss due to missing or invalid ATR data.")
        return None
    last_close = entry_price if entry_price else data['Close'].iloc[-1]
    last_atr = data['ATR'].iloc[-1]
    
    # Intraday Risk Management (Tighter SL)
    # Breakout: 1.8 - 2.2 ATR (Max 2.2 to survive noise)
    # Pullback: 1.5 ATR (Standard)
    # We use a base of 1.5, can be overridden by caller or adjusted here
    
    # If high volatility, maybe tighten to protect capital? Or widen to avoid chop?
    # User Request: Breakout = 1.8-2.2 ATR, Pullback = 1.5 ATR
    # For now, we update default to reflect Intraday preference
    
    stop_loss = last_close - (atr_multiplier * last_atr)
    
    # Ensure SL is below entry
    if stop_loss > last_close: 
        stop_loss = last_close - last_atr 
    
    return round(stop_loss, 2)

def calculate_target(data, risk_reward_ratio=3, entry_price=None):
    stop_loss = calculate_stop_loss(data, entry_price=entry_price)
    if stop_loss is None:
        st.warning("⚠️ Cannot calculate Target due to missing Stop Loss data.")
        return None
    last_close = entry_price if entry_price else data['Close'].iloc[-1]
    risk = last_close - stop_loss
    adjusted_ratio = min(risk_reward_ratio, 5) if data['ADX'].iloc[-1] is not None and data['ADX'].iloc[-1] > 25 else min(risk_reward_ratio, 3)
    target = last_close + (risk * adjusted_ratio)
    if target > last_close * 1.2:
        target = last_close * 1.2
    return round(target, 2)

def calculate_buy_at_row(row):
    if pd.notnull(row.get('ATR')):
        current_close = row['Close']
        atr = row['ATR']
        adx = row.get('ADX', 0)
        upper_band = row.get('Upper_Band', float('inf'))
        
        # Context-aware logic for row-based calculation
        if current_close > upper_band or (pd.notnull(adx) and adx > 25):
             buy_at = current_close - (0.2 * atr)
             if upper_band != float('inf'):
                 buy_at = min(buy_at, upper_band * 1.005)
             return round(buy_at, 2)
        elif pd.notnull(adx) and adx < 20:
             return None # Skip in choppy
        else:
             buy_at = current_close - (0.5 * atr)
             if pd.notnull(row.get('VWAP')):
                 buy_at = min(buy_at, row['VWAP'] * 1.01)
             return round(buy_at, 2)

    elif pd.notnull(row.get('RSI')) and row['RSI'] < 30:
        return round(row['Close'] * 0.99, 2)
    return round(row['Close'] * 0.995, 2)

def calculate_stop_loss_row(row, atr_multiplier=2.5):
    if pd.notnull(row['ATR']):
        atr_multiplier = 3.0 if pd.notnull(row['ADX']) and row['ADX'] > 25 else 1.5
        stop_loss = row['Close'] - (atr_multiplier * row['ATR'])
        if stop_loss < row['Close'] * 0.9:
            stop_loss = row['Close'] * 0.9
        return round(stop_loss, 2)
    return None

def calculate_target_row(row, risk_reward_ratio=3):
    stop_loss = calculate_stop_loss_row(row)
    if stop_loss is not None:
        risk = row['Close'] - stop_loss
        adjusted_ratio = min(risk_reward_ratio, 5) if pd.notnull(row['ADX']) and row['ADX'] > 25 else min(risk_reward_ratio, 3)
        target = row['Close'] + (risk * adjusted_ratio)
        if target > row['Close'] * 1.2:
            target = row['Close'] * 1.2
        return round(target, 2)
    return None

def fetch_fundamentals(symbol):
    try:
        smart_api = init_smartapi_client()
        if not smart_api:
            return {'P/E': float('inf'), 'EPS': 0, 'RevenueGrowth': 0}
        return {'P/E': float('inf'), 'EPS': 0, 'RevenueGrowth': 0}
    except Exception:
        return {'P/E': float('inf'), 'EPS': 0, 'RevenueGrowth': 0}

# Improved strategy logic using adaptive regime detection, signal scoring, and volatility-aware filters

def classify_market_regime(data):
    """Classifies regime based on volatility and trend"""
    data['ATR_pct'] = data['ATR'] / data['Close']
    if data['ATR_pct'].iloc[-1] > 0.03:
        return 'volatile'
    elif data['Close'].iloc[-1] > data['SMA_50'].iloc[-1]:
        return 'bullish'
    else:
        return 'neutral'

def compute_signal_score(data, symbol=None):
    """
    Computes a weighted score based on normalized technical and fundamental indicators.
    Returns a score between -10 and 10, with negative scores indicating no trade.
    """
    score = 0.0
    weights = {
        'RSI': 1.5,
        'MACD': 1.2,
        'Ichimoku': 1.5,
        'CMF': 0.5,
        'ATR_Volatility': 1.0,
        'Breakout': 1.2,
        'Fundamentals': 1.0
    }

    # Volume filter: skip low volume days
    if data['Volume'].iloc[-1] < data['Avg_Volume'].iloc[-1] * 0.5:
        return -10  # Force no trade

    # RSI: Context-Aware Scoring (Momentum vs Mean Reversion)
    if 'RSI' in data.columns and pd.notnull(data['RSI'].iloc[-1]):
        rsi = data['RSI'].iloc[-1]
        
        # Determine Market Context
        adx = data['ADX'].iloc[-1] if 'ADX' in data.columns else 0
        is_trending = adx > 25
        
        if is_trending:
            # Momentum / Breakout Mode
            # Bullish: RSI 55-70 (Strong momentum but not exhausted)
            if 55 <= rsi <= 75:
                score += weights['RSI'] * 1.0
            elif rsi > 75:
                score -= weights['RSI'] * 0.5 # Getting overextended
            elif rsi < 40:
                score -= weights['RSI'] * 1.0 # Loss of momentum
        else:
            # Pullback / Range Mode
            # Bullish: RSI 35-50 (Healthy pullback / oversold in range)
            if 35 <= rsi <= 55:
                score += weights['RSI'] * 1.0
            elif rsi < 30:
                score += weights['RSI'] * 1.5 # Deep value
            elif rsi > 70:
                score -= weights['RSI'] * 1.0 # Overbought
                
    # MACD: Check crossover with signal line
    if 'MACD' in data.columns and 'MACD_signal' in data.columns and pd.notnull(data['MACD'].iloc[-1]) and pd.notnull(data['MACD_signal'].iloc[-1]):
        macd_diff = data['MACD'].iloc[-1] - data['MACD_signal'].iloc[-1]
        macd_normalized = macd_diff / (data['MACD'].std() + 1e-10)  # Normalize by volatility
        if macd_diff > 0:
            score += weights['MACD'] * max(macd_normalized, 0)
        else:
            score -= weights['MACD'] * min(macd_normalized, 0)

    # Ichimoku: Check price vs cloud
    if 'Ichimoku_Span_A' in data.columns and 'Ichimoku_Span_B' in data.columns and pd.notnull(data['Ichimoku_Span_A'].iloc[-1]):
        close = data['Close'].iloc[-1]
        span_a, span_b = data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]
        if close > max(span_a, span_b):
            score += weights['Ichimoku']
        elif close < min(span_a, span_b):
            score -= weights['Ichimoku']

    # CMF: Money flow
    if 'CMF' in data.columns and pd.notnull(data['CMF'].iloc[-1]):
        cmf = data['CMF'].iloc[-1]
        score += weights['CMF'] * cmf  # CMF is already in [-1, 1]

    # ATR Volatility: Penalize high volatility
    if 'ATR' in data.columns and pd.notnull(data['ATR'].iloc[-1]):
        atr_pct = data['ATR'].iloc[-1] / data['Close'].iloc[-1]
        if atr_pct > 0.04:
            score -= weights['ATR_Volatility'] * (atr_pct / 0.04)

    # Donchian Breakout
    if 'Donchian_Upper' in data.columns and pd.notnull(data['Donchian_Upper'].iloc[-1]):
        if data['Close'].iloc[-1] > data['Donchian_Upper'].iloc[-1]:
            score += weights['Breakout']
        elif data['Close'].iloc[-1] < data['Donchian_Lower'].iloc[-1]:
            score -= weights['Breakout']

    # Fundamentals
    if symbol:
        fundamentals = fetch_fundamentals(symbol)
        if fundamentals['P/E'] < 15 and fundamentals['EPS'] > 0:
            score += weights['Fundamentals'] * 0.5
        elif fundamentals['P/E'] > 30 or fundamentals['EPS'] < 0:
            score -= weights['Fundamentals'] * 0.5
        if fundamentals['RevenueGrowth'] > 0.1:
            score += weights['Fundamentals'] * 0.3

    return min(max(score, -10), 10)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def adaptive_recommendation(data, symbol=None):
    """
    Generate a trading recommendation based on market regime and technical indicators.
    Returns a dictionary with all required fields, even in edge cases.
    """
    try:
        if not validate_data(data, min_length=50):
            logging.warning("Insufficient data for adaptive recommendation")
            return {
                "Current Price": None,
                "Buy At": None,
                "Stop Loss": None,
                "Target": None,
                "Recommendation": "Hold",
                "Score": 0,
                "Regime": "Unknown",
                "Position Size": None,
                "Trailing Stop": None,
                "Reason": "Insufficient data"
            }

        # Extract latest data
        current_price = data['Close'].iloc[-1] if 'Close' in data else None
        if current_price is None or pd.isna(current_price):
            logging.warning("No valid close price available")
            return {
                "Current Price": None,
                "Buy At": None,
                "Stop Loss": None,
                "Target": None,
                "Recommendation": "Hold",
                "Score": 0,
                "Regime": "Unknown",
                "Position Size": None,
                "Trailing Stop": None,
                "Reason": "No valid close price"
            }

        # Market regime classification
        atr = data['ATR'].iloc[-1] if 'ATR' in data and pd.notnull(data['ATR'].iloc[-1]) else 0
        sma_50 = data['SMA_50'].iloc[-1] if 'SMA_50' in data and pd.notnull(data['SMA_50'].iloc[-1]) else current_price
        regime = ("High Volatility" if atr > 0.05 * current_price else
                 "Bullish" if current_price > sma_50 else "Neutral")

        # Compute signal score
        score = compute_signal_score(data, symbol)

        # Filters
        if current_price < 100 or atr < 5 or data['Volume'].iloc[-1] < 5000:
            logging.info("Stock filtered out due to low price, ATR, or volume")
            return {
                "Current Price": current_price,
                "Buy At": None,
                "Stop Loss": None,
                "Target": None,
                "Recommendation": "Hold",
                "Score": score,
                "Regime": regime,
                "Position Size": None,
                "Trailing Stop": None,
                "Reason": "Low price, ATR, or volume"
            }

        # Recommendation logic with confidence threshold
        confidence_threshold = 1.0
        if score > confidence_threshold:
            recommendation = "Buy"
            reason = f"Bullish signals (Score: {score:.2f}) in {regime} regime"
        elif score < -confidence_threshold:
            recommendation = "Sell"
            reason = f"Bearish signals (Score: {score:.2f}) in {regime} regime"
        else:
            recommendation = "Hold"
            reason = f"Neutral signals (Score: {score:.2f}) in {regime} regime"

        # Trading parameters
        buy_at = current_price * 1.01 if recommendation == "Buy" else None
        stop_loss = current_price * 0.95 if recommendation == "Buy" else current_price * 1.05 if recommendation == "Sell" else None
        target = current_price * 1.05 if recommendation == "Buy" else current_price * 0.95 if recommendation == "Sell" else None
        position_size = min(100000 / current_price, 100) if recommendation in ["Buy", "Sell"] else None
        trailing_stop = current_price - (atr * 2) if recommendation == "Buy" else current_price + (atr * 2) if recommendation == "Sell" else None

        return {
            "Current Price": current_price,
            "Buy At": buy_at,
            "Stop Loss": stop_loss,
            "Target": target,
            "Recommendation": recommendation,
            "Score": score,
            "Regime": regime,
            "Position Size": position_size,
            "Trailing Stop": trailing_stop,
            "Reason": reason
        }
    except Exception as e:
        logging.error(f"Error in adaptive_recommendation: {str(e)}")
        return {
            "Current Price": None,
            "Buy At": None,
            "Stop Loss": None,
            "Target": None,
            "Recommendation": "Hold",
            "Score": 0,
            "Regime": "Unknown",
            "Position Size": None,
            "Trailing Stop": None,
            "Reason": f"Error: {str(e)}"
        }
        
def generate_recommendations(data, symbol=None):
    recommendations = {
        "Intraday": "Hold", "Swing": "Hold",
        "Short-Term": "Hold", "Long-Term": "Hold",
        "Mean_Reversion": "Hold", "Breakout": "Hold", "Ichimoku_Trend": "Hold",
        "Current Price": None, "Buy At": None,
        "Stop Loss": None, "Target": None, "Score": 0
    }

    if not validate_data(data, min_length=27):
        return recommendations

    if data.empty or len(data) < 27 or 'Close' not in data.columns or data['Close'].iloc[-1] is None:
        st.warning("⚠️ Insufficient data for recommendations.")
        return recommendations

    try:
        recommendations["Current Price"] = float(data['Close'].iloc[-1])
        buy_score = 0
        sell_score = 0

        if 'RSI' in data.columns and data['RSI'].iloc[-1] is not None and len(data['RSI'].dropna()) >= 1:
            if isinstance(data['RSI'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['RSI'].iloc[-1] <= 20:
                    buy_score += 4
                elif data['RSI'].iloc[-1] < 30:
                    buy_score += 2
                elif data['RSI'].iloc[-1] > 70:
                    sell_score += 2

        if 'MACD' in data.columns and 'MACD_signal' in data.columns and data['MACD'].iloc[-1] is not None and data['MACD_signal'].iloc[-1] is not None and len(data['MACD'].dropna()) >= 1:
            if isinstance(data['MACD'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['MACD_signal'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1]:
                    buy_score += 1
                elif data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1]:
                    sell_score += 1

        if 'Close' in data.columns and 'Lower_Band' in data.columns and 'Upper_Band' in data.columns and data['Close'].iloc[-1] is not None and len(data['Lower_Band'].dropna()) >= 1:
            if isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Lower_Band'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Upper_Band'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['Close'].iloc[-1] < data['Lower_Band'].iloc[-1]:
                    buy_score += 1
                elif data['Close'].iloc[-1] > data['Upper_Band'].iloc[-1]:
                    sell_score += 1

        if 'VWAP' in data.columns and data['VWAP'].iloc[-1] is not None and data['Close'].iloc[-1] is not None and len(data['VWAP'].dropna()) >= 1:
            if isinstance(data['VWAP'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['Close'].iloc[-1] > data['VWAP'].iloc[-1]:
                    buy_score += 1
                elif data['Close'].iloc[-1] < data['VWAP'].iloc[-1]:
                    sell_score += 1

        if ('Volume' in data.columns and data['Volume'].iloc[-1] is not None and 
            'Avg_Volume' in data.columns and data['Avg_Volume'].iloc[-1] is not None and len(data['Volume'].dropna()) >= 2):
            volume_ratio = data['Volume'].iloc[-1] / data['Avg_Volume'].iloc[-1]
            if isinstance(volume_ratio, (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-2], (int, float, np.integer, np.floating)):
                if volume_ratio > 1.5 and data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                    buy_score += 2
                elif volume_ratio > 1.5 and data['Close'].iloc[-1] < data['Close'].iloc[-2]:
                    sell_score += 2
                elif volume_ratio < 0.5:
                    sell_score += 1

        if 'Volume_Spike' in data.columns and data['Volume_Spike'].iloc[-1] is not None and len(data['Volume_Spike'].dropna()) >= 1:
            if data['Volume_Spike'].iloc[-1] and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-2], (int, float, np.integer, np.floating)):
                if data['Close'].iloc[-1] > data['Close'].iloc[-2]:
                    buy_score += 1
                else:
                    sell_score += 1

        if 'Divergence' in data.columns and data['Divergence'].iloc[-1] is not None:
            if data['Divergence'].iloc[-1] == "Bullish Divergence":
                buy_score += 1
            elif data['Divergence'].iloc[-1] == "Bearish Divergence":
                sell_score += 1

        if 'Ichimoku_Span_A' in data.columns and 'Ichimoku_Span_B' in data.columns and data['Close'].iloc[-1] is not None and len(data['Ichimoku_Span_A'].dropna()) >= 1:
            if (isinstance(data['Ichimoku_Span_A'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Ichimoku_Span_B'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating))):
                if data['Close'].iloc[-1] > max(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
                    buy_score += 1
                    recommendations["Ichimoku_Trend"] = "Buy"
                elif data['Close'].iloc[-1] < min(data['Ichimoku_Span_A'].iloc[-1], data['Ichimoku_Span_B'].iloc[-1]):
                    sell_score += 1
                    recommendations["Ichimoku_Trend"] = "Sell"

        if 'CMF' in data.columns and data['CMF'].iloc[-1] is not None and len(data['CMF'].dropna()) >= 1:
            if isinstance(data['CMF'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['CMF'].iloc[-1] > 0:
                    buy_score += 1
                elif data['CMF'].iloc[-1] < 0:
                    sell_score += 1

        if 'Donchian_Upper' in data.columns and 'Donchian_Lower' in data.columns and data['Close'].iloc[-1] is not None and len(data['Donchian_Upper'].dropna()) >= 1:
            if (isinstance(data['Donchian_Upper'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Donchian_Lower'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating))):
                if data['Close'].iloc[-1] > data['Donchian_Upper'].iloc[-1]:
                    buy_score += 1
                    recommendations["Breakout"] = "Buy"
                elif data['Close'].iloc[-1] < data['Donchian_Lower'].iloc[-1]:
                    sell_score += 1
                    recommendations["Breakout"] = "Sell"

        if 'RSI' in data.columns and 'Lower_Band' in data.columns and 'Upper_Band' in data.columns and data['Close'].iloc[-1] is not None and len(data['RSI'].dropna()) >= 1:
            if (isinstance(data['RSI'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Lower_Band'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Upper_Band'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating))):
                if data['RSI'].iloc[-1] < 30 and data['Close'].iloc[-1] >= data['Lower_Band'].iloc[-1]:
                    buy_score += 2
                    recommendations["Mean_Reversion"] = "Buy"
                elif data['RSI'].iloc[-1] > 70 and data['Close'].iloc[-1] >= data['Upper_Band'].iloc[-1]:
                    sell_score += 2
                    recommendations["Mean_Reversion"] = "Sell"

        if 'Ichimoku_Tenkan' in data.columns and 'Ichimoku_Kijun' in data.columns and data['Close'].iloc[-1] is not None and len(data['Ichimoku_Tenkan'].dropna()) >= 1:
            if (isinstance(data['Ichimoku_Tenkan'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Ichimoku_Kijun'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Ichimoku_Span_A'].iloc[-1], (int, float, np.integer, np.floating))):
                if (data['Ichimoku_Tenkan'].iloc[-1] > data['Ichimoku_Kijun'].iloc[-1] and
                    data['Close'].iloc[-1] > data['Ichimoku_Span_A'].iloc[-1]):
                    buy_score += 1
                    recommendations["Ichimoku_Trend"] = "Strong Buy"
                elif (data['Ichimoku_Tenkan'].iloc[-1] < data['Ichimoku_Kijun'].iloc[-1] and
                      data['Close'].iloc[-1] < data['Ichimoku_Span_B'].iloc[-1]):
                    sell_score += 1
                    recommendations["Ichimoku_Trend"] = "Strong Sell"

        if ('Keltner_Upper' in data.columns and 'Keltner_Lower' in data.columns and 
            data['Close'].iloc[-1] is not None and len(data['Keltner_Upper'].dropna()) >= 1):
            if (isinstance(data['Keltner_Upper'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Keltner_Lower'].iloc[-1], (int, float, np.integer, np.floating)) and 
                isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating))):
                if data['Close'].iloc[-1] < data['Keltner_Lower'].iloc[-1]:
                    buy_score += 1
                elif data['Close'].iloc[-1] > data['Keltner_Upper'].iloc[-1]:
                    sell_score += 1

        if 'TRIX' in data.columns and data['TRIX'].iloc[-1] is not None and len(data['TRIX'].dropna()) >= 2:
            if isinstance(data['TRIX'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['TRIX'].iloc[-2], (int, float, np.integer, np.floating)):
                if data['TRIX'].iloc[-1] > 0 and data['TRIX'].iloc[-1] > data['TRIX'].iloc[-2]:
                    buy_score += 1
                elif data['TRIX'].iloc[-1] < 0 and data['TRIX'].iloc[-1] < data['TRIX'].iloc[-2]:
                    sell_score += 1

        if 'Ultimate_Osc' in data.columns and data['Ultimate_Osc'].iloc[-1] is not None and len(data['Ultimate_Osc'].dropna()) >= 1:
            if isinstance(data['Ultimate_Osc'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['Ultimate_Osc'].iloc[-1] < 30:
                    buy_score += 1
                elif data['Ultimate_Osc'].iloc[-1] > 70:
                    sell_score += 1

        if 'CMO' in data.columns and data['CMO'].iloc[-1] is not None and len(data['CMO'].dropna()) >= 1:
            if isinstance(data['CMO'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['CMO'].iloc[-1] < -50:
                    buy_score += 1
                elif data['CMO'].iloc[-1] > 50:
                    sell_score += 1

        if 'VPT' in data.columns and data['VPT'].iloc[-1] is not None and len(data['VPT'].dropna()) >= 2:
            if isinstance(data['VPT'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['VPT'].iloc[-2], (int, float, np.integer, np.floating)):
                if data['VPT'].iloc[-1] > data['VPT'].iloc[-2]:
                    buy_score += 1
                elif data['VPT'].iloc[-1] < data['VPT'].iloc[-2]:
                    sell_score += 1

        if ('Fib_23.6' in data.columns and 'Fib_38.2' in data.columns and 
            data['Close'].iloc[-1] is not None and len(data['Fib_23.6'].dropna()) >= 1):
            current_price = data['Close'].iloc[-1]
            fib_levels = [data['Fib_23.6'].iloc[-1], data['Fib_38.2'].iloc[-1], 
                          data['Fib_50.0'].iloc[-1], data['Fib_61.8'].iloc[-1]]
            for level in fib_levels:
                if isinstance(level, (int, float, np.integer, np.floating)) and abs(current_price - level) / current_price < 0.01:
                    if current_price > level:
                        buy_score += 1
                    else:
                        sell_score += 1

        if ('Parabolic_SAR' in data.columns and data['Parabolic_SAR'].iloc[-1] is not None and 
            data['Close'].iloc[-1] is not None and len(data['Parabolic_SAR'].dropna()) >= 1):
            if isinstance(data['Parabolic_SAR'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['Close'].iloc[-1], (int, float, np.integer, np.floating)):
                if data['Close'].iloc[-1] > data['Parabolic_SAR'].iloc[-1]:
                    buy_score += 1
                elif data['Close'].iloc[-1] < data['Parabolic_SAR'].iloc[-1]:
                    sell_score += 1

        if ('OBV' in data.columns and data['OBV'].iloc[-1] is not None and 
            data['OBV'].iloc[-2] is not None and len(data['OBV'].dropna()) >= 2):
            if isinstance(data['OBV'].iloc[-1], (int, float, np.integer, np.floating)) and isinstance(data['OBV'].iloc[-2], (int, float, np.integer, np.floating)):
                if data['OBV'].iloc[-1] > data['OBV'].iloc[-2]:
                    buy_score += 1
                elif data['OBV'].iloc[-1] < data['OBV'].iloc[-2]:
                    sell_score += 1

        if symbol:
            fundamentals = fetch_fundamentals(symbol)
            if fundamentals['P/E'] < 15 and fundamentals['EPS'] > 0:
                buy_score += 2
            elif fundamentals['P/E'] > 30 or fundamentals['EPS'] < 0:
                sell_score += 1
            if fundamentals['RevenueGrowth'] > 0.1:
                buy_score += 1
            elif fundamentals['RevenueGrowth'] < 0:
                sell_score += 0.5

        net_score = buy_score - sell_score
        if buy_score > sell_score and buy_score >= 4:
            recommendations["Intraday"] = "Strong Buy"
            recommendations["Swing"] = "Buy" if buy_score >= 3 else "Hold"
            recommendations["Short-Term"] = "Buy" if buy_score >= 2 else "Hold"
            recommendations["Long-Term"] = "Buy" if buy_score >= 1 else "Hold"
        elif sell_score > buy_score and sell_score >= 4:
            recommendations["Intraday"] = "Strong Sell"
            recommendations["Swing"] = "Sell" if sell_score >= 3 else "Hold"
            recommendations["Short-Term"] = "Sell" if sell_score >= 2 else "Hold"
            recommendations["Long-Term"] = "Sell" if sell_score >= 1 else "Hold"
        elif net_score > 0:
            recommendations["Intraday"] = "Buy" if net_score >= 3 else "Hold"
            recommendations["Swing"] = "Buy" if net_score >= 2 else "Hold"
            recommendations["Short-Term"] = "Buy" if net_score >= 1 else "Hold"
            recommendations["Long-Term"] = "Hold"
        elif net_score < 0:
            recommendations["Intraday"] = "Sell" if net_score <= -3 else "Hold"
            recommendations["Swing"] = "Sell" if net_score <= -2 else "Hold"
            recommendations["Short-Term"] = "Sell" if net_score <= -1 else "Hold"
            recommendations["Long-Term"] = "Hold"

        recommendations["Buy At"] = calculate_buy_at(data)
        recommendations["Stop Loss"] = calculate_stop_loss(data)
        recommendations["Target"] = calculate_target(data)

        recommendations["Score"] = min(max(buy_score - sell_score, -7), 7)
    except Exception as e:
        st.warning(f"⚠️ Error generating recommendations: {str(e)}")
    return recommendations

@st.cache_data(ttl=3600)  # Cache results for 1 hour to avoid repeated API hits
def get_top_sectors_cached(rate_limit_delay=2, stocks_per_sector=2):
    sector_scores = {}
    for sector, stocks in SECTORS.items():
        total_score = 0
        count = 0
        for symbol in stocks[:stocks_per_sector]:  # Only analyze top N stocks per sector
            data = fetch_stock_data_cached(symbol)
            if data.empty:
                continue
            data = analyze_stock(data)
            rec = generate_recommendations(data, symbol)
            total_score += rec.get("Score", 0)
            count += 1
            rec = generate_recommendations(data, symbol)
            total_score += rec.get("Score", 0)
            count += 1
            # Rate limiting is handled globally now
        avg_score = total_score / count if count else 0
        sector_scores[sector] = avg_score
        avg_score = total_score / count if count else 0
        sector_scores[sector] = avg_score
        # Removed redundant sleep
    return sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)[:3]

@st.cache_data
@st.cache_data(ttl=3600)
def backtest_stock(data, symbol, strategy="Swing", _data_hash=None):
    results = {
        "total_return": 0,
        "annual_return": 0,
        "sharpe_ratio": 0,
        "max_drawdown": 0,
        "trades": 0,
        "win_rate": 0,
        "buy_signals": [],
        "sell_signals": [],
        "trade_details": []
    }
    recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
    
    position = None
    entry_price = 0
    entry_date = None
    trades = []
    returns = []
    
    for i in range(1, len(data)):
        sliced_data = data.iloc[:i+1]
        if recommendation_mode == "Adaptive":
            rec = adaptive_recommendation(sliced_data)
            signal = rec["Recommendation"]
        else:
            rec = generate_recommendations(sliced_data, symbol)
            signal = rec[strategy] if strategy in rec else "Hold"
        
        current_price = data['Close'].iloc[i]
        current_date = data.index[i]
        
        if signal == "Buy" and position is None:
            position = "Long"
            entry_price = current_price
            entry_date = current_date
            results["buy_signals"].append((current_date, current_price))
        
        elif signal == "Sell" and position == "Long":
            position = None
            profit = current_price - entry_price
            returns.append(profit / entry_price)
            trades.append({
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_date": current_date,
                "exit_price": current_price,
                "profit": profit
            })
            results["sell_signals"].append((current_date, current_price))
            entry_price = 0
            entry_date = None
    
    if trades:
        results["trade_details"] = trades
        results["trades"] = len(trades)
        results["total_return"] = sum([t["profit"]/t["entry_price"] for t in trades]) * 100
        results["win_rate"] = len([t for t in trades if t["profit"] > 0]) / len(trades) * 100
        if returns:
            results["annual_return"] = (np.mean(returns) * 252) * 100
            results["sharpe_ratio"] = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
        drawdowns = [t["profit"]/t["entry_price"] for t in trades]
        results["max_drawdown"] = min(drawdowns, default=0) * 100 if drawdowns else 0
    
    return results
    
def init_database():
    conn = sqlite3.connect('stock_picks.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS daily_picks (
            date TEXT,
            symbol TEXT,
            score REAL,
            current_price REAL,
            buy_at REAL,
            stop_loss REAL,
            target REAL,
            intraday TEXT,
            swing TEXT,
            short_term TEXT,
            long_term TEXT,
            mean_reversion TEXT,
            breakout TEXT,
            ichimoku_trend TEXT,
            recommendation TEXT,
            regime TEXT,
            position_size REAL,
            trailing_stop REAL,
            reason TEXT,
            pick_type TEXT,
            PRIMARY KEY (date, symbol)
        )
    ''')
    conn.close()

def insert_top_picks(results_df, pick_type="daily"):
    conn = sqlite3.connect('stock_picks.db')
    cursor = conn.cursor()
    
    data_to_insert = []
    for _, row in results_df.head(5).iterrows():
        data_to_insert.append((
            datetime.now().strftime('%Y-%m-%d'),
            row.get('Symbol'),
            row.get('Score', 0),
            row.get('Current Price'),
            row.get('Buy At'),
            row.get('Stop Loss'),
            row.get('Target'),
            row.get('Intraday'),
            row.get('Swing'),
            row.get('Short-Term'),
            row.get('Long-Term'),
            row.get('Mean_Reversion'),
            row.get('Breakout'),
            row.get('Ichimoku_Trend'),
            row.get('Recommendation'),
            row.get('Regime'),
            row.get('Position Size'),
            row.get('Trailing Stop'),
            row.get('Reason'),
            pick_type
        ))

    cursor.executemany('''
        INSERT OR IGNORE INTO daily_picks (
            date, symbol, score, current_price, buy_at, stop_loss, target,
            intraday, swing, short_term, long_term, mean_reversion, breakout,
            ichimoku_trend, recommendation, regime, position_size, trailing_stop,
            reason, pick_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', data_to_insert)
    
    conn.commit()
    conn.close()

def analyze_batch(stock_batch, patience="high"):
    """
    Analyzes a batch of stocks in parallel.
    Returns a list of results (dictionaries) for ALL processed stocks, including failures.
    """
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(analyze_stock_parallel, symbol, patience): symbol for symbol in stock_batch}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Fallback for unexpected crashe within the thread handling itself
                results.append({
                    "Symbol": symbol,
                    "Status": "Critical Error",
                    "Error": str(e),
                    "Score": 0,
                    "Recommendation": "N/A"
                })
                logging.error(f"Critical error processing {symbol}: {str(e)}")
    return results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_stock_parallel(symbol, patience="high"):
    """
    Analyzes a single stock.
    Returns a dictionary with 'Status' (Success, No Data, Error) and detailed analysis or error info.
    """
    try:
        logging.info(f"Starting analysis for {symbol}")
        data = fetch_stock_data_cached(symbol)
        
        if data.empty or len(data) < 50:
            logging.warning(f"No sufficient data for {symbol}: {len(data) if data is not None else 0} rows")
            return {
                "Symbol": symbol,
                "Status": "No Data",
                "Error": "Insufficient/Empty Data",
                "Score": 0,
                "Recommendation": "N/A",
                "Current Price": 0
            }
        
        data = analyze_stock(data)
        recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
        logging.info(f"Analyzing {symbol} in {recommendation_mode} mode")
        
        if recommendation_mode == "Adaptive":
            rec = adaptive_recommendation(data, symbol)
            # Override Buy At if patience is low (e.g. Intraday)
            if patience == "low" and rec.get("Current Price"):
                rec["Buy At"], rec["Entry Type"] = calculate_buy_at(data, patience="low")
                # RECALCULATE Risk Management based on new Entry!
                if rec["Buy At"]:
                    entry_type = rec["Entry Type"]
                    # Adjust SL Multiplier based on Entry Type (User Request)
                    sl_mult = 2.0 if entry_type == "Breakout" else 1.5
                    rec["Stop Loss"] = calculate_stop_loss(data, atr_multiplier=sl_mult, entry_price=rec["Buy At"])
                    rec["Target"] = calculate_target(data, risk_reward_ratio=2.5, entry_price=rec["Buy At"]) # Realistic 2.5R
            else:
                rec["Entry Type"] = "Standard"
            
            if not rec or not rec.get('Recommendation'):
                return {
                    "Symbol": symbol,
                    "Status": "Analysis Failed",
                    "Error": "Adaptive Recommendation returned empty",
                    "Score": 0,
                    "Recommendation": "N/A"
                }

            return {
                "Symbol": symbol,
                "Status": "Success",
                "Current Price": rec.get("Current Price"),
                "Buy At": rec.get("Buy At"),
                "Stop Loss": rec.get("Stop Loss"),
                "Target": rec.get("Target"),
                "Recommendation": rec.get("Recommendation", "Hold"),
                "Score": rec.get("Score", 0),
                "Regime": rec.get("Regime"),
                "Position Size": rec.get("Position Size"),
                "Trailing Stop": rec.get("Trailing Stop"),
                "Entry Type": rec.get("Entry Type", "Standard"),
                "Reason": rec.get("Reason"),
                "Intraday": None,
                "Swing": None,
                "Short-Term": None,
                "Long-Term": None,
                "Mean_Reversion": None,
                "Breakout": None,
                "Ichimoku_Trend": None
            }
        else:
            rec = generate_recommendations(data, symbol)
            # Override Buy At if patience is low
            if patience == "low" and rec.get("Current Price"):
                 rec["Buy At"], rec["Entry Type"] = calculate_buy_at(data, patience="low")
                 # RECALCULATE Risk Management based on new Entry!
                 if rec["Buy At"]:
                    entry_type = rec.get("Entry Type", "Standard")
                    sl_mult = 2.0 if entry_type == "Breakout" else 1.5
                    rec["Stop Loss"] = calculate_stop_loss(data, atr_multiplier=sl_mult, entry_price=rec["Buy At"])
                    rec["Target"] = calculate_target(data, risk_reward_ratio=2.5, entry_price=rec["Buy At"])
            else:
                 rec["Entry Type"] = "Standard"

            if not rec or not rec.get('Intraday'):
                return {
                    "Symbol": symbol,
                    "Status": "Analysis Failed",
                    "Error": "Standard Recommendation returned empty",
                    "Score": 0,
                    "Recommendation": "N/A"
                }

            return {
                "Symbol": symbol,
                "Status": "Success",
                "Current Price": rec.get("Current Price"),
                "Buy At": rec.get("Buy At"),
                "Stop Loss": rec.get("Stop Loss"),
                "Target": rec.get("Target"),
                "Intraday": rec.get("Intraday", "Hold"),
                "Swing": rec.get("Swing", "Hold"),
                "Short-Term": rec.get("Short-Term", "Hold"),
                "Long-Term": rec.get("Long-Term", "Hold"),
                "Mean_Reversion": rec.get("Mean_Reversion", "Hold"),
                "Breakout": rec.get("Breakout", "Hold"),
                "Ichimoku_Trend": rec.get("Ichimoku_Trend", "Hold"),
                "Score": rec.get("Score", 0),
                "Entry Type": rec.get("Entry Type", "Standard"),
                "Recommendation": None,
                "Regime": None,
                "Position Size": None,
                "Trailing Stop": None,
                "Reason": None
            }
    except Exception as e:
        error_msg = f"Error in analyze_stock_parallel for {symbol}: {str(e)}"
        logging.error(error_msg)
        return {
            "Symbol": symbol,
            "Status": "Error",
            "Error": str(e),
            "Score": 0,
            "Recommendation": "N/A"
        }

def analyze_all_stocks(stock_list, batch_size=10, progress_callback=None):
    results = []
    # No need to calculate total_batches for the loop logic itself, just for progress
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list[i:i + batch_size]
        batch_results = analyze_batch(batch)
        results.extend(batch_results)
        if progress_callback:
            progress_callback((i + len(batch)) / len(stock_list))
        # Removed redundant sleep
    
    results_df = pd.DataFrame(results)
    if results_df.empty:
        st.warning("⚠️ No valid stock data retrieved.")
        return pd.DataFrame(), pd.DataFrame() # Return empty pair
    
    # Fill missing columns for consistent structure
    expected_cols = ["Score", "Current Price", "Recommendation", "Intraday", "Status", "Error"]
    for col in expected_cols:
         if col not in results_df.columns:
             results_df[col] = None

    # Filter for Top Picks (Success only)
    success_df = results_df[results_df["Status"] == "Success"].copy()

    # Sort logic for Top Picks
    recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
    if recommendation_mode == "Adaptive":
        top_picks_df = success_df[success_df["Recommendation"].str.contains("Buy|Sell", na=False)]
    else:
        # For standard, maybe just default sorts
        top_picks_df = success_df

    top_picks_df = top_picks_df.sort_values(by="Score", ascending=False).head(5)
    
    return top_picks_df, results_df

def calculate_sector_performance():
    """
    Calculates the real-time performance of each sector based on constituent stocks.
    Returns a DataFrame sorted by % Change.
    """
    try:
        sector_performance = []
        
        # Flatten all symbols to fetch data in one batch
        all_symbols = []
        for sector, symbols in SECTORS.items():
            all_symbols.extend(symbols)
        all_symbols = list(set(all_symbols))
        
        # Helper to fetch change
        live_data = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {executor.submit(fetch_stock_data, symbol, "5d"): symbol for symbol in all_symbols} # Fetch 5d for safety
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if not data.empty and len(data) >= 2:
                        current = data['Close'].iloc[-1]
                        prev_close = data['Close'].iloc[-2]
                        change = ((current - prev_close) / prev_close) * 100
                        live_data[symbol] = change
                except:
                    pass

        # Aggregate
        for sector, symbols in SECTORS.items():
            sector_changes = []
            for symbol in symbols:
                if symbol in live_data:
                    sector_changes.append(live_data[symbol])
            
            if sector_changes:
                avg_change = sum(sector_changes) / len(sector_changes)
                # Sentiment Logic
                if avg_change > 0.5: sentiment = "🟢 Strong"
                elif avg_change > 0: sentiment = "🟢 Bullish"
                elif avg_change < -0.5: sentiment = "🔴 Weak"
                else: sentiment = "🔴 Bearish"
                
                sector_performance.append({
                    "Sector": sector,
                    "% Change": round(avg_change, 2),
                    "Sentiment": sentiment
                })
        
        if not sector_performance:
            return pd.DataFrame()
            
        df = pd.DataFrame(sector_performance)
        return df.sort_values(by="% Change", ascending=False)
    except Exception as e:
        logging.error(f"Sector Perf Error: {e}")
        return pd.DataFrame()


def calculate_sector_performance():
    """
    Calculates the real-time performance of each sector based on constituent stocks.
    Returns a DataFrame sorted by % Change.
    """
    try:
        sector_performance = []
        
        # Flatten all symbols to fetch data in one batch
        all_symbols = []
        for sector, symbols in SECTORS.items():
            all_symbols.extend(symbols)
        all_symbols = list(set(all_symbols))
        
        # Helper to fetch change
        live_data = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {executor.submit(fetch_stock_data_cached, symbol, "5d"): symbol for symbol in all_symbols} # Fetch 5d for safety
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if not data.empty and len(data) >= 2:
                        current = data['Close'].iloc[-1]
                        prev_close = data['Close'].iloc[-2]
                        change = ((current - prev_close) / prev_close) * 100
                        live_data[symbol] = change
                except:
                    pass

        # Aggregate
        for sector, symbols in SECTORS.items():
            sector_changes = []
            for symbol in symbols:
                if symbol in live_data:
                    sector_changes.append(live_data[symbol])
            
            if sector_changes:
                avg_change = sum(sector_changes) / len(sector_changes)
                # Sentiment Logic
                if avg_change > 0.5: sentiment = "🟢 Strong"
                elif avg_change > 0: sentiment = "🟢 Bullish"
                elif avg_change < -0.5: sentiment = "🔴 Weak"
                else: sentiment = "🔴 Bearish"
                
                sector_performance.append({
                    "Sector": sector,
                    "% Change": round(avg_change, 2),
                    "Sentiment": sentiment
                })
        
        if not sector_performance:
            return pd.DataFrame()
            
        df = pd.DataFrame(sector_performance)
        return df.sort_values(by="% Change", ascending=False)
    except Exception as e:
        logging.error(f"Sector Perf Error: {e}")
        return pd.DataFrame()


def analyze_intraday_stocks(stock_list, batch_size=10, progress_callback=None):
    results = []
    total_batches = (len(stock_list) // batch_size) + (1 if len(stock_list) % batch_size != 0 else 0)
    for i in range(0, len(stock_list), batch_size):
        batch = stock_list[i:i + batch_size]
        # Pass patience="low" for Intraday scans
        batch_results = analyze_batch(batch, patience="low")
        results.extend([r for r in batch_results if r is not None])
        if progress_callback:
            progress_callback((i + len(batch)) / len(stock_list))
        # Removed redundant sleep
    
    results_df = pd.DataFrame(results)
    if results_df.empty:
        return pd.DataFrame()
    if "Score" not in results_df.columns:
        results_df["Score"] = 0
    if "Current Price" not in results_df.columns:
        results_df["Current Price"] = None
        
    # Filter out invalid 'Buy At' entries (e.g. from Choppy Markets)
    if "Buy At" in results_df.columns:
        results_df = results_df.dropna(subset=["Buy At"])
        
    recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
    if recommendation_mode == "Adaptive":
        results_df = results_df[results_df["Recommendation"].str.contains("Buy", na=False)]
    else:
        results_df = results_df[results_df["Intraday"].str.contains("Buy", na=False)]
    return results_df.sort_values(by="Score", ascending=False).head(5)

def colored_recommendation(recommendation):
    if recommendation is None or not isinstance(recommendation, str):
        return "⚪ N/A"
    if "Buy" in recommendation:
        return f"🟢 {recommendation}"
    elif "Sell" in recommendation:
        return f"🔴 {recommendation}"
    else:
        return f"⚪ {recommendation}"

def update_progress(progress_bar, loading_text, progress_value, loading_messages):
    progress_bar.progress(progress_value)
    loading_message = next(loading_messages)
    dots = "." * int((progress_value * 10) % 4)
    loading_text.text(f"{loading_message}{dots}")

def display_dashboard(symbol=None, data=None, recommendations=None):
    # Initialize session state
    if 'selected_sectors' not in st.session_state:
        st.session_state.selected_sectors = ["Bank"]
    if 'symbol' not in st.session_state:
        st.session_state.symbol = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'backtest_results_swing' not in st.session_state:
        st.session_state.backtest_results_swing = None
    if 'backtest_results_intraday' not in st.session_state:
        st.session_state.backtest_results_intraday = None
    if 'recommendation_mode' not in st.session_state:
        st.session_state.recommendation_mode = "Standard"

    # Update session state if new data is provided
    if symbol and data is not None and recommendations is not None:
        st.session_state.symbol = symbol
        st.session_state.data = data
        st.session_state.recommendations = recommendations

    st.title("📊 StockGenie Pro - NSE Analysis")
    st.subheader(f"📅 Analysis for {datetime.now().strftime('%d %b %Y')}")

    # Sector selection
    sector_options = ["All"] + list(SECTORS.keys())
    st.session_state.selected_sectors = st.sidebar.multiselect(
        "Select Sectors",
        options=sector_options,
        default=st.session_state.selected_sectors,
        help="Choose one or more sectors to analyze. Select 'All' to include all sectors."
    )

    if "All" in st.session_state.selected_sectors:
        selected_stocks = list(set([stock for sector in SECTORS.values() for stock in sector]))
    else:
        selected_stocks = list(set([stock for sector in st.session_state.selected_sectors for stock in SECTORS.get(sector, [])]))

    if not selected_stocks:
        st.warning("⚠️ No stocks selected. Please choose at least one sector.")
        return

    # Top sectors button
    if st.button("🔎 Analyze Top Performing Sectors"):
        with st.spinner("🔍 Crunching sector data ..."):
            top_sectors = get_top_sectors_cached(rate_limit_delay=2, stocks_per_sector=2)
            st.subheader("🔝 Top 3 Performing Sectors Today")
            for name, score in top_sectors:
                st.markdown(f"- **{name}**: {score:.2f}/7")

    # Daily top picks button
    if st.button("🚀 Generate Daily Top Picks"):
        progress_bar = st.progress(0)
        loading_text = st.empty()
        loading_messages = itertools.cycle([
            "Analyzing trends...", "Fetching data...", "Crunching numbers...",
            "Evaluating indicators...", "Finalizing results..."
        ])
        
        # Unpack the two dataframes
        top_picks_df, full_report_df = analyze_all_stocks(
            selected_stocks,
            batch_size=10,
            progress_callback=lambda x: update_progress(progress_bar, loading_text, x, loading_messages)
        )
        
        # Insert top picks as before
        if not top_picks_df.empty:
            insert_top_picks(top_picks_df, pick_type="daily")
            
        progress_bar.empty()
        loading_text.empty()
        
        # Display Top 5
        if not top_picks_df.empty:
            st.subheader("🏆 Today's Top 5 Stocks")
            for _, row in top_picks_df.iterrows():
                with st.expander(f"{row['Symbol']} - {tooltip('Score', TOOLTIPS['Score'])}: {row['Score']}/7"):
                    current_price = row.get('Current Price', 'N/A')
                    buy_at = row.get('Buy At', 'N/A')
                    stop_loss = row.get('Stop Loss', 'N/A')
                    target = row.get('Target', 'N/A')
                    if st.session_state.recommendation_mode == "Adaptive":
                        st.markdown(f"""
                        {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: ₹{current_price}  
                        Buy At: ₹{buy_at} | Stop Loss: ₹{stop_loss}  
                        Target: ₹{target}  
                        Recommendation: {colored_recommendation(row.get('Recommendation', 'N/A'))}  
                        Regime: {row.get('Regime', 'N/A')}  
                        Position Size (₹): {row.get('Position Size', 'N/A')}  
                        Trailing Stop: ₹{row.get('Trailing Stop', 'N/A')}  
                        Reason: {row.get('Reason', 'N/A')}
                        """)
                    else:
                        st.markdown(f"""
                        {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: ₹{current_price}  
                        Buy At: ₹{buy_at} | Stop Loss: ₹{stop_loss}  
                        Target: ₹{target}  
                        Intraday: {colored_recommendation(row.get('Intraday', 'N/A'))}  
                        Swing: {colored_recommendation(row.get('Swing', 'N/A'))}  
                        Short-Term: {colored_recommendation(row.get('Short-Term', 'N/A'))}  
                        Long-Term: {colored_recommendation(row.get('Long-Term', 'N/A'))}  
                        Mean Reversion: {colored_recommendation(row.get('Mean_Reversion', 'N/A'))}  
                        Breakout: {colored_recommendation(row.get('Breakout', 'N/A'))}  
                        Ichimoku Trend: {colored_recommendation(row.get('Ichimoku_Trend', 'N/A'))}
                        """)
        else:
            st.warning("⚠️ No top picks available due to data issues.")
            
        # --- STRATEGY EXECUTION DETAILS (New Section) ---
        if not full_report_df.empty:
            with st.expander("📊 Strategy Execution Details", expanded=True):
                total = len(full_report_df)
                success = len(full_report_df[full_report_df['Status'] == 'Success'])
                failed = len(full_report_df[full_report_df['Status'] != 'Success'])
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Processed", total)
                c2.metric("Successful Runs", success)
                c3.metric("Failures/No Data", failed)
                
                # Show failures if any
                if failed > 0:
                    st.error(f"Failed to process {failed} stocks.")
                    st.dataframe(full_report_df[full_report_df['Status'] != 'Success'][['Symbol', 'Status', 'Error']])
                
                st.download_button(
                    label="📥 Download Full Strategy Report (CSV)",
                    data=full_report_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"strategy_report_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                )


    # Intraday top picks button
    if st.button("⚡ Generate Intraday Top 5 Picks"):
        # [NEW] Sector Performance Overview (Bloomberg Style)
        with st.expander("📊 Market Sector Performance (Live)", expanded=False):
            with st.spinner("Analyzing Sectors..."):
                sector_df = calculate_sector_performance()
                if not sector_df.empty:
                    # Color Styling for DataFrame
                    def color_survived(val):
                        color = '#90EE90' if val > 0 else '#FFB6C1'
                        return f'background-color: {color}; color: black'
                    
                    st.dataframe(
                        sector_df.style.applymap(color_survived, subset=['% Change']),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("Sector data loading...")

        progress_bar = st.progress(0)
        loading_text = st.empty()
        loading_messages = itertools.cycle([
            "Scanning intraday trends...", "Detecting buy signals...", "Calculating stop-loss levels...",
            "Optimizing targets...", "Finalizing top picks..."
        ])
        intraday_results = analyze_intraday_stocks(
            selected_stocks,
            batch_size=10,
            progress_callback=lambda x: update_progress(progress_bar, loading_text, x, loading_messages)
        )
        insert_top_picks(intraday_results, pick_type="intraday")
        progress_bar.empty()
        loading_text.empty()
        if not intraday_results.empty:
            st.subheader("🏆 Top 5 Intraday Stocks (⚡ Fast Exit)")
            for _, row in intraday_results.iterrows():
                with st.expander(f"{row['Symbol']} - {tooltip('Score', TOOLTIPS['Score'])}: {row['Score']}/7"):
                    current_price = row.get('Current Price', 'N/A')
                    buy_at = row.get('Buy At', 'N/A')
                    stop_loss = row.get('Stop Loss', 'N/A')
                    target = row.get('Target', 'N/A')
                    if st.session_state.recommendation_mode == "Adaptive":
                        buy_label = "Buy At"
                        buy_icon = ""
                        entry_type = row.get('Entry Type', 'Standard')
                        if entry_type == "Breakout":
                             buy_label = "Buy Above (Breakout)"
                             buy_icon = "🟢"
                        elif entry_type == "Pullback":
                             buy_label = "Buy On Pullback"
                             buy_icon = "🔵"
                             
                        st.markdown(f"""
                        {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: ₹{current_price}  
                        {buy_icon} {buy_label}: ₹{buy_at} | Stop Loss: ₹{stop_loss}  
                        Target: ₹{target}  
                        Recommendation: {colored_recommendation(row.get('Recommendation', 'N/A'))}  
                        Regime: {row.get('Regime', 'N/A')}  
                        Position Size (₹): {row.get('Position Size', 'N/A')}  
                        Trailing Stop: ₹{row.get('Trailing Stop', 'N/A')}  
                        Reason: {row.get('Reason', 'N/A')}
                        """)
                    else:
                        buy_label = "Buy At"
                        buy_icon = ""
                        entry_type = row.get('Entry Type', 'Standard')
                        if entry_type == "Breakout":
                             buy_label = "Buy Above (Breakout)"
                             buy_icon = "🟢"
                        elif entry_type == "Pullback":
                             buy_label = "Buy On Pullback"
                             buy_icon = "🔵"

                        st.markdown(f"""
                        {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: ₹{current_price}  
                        {buy_icon} {buy_label}: ₹{buy_at} | Stop Loss: ₹{stop_loss}  
                        Target: ₹{target}  
                        Intraday: {colored_recommendation(row.get('Intraday', 'N/A'))}
                        """)
        else:
            st.warning("⚠️ No intraday picks available due to data issues.")

    # Historical picks button
    if st.button("📜 View Historical Picks"):
        conn = sqlite3.connect('stock_picks.db')
        history_df = pd.read_sql_query("SELECT * FROM daily_picks ORDER BY date DESC", conn)
        conn.close()
        if not history_df.empty:
            st.subheader("📜 Historical Top Picks")
            all_dates = sorted(history_df['date'].unique(), reverse=True)
            date_filter = st.selectbox("Filter by Date", ["All"] + all_dates)
            pick_type_filter = st.selectbox("Filter by Pick Type", ["All", "daily", "intraday"])
            filtered_df = history_df.copy()
            if pick_type_filter != "All":
                filtered_df = filtered_df[filtered_df['pick_type'] == pick_type_filter]
            if date_filter != "All":
                filtered_df = filtered_df[filtered_df['date'] == date_filter]
            st.dataframe(filtered_df)
        else:
            st.warning("⚠️ No historical data available.")

    # Display stock analysis if symbol is available
    if st.session_state.symbol and st.session_state.data is not None and st.session_state.recommendations is not None:
        symbol = st.session_state.symbol
        data = st.session_state.data
        recommendations = st.session_state.recommendations

        st.header(f"📋 {symbol.split('-')[0]} Analysis")
        
        # --- TABBED INTERFACE ---
        tab_overview, tab_technical, tab_backtest = st.tabs(["Overview", "Technical Analysis", "Backtesting"])

        # 1. OVERVIEW TAB
        with tab_overview:
            st.subheader("✨ Key Metrics & Recommendations")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                current_price = recommendations.get('Current Price', 'N/A')
                st.metric(tooltip("Current Price", TOOLTIPS['RSI']), f"₹{current_price}")
            with col2:
                buy_at = recommendations.get('Buy At', 'N/A')
                entry_type = recommendations.get('Entry Type', 'Standard')
                label = "Buy At"
                if entry_type == "Breakout":
                    label = "🟢 Buy Above"
                elif entry_type == "Pullback":
                    label = "🔵 Buy Pullback"
                elif entry_type == "Choppy":
                    label = "⚠️ No Trade"
                    buy_at = "Choppy"
                
                st.metric(label, f"₹{buy_at}" if isinstance(buy_at, (int, float)) else buy_at)
            with col3:
                stop_loss = recommendations.get('Stop Loss', 'N/A')
                st.metric(tooltip("Stop Loss", TOOLTIPS['Stop Loss']), f"₹{stop_loss}")
            with col4:
                target = recommendations.get('Target', 'N/A')
                st.metric("Target", f"₹{target}")
            with col5:
                regime = recommendations.get('Regime', 'N/A') if st.session_state.recommendation_mode == "Adaptive" else 'N/A'
                st.metric("Market Regime", regime)

            st.markdown("---")
            st.subheader("📈 Trading Signals")
            if st.session_state.recommendation_mode == "Adaptive":
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Recommendation**: {colored_recommendation(recommendations.get('Recommendation', 'N/A'))}")
                    st.write(f"**Reason**: {recommendations.get('Reason', 'N/A')}")
                with col2:
                    st.write(f"**{tooltip('Score', TOOLTIPS['Score'])}**: {recommendations.get('Score', 'N/A')}/7")
                    st.write(f"**Position Size (₹)**: {recommendations.get('Position Size', 'N/A')}")
                with col3:
                    st.write(f"**Trailing Stop**: ₹{recommendations.get('Trailing Stop', 'N/A')}")
                    st.write(f"**Volatility**: {assess_risk(data)}")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Intraday**: {colored_recommendation(recommendations.get('Intraday', 'N/A'))}")
                    st.write(f"**Swing**: {colored_recommendation(recommendations.get('Swing', 'N/A'))}")
                with col2:
                    st.write(f"**Short-Term**: {colored_recommendation(recommendations.get('Short-Term', 'N/A'))}")
                    st.write(f"**Long-Term**: {colored_recommendation(recommendations.get('Long-Term', 'N/A'))}")
                with col3:
                    st.write(f"**Mean Reversion**: {colored_recommendation(recommendations.get('Mean_Reversion', 'N/A'))}")
                    st.write(f"**Breakout**: {colored_recommendation(recommendations.get('Breakout', 'N/A'))}")
                    st.write(f"**Ichimoku Trend**: {colored_recommendation(recommendations.get('Ichimoku_Trend', 'N/A'))}")
                st.write(f"**{tooltip('Score', TOOLTIPS['Score'])}**: {recommendations.get('Score', 'N/A')}/7")
                st.write(f"**Volatility**: {assess_risk(data)}")

            st.markdown("---")
            # Monte Carlo Simulation
            st.subheader("🎲 Monte Carlo Projection (30 Days)")
            simulations = monte_carlo_simulation(data)
            sim_df = pd.DataFrame(simulations).T
            sim_df.index = [data.index[-1] + timedelta(days=i) for i in range(len(sim_df))]
            fig_sim = px.line(sim_df, title="Price Projections")
            st.plotly_chart(fig_sim, use_container_width=True)

        # 2. TECHNICAL ANALYSIS TAB
        with tab_technical:
            st.subheader("📊 Technical Indicators")
            indicators = [
                ("RSI", data['RSI'].iloc[-1], TOOLTIPS['RSI']),
                ("MACD", data['MACD'].iloc[-1], TOOLTIPS['MACD']),
                ("ATR", data['ATR'].iloc[-1], TOOLTIPS['ATR']),
                ("ADX", data['ADX'].iloc[-1], TOOLTIPS['ADX']),
                ("Bollinger Upper", data['Upper_Band'].iloc[-1], TOOLTIPS['Bollinger']),
                ("Bollinger Lower", data['Lower_Band'].iloc[-1], TOOLTIPS['Bollinger']),
                ("VWAP", data['VWAP'].iloc[-1], TOOLTIPS['VWAP']),
                ("Ichimoku Span A", data['Ichimoku_Span_A'].iloc[-1], TOOLTIPS['Ichimoku']),
                ("CMF", data['CMF'].iloc[-1], TOOLTIPS['CMF']),
            ]
            
            # Display indicators in a cleaner grid (4 cols)
            cols = st.columns(4)
            for i, (name, value, tooltip_text) in enumerate(indicators):
                with cols[i % 4]:
                    val = round(value, 2) if pd.notnull(value) else "N/A"
                    st.metric(tooltip(name, tooltip_text), val)

            st.markdown("---")
            # Price Chart
            st.subheader("📈 Interactive Price Chart")
            fig = px.line(data, x=data.index, y='Close', title=f"{symbol.split('-')[0]} Price Action")
            if 'SMA_50' in data.columns and data['SMA_50'].notnull().any():
                fig.add_scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50', line=dict(color='orange'))
            if 'SMA_200' in data.columns and data['SMA_200'].notnull().any():
                fig.add_scatter(x=data.index, y=data['SMA_200'], mode='lines', name='SMA 200', line=dict(color='red'))
            if 'Upper_Band' in data.columns and data['Upper_Band'].notnull().any():
                fig.add_scatter(x=data.index, y=data['Upper_Band'], mode='lines', name='Bollinger Upper', line=dict(color='green', dash='dash'))
            if 'Lower_Band' in data.columns and data['Lower_Band'].notnull().any():
                fig.add_scatter(x=data.index, y=data['Lower_Band'], mode='lines', name='Bollinger Lower', line=dict(color='green', dash='dash'))
            if 'Ichimoku_Span_A' in data.columns and data['Ichimoku_Span_A'].notnull().any():
                fig.add_scatter(x=data.index, y=data['Ichimoku_Span_A'], mode='lines', name='Ichimoku Span A', line=dict(color='purple'))
            if 'Ichimoku_Span_B' in data.columns and data['Ichimoku_Span_B'].notnull().any():
                fig.add_scatter(x=data.index, y=data['Ichimoku_Span_B'], mode='lines', name='Ichimoku Span B', line=dict(color='purple', dash='dash'))
            st.plotly_chart(fig, use_container_width=True)

            # Sub-charts
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("RSI")
                fig_ind = px.line(data, x=data.index, y='RSI')
                fig_ind.add_hline(y=70, line_dash="dash", line_color="red")
                fig_ind.add_hline(y=30, line_dash="dash", line_color="green")
                st.plotly_chart(fig_ind, use_container_width=True)
            with c2:
                st.subheader("MACD")
                fig_macd = px.line(data, x=data.index, y=['MACD', 'MACD_signal'])
                st.plotly_chart(fig_macd, use_container_width=True)

            # Volume Analysis
            st.subheader("📊 Volume Analysis")
            fig_vol = px.bar(data, x=data.index, y='Volume')
            if 'Volume_Spike' in data.columns:
                spike_data = data[data['Volume_Spike'] == True]
                if not spike_data.empty:
                    fig_vol.add_scatter(x=spike_data.index, y=spike_data['Volume'], mode='markers', name='Volume Spike',
                                       marker=dict(color='red', size=10))
            st.plotly_chart(fig_vol, use_container_width=True)

        # 3. BACKTESTING TAB
        with tab_backtest:
            st.subheader("🧪 Strategy Backtester")
            
            # Backtest form
            with st.form(key="backtest_form"):
                col1, col2 = st.columns(2)
                with col1:
                    swing_button = st.form_submit_button("🔍 Backtest Swing Strategy")
                with col2:
                    intraday_button = st.form_submit_button("🔍 Backtest Intraday Strategy")
                
                if swing_button or intraday_button:
                    strategy = "Swing" if swing_button else "Intraday"
                    with st.spinner(f"Running {strategy} Strategy backtest..."):
                        data_hash = hash(data.to_string())
                        backtest_results = backtest_stock(data, symbol, strategy=strategy, _data_hash=data_hash)
                        if strategy == "Swing":
                            st.session_state.backtest_results_swing = backtest_results
                        else:
                            st.session_state.backtest_results_intraday = backtest_results

            # Backtest results
            for strategy, results_key in [("Swing", "backtest_results_swing"), ("Intraday", "backtest_results_intraday")]:
                backtest_results = st.session_state.get(results_key)
                if backtest_results:
                    st.divider()
                    st.subheader(f"Results: {strategy} Strategy")
                    
                    # Metrics Grid
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Return", f"{backtest_results['total_return']:.2f}%")
                    m2.metric("Win Rate", f"{backtest_results['win_rate']:.2f}%")
                    m3.metric("Trades", backtest_results['trades'])
                    m4.metric("Sharpe Ratio", f"{backtest_results['sharpe_ratio']:.2f}")

                    # Detailed Trades
                    with st.expander("📝 Trade Log"):
                        for trade in backtest_results["trade_details"]:
                            profit = trade.get("profit", 0)
                            color = "green" if profit > 0 else "red"
                            st.markdown(f"**{trade['entry_date'].date()}**: Buy @ {trade['entry_price']:.2f} ➔ Sell @ {trade['exit_price']:.2f} | Profit: :{color}[{profit:.2f}]")

                    # Signal Chart
                    st.subheader("Signal Visualization")
                    fig = px.line(data, x=data.index, y='Close', title=f"Trade Signals on Price")
                    if backtest_results["buy_signals"]:
                        buy_dates, buy_prices = zip(*backtest_results["buy_signals"])
                        fig.add_scatter(x=buy_dates, y=buy_prices, mode='markers', name='Buy Signals',
                                       marker=dict(color='green', symbol='triangle-up', size=15))
                    if backtest_results["sell_signals"]:
                        sell_dates, sell_prices = zip(*backtest_results["sell_signals"])
                        fig.add_scatter(x=sell_dates, y=sell_prices, mode='markers', name='Sell Signals',
                                       marker=dict(color='red', symbol='triangle-down', size=15))
                    st.plotly_chart(fig, use_container_width=True)
    
            
def main():
    init_database()
    st.sidebar.title("🔍 Stock Selection")
    stock_list = fetch_nse_stock_list()

    if 'symbol' not in st.session_state:
        st.session_state.symbol = stock_list[0]
    if 'recommendation_mode' not in st.session_state:
        st.session_state.recommendation_mode = "Standard"

    symbol = st.sidebar.selectbox(
        "Select Stock",
        stock_list,
        key="stock_select",
        index=stock_list.index(st.session_state.symbol) if st.session_state.symbol in stock_list else 0
    )

    recommendation_mode = st.sidebar.radio(
        "Recommendation Mode",
        ["Standard", "Adaptive"],
        index=0 if st.session_state.recommendation_mode == "Standard" else 1,
        help="Standard: Timeframe-specific recommendations. Adaptive: Regime-based with position sizing."
    )
    st.session_state.recommendation_mode = recommendation_mode

    st.session_state.recommendation_mode = recommendation_mode




    if st.sidebar.button("Analyze Selected Stock"):
        if symbol:
            with st.spinner("Loading stock data..."):
                data = fetch_stock_data_with_auth(symbol)
                if not data.empty:
                    data = analyze_stock(data)
                    recommendations = (adaptive_recommendation(data) if recommendation_mode == "Adaptive"
                                      else generate_recommendations(data, symbol))
                    st.session_state.symbol = symbol
                    st.session_state.data = data
                    st.session_state.recommendations = recommendations
                    st.session_state.backtest_results_swing = None
                    st.session_state.backtest_results_intraday = None
                    display_dashboard(symbol, data, recommendations)
                else:
                    st.warning("⚠️ No data available for the selected stock.")
    else:
        display_dashboard()

    # Add Validation Tool in Sidebar (Moved to bottom)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠️ Diagnostics")
    if st.sidebar.button("✅ Validate All Tickers"):
        all_stocks = list(set([stock for sector in SECTORS.values() for stock in sector]))
        st.write(f"### 🔍 Validating {len(all_stocks)} Tickers...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        valid_count = 0
        invalid_count = 0
        
        # Create a placeholder for live results
        result_table = st.empty()
        
        for i, symbol in enumerate(all_stocks):
            status_text.text(f"Checking {symbol} ({i+1}/{len(all_stocks)})...")
            
            # Use a short period to just check connectivity
            data = fetch_stock_data_with_auth(symbol, period="1mo", interval="1d")
            
            if not data.empty:
                results.append({"Symbol": symbol, "Status": "✅ Pass", "Last Price": f"₹{data['Close'].iloc[-1]:.2f}"})
                valid_count += 1
            else:
                results.append({"Symbol": symbol, "Status": "❌ Fail", "Info": "No Data"})
                invalid_count += 1
            
            progress_bar.progress((i + 1) / len(all_stocks))
            
            # Update table every 5 stocks to keep UI responsive
            if i % 5 == 0:
                 result_table.dataframe(pd.DataFrame(results))

        progress_bar.empty()
        status_text.text(f"Validation Complete: {valid_count} Passed, {invalid_count} Failed")
        result_table.dataframe(pd.DataFrame(results))
        
        if invalid_count > 0:
            st.error(f"Found {invalid_count} invalid tickers. Run validation to see list.")
        else:
            st.success("All tickers are valid! 🎉")
if __name__ == "__main__":
    main()