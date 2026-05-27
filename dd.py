import pandas as pd
import ta
import threading
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

def get_config_value_with_source(*names):
    secret_sections = ("angelone", "smartapi", "smart_api", "broker")
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip() if isinstance(value, str) else value, f"env:{name}"

    try:
        for name in names:
            value = st.secrets.get(name)
            if value:
                return value.strip() if isinstance(value, str) else value, f"secrets:{name}"

        for section in secret_sections:
            values = st.secrets.get(section, {})
            for name in names:
                value = values.get(name) if hasattr(values, "get") else None
                if value:
                    return value.strip() if isinstance(value, str) else value, f"secrets:{section}.{name}"
    except Exception:
        pass

    return None, None

def get_config_value(*names):
    value, _ = get_config_value_with_source(*names)
    return value

def mask_secret(value):
    if not value:
        return "missing"
    text = str(value)
    if len(text) <= 4:
        return "*" * len(text)
    return f"{text[:2]}{'*' * max(len(text) - 4, 4)}{text[-2:]}"

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

# Suppress "missing ScriptRunContext" warnings from threads
class ContextWarningFilter(logging.Filter):
    def filter(self, record):
        return "missing ScriptRunContext" not in record.getMessage()

logging.getLogger().addFilter(ContextWarningFilter())
# Also try to hush the specific logger used by Streamlit runner
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").addFilter(ContextWarningFilter())

CLIENT_ID = get_config_value("CLIENT_ID", "ANGEL_CLIENT_ID", "client_id")
PASSWORD = get_config_value("PASSWORD", "PIN", "MPIN", "password")
TOTP_SECRET = get_config_value("TOTP_SECRET", "TOTP", "totp_secret", "totp")
HISTORICAL_API_KEY, HISTORICAL_API_KEY_SOURCE = get_config_value_with_source("API_KEY", "TRADING_API_KEY", "HISTORICAL_API_KEY", "api_key")
API_KEYS = {
    "Historical": HISTORICAL_API_KEY,
    "Trading": get_config_value("TRADING_API_KEY", "API_KEY", "api_key"),
    "Market": get_config_value("MARKET_API_KEY", "market_api_key")
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
smartapi_auth_error = None
smartapi_auth_lock = threading.Lock()
NIFTY_50_TOKEN = "99926000"
MIN_TOP_PICK_SCORE = 5
RANKING_WEIGHTS = {
    "relative_strength": 0.35,
    "rvol": 0.25,
    "sector": 0.20,
    "liquidity": 0.10,
    "entry": 0.10,
}
INTRADAY_RANKING_WEIGHTS = {
    "rvol": 0.30,
    "liquidity": 0.25,
    "entry": 0.20,
    "sector": 0.15,
    "relative_strength": 0.10,
}
OPPORTUNITY_SCORE_SCALE = 100
OPPORTUNITY_SCORE_CURVE_SCALE = 1.25
MAX_RANKED_ENTRY_GAP_PERCENT = 3.0
FRESH_BREAKOUT_LOOKBACK = 20
FRESH_BREAKOUT_MAX_AGE = 3
FRESH_BREAKOUT_DECAY_BONUSES = {
    1: 0.5,
    2: 0.3,
    3: 0.1,
}
SECTOR_EXHAUSTION_MOVE_THRESHOLD = 10.0
SECTOR_EXHAUSTION_RANKING_PENALTY = 0.5
TREND_PERSISTENCE_LOOKBACK = 5
MAX_TREND_PERSISTENCE_RANKING_ADJUSTMENT = 0.8
MAX_SECTOR_LEADER_RANKING_ADJUSTMENT = 0.6
MIN_INTRADAY_LIQUIDITY_CR = 10
MIN_INTRADAY_LIQUIDITY_VALUE = MIN_INTRADAY_LIQUIDITY_CR * 10_000_000
MIN_INTRADAY_RS = 1.0
MIN_INTRADAY_BREAKOUT_RS = 2.5
MIN_INTRADAY_SECTOR_RELATIVE_STRENGTH = 0.25
MIN_SWING_LIQUIDITY_CR = 10
MIN_SWING_LIQUIDITY_VALUE = MIN_SWING_LIQUIDITY_CR * 10_000_000
MIN_SWING_SECTOR_RELATIVE_STRENGTH = 0.25
EXHAUSTION_RVOL_THRESHOLD = 5.0
EXHAUSTION_EMA20_DISTANCE_THRESHOLD = 8.0
EXHAUSTION_DAILY_MOVE_THRESHOLD = 8.0
MAX_EXHAUSTION_RANKING_PENALTY = 2.0
INTRADAY_EXHAUSTION_RVOL_THRESHOLD = 8.0
INTRADAY_EXHAUSTION_EMA20_DISTANCE_THRESHOLD = 10.0
INTRADAY_EXHAUSTION_DAILY_MOVE_THRESHOLD = 12.0
INTRADAY_MAX_EXHAUSTION_RANKING_PENALTY = 1.0
INTRADAY_GAP_RISK_MOVE_THRESHOLD = 6.0
INTRADAY_OVERNIGHT_GAP_THRESHOLD = 3.0
INTRADAY_GAP_RISK_PENALTY = 0.5

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
        "TCS-EQ", "INFY-EQ", "HCLTECH-EQ", "WIPRO-EQ", "TECHM-EQ", "LTM-EQ",
        "MPHASIS-EQ", "FSL-EQ", "BSOFT-EQ", "NEWGEN-EQ", "ZENSARTECH-EQ",
        "RATEGAIN-EQ", "TANLA-EQ", "COFORGE-EQ", "PERSISTENT-EQ", "CYIENT-EQ",
        "SONATSOFTW-EQ", "KPITTECH-EQ", "TATAELXSI-EQ",
        "INTELLECT-EQ", "HAPPSTMNDS-EQ", "MASTEK-EQ", "ECLERX-EQ", "NIITLTD-EQ",
        "RSYSTEMS-EQ", "OFSS-EQ", "AURIONPRO-EQ", "DATAMATICS-EQ",
        "QUICKHEAL-EQ", "SAGILITY-EQ", "ALLDIGI-EQ","BLS-EQ"
    ],
    "Finance": [
        "HDFCBANK-EQ", "ICICIBANK-EQ", "SBIN-EQ", "KOTAKBANK-EQ", "BAJFINANCE-EQ",
        "AXISBANK-EQ", "BAJAJFINSV-EQ", "INDUSINDBK-EQ", "SHRIRAMFIN-EQ", "CHOLAFIN-EQ",
        "SBICARD-EQ", "M&MFIN-EQ", "MUTHOOTFIN-EQ", "LICHSGFIN-EQ", "IDFCFIRSTB-EQ",
        "AUBANK-EQ", "POONAWALLA-EQ", "SUNDARMFIN-EQ", "IIFL-EQ", "ABCAPITAL-EQ",
        "LTF-EQ", "CREDITACC-EQ", "MANAPPURAM-EQ", "JMFINANCIL-EQ",
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
        "SATIN-EQ", "JIOFIN-EQ", "NUVAMA-EQ"
    ],
    "Auto": [
        "MARUTI-EQ","BELRISE-EQ", "TMPV-EQ", "M&M-EQ", "BAJAJ-AUTO-EQ", "HEROMOTOCO-EQ",
        "EICHERMOT-EQ", "TVSMOTOR-EQ", "ASHOKLEY-EQ", "MRF-EQ", "BALKRISIND-EQ",
        "APOLLOTYRE-EQ", "CEATLTD-EQ", "JKTYRE-EQ", "MOTHERSON-EQ", "BHARATFORG-EQ",
        "SUNDRMFAST-EQ", "EXIDEIND-EQ", "BOSCHLTD-EQ", "ENDURANCE-EQ",
        "UNOMINDA-EQ", "ZFCVINDIA-EQ", "GABRIEL-EQ", "SUPRAJIT-EQ", "LUMAXTECH-EQ",
        "FIEMIND-EQ", "SUBROS-EQ", "JAMNAAUTO-EQ", "SHRIRAMFIN-EQ", "ESCORTS-EQ",
        "ATULAUTO-EQ", "OLECTRA-EQ", "GREAVESCOT-EQ", "SMLMAH-EQ", "VSTTILLERS-EQ",
        "MAHSCOOTER-EQ", "SONACOMS-EQ", "CRAFTSMAN-EQ"
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
        "BLISSGVS-EQ", "MOREPENLAB-EQ", "RPGLIFE-EQ", "YATHARTH-EQ"
    ],
    "Metals": [
        "TATASTEEL-EQ", "JSWSTEEL-EQ", "HINDALCO-EQ", "VEDL-EQ", "SAIL-EQ",
        "NMDC-EQ", "HINDZINC-EQ", "NATIONALUM-EQ", "JINDALSTEL-EQ", "MOIL-EQ",
        "APLAPOLLO-EQ", "RATNAMANI-EQ", "JSL-EQ", "WELCORP-EQ",
        "SHYAMMETL-EQ", "MIDHANI-EQ", "GRAVITA-EQ", "SARDAEN-EQ", "ASHAPURMIN-EQ",
        "JTLIND-EQ", "MAITHANALL-EQ", "KIOCL-EQ", "IMFA-EQ",
        "GMDCLTD-EQ", "VISHNU-EQ", "SANDUMA-EQ", "VRAJ-EQ", "COALINDIA-EQ", "JINDALSAW-EQ"
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
        "TARIL-EQ", "TDPOWERSYS-EQ", "ACMESOLAR-EQ", "WAAREEENER-EQ", "PREMIERENE-EQ", "GENUSPOWER-EQ"
    ],
    "Capital Goods": [
        "LT-EQ", "SIEMENS-EQ", "ABB-EQ", "BEL-EQ", "BHEL-EQ", "HAL-EQ",
        "CUMMINSIND-EQ", "THERMAX-EQ", "AIAENG-EQ", "SKFINDIA-EQ", "GRINDWELL-EQ",
        "TIMKEN-EQ", "KSB-EQ", "ELGIEQUIP-EQ", "LMW-EQ", "KIRLOSENG-EQ",
        "GREAVESCOT-EQ", "TRITURBINE-EQ", "VOLTAS-EQ", "BLUESTARCO-EQ", "HAVELLS-EQ",
        "DIXON-EQ", "KAYNES-EQ", "SYRMA-EQ", "AMBER-EQ", "SUZLON-EQ", "CGPOWER-EQ",
        "APARINDS-EQ", "HBLENGINE-EQ", "KEI-EQ", "POLYCAB-EQ", "RRKABEL-EQ",
        "SCHNEIDER-EQ", "TDPOWERSYS-EQ", "KIRLOSBROS-EQ", "JYOTICNC-EQ", "DATAPATTNS-EQ",
        "INOXWIND-EQ", "KPIL-EQ", "MAZDOCK-EQ", "COCHINSHIP-EQ", "GRSE-EQ",
        "POWERMECH-EQ", "ISGEC-EQ", "DYNAMATECH-EQ",
        "GMMPFAUDLR-EQ", "ESABINDIA-EQ", "TITAGARH-EQ",
        "VGUARD-EQ", "WABAG-EQ", "AZAD-EQ", "PGEL-EQ", "AVALON-EQ", "NETWEB-EQ", "MOSCHIP-EQ",
        "SOLARINDS-EQ", "POWERINDIA-EQ"
    ],
    "Oil & Gas": [
        "RELIANCE-EQ", "ONGC-EQ", "IOC-EQ", "BPCL-EQ", "HINDPETRO-EQ", "GAIL-EQ",
        "PETRONET-EQ", "OIL-EQ", "IGL-EQ", "MGL-EQ", "GUJGASLTD-EQ",
        "AEGISLOG-EQ", "CHENNPETRO-EQ", "MRPL-EQ", "FLUOROCHEM-EQ", "CASTROLIND-EQ",
        "SOTL-EQ", "PANAMAPET-EQ", "GOCLCORP-EQ"
    ],
    "Chemicals": [
        "PIDILITIND-EQ", "SRF-EQ", "DEEPAKNTR-EQ", "ATUL-EQ", "AARTIIND-EQ",
        "NAVINFLUOR-EQ", "VINATIORGA-EQ", "FINEORG-EQ", "ALKYLAMINE-EQ", "BALAMINES-EQ",
        "FLUOROCHEM-EQ", "CLEAN-EQ", "JUBLINGREA-EQ", "GALAXYSURF-EQ", "PCBL-EQ",
        "NOCIL-EQ", "BASF-EQ", "SUDARSCHEM-EQ", "NEOGEN-EQ", "PRIVISCL-EQ",
        "ROSSARI-EQ", "LXCHEM-EQ", "ANURAS-EQ", "CHEMCON-EQ",
        "DMCC-EQ", "TATACHEM-EQ", "COROMANDEL-EQ", "UPL-EQ",
        "SUMICHEM-EQ", "PIIND-EQ", "EIDPARRY-EQ", "CHEMPLASTS-EQ",
        "IGPL-EQ", "TIRUMALCHM-EQ", "RALLIS-EQ"
    ],
    "Telecom": [
        "BHARTIARTL-EQ", "INDUSTOWER-EQ", "TATACOMM-EQ",
        "HFCL-EQ", "TEJASNET-EQ"
    ],
    "Infrastructure": [
        "LT-EQ", "GMRAIRPORT-EQ", "IRB-EQ", "NBCC-EQ", "RVNL-EQ", "KEC-EQ",
        "PNCINFRA-EQ", "GRINFRA-EQ", "NCC-EQ", "HGINFRA-EQ",
        "ASHOKA-EQ", "JWL-EQ", "KPIL-EQ",
        "IRCON-EQ", "ENGINERSIN-EQ", "AHLUCONT-EQ", "PSPPROJECT-EQ", "CAPACITE-EQ",
        "WELSPUNLIV-EQ", "MANINFRA-EQ", "ADANIPORTS-EQ", "JSWINFRA-EQ"
    ],
    "Insurance": [
        "SBILIFE-EQ", "HDFCLIFE-EQ", "ICICIGI-EQ", "ICICIPRULI-EQ", "LICI-EQ",
        "GICRE-EQ", "NIACL-EQ", "STARHEALTH-EQ", "MFSL-EQ"
    ],
    "Diversified": [
        "ADANIENT-EQ", "GRASIM-EQ",
        "DCMSHRIRAM-EQ", "3MINDIA-EQ", "CENTURYPLY-EQ", "KFINTECH-EQ", "BALMLAWRIE-EQ",
        "GODREJIND-EQ", "BIRLACORPN-EQ"
    ],
    "Cement": [
        "ULTRACEMCO-EQ", "SHREECEM-EQ", "AMBUJACEM-EQ", "ACC-EQ", "JKCEMENT-EQ",
        "DALBHARAT-EQ", "RAMCOCEM-EQ", "NUVOCO-EQ", "JKLAKSHMI-EQ",
        "HEIDELBERG-EQ", "INDIACEM-EQ", "PRSMJOHNSN-EQ", "STARCEMENT-EQ", "SAGCEM-EQ",
        "DECCANCE-EQ", "KCP-EQ", "ORIENTCEM-EQ", "BIRLANU-EQ", "EVERESTIND-EQ",
        "VISAKAIND-EQ", "BIGBLOC-EQ"
    ],
    "Realty": [
        "DLF-EQ", "GODREJPROP-EQ", "OBEROIRLTY-EQ", "PHOENIXLTD-EQ", "PRESTIGE-EQ",
        "BRIGADE-EQ", "SOBHA-EQ", "SUNTECK-EQ", "MAHLIFE-EQ", "ANANTRAJ-EQ", "LODHA-EQ",
        "KOLTEPATIL-EQ", "PURVA-EQ", "ARVSMART-EQ", "RUSTOMJEE-EQ", "DBREALTY-EQ",
        "OMAXE-EQ", "ASHIANA-EQ", "ELDEHSG-EQ", "TARC-EQ"
    ],
    "Aviation": [
        "INDIGO-EQ", "SPICEJET-EQ", "GMRAIRPORT-EQ"
    ],
    "Retail": [
        "DMART-EQ", "TRENT-EQ", "ABFRL-EQ", "VMART-EQ", "SHOPERSTOP-EQ",
        "BATAINDIA-EQ", "METROBRAND-EQ", "ARVINDFASN-EQ", "CANTABIL-EQ", "ETERNAL-EQ",
        "NYKAA-EQ", "MANYAVAR-EQ", "LANDMARK-EQ", "V2RETAIL-EQ",
        "THANGAMAYL-EQ", "KALYANKJIL-EQ", "TITAN-EQ"
    ],
    "Media": [
        "ZEEL-EQ", "SUNTV-EQ", "TVTODAY-EQ",
        "SAREGAMA-EQ"
    ],
    "Consumer Durables": [
        "WHIRLPOOL-EQ", "DIXON-EQ", "AMBER-EQ", "VOLTAS-EQ", "BLUESTARCO-EQ",
        "HAVELLS-EQ", "CROMPTON-EQ", "VGUARD-EQ", "ORIENTELEC-EQ", "KIRIINDUS-EQ",
        "SYMPHONY-EQ", "TITAN-EQ", "KPIL-EQ", "RELAXO-EQ",
        "TTKHLTCARE-EQ", "VAIBHAVGBL-EQ", "BAJAJELEC-EQ", "FINEORG-EQ", "CHOLAHLDNG-EQ",
        "SUPRAJIT-EQ", "NIITLTD-EQ", "APARINDS-EQ"
    ],
    "Defence": [
        "HAL-EQ", "BEL-EQ", "BDL-EQ", "PARAS-EQ", "BEML-EQ",
        "MAZDOCK-EQ", "COCHINSHIP-EQ", "GRSE-EQ", "DATAPATTNS-EQ", "AZAD-EQ",
        "SOLARINDS-EQ"
    ],
    "Consumer Services": [
        "ETERNAL-EQ", "NYKAA-EQ", "ADANIPORTS-EQ", "IRCTC-EQ", "PAYTM-EQ", "INDHOTEL-EQ", "NAUKRI-EQ", "DELHIVERY-EQ",
        "JUBLFOOD-EQ", "DEVYANI-EQ", "WESTLIFE-EQ", "SAPPHIRE-EQ", "BIKAJI-EQ",
        "IXIGO-EQ", "TEAMLEASE-EQ", "QUESS-EQ", "FSL-EQ",
        "MINDSPACE-EQ", "CIEINDIA-EQ", "VMART-EQ", "SHOPERSTOP-EQ",
        "TRENT-EQ", "DMART-EQ", "ABFRL-EQ", "MANYAVAR-EQ", "V2RETAIL-EQ"
    ]
}

@st.cache_resource(ttl=86400)
def get_smartapi_session(api_key, client_id, password, totp_secret):
    """
    Initializes and caches the SmartAPI session to avoid repeated logins.
    Logins are limited to 1 per second, but we should only need one per day/session.
    """
    missing = [
        name for name, value in {
            "CLIENT_ID": client_id,
            "PASSWORD": password,
            "TOTP_SECRET": totp_secret,
            "HISTORICAL_API_KEY": api_key,
        }.items()
        if not value
    ]
    if missing:
        st.error(f"SmartAPI credentials missing in .env or Streamlit secrets: {', '.join(missing)}")
        return None

    try:
        logging.info(
            "Using SmartAPI historical key from %s: %s",
            HISTORICAL_API_KEY_SOURCE or "unknown",
            mask_secret(api_key),
        )
        smart_api = SmartConnect(api_key=api_key)
        totp = pyotp.TOTP(totp_secret)
        data = smart_api.generateSession(client_id, password, totp.now())
        if data and isinstance(data, dict) and data.get('status'):
            clear_smartapi_auth_error()
            return smart_api
        elif isinstance(data, dict):
            message = data.get('message', 'Unknown authentication error')
            error_code = data.get('errorCode') or data.get('errorcode')
            if error_code:
                message = f"{message} ({error_code})"
            st.error(f"SmartAPI authentication failed: {message}")
            return None
        else:
            st.error("SmartAPI authentication failed: generateSession returned an empty or invalid response")
            return None
    except Exception as e:
        st.error(f"Error initializing SmartAPI: {str(e)}")
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

# Thread-safe rate limiter
rate_limit_lock = threading.Lock()

def enforce_rate_limit(min_interval=0.5): # Increased to 0.5s (2 req/s) for safety
    global last_api_call_time
    with rate_limit_lock:
        current_time = time.time()
        elapsed = current_time - last_api_call_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        last_api_call_time = time.time()

def set_smartapi_auth_error(message):
    global smartapi_auth_error
    with smartapi_auth_lock:
        smartapi_auth_error = message

def get_smartapi_auth_error():
    with smartapi_auth_lock:
        return smartapi_auth_error

def clear_smartapi_auth_error():
    global smartapi_auth_error
    with smartapi_auth_lock:
        smartapi_auth_error = None

@retry(max_retries=5, delay=5)
def fetch_stock_data_with_auth(symbol, period="2y", interval="1d"):
    cache_key = f"{symbol}_{period}_{interval}"
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return pd.read_pickle(io.BytesIO(cached_data))

    try:
        auth_error = get_smartapi_auth_error()
        if auth_error:
            logging.warning(f"Skipping SmartAPI request for {symbol}: {auth_error}")
            return pd.DataFrame()

        if "-EQ" not in symbol:
            symbol = f"{symbol.split('.')[0]}-EQ"

        # Use the cached session instead of creating a new one every time
        smart_api = get_smartapi_session(API_KEYS["Historical"], CLIENT_ID, PASSWORD, TOTP_SECRET)
        if not smart_api:
            # If session failed, try to re-initialize once (maybe expired)
            st.cache_resource.clear()
            smart_api = get_smartapi_session(API_KEYS["Historical"], CLIENT_ID, PASSWORD, TOTP_SECRET)
            if not smart_api:
                 raise ValueError("SmartAPI client initialization failed")

        end_date = datetime.now()
        if period == "2y":
            start_date = end_date - timedelta(days=2 * 365)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "1mo":
            start_date = end_date - timedelta(days=30)
        elif period == "5d":
            start_date = end_date - timedelta(days=5)
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
            logging.warning(f"⚠️ Token not found for symbol: {symbol}")
            return pd.DataFrame()

        # Enforce rate limit before making the API call
        enforce_rate_limit()

        # Retry logic for API instability
        for attempt in range(3):
            try:
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
                    
                    # Dynamic Cache Expiry: Intraday needs freshness!
                    if interval in ['5m', '15m']:
                        expire_time = 300 # 5 minutes for intraday
                    elif interval == '1h':
                        expire_time = 1800 # 30 mins for hourly
                    else:
                        expire_time = 43200 # 12 hours for daily
                        
                    cache.set(cache_key, buffer.getvalue(), expire=expire_time)
                    return data
                
                # Handling INVALID TOKEN (AG8001) - Force Re-login
                elif historical_data and isinstance(historical_data, dict) and historical_data.get('errorCode') == 'AG8001':
                    logging.warning(f"⚠️ Invalid Token for {symbol} (AG8001). Clearing cache & re-logging in...")
                    st.cache_resource.clear() # Clear cached session
                    smart_api = get_smartapi_session(API_KEYS["Historical"], CLIENT_ID, PASSWORD, TOTP_SECRET) # Get fresh session
                    time.sleep(1) # Slight pause before retry
                    continue # Retry loop with new session

                elif historical_data and isinstance(historical_data, dict) and (historical_data.get('errorCode') == 'AG8004' or historical_data.get('errorcode') == 'AG8004'):
                    message = historical_data.get('message', 'Invalid SmartAPI API key')
                    auth_error = f"{message} (AG8004)"
                    set_smartapi_auth_error(auth_error)
                    logging.error(f"SmartAPI authentication failed for {symbol}: {auth_error}")
                    st.cache_resource.clear()
                    return pd.DataFrame()

                elif historical_data and isinstance(historical_data, dict) and (historical_data.get('errorcode') == 'AB1004' or historical_data.get('message') == 'Internal Server Error'):
                     # Retry on recognized temporary server errors
                     logging.warning(f"Server error for {symbol}, retrying ({attempt+1}/3)...")
                     time.sleep(1 * (attempt + 1))
                     continue
                else:
                    # Data missing but no error code, likely valid empty response
                    return pd.DataFrame()

            except Exception as e:
                # Catch connection errors/timeouts during call
                 logging.warning(f"Exception for {symbol}, retrying ({attempt+1}/3): {str(e)}")
                 time.sleep(1 * (attempt + 1))

        return pd.DataFrame()

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logging.warning(f"⚠️ Rate limit exceeded for {symbol}. Skipping...")
            return pd.DataFrame()
        raise e
    except Exception as e:
        # Check for specific "Rate Limit" string in exception message
        if "exceeding access rate" in str(e):
             logging.warning(f"Rate limit hit for {symbol}. Prioritizing safety sleep...")
             time.sleep(5) # Long sleep if hit hard limit
             return pd.DataFrame()
        logging.warning(f"⚠️ Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

@lru_cache(maxsize=1000)
def fetch_stock_data_cached(symbol, period="2y", interval="1d"):
    return fetch_stock_data_with_auth(symbol, period, interval)

@st.cache_data(ttl=1800)
def fetch_nifty_recent_return(interval="ONE_DAY", lookback_days=10, candles=5):
    try:
        auth_error = get_smartapi_auth_error()
        if auth_error:
            logging.warning(f"Skipping NIFTY benchmark request: {auth_error}")
            return 0.0

        smart_api = get_smartapi_session(API_KEYS["Historical"], CLIENT_ID, PASSWORD, TOTP_SECRET)
        if not smart_api:
            return 0.0

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        enforce_rate_limit()
        historical_data = smart_api.getCandleData({
            "exchange": "NSE",
            "symboltoken": NIFTY_50_TOKEN,
            "interval": interval,
            "fromdate": start_date.strftime("%Y-%m-%d %H:%M"),
            "todate": end_date.strftime("%Y-%m-%d %H:%M")
        })

        if not historical_data or not isinstance(historical_data, dict) or not historical_data.get("data"):
            return 0.0

        data = pd.DataFrame(historical_data["data"], columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        return_value = calculate_recent_return(data, candles=candles)
        return 0.0 if pd.isna(return_value) else float(return_value)
    except Exception as e:
        logging.warning(f"Failed to compute NIFTY relative-strength benchmark: {str(e)}")
        return 0.0

def fetch_nifty_5d_return():
    return fetch_nifty_recent_return(interval="ONE_DAY", lookback_days=10, candles=5)

def fetch_nifty_intraday_return():
    return fetch_nifty_recent_return(interval="FIFTEEN_MINUTE", lookback_days=5, candles=5)

def calculate_advance_decline_ratio(stock_list):
    advances = 0
    declines = 0
    for symbol in stock_list:
        data = fetch_stock_data_cached(symbol)
        if not data.empty and len(data) >= 2:
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

def get_dynamic_rsi_window(data):
    try:
        atr = to_float_or_none(data['ATR'].iloc[-1]) if 'ATR' in data.columns else None
        close = to_float_or_none(data['Close'].iloc[-1]) if 'Close' in data.columns else None
        if not atr or not close:
            return 14
        atr_pct = atr / close
        return 9 if atr_pct > 0.03 else 14
    except Exception:
        return 14

def detect_divergence(data):
    recent = data[['Close', 'RSI']].dropna().tail(5)
    if len(recent) < 5:
        return "No Divergence"

    price = recent['Close'].reset_index(drop=True)
    rsi = recent['RSI'].reset_index(drop=True)
    recent_highs = int(price.idxmax())
    recent_lows = int(price.idxmin())
    rsi_highs = int(rsi.idxmax())
    rsi_lows = int(rsi.idxmin())
    bullish_div = (recent_lows > rsi_lows) and (price.iloc[recent_lows] < price.iloc[-1]) and (rsi.iloc[rsi_lows] < rsi.iloc[-1])
    bearish_div = (recent_highs < rsi_highs) and (price.iloc[recent_highs] > price.iloc[-1]) and (rsi.iloc[rsi_highs] > rsi.iloc[-1])
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

def analyze_stock(data, interval="1d"):
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
            rsi_window = get_dynamic_rsi_window(data)
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
        if interval in ["5m", "15m"] and can_compute_indicator(data, 'VWAP'):
            typical_price_volume = ((data['High'] + data['Low'] + data['Close']) / 3) * data['Volume']
            session_key = data.index.date if isinstance(data.index, pd.DatetimeIndex) else pd.Series(0, index=data.index)
            session_tp_volume = typical_price_volume.groupby(session_key).cumsum()
            session_volume = data['Volume'].groupby(session_key).cumsum()
            data['VWAP'] = session_tp_volume / session_volume.replace(0, np.nan)
        else:
            data['VWAP'] = np.nan
    except Exception as e:
        logging.warning(f"Failed to compute VWAP: {str(e)}")
        data['VWAP'] = np.nan

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
    if data.empty or 'RSI' not in data.columns or pd.isna(data['RSI'].iloc[-1]):
        st.warning("⚠️ Cannot calculate Buy At due to missing or invalid RSI data.")
        return None, "Unavailable"
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

def calculate_target(data, risk_reward_ratio=3, entry_price=None, stop_loss=None):
    stop_loss = stop_loss if stop_loss is not None else calculate_stop_loss(data, entry_price=entry_price)
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
    # SmartAPI historical data does not provide fundamentals in this app.
    return {'P/E': None, 'EPS': None, 'RevenueGrowth': None}

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

    avg_volume = data['Avg_Volume'].iloc[-1] if 'Avg_Volume' in data.columns else None
    if pd.notnull(avg_volume) and avg_volume > 0 and data['Volume'].iloc[-1] < avg_volume * 0.5:
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
        pe = fundamentals.get('P/E')
        eps = fundamentals.get('EPS')
        revenue_growth = fundamentals.get('RevenueGrowth')
        if pd.notnull(pe) and pd.notnull(eps) and pe < 15 and eps > 0:
            score += weights['Fundamentals'] * 0.5
        elif (pd.notnull(pe) and pe > 30) or (pd.notnull(eps) and eps < 0):
            score -= weights['Fundamentals'] * 0.5
        if pd.notnull(revenue_growth) and revenue_growth > 0.1:
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
        
def detect_advanced_patterns(data, window=20):
    """
    Detects advanced breakout patterns with STRICT filters:
    1. Increasing Demand (Ascending Triangle) - Requires Trend & Quality Slope
    2. Fake-out Reversal (Bear Trap)
    Returns a dictionary with pattern detected and description.
    """
    if data is None or len(data) < window + 5:
        return None

    try:
        # Get recent data
        recent = data.iloc[-window:]
        current_close = recent['Close'].iloc[-1]
        
        # --- PRE-FILTERS ---
        # 1. Trend Filter: Must be above EMA 50 to ensure we aren't catching falling knives
        if 'EMA_50' in recent.columns and pd.notnull(recent['EMA_50'].iloc[-1]):
             if current_close < recent['EMA_50'].iloc[-1]:
                 return None # Downtrend -> Ignore all bullish patterns
        
        # 2. Volume Check: Recent action must have some volume (avoid dead stocks)
        avg_vol = recent['Volume'].mean()
        current_vol = recent['Volume'].iloc[-1]
        if current_vol < (avg_vol * 0.5): # At least 50% of avg volume required
            return None

        # --- PATTERN 1: Increasing Demand (Ascending Triangle) ---
        # Logic: Highs are relatively flat (resistance), Lows are making higher lows
        highs = recent['High'].values
        lows = recent['Low'].values
        
        # Check for resistance (flat highs)
        avg_high = np.mean(highs[-5:]) # Last 5 bars
        resistance_variance = np.var(highs[-5:])
        is_resistance_flat = resistance_variance < (current_close * 0.005)
        
        # Check for higher lows (Positive Slope + Quality Fit)
        x = np.arange(len(lows))
        slope, _ = np.polyfit(x, lows, 1)
        
        # Calculate R-Squared to verify it's a real line, not noise
        correlation_matrix = np.corrcoef(x, lows)
        correlation_xy = correlation_matrix[0,1]
        r_squared = correlation_xy**2
        
        is_demand_increasing = slope > 0.05 and r_squared > 0.6 # Stricter: Real positive trend
        
        # Breakout Potential: Close must be near resistance
        near_resistance = current_close >= (avg_high * 0.98)

        if is_resistance_flat and is_demand_increasing and near_resistance:
            return {
                "pattern": "Increasing Demand",
                "action": "BUY",
                "confidence": "High",
                "desc": "Higher lows into resistance (Ascending Triangle). Buyers absorbing supply.",
                "breakout_level": avg_high
            }

        # --- PATTERN 2: Fake-out Reversal (Bear Trap) ---
        # Logic: Price dipped below recent support (last 10-20 bars) but closed strong
        recent_support = data['Low'].iloc[-(window+10):-5].min() 
        recent_low = recent['Low'].min()
        
        # Trap Logic:
        # 1. Sweep: Low went below support
        liquidity_sweep = recent_low < recent_support
        # 2. Rejection: Close is back above support
        strong_close = current_close > recent_support
        # 3. Shape: Bullish Candle
        current_open = recent['Open'].iloc[-1]
        is_bullish_candle = current_close > current_open
        # 4. Proximity: The dip shouldn't be a massive crash (e.g. < 3% drop below support)
        # If it dropped 10% then came back, that's too volatile.
        valid_depth = (recent_support - recent_low) / recent_support < 0.03
        
        if liquidity_sweep and strong_close and is_bullish_candle and valid_depth:
             return {
                "pattern": "Fake-out Reversal",
                "action": "STRONG BUY",
                "confidence": "Very High",
                "desc": "Liquidity sweep below support followed by strong rejection (Bear Trap).",
                "breakout_level": current_close
            }

    except Exception as e:
        logging.warning(f"Error in detect_advanced_patterns: {e}")
    
    return None

def generate_recommendations(data, symbol=None):
    recommendations = {
        "Intraday": "Hold", "Swing": "Hold",
        "Short-Term": "Hold", "Long-Term": "Hold",
        "Mean_Reversion": "Hold", "Breakout": "Hold", "Ichimoku_Trend": "Hold",
        "Current Price": None, "Buy At": None,
        "Stop Loss": None, "Target": None, "Score": 0,
        "Major Trend Conflict": False,
        "Pattern Notes": None, "Entry Strategy": None # Added for advanced details
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
        
        # --- Advanced Pattern Detection Integration ---
        adv_pattern = detect_advanced_patterns(data)
        if adv_pattern:
            breakout_lvl = adv_pattern.get('breakout_level', recommendations["Current Price"])
            if adv_pattern['action'] == "BUY":
                buy_score += 5 # High weight
                recommendations["Breakout"] = "Buy"
                recommendations["Pattern Notes"] = f"✅ {adv_pattern['pattern']}: {adv_pattern['desc']}"
                recommendations["Entry Strategy"] = f"⚠️ Pyramiding: Buy 25% qty Now. Add remaining 75% above ₹{breakout_lvl:.2f}."
            elif adv_pattern['action'] == "STRONG BUY":
                buy_score += 8 # Very High weight
                recommendations["Breakout"] = "Strong Buy"
                recommendations["Pattern Notes"] = f"🚀 {adv_pattern['pattern']}: {adv_pattern['desc']}"
                recommendations["Entry Strategy"] = f"⚠️ Pyramiding: Buy 25% qty Now. Add remaining 75% above ₹{breakout_lvl:.2f} (Confirmation)."
        # ----------------------------------------------

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
            pe = fundamentals.get('P/E')
            eps = fundamentals.get('EPS')
            revenue_growth = fundamentals.get('RevenueGrowth')
            if pd.notnull(pe) and pd.notnull(eps) and pe < 15 and eps > 0:
                buy_score += 2
            elif (pd.notnull(pe) and pe > 30) or (pd.notnull(eps) and eps < 0):
                sell_score += 1
            if pd.notnull(revenue_growth) and revenue_growth > 0.1:
                buy_score += 1
            elif pd.notnull(revenue_growth) and revenue_growth < 0:
                sell_score += 0.5

        major_trend_conflict = recommendations["Ichimoku_Trend"] == "Strong Sell"
        if major_trend_conflict:
            buy_score = max(0, buy_score - 2)
            sell_score += 2

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

        if recommendations["Mean_Reversion"] == "Sell" and recommendations["Swing"] == "Buy":
            buy_score = max(0, buy_score - 1)

        recommendations["Major Trend Conflict"] = has_major_trend_conflict(recommendations)
        if recommendations["Major Trend Conflict"]:
            for signal in ("Intraday", "Swing", "Short-Term", "Long-Term", "Breakout"):
                if is_buy_signal(recommendations.get(signal)):
                    recommendations[signal] = "Hold"
            conflict_note = "Major trend conflict: Ichimoku Strong Sell blocked bullish recommendation."
            existing_notes = recommendations.get("Pattern Notes")
            recommendations["Pattern Notes"] = f"{existing_notes} | {conflict_note}" if existing_notes else conflict_note

        recommendations["Buy At"], recommendations["Entry Type"] = calculate_buy_at(data)
        if is_valid_price(recommendations["Buy At"]):
            recommendations["Stop Loss"] = calculate_stop_loss(data, entry_price=recommendations["Buy At"])
            recommendations["Target"] = calculate_target(
                data,
                entry_price=recommendations["Buy At"],
                stop_loss=recommendations["Stop Loss"],
            )
        else:
            recommendations["Stop Loss"] = None
            recommendations["Target"] = None

        final_score = buy_score - sell_score
        if recommendations["Major Trend Conflict"]:
            final_score = min(final_score, MIN_TOP_PICK_SCORE - 1)
        recommendations["Score"] = min(max(final_score, -7), 7)
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
            data = analyze_stock(data, interval="1d")
            rec = generate_recommendations(data, symbol)
            total_score += rec.get("Score", 0)
            count += 1
            # Rate limiting is handled globally now
        avg_score = total_score / count if count else 0
        sector_scores[sector] = avg_score
        # Removed redundant sleep
    return sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)[:3]

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
        
        if isinstance(signal, str) and "Buy" in signal and position is None:
            position = "Long"
            entry_price = current_price
            entry_date = current_date
            results["buy_signals"].append((current_date, current_price))
        
        elif isinstance(signal, str) and "Sell" in signal and position == "Long":
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

    if position == "Long" and entry_price:
        current_price = data['Close'].iloc[-1]
        current_date = data.index[-1]
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
    expected_columns = {
        "score": "REAL",
        "current_price": "REAL",
        "buy_at": "REAL",
        "stop_loss": "REAL",
        "target": "REAL",
        "intraday": "TEXT",
        "swing": "TEXT",
        "short_term": "TEXT",
        "long_term": "TEXT",
        "mean_reversion": "TEXT",
        "breakout": "TEXT",
        "ichimoku_trend": "TEXT",
        "recommendation": "TEXT",
        "regime": "TEXT",
        "position_size": "REAL",
        "trailing_stop": "REAL",
        "reason": "TEXT",
        "pick_type": "TEXT",
    }
    existing_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(daily_picks)").fetchall()
    }
    for column, column_type in expected_columns.items():
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE daily_picks ADD COLUMN {column} {column_type}")
    conn.commit()
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

def analyze_batch(stock_batch, patience="high", interval="1d"):
    """
    Analyzes a batch of stocks in parallel.
    Returns a list of results (dictionaries) for ALL processed stocks, including failures.
    """
    # Capture Streamlit state in the main thread
    recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
    
    results = []
    # Reduced max_workers to 2 to prevent API Rate Limit hits
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Pass recommendation_mode explicitly to the worker
        futures = {executor.submit(analyze_stock_parallel, symbol, patience, interval, recommendation_mode): symbol for symbol in stock_batch}
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

def analyze_stock_parallel(symbol, patience="high", interval="1d", recommendation_mode="Standard"):
    """
    Analyzes a single stock.
    Returns a dictionary with 'Status' (Success, No Data, Error) and detailed analysis or error info.
    """
    try:
        logging.info(f"Starting analysis for {symbol}")
        # Adjust period based on interval for efficiency
        period = "2y" if interval == "1d" else "1mo" 
        data = fetch_stock_data_cached(symbol, period=period, interval=interval)
        
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
        
        data = analyze_stock(data, interval=interval)
        recent_return = calculate_recent_return(data)
        trend_persistence = calculate_trend_persistence_score(data)
        latest_move_pct, ema20_distance_pct = calculate_momentum_extension_metrics(data)
        previous_day_move_pct, overnight_gap_pct = calculate_session_gap_metrics(data)
        fresh_breakout_age = calculate_fresh_breakout_age(data)
        rvol, avg_volume_value = calculate_volume_metrics(data)
        logging.info(f"Analyzing {symbol} in {recommendation_mode} mode")
        
        if recommendation_mode == "Adaptive":
            rec = adaptive_recommendation(data, symbol)
            # Override Buy At if patience is low (e.g. Intraday)
            if patience == "low" and rec.get("Current Price"):
                rec["Buy At"], rec["Entry Type"] = calculate_buy_at(data, patience="low")
                # RECALCULATE Risk Management based on new Entry!
                if is_valid_price(rec["Buy At"]):
                    entry_type = rec["Entry Type"]
                    # Adjust SL Multiplier based on Entry Type (User Request)
                    sl_mult = 2.0 if entry_type == "Breakout" else 1.5
                    rec["Stop Loss"] = calculate_stop_loss(data, atr_multiplier=sl_mult, entry_price=rec["Buy At"])
                    rec["Target"] = calculate_target(
                        data,
                        risk_reward_ratio=2.5,
                        entry_price=rec["Buy At"],
                        stop_loss=rec["Stop Loss"],
                    ) # Realistic 2.5R
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
                "Recent Return": recent_return,
                "Trend Persistence": trend_persistence,
                "Latest Move %": latest_move_pct,
                "Previous Day Move %": previous_day_move_pct,
                "Overnight Gap %": overnight_gap_pct,
                "EMA20 Distance %": ema20_distance_pct,
                "Fresh Breakout Age": fresh_breakout_age,
                "RVOL": rvol,
                "Avg Volume Value": avg_volume_value,
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
                "Pattern Notes": rec.get("Pattern Notes"), # Pass through
                "Entry Strategy": rec.get("Entry Strategy"), # Pass through
                "Intraday": rec.get("Intraday", "Hold"),
                "Swing": rec.get("Swing", "Hold"),
                "Short-Term": None,
                "Long-Term": None,
                "Mean_Reversion": None,
                "Breakout": None,
                "Ichimoku_Trend": None,
                "Major Trend Conflict": rec.get("Major Trend Conflict", False)
            }
        else:
            rec = generate_recommendations(data, symbol)
            # Override Buy At if patience is low
            if patience == "low" and rec.get("Current Price"):
                 rec["Buy At"], rec["Entry Type"] = calculate_buy_at(data, patience="low")
                 # RECALCULATE Risk Management based on new Entry!
                 if is_valid_price(rec["Buy At"]):
                    entry_type = rec.get("Entry Type", "Standard")
                    sl_mult = 2.0 if entry_type == "Breakout" else 1.5
                    rec["Stop Loss"] = calculate_stop_loss(data, atr_multiplier=sl_mult, entry_price=rec["Buy At"])
                    rec["Target"] = calculate_target(
                        data,
                        risk_reward_ratio=2.5,
                        entry_price=rec["Buy At"],
                        stop_loss=rec["Stop Loss"],
                    )
            else:
                 rec["Entry Type"] = "Standard"

            if not rec or not rec.get('Intraday'):
                return {
                    "Symbol": symbol,
                    "Status": "Analysis Failed",
                    "Error": "Standard Recommendation returned empty",
                    "Score": 0,
                    "Recommendation": "N/A",
                    "Intraday": "Hold", 
                    "Swing": "Hold"
                }

            return {
                "Symbol": symbol,
                "Status": "Success",
                "Current Price": rec.get("Current Price"),
                "Recent Return": recent_return,
                "Trend Persistence": trend_persistence,
                "Latest Move %": latest_move_pct,
                "Previous Day Move %": previous_day_move_pct,
                "Overnight Gap %": overnight_gap_pct,
                "EMA20 Distance %": ema20_distance_pct,
                "Fresh Breakout Age": fresh_breakout_age,
                "RVOL": rvol,
                "Avg Volume Value": avg_volume_value,
                "Buy At": rec.get("Buy At"),
                "Stop Loss": rec.get("Stop Loss"),
                "Target": rec.get("Target"),
                "Pattern Notes": rec.get("Pattern Notes"), # Pass through
                "Entry Strategy": rec.get("Entry Strategy"), # Pass through
                "Intraday": rec.get("Intraday", "Hold"),
                "Swing": rec.get("Swing", "Hold"),
                "Short-Term": rec.get("Short-Term", "Hold"),
                "Long-Term": rec.get("Long-Term", "Hold"),
                "Mean_Reversion": rec.get("Mean_Reversion", "Hold"),
                "Breakout": rec.get("Breakout", "Hold"),
                "Ichimoku_Trend": rec.get("Ichimoku_Trend", "Hold"),
                "Major Trend Conflict": rec.get("Major Trend Conflict", False),
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
            "Recommendation": "N/A",
            "Intraday": "Hold",
            "Swing": "Hold"
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
    expected_cols = [
        "Symbol", "Score", "Current Price", "Recent Return", "Trend Persistence", "Latest Move %",
        "EMA20 Distance %", "Fresh Breakout Age", "RVOL", "Avg Volume Value",
        "Buy At", "Stop Loss", "Target", "Recommendation", "Intraday", "Swing", "Short-Term",
        "Long-Term", "Mean_Reversion", "Breakout", "Ichimoku_Trend", "Major Trend Conflict",
        "Status", "Error"
    ]
    for col in expected_cols:
         if col not in results_df.columns:
             results_df[col] = None

    # Filter for Top Picks (Success only)
    success_df = results_df[results_df["Status"] == "Success"].copy()
    sector_momentum = calculate_sector_momentum_map(success_df)
    nifty_5d_return = fetch_nifty_5d_return()
    success_df["Score"] = pd.to_numeric(success_df["Score"], errors="coerce").fillna(0)
    success_df = success_df[success_df.apply(is_actionable_entry, axis=1)]
    success_df = success_df[success_df["Score"] >= MIN_TOP_PICK_SCORE]
    ranked_success_df = add_entry_quality_columns(success_df, sector_momentum, nifty_5d_return)

    # Sort logic for Top Picks
    recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
    if recommendation_mode == "Adaptive":
        top_picks_df = ranked_success_df[ranked_success_df["Recommendation"].str.contains("Buy", na=False)]
    else:
        buy_columns = ["Swing", "Short-Term", "Long-Term", "Breakout", "Ichimoku_Trend"]
        buy_signal = ranked_success_df[buy_columns].apply(
            lambda row: row.astype(str).str.contains("Buy", na=False).any(),
            axis=1
        )
        top_picks_df = ranked_success_df[buy_signal]

    if not top_picks_df.empty:
        top_picks_df = top_picks_df[top_picks_df.apply(is_swing_quality_setup, axis=1)]
    top_picks_df = top_picks_df.sort_values(
        by=["Ranking Score", "Reward/Risk", "Score"],
        ascending=[False, False, False]
    )
    top_picks_df = limit_top_picks_by_sector(top_picks_df, max_per_sector=2, limit=5)
    
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
        
        # Helper to fetch swing-window change.
        live_data = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {executor.submit(fetch_stock_data_cached, symbol, "5d"): symbol for symbol in all_symbols}
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    if not data.empty and len(data) >= 2:
                        first_close = data["Close"].iloc[0]
                        last_close = data["Close"].iloc[-1]
                        if first_close <= 0:
                            continue
                        change = ((last_close - first_close) / first_close) * 100
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
        # Pass patience="low" AND interval="15m" for True Intraday scans
        batch_results = analyze_batch(batch, patience="low", interval="15m")
        results.extend([r for r in batch_results if r is not None])
        if progress_callback:
            progress_callback((i + len(batch)) / len(stock_list))
        # Removed redundant sleep
    
    results_df = pd.DataFrame(results)
    if results_df.empty:
        return pd.DataFrame()
    
    # Ensure all required columns exist to avoid KeyError
    expected_cols = [
        "Symbol", "Score", "Current Price", "Recent Return", "Trend Persistence", "Latest Move %",
        "Previous Day Move %", "Overnight Gap %", "EMA20 Distance %", "Fresh Breakout Age", "RVOL", "Avg Volume Value",
        "Intraday", "Recommendation", "Buy At", "Stop Loss", "Target",
        "Ichimoku_Trend", "Major Trend Conflict", "Entry Type"
    ]
    for col in expected_cols:
        if col not in results_df.columns:
            results_df[col] = None 

    if "Score" not in results_df.columns:
        results_df["Score"] = 0

    sector_momentum = calculate_sector_momentum_map(results_df)
    nifty_5d_return = fetch_nifty_intraday_return()
    results_df["Score"] = pd.to_numeric(results_df["Score"], errors="coerce").fillna(0)
    results_df = results_df[results_df.apply(is_actionable_entry, axis=1)]
    results_df = results_df[results_df["Score"] >= MIN_TOP_PICK_SCORE]
        
    recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
    if recommendation_mode == "Adaptive":
        results_df = results_df[results_df["Recommendation"].str.contains("Buy", na=False)]
    else:
        results_df = results_df[results_df["Intraday"].str.contains("Buy", na=False)]
    results_df = add_entry_quality_columns(
        results_df,
        sector_momentum,
        nifty_5d_return,
        ranking_weights=INTRADAY_RANKING_WEIGHTS,
        intraday=True,
    )
    if results_df.empty:
        return pd.DataFrame()
    results_df = results_df[results_df.apply(is_intraday_quality_setup, axis=1)]
    results_df = results_df.sort_values(
        by=["Ranking Score", "Reward/Risk", "Score"],
        ascending=[False, False, False]
    )
    return limit_top_picks_by_sector(results_df, max_per_sector=2, limit=5)

def colored_recommendation(recommendation):
    if recommendation is None or not isinstance(recommendation, str):
        return "⚪ N/A"
    if "Buy" in recommendation:
        return f"🟢 {recommendation}"
    elif "Sell" in recommendation:
        return f"🔴 {recommendation}"
    else:
        return f"⚪ {recommendation}"

def clean_display_text(value, fallback="—"):
    if isinstance(value, tuple):
        value = value[0]
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except TypeError:
        pass
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return fallback
    return text

def is_valid_price(value):
    if isinstance(value, tuple):
        value = value[0]
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False

def to_float_or_none(value):
    if isinstance(value, tuple):
        value = value[0]
    if not is_valid_price(value):
        return None
    return float(value)

def to_number_or_none(value):
    if isinstance(value, tuple):
        value = value[0]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def calculate_recent_return(data, candles=5):
    if data.empty or "Close" not in data.columns or len(data) < 2:
        return np.nan
    window = data.tail(candles)
    first_close = to_float_or_none(window["Close"].iloc[0])
    last_close = to_float_or_none(window["Close"].iloc[-1])
    if not first_close or not last_close:
        return np.nan
    return ((last_close - first_close) / first_close) * 100

def calculate_trend_persistence_score(data, candles=TREND_PERSISTENCE_LOOKBACK):
    if data.empty or not {"High", "Low", "Close"}.issubset(data.columns) or len(data) < candles:
        return np.nan

    window = data.tail(candles).copy()
    highs = pd.to_numeric(window["High"], errors="coerce")
    lows = pd.to_numeric(window["Low"], errors="coerce")
    closes = pd.to_numeric(window["Close"], errors="coerce")
    if highs.isna().any() or lows.isna().any() or closes.isna().any():
        return np.nan

    candle_range = (highs - lows).replace(0, np.nan)
    close_location = ((closes - lows) / candle_range).clip(0, 1).mean()
    close_changes = closes.diff().dropna()
    if close_changes.empty:
        return np.nan

    advancing_close_rate = (close_changes > 0).mean()
    path_length = close_changes.abs().sum()
    smoothness = abs(closes.iloc[-1] - closes.iloc[0]) / path_length if path_length else 0.0
    smoothness = min(max(smoothness, 0.0), 1.0)

    persistence_score = (
        (close_location * 0.45)
        + (advancing_close_rate * 0.35)
        + (smoothness * 0.20)
    ) * 100
    return round(float(persistence_score), 1)

def calculate_momentum_extension_metrics(data):
    if data.empty or "Close" not in data.columns or len(data) < 2:
        return np.nan, np.nan

    previous_close = to_float_or_none(data["Close"].iloc[-2])
    current_close = to_float_or_none(data["Close"].iloc[-1])
    if not previous_close or not current_close:
        latest_move_pct = np.nan
    else:
        latest_move_pct = ((current_close - previous_close) / previous_close) * 100

    ema20_distance_pct = np.nan
    if "EMA_20" in data.columns:
        ema20 = to_float_or_none(data["EMA_20"].iloc[-1])
        if ema20 and current_close:
            ema20_distance_pct = ((current_close - ema20) / ema20) * 100

    return latest_move_pct, ema20_distance_pct

def calculate_session_gap_metrics(data):
    if data.empty or not {"Open", "Close"}.issubset(data.columns) or len(data) < 2:
        return np.nan, np.nan
    if not isinstance(data.index, pd.DatetimeIndex):
        return np.nan, np.nan

    session_df = data[["Open", "Close"]].copy()
    session_df["_SessionDate"] = session_df.index.date
    sessions = session_df.groupby("_SessionDate").agg({"Open": "first", "Close": "last"})
    sessions = sessions.dropna()
    if len(sessions) < 2:
        return np.nan, np.nan

    previous_session = sessions.iloc[-2]
    current_session = sessions.iloc[-1]
    previous_open = to_float_or_none(previous_session["Open"])
    previous_close = to_float_or_none(previous_session["Close"])
    current_open = to_float_or_none(current_session["Open"])
    if not previous_open or not previous_close or not current_open:
        return np.nan, np.nan

    previous_day_move_pct = ((previous_close - previous_open) / previous_open) * 100
    overnight_gap_pct = ((current_open - previous_close) / previous_close) * 100
    return previous_day_move_pct, overnight_gap_pct

def calculate_fresh_breakout_age(data, lookback=FRESH_BREAKOUT_LOOKBACK, max_age=FRESH_BREAKOUT_MAX_AGE):
    if data.empty or not {"High", "Close"}.issubset(data.columns):
        return np.nan
    if len(data) < lookback + max_age + 2:
        return np.nan

    highs = pd.to_numeric(data["High"], errors="coerce")
    closes = pd.to_numeric(data["Close"], errors="coerce")
    for age in range(1, max_age + 1):
        breakout_idx = len(data) - age
        previous_idx = breakout_idx - 1
        prior_start = breakout_idx - lookback
        previous_prior_start = previous_idx - lookback
        if prior_start < 0 or previous_prior_start < 0:
            continue

        prior_high = highs.iloc[prior_start:breakout_idx].max()
        previous_prior_high = highs.iloc[previous_prior_start:previous_idx].max()
        breakout_close = closes.iloc[breakout_idx]
        previous_close = closes.iloc[previous_idx]
        if pd.isna(prior_high) or pd.isna(previous_prior_high) or pd.isna(breakout_close) or pd.isna(previous_close):
            continue
        if breakout_close > prior_high and previous_close <= previous_prior_high:
            return age
    return np.nan

def calculate_volume_metrics(data):
    if data.empty or not {"Close", "Volume"}.issubset(data.columns):
        return np.nan, np.nan

    current_close = to_float_or_none(data["Close"].iloc[-1])
    current_volume = to_float_or_none(data["Volume"].iloc[-1])
    if not current_close or not current_volume:
        return np.nan, np.nan

    if "Avg_Volume" in data.columns:
        avg_volume = to_float_or_none(data["Avg_Volume"].iloc[-1])
    else:
        avg_volume = to_float_or_none(data["Volume"].tail(10).mean())

    rvol = current_volume / avg_volume if avg_volume else np.nan
    avg_volume_value = avg_volume * current_close if avg_volume else np.nan
    return rvol, avg_volume_value

def is_buy_signal(value):
    return isinstance(value, str) and "Buy" in value

def has_major_trend_conflict(row):
    conflict_flag = row.get("Major Trend Conflict", False)
    if conflict_flag is True or (
        isinstance(conflict_flag, str) and conflict_flag.strip().lower() == "true"
    ):
        return True
    if row.get("Ichimoku_Trend") != "Strong Sell":
        return False
    buy_signal_columns = (
        "Recommendation", "Intraday", "Swing", "Short-Term",
        "Long-Term", "Mean_Reversion", "Breakout"
    )
    return any(is_buy_signal(row.get(column)) for column in buy_signal_columns)

def is_actionable_entry(row, max_distance_pct=0.08, min_reward_risk=1.8):
    if has_major_trend_conflict(row):
        return False

    current_price = to_float_or_none(row.get("Current Price"))
    buy_at = to_float_or_none(row.get("Buy At"))
    stop_loss = to_float_or_none(row.get("Stop Loss"))
    target = to_float_or_none(row.get("Target"))
    if not all([current_price, buy_at, stop_loss, target]):
        return False
    distance_pct = abs(buy_at - current_price) / current_price
    risk = buy_at - stop_loss
    reward = target - buy_at
    if risk <= 0:
        return False
    reward_risk = reward / risk
    return (
        buy_at > stop_loss
        and target > buy_at
        and distance_pct <= max_distance_pct
        and reward_risk >= min_reward_risk
    )

def get_stock_sector(symbol):
    if not isinstance(symbol, str):
        return "Other"
    symbol = symbol.upper().strip()
    for sector, symbols in SECTORS.items():
        if symbol in symbols:
            return sector
    return "Other"

def calculate_sector_momentum_map(df):
    if df.empty or "Recent Return" not in df.columns:
        return {}
    momentum_df = df.copy()
    if "Sector" not in momentum_df.columns:
        momentum_df["Sector"] = momentum_df["Symbol"].apply(get_stock_sector)
    momentum_df["Recent Return"] = pd.to_numeric(momentum_df["Recent Return"], errors="coerce")
    momentum_df = momentum_df.dropna(subset=["Recent Return"])
    if momentum_df.empty:
        return {}
    return momentum_df.groupby("Sector")["Recent Return"].mean().to_dict()

def sector_momentum_adjustment(sector_perf):
    sector_perf = to_number_or_none(sector_perf)
    if sector_perf is None:
        return 0.0
    if sector_perf > 4:
        return 2.0
    if sector_perf > 2:
        return 1.5
    if sector_perf > 1:
        return 1.0
    if sector_perf < -1:
        return -1.0
    return 0.0

def relative_strength_adjustment(relative_strength):
    relative_strength = to_number_or_none(relative_strength)
    if relative_strength is None:
        return 0.0
    if relative_strength > 3:
        return 2.0
    if relative_strength > 1:
        return 1.0
    if relative_strength < -2:
        return -2.0
    return 0.0

def entry_distance_adjustment(distance_pct):
    distance_pct = to_number_or_none(distance_pct)
    if distance_pct is None:
        return 0.0
    if distance_pct <= 1:
        return 1.5
    if distance_pct <= 2:
        return 1.0
    if distance_pct <= 3:
        return 0.5
    return 0.0

def liquidity_adjustment(avg_volume_value):
    avg_volume_value = to_number_or_none(avg_volume_value)
    if avg_volume_value is None or avg_volume_value <= 0:
        return 0.0
    turnover_cr = avg_volume_value / 10_000_000
    if turnover_cr < 10:
        return 0.0
    if turnover_cr < 20:
        return 0.3
    if turnover_cr < 50:
        return 0.6
    if turnover_cr < 100:
        return 1.0
    if turnover_cr < 250:
        return 1.3
    if turnover_cr < 500:
        return 1.6
    return 2.0

def rvol_adjustment(rvol):
    rvol = to_number_or_none(rvol)
    if rvol is None:
        return 0.0
    if rvol > 2:
        return 2.0
    if rvol > 1.5:
        return 1.0
    return 0.0

def intraday_rvol_adjustment(rvol):
    rvol = to_number_or_none(rvol)
    if rvol is None:
        return 0.0
    if rvol > 3:
        return 2.0
    if rvol > 2:
        return 1.0
    return 0.0

def intraday_liquidity_factor(avg_volume_value):
    avg_volume_value = to_number_or_none(avg_volume_value)
    if avg_volume_value is None or avg_volume_value <= 0:
        return 0.0
    turnover_cr = avg_volume_value / 10_000_000
    if turnover_cr < MIN_INTRADAY_LIQUIDITY_CR:
        return 0.0
    return min(1.5, max(0.5, turnover_cr / 20))

def intraday_gap_risk_penalty(row):
    previous_day_move_pct = to_number_or_none(row.get("Previous Day Move %"))
    overnight_gap_pct = to_number_or_none(row.get("Overnight Gap %"))
    if previous_day_move_pct is not None and previous_day_move_pct > INTRADAY_GAP_RISK_MOVE_THRESHOLD:
        return -INTRADAY_GAP_RISK_PENALTY
    if overnight_gap_pct is not None and abs(overnight_gap_pct) > INTRADAY_OVERNIGHT_GAP_THRESHOLD:
        return -INTRADAY_GAP_RISK_PENALTY
    return 0.0

def fresh_breakout_bonus(row):
    breakout_age = to_number_or_none(row.get("Fresh Breakout Age"))
    if breakout_age is None:
        return 0.0
    return FRESH_BREAKOUT_DECAY_BONUSES.get(int(breakout_age), 0.0)

def sector_exhaustion_penalty(row, intraday=False):
    if intraday:
        return 0.0
    sector_perf = to_number_or_none(row.get("Sector Performance %"))
    if sector_perf is None or sector_perf <= SECTOR_EXHAUSTION_MOVE_THRESHOLD:
        return 0.0
    return -SECTOR_EXHAUSTION_RANKING_PENALTY

def trend_persistence_adjustment(row):
    trend_persistence = to_number_or_none(row.get("Trend Persistence"))
    if trend_persistence is None:
        return 0.0
    centered_score = (trend_persistence - 50.0) / 50.0
    adjustment = centered_score * MAX_TREND_PERSISTENCE_RANKING_ADJUSTMENT
    return round(
        max(
            -MAX_TREND_PERSISTENCE_RANKING_ADJUSTMENT,
            min(MAX_TREND_PERSISTENCE_RANKING_ADJUSTMENT, adjustment),
        ),
        2,
    )

def normalize_opportunity_score(raw_score):
    return OPPORTUNITY_SCORE_SCALE * (
        1 - np.exp(-np.maximum(raw_score, 0) / OPPORTUNITY_SCORE_CURVE_SCALE)
    )

def sector_leader_adjustment_columns(ranked_df):
    ranked_df["Sector Leader Score"] = 0.5
    ranked_df["Sector Leader Adjustment"] = 0.0
    if ranked_df.empty:
        return ranked_df

    metrics = ["Relative Strength", "Avg Volume Value", "Trend Persistence"]
    for metric in metrics:
        ranked_df[metric] = pd.to_numeric(ranked_df[metric], errors="coerce")

    for _, sector_df in ranked_df.groupby("Sector", dropna=False):
        if len(sector_df) < 2:
            index = sector_df.index[0]
            row = sector_df.iloc[0]
            relative_strength = to_number_or_none(row.get("Relative Strength")) or 0.0
            avg_volume_value = to_number_or_none(row.get("Avg Volume Value")) or 0.0
            trend_persistence = to_number_or_none(row.get("Trend Persistence")) or 0.0
            turnover_cr = avg_volume_value / 10_000_000
            singleton_score = (
                (1.0 if relative_strength >= 3 else 0.7 if relative_strength >= 1 else 0.5 if relative_strength > 0 else 0.0)
                + (1.0 if turnover_cr >= 100 else 0.8 if turnover_cr >= 50 else 0.6 if turnover_cr >= 20 else 0.4 if turnover_cr >= 10 else 0.0)
                + (1.0 if trend_persistence >= 75 else 0.7 if trend_persistence >= 65 else 0.5 if trend_persistence >= 55 else 0.0)
            ) / 3
            singleton_adjustment = max(
                0.0,
                (singleton_score - 0.5) * 2 * MAX_SECTOR_LEADER_RANKING_ADJUSTMENT,
            )
            ranked_df.loc[index, "Sector Leader Score"] = round(singleton_score, 2)
            ranked_df.loc[index, "Sector Leader Adjustment"] = round(singleton_adjustment, 2)
            continue

        metric_ranks = []
        for metric in metrics:
            values = sector_df[metric].fillna(sector_df[metric].min())
            if values.isna().all() or values.nunique(dropna=False) <= 1:
                metric_ranks.append(pd.Series(0.5, index=sector_df.index))
                continue
            zero_to_one_rank = (values.rank(method="average") - 1) / (len(values) - 1)
            metric_ranks.append(zero_to_one_rank)

        leader_score = sum(metric_ranks) / len(metric_ranks)
        leader_adjustment = (
            (leader_score - 0.5)
            * 2
            * MAX_SECTOR_LEADER_RANKING_ADJUSTMENT
        ).clip(
            lower=-MAX_SECTOR_LEADER_RANKING_ADJUSTMENT,
            upper=MAX_SECTOR_LEADER_RANKING_ADJUSTMENT,
        )
        ranked_df.loc[sector_df.index, "Sector Leader Score"] = leader_score.round(2)
        ranked_df.loc[sector_df.index, "Sector Leader Adjustment"] = leader_adjustment.round(2)

    return ranked_df

def momentum_exhaustion_penalty(row, intraday=False):
    rvol = to_number_or_none(row.get("RVOL"))
    latest_move_pct = to_number_or_none(row.get("Latest Move %"))
    ema20_distance_pct = to_number_or_none(row.get("EMA20 Distance %"))
    rvol_threshold = INTRADAY_EXHAUSTION_RVOL_THRESHOLD if intraday else EXHAUSTION_RVOL_THRESHOLD
    move_threshold = INTRADAY_EXHAUSTION_DAILY_MOVE_THRESHOLD if intraday else EXHAUSTION_DAILY_MOVE_THRESHOLD
    ema20_threshold = INTRADAY_EXHAUSTION_EMA20_DISTANCE_THRESHOLD if intraday else EXHAUSTION_EMA20_DISTANCE_THRESHOLD
    max_penalty = INTRADAY_MAX_EXHAUSTION_RANKING_PENALTY if intraday else MAX_EXHAUSTION_RANKING_PENALTY

    if rvol is None or rvol <= rvol_threshold:
        return 0.0

    move_excess = max(
        0.0,
        (latest_move_pct or 0.0) - move_threshold
    )
    ema_excess = max(
        0.0,
        (ema20_distance_pct or 0.0) - ema20_threshold
    )
    extension_excess = max(move_excess, ema_excess)
    if extension_excess <= 0:
        return 0.0

    penalty = min(
        max_penalty,
        ((rvol - rvol_threshold) * 0.25)
        + (extension_excess * 0.15)
    )
    return -round(penalty, 2)

def is_intraday_quality_setup(row):
    avg_volume_value = to_number_or_none(row.get("Avg Volume Value"))
    if avg_volume_value is None or avg_volume_value < MIN_INTRADAY_LIQUIDITY_VALUE:
        return False

    entry_type = str(row.get("Entry Type") or "").strip().lower()
    relative_strength = to_number_or_none(row.get("Relative Strength"))
    sector_perf = to_number_or_none(row.get("Sector Relative Strength %"))
    is_breakout = entry_type == "breakout"

    if relative_strength is None or relative_strength <= MIN_INTRADAY_RS:
        return False
    if sector_perf is None or sector_perf <= MIN_INTRADAY_SECTOR_RELATIVE_STRENGTH:
        return False
    if is_breakout and (relative_strength is None or relative_strength <= MIN_INTRADAY_BREAKOUT_RS):
        return False
    return True

def is_swing_quality_setup(row):
    avg_volume_value = to_number_or_none(row.get("Avg Volume Value"))
    sector_perf = to_number_or_none(row.get("Sector Relative Strength %"))
    weak_liquidity = avg_volume_value is None or avg_volume_value < MIN_SWING_LIQUIDITY_VALUE
    weak_sector = sector_perf is None or sector_perf < MIN_SWING_SECTOR_RELATIVE_STRENGTH
    return not (weak_liquidity and weak_sector)

def calculate_entry_metrics(row, max_distance_pct=0.08):
    current_price = to_float_or_none(row.get("Current Price"))
    buy_at = to_float_or_none(row.get("Buy At"))
    stop_loss = to_float_or_none(row.get("Stop Loss"))
    target = to_float_or_none(row.get("Target"))

    if not all([current_price, buy_at, stop_loss, target]):
        return pd.Series({
            "Entry Distance %": np.nan,
            "Reward/Risk": np.nan,
            "Entry Quality": 0.0
        })

    distance_pct = abs(buy_at - current_price) / current_price
    risk = buy_at - stop_loss
    reward = target - buy_at
    reward_risk = reward / risk if risk > 0 else np.nan
    entry_quality = max(0.0, 1.0 - (distance_pct / max_distance_pct))

    return pd.Series({
        "Entry Distance %": distance_pct * 100,
        "Reward/Risk": reward_risk,
        "Entry Quality": entry_quality
    })

def add_entry_quality_columns(
    df,
    sector_momentum=None,
    nifty_5d_return=0.0,
    ranking_weights=None,
    intraday=False,
):
    sector_momentum = sector_momentum or {}
    nifty_5d_return = to_number_or_none(nifty_5d_return) or 0.0
    ranking_weights = ranking_weights or RANKING_WEIGHTS
    ranked_df = df.copy()
    if "Symbol" not in ranked_df.columns:
        ranked_df["Symbol"] = None
    if ranked_df.empty:
        ranked_df["Entry Distance %"] = np.nan
        ranked_df["Reward/Risk"] = np.nan
        ranked_df["Entry Quality"] = 0.0
        ranked_df["Sector Performance %"] = np.nan
        ranked_df["Sector Momentum Score"] = 0.0
        ranked_df["Sector Relative Strength %"] = np.nan
        ranked_df["Relative Strength"] = np.nan
        ranked_df["Relative Strength Score"] = 0.0
        ranked_df["Entry Distance Score"] = 0.0
        ranked_df["Liquidity Score"] = 0.0
        ranked_df["RVOL Score"] = 0.0
        ranked_df["Effective RVOL"] = np.nan
        ranked_df["Intraday Liquidity Factor"] = 0.0
        ranked_df["Gap Risk Penalty"] = 0.0
        ranked_df["Fresh Breakout Bonus"] = 0.0
        ranked_df["Sector Exhaustion Penalty"] = 0.0
        ranked_df["Trend Persistence Adjustment"] = 0.0
        ranked_df["Sector Leader Score"] = 0.5
        ranked_df["Sector Leader Adjustment"] = 0.0
        ranked_df["Exhaustion Penalty"] = 0.0
        ranked_df["Raw Ranking Score"] = pd.Series(dtype=float)
        ranked_df["Ranking Score"] = pd.Series(dtype=float)
        ranked_df["Sector"] = pd.Series(dtype=object)
        return ranked_df
    metrics = ranked_df.apply(calculate_entry_metrics, axis=1)
    ranked_df = pd.concat([ranked_df, metrics], axis=1)
    ranked_df["Entry Distance %"] = pd.to_numeric(ranked_df["Entry Distance %"], errors="coerce")
    ranked_df = ranked_df[
        ranked_df["Entry Distance %"].notna()
        & (ranked_df["Entry Distance %"] <= MAX_RANKED_ENTRY_GAP_PERCENT)
    ].copy()
    ranked_df["Score"] = pd.to_numeric(ranked_df["Score"], errors="coerce").fillna(0)
    ranked_df["Sector"] = ranked_df["Symbol"].apply(get_stock_sector)
    ranked_df["Sector Performance %"] = ranked_df["Sector"].map(sector_momentum).fillna(0.0)
    ranked_df["Sector Relative Strength %"] = ranked_df["Sector Performance %"] - nifty_5d_return
    ranked_df["Sector Momentum Score"] = ranked_df["Sector Relative Strength %"].apply(sector_momentum_adjustment)
    ranked_df["Recent Return"] = pd.to_numeric(ranked_df["Recent Return"], errors="coerce")
    ranked_df["Relative Strength"] = ranked_df["Recent Return"] - nifty_5d_return
    ranked_df["Relative Strength Score"] = ranked_df["Relative Strength"].apply(relative_strength_adjustment)
    if "Trend Persistence" not in ranked_df.columns:
        ranked_df["Trend Persistence"] = np.nan
    ranked_df["Trend Persistence"] = pd.to_numeric(ranked_df["Trend Persistence"], errors="coerce")
    ranked_df["RVOL"] = pd.to_numeric(ranked_df["RVOL"], errors="coerce")
    ranked_df["Avg Volume Value"] = pd.to_numeric(ranked_df["Avg Volume Value"], errors="coerce")
    if "Latest Move %" not in ranked_df.columns:
        ranked_df["Latest Move %"] = np.nan
    if "EMA20 Distance %" not in ranked_df.columns:
        ranked_df["EMA20 Distance %"] = np.nan
    if "Fresh Breakout Age" not in ranked_df.columns:
        ranked_df["Fresh Breakout Age"] = np.nan
    ranked_df["Latest Move %"] = pd.to_numeric(ranked_df["Latest Move %"], errors="coerce")
    ranked_df["EMA20 Distance %"] = pd.to_numeric(ranked_df["EMA20 Distance %"], errors="coerce")
    ranked_df["Fresh Breakout Age"] = pd.to_numeric(ranked_df["Fresh Breakout Age"], errors="coerce")
    ranked_df["Entry Distance Score"] = ranked_df["Entry Distance %"].apply(entry_distance_adjustment)
    ranked_df["Liquidity Score"] = ranked_df["Avg Volume Value"].apply(liquidity_adjustment)
    if intraday:
        ranked_df["Intraday Liquidity Factor"] = ranked_df["Avg Volume Value"].apply(intraday_liquidity_factor)
        ranked_df["Effective RVOL"] = ranked_df["RVOL"] * ranked_df["Intraday Liquidity Factor"]
        ranked_df["RVOL Score"] = ranked_df["Effective RVOL"].apply(intraday_rvol_adjustment)
        ranked_df["Gap Risk Penalty"] = ranked_df.apply(intraday_gap_risk_penalty, axis=1)
    else:
        ranked_df["Intraday Liquidity Factor"] = 1.0
        ranked_df["Effective RVOL"] = ranked_df["RVOL"]
        ranked_df["RVOL Score"] = ranked_df["RVOL"].apply(rvol_adjustment)
        ranked_df["Gap Risk Penalty"] = 0.0
    ranked_df["Fresh Breakout Bonus"] = ranked_df.apply(fresh_breakout_bonus, axis=1)
    ranked_df["Sector Exhaustion Penalty"] = ranked_df.apply(
        lambda row: sector_exhaustion_penalty(row, intraday=intraday),
        axis=1,
    )
    ranked_df["Trend Persistence Adjustment"] = ranked_df.apply(trend_persistence_adjustment, axis=1)
    ranked_df = sector_leader_adjustment_columns(ranked_df)
    ranked_df["Exhaustion Penalty"] = ranked_df.apply(
        lambda row: momentum_exhaustion_penalty(row, intraday=intraday),
        axis=1,
    )
    raw_opportunity_score = (
        (ranked_df["Relative Strength Score"] * ranking_weights["relative_strength"])
        + (ranked_df["RVOL Score"] * ranking_weights["rvol"])
        + (ranked_df["Sector Momentum Score"] * ranking_weights["sector"])
        + (ranked_df["Liquidity Score"] * ranking_weights["liquidity"])
        + (ranked_df["Entry Distance Score"] * ranking_weights["entry"])
        + ranked_df["Fresh Breakout Bonus"]
        + ranked_df["Trend Persistence Adjustment"]
        + ranked_df["Sector Leader Adjustment"]
        + ranked_df["Sector Exhaustion Penalty"]
        + ranked_df["Exhaustion Penalty"]
        + ranked_df["Gap Risk Penalty"]
    )
    ranked_df["Raw Ranking Score"] = raw_opportunity_score.round(3)
    ranked_df["Ranking Score"] = normalize_opportunity_score(raw_opportunity_score).round(1)
    return ranked_df

def limit_top_picks_by_sector(df, max_per_sector=2, limit=5):
    if df.empty:
        return df.copy()

    selected_indices = []
    sector_counts = {}
    for index, row in df.iterrows():
        sector = row.get("Sector") or "Other"
        if sector_counts.get(sector, 0) >= max_per_sector:
            continue
        selected_indices.append(index)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        if len(selected_indices) >= limit:
            break

    if not selected_indices:
        return df.head(0).copy()
    return df.loc[selected_indices].reset_index(drop=True)

def format_currency(value):
    if isinstance(value, tuple):
        value = value[0]
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except TypeError:
        pass
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"₹{float(value):.2f}"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "n/a"}:
        return "N/A"
    if text.startswith("₹"):
        return text
    try:
        return f"₹{float(text):.2f}"
    except ValueError:
        return text

def format_number(value, decimals=2):
    value = to_number_or_none(value)
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"

def format_percent(value, decimals=2):
    value = to_number_or_none(value)
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}%"

def format_compact_currency(value):
    value = to_number_or_none(value)
    if value is None:
        return "N/A"
    if abs(value) >= 10_000_000:
        return f"₹{value / 10_000_000:.2f}Cr"
    if abs(value) >= 100_000:
        return f"₹{value / 100_000:.2f}L"
    return f"₹{value:.0f}"

def ranking_audit_text(row):
    exhaustion_penalty = to_number_or_none(row.get("Exhaustion Penalty")) or 0.0
    gap_risk_penalty = to_number_or_none(row.get("Gap Risk Penalty")) or 0.0
    fresh_breakout_bonus_value = to_number_or_none(row.get("Fresh Breakout Bonus")) or 0.0
    sector_exhaustion_penalty_value = to_number_or_none(row.get("Sector Exhaustion Penalty")) or 0.0
    trend_persistence_adjustment_value = to_number_or_none(row.get("Trend Persistence Adjustment")) or 0.0
    sector_leader_adjustment_value = to_number_or_none(row.get("Sector Leader Adjustment")) or 0.0
    exhaustion_text = ""
    if exhaustion_penalty < 0:
        exhaustion_text = (
            f" | Exhaustion: {format_number(exhaustion_penalty, 1)} "
            f"(Move: {format_percent(row.get('Latest Move %'))}, "
            f"EMA20 Gap: {format_percent(row.get('EMA20 Distance %'))})"
        )
    gap_risk_text = ""
    if gap_risk_penalty < 0:
        gap_risk_text = (
            f" | Gap Risk: {format_number(gap_risk_penalty, 1)} "
            f"(Prev Day: {format_percent(row.get('Previous Day Move %'))}, "
            f"Gap: {format_percent(row.get('Overnight Gap %'))})"
        )

    effective_rvol = to_number_or_none(row.get("Effective RVOL"))
    rvol = to_number_or_none(row.get("RVOL"))
    effective_rvol_text = ""
    if effective_rvol is not None and rvol is not None and abs(effective_rvol - rvol) > 0.01:
        effective_rvol_text = f", effective {format_number(effective_rvol)}"

    fresh_breakout_text = ""
    if fresh_breakout_bonus_value > 0:
        fresh_breakout_text = (
            f" | Fresh Breakout: +{format_number(fresh_breakout_bonus_value, 1)} "
            f"({format_number(row.get('Fresh Breakout Age'), 0)} candles)"
        )
    sector_exhaustion_text = ""
    if sector_exhaustion_penalty_value < 0:
        sector_exhaustion_text = (
            f" | Sector Exhaustion: {format_number(sector_exhaustion_penalty_value, 1)} "
            f"({format_percent(row.get('Sector Performance %'))})"
        )

    trend_persistence_text = ""
    if abs(trend_persistence_adjustment_value) > 0:
        trend_persistence_text = (
            f" | Persistence: {format_number(row.get('Trend Persistence'), 1)} "
            f"({format_number(trend_persistence_adjustment_value, 1)})"
        )

    sector_leader_text = ""
    if abs(sector_leader_adjustment_value) > 0:
        sector_leader_text = (
            f" | Sector Leader: {format_number(row.get('Sector Leader Score'), 2)} "
            f"({format_number(sector_leader_adjustment_value, 1)})"
        )

    return (
        f"Opportunity Score: {format_number(row.get('Ranking Score'))} | "
        f"RS: {format_percent(row.get('Relative Strength'))} "
        f"({format_number(row.get('Relative Strength Score'), 1)}) | "
        f"RVOL: {format_number(row.get('RVOL'))} "
        f"({format_number(row.get('RVOL Score'), 1)}{effective_rvol_text}) | "
        f"Sector: {row.get('Sector', 'Other')} "
        f"{format_percent(row.get('Sector Performance %'))} "
        f"(Rel: {format_percent(row.get('Sector Relative Strength %'))}) "
        f"({format_number(row.get('Sector Momentum Score'), 1)}) | "
        f"RR: {format_number(row.get('Reward/Risk'))} | "
        f"Entry Gap: {format_percent(row.get('Entry Distance %'))} "
        f"({format_number(row.get('Entry Distance Score'), 1)}) | "
        f"Liquidity: {format_compact_currency(row.get('Avg Volume Value'))} "
        f"({format_number(row.get('Liquidity Score'), 1)})"
        f"{fresh_breakout_text}"
        f"{trend_persistence_text}"
        f"{sector_leader_text}"
        f"{sector_exhaustion_text}"
        f"{exhaustion_text}"
        f"{gap_risk_text}"
    )

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
    st.session_state.selected_sectors = [
        sector for sector in st.session_state.selected_sectors if sector in sector_options
    ]
    selected_sectors = st.sidebar.multiselect(
        "Select Sectors",
        options=sector_options,
        key="selected_sectors",
        help="Choose one or more sectors to analyze. Select 'All' to include all sectors."
    )

    if "All" in selected_sectors:
        selected_stocks = list(set([stock for sector in SECTORS.values() for stock in sector]))
    else:
        selected_stocks = list(set([stock for sector in selected_sectors for stock in SECTORS.get(sector, [])]))

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
                        {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: {format_currency(current_price)}  
                        Buy At: {format_currency(buy_at)} | Stop Loss: {format_currency(stop_loss)}  
                        Target: {format_currency(target)}  
                        **Audit**: {ranking_audit_text(row)}
                        Recommendation: {colored_recommendation(row.get('Recommendation', 'N/A'))}  
                        Regime: {row.get('Regime', 'N/A')}  
                        Position Size (₹): {row.get('Position Size', 'N/A')}  
                        Trailing Stop: ₹{row.get('Trailing Stop', 'N/A')}  
                        Reason: {row.get('Reason', 'N/A')}
                        """)
                    else:
                        st.markdown(f"""
                        {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: {format_currency(current_price)}  
                        Buy At: {format_currency(buy_at)} | Stop Loss: {format_currency(stop_loss)}  
                        Target: {format_currency(target)}  
                        **Audit**: {ranking_audit_text(row)}
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
                        {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: {format_currency(current_price)}  
                        {buy_icon} {buy_label}: {format_currency(buy_at)} | Stop Loss: {format_currency(stop_loss)}  
                        Target: {format_currency(target)}  
                        **Audit**: {ranking_audit_text(row)}
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
                        {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: {format_currency(current_price)}  
                        {buy_icon} {buy_label}: {format_currency(buy_at)} | Stop Loss: {format_currency(stop_loss)}  
                        Target: {format_currency(target)}  
                        **Audit**: {ranking_audit_text(row)}
                        Intraday: {colored_recommendation(row.get('Intraday', 'N/A'))}
                        
                        **Strategy Notes:**
                        {clean_display_text(row.get('Pattern Notes'))}
                        
                        **Entry Advice:**
                        {clean_display_text(row.get('Entry Strategy'))}
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
                st.metric(tooltip("Current Price", TOOLTIPS['RSI']), format_currency(current_price))
            with col2:
                buy_at = recommendations.get('Buy At', 'N/A')
                entry_type = recommendations.get('Entry Type', 'Standard')
                if isinstance(buy_at, tuple):
                    buy_at, tuple_entry_type = buy_at
                    entry_type = tuple_entry_type or entry_type
                label = "Buy At"
                if entry_type == "Breakout":
                    label = "🟢 Buy Above"
                elif entry_type == "Pullback":
                    label = "🔵 Buy Pullback"
                elif entry_type == "Choppy":
                    label = "⚠️ No Trade"
                    buy_at = "Choppy"
                
                st.metric(label, format_currency(buy_at))
            with col3:
                stop_loss = recommendations.get('Stop Loss', 'N/A')
                st.metric(tooltip("Stop Loss", TOOLTIPS['Stop Loss']), format_currency(stop_loss))
            with col4:
                target = recommendations.get('Target', 'N/A')
                st.metric("Target", format_currency(target))
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
            # Monte Carlo/GARCH is intentionally opt-in. Streamlit evaluates all tabs on
            # each rerun, so running this unconditionally slows down batch scans.
            st.subheader("🎲 Monte Carlo Projection (30 Days)")
            if st.button("Run Monte Carlo / GARCH Projection", key=f"mc_projection_{symbol}"):
                with st.spinner("Running Monte Carlo / GARCH projection..."):
                    simulations = monte_carlo_simulation(data)
                    sim_df = pd.DataFrame(simulations).T
                    sim_df.index = [data.index[-1] + timedelta(days=i) for i in range(len(sim_df))]
                    fig_sim = px.line(sim_df, title="Price Projections")
                    st.plotly_chart(fig_sim, use_container_width=True)
            else:
                st.caption("Optional advanced analysis. Skipped during scans.")

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

    if st.sidebar.button("Analyze Selected Stock"):
        if symbol:
            with st.spinner("Loading stock data..."):
                data = fetch_stock_data_with_auth(symbol)
                if not data.empty:
                    data = analyze_stock(data, interval="1d")
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
