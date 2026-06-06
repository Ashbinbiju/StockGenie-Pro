import pandas as pd
import ta
import threading
import logging
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from tqdm import tqdm
import plotly.express as px
import time
import requests
import io
import random
import shutil
import spacy
import tempfile
from pytrends.request import TrendReq
import numpy as np
import itertools
from arch import arch_model
import warnings
import sqlite3
import base64
from diskcache import Cache
from SmartApi import SmartConnect
import pyotp
import os
from dotenv import load_dotenv
from streamlit import cache_data

load_dotenv()
APP_TIMEZONE = ZoneInfo("Asia/Kolkata")
DEFAULT_DB_PATH = Path(__file__).resolve().with_name("stock_picks.db")

def resolve_database_path():
    configured_path = (
        os.getenv("STOCKGENIE_DB_PATH")
        or os.getenv("DATABASE_PATH")
        or os.getenv("DB_PATH")
    )
    if not configured_path:
        try:
            configured_path = (
                st.secrets.get("STOCKGENIE_DB_PATH")
                or st.secrets.get("DATABASE_PATH")
                or st.secrets.get("DB_PATH")
            )
        except Exception:
            configured_path = None

    if not configured_path:
        return DEFAULT_DB_PATH

    db_path = Path(str(configured_path)).expanduser()
    if not db_path.is_absolute():
        db_path = Path(__file__).resolve().parent / db_path
    return db_path

DB_PATH = resolve_database_path()
SYMBOL_ALIASES = {
    "DBREALTY-EQ": "DBREALTY-BE",
    "INDIGRID-EQ": "INDIGRID-IV",
    "MINDSPACE-EQ": "MINDSPACE-RR",
    "SPICEJET-EQ": "SPICEJET",
    "STLTECH-EQ": "STLTECH-BE",
}

def app_now():
    return datetime.now(APP_TIMEZONE)

def app_date_string():
    return app_now().strftime('%Y-%m-%d')

def app_timestamp_string():
    return app_now().strftime('%Y-%m-%d %H:%M:%S')

def get_db_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

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

def github_history_backup_config():
    token = get_config_value("STOCKGENIE_HISTORY_GITHUB_TOKEN", "HISTORY_GITHUB_TOKEN")
    repo = get_config_value(
        "STOCKGENIE_HISTORY_GITHUB_REPO",
        "HISTORY_GITHUB_REPO",
        "GITHUB_REPOSITORY",
    )
    if not token or not repo:
        return None

    return {
        "token": token,
        "repo": str(repo).strip(),
        "branch": (
            get_config_value("STOCKGENIE_HISTORY_GITHUB_BRANCH", "HISTORY_GITHUB_BRANCH")
            or "history"
        ),
        "path": (
            get_config_value("STOCKGENIE_HISTORY_GITHUB_PATH", "HISTORY_GITHUB_PATH")
            or "stock_picks.db"
        ),
    }

def github_history_backup_url(config):
    return f"https://api.github.com/repos/{config['repo']}/contents/{config['path']}"

def github_history_backup_headers(config):
    return {
        "Authorization": f"Bearer {config['token']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

def local_history_row_count():
    if not DB_PATH.exists():
        return 0
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        has_table = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'daily_picks'"
        ).fetchone()
        if not has_table:
            return 0
        return int(conn.execute("SELECT COUNT(*) FROM daily_picks").fetchone()[0] or 0)
    except sqlite3.Error as e:
        logging.warning(f"Failed to inspect local history database at {DB_PATH}: {str(e)}")
        return 0
    finally:
        if conn is not None:
            conn.close()

def fetch_github_history_backup():
    config = github_history_backup_config()
    if not config:
        return None

    try:
        response = requests.get(
            github_history_backup_url(config),
            headers=github_history_backup_headers(config),
            params={"ref": config["branch"]},
            timeout=20,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.json()
        content = payload.get("content")
        if not content:
            return None
        return base64.b64decode(content), payload.get("sha")
    except Exception as e:
        logging.warning(f"Failed to fetch GitHub history backup: {str(e)}")
        return None

def restore_history_backup_if_empty():
    if local_history_row_count() > 0:
        return False

    backup = fetch_github_history_backup()
    if not backup:
        return False

    backup_bytes, _ = backup
    tmp_path = DB_PATH.with_suffix(f"{DB_PATH.suffix}.restore.tmp")
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(backup_bytes)
        conn = sqlite3.connect(tmp_path)
        try:
            conn.execute("SELECT COUNT(*) FROM daily_picks").fetchone()
        finally:
            conn.close()
        shutil.move(str(tmp_path), str(DB_PATH))
        logging.info(f"Restored historical picks database from GitHub backup to {DB_PATH}")
        return True
    except Exception as e:
        logging.warning(f"Failed to restore GitHub history backup: {str(e)}")
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False

def sync_history_backup():
    config = github_history_backup_config()
    if not config or not DB_PATH.exists():
        return False

    try:
        existing_sha = None
        get_response = requests.get(
            github_history_backup_url(config),
            headers=github_history_backup_headers(config),
            params={"ref": config["branch"]},
            timeout=20,
        )
        if get_response.status_code == 200:
            existing_sha = get_response.json().get("sha")
        elif get_response.status_code != 404:
            get_response.raise_for_status()

        payload = {
            "message": f"Update StockGenie history backup {app_timestamp_string()}",
            "content": base64.b64encode(DB_PATH.read_bytes()).decode("ascii"),
            "branch": config["branch"],
        }
        if existing_sha:
            payload["sha"] = existing_sha

        put_response = requests.put(
            github_history_backup_url(config),
            headers=github_history_backup_headers(config),
            json=payload,
            timeout=30,
        )
        if put_response.status_code == 404:
            logging.warning(
                "GitHub history backup branch/path was not found. "
                "Create the configured branch or update STOCKGENIE_HISTORY_GITHUB_BRANCH."
            )
            return False
        put_response.raise_for_status()
        return True
    except Exception as e:
        logging.warning(f"Failed to sync GitHub history backup: {str(e)}")
        return False

def history_storage_notice():
    config = github_history_backup_config()
    if config:
        return (
            f"History backup: GitHub {config['repo']}@{config['branch']}:"
            f"{config['path']}"
        )
    if DB_PATH != DEFAULT_DB_PATH:
        return f"History database: {DB_PATH}"
    return (
        "History is stored in local SQLite only. On Streamlit Cloud this can reset "
        "after the app sleeps or restarts; configure persistent storage for durable history."
    )

@st.cache_data(ttl=86400)
def load_symbol_token_map():
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        token_map = {entry["symbol"]: entry["token"] for entry in data if "symbol" in entry and "token" in entry}
        for requested_symbol, tradable_symbol in SYMBOL_ALIASES.items():
            if tradable_symbol in token_map:
                token_map[requested_symbol] = token_map[tradable_symbol]
        return token_map
    except Exception as e:
        st.warning(f"⚠️ Failed to load instrument list: {str(e)}")
        return {}

@st.cache_data(ttl=86400)
def load_symbol_exchange_map():
    try:
        url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        exchange_map = {entry["symbol"]: entry["exch_seg"] for entry in data if "symbol" in entry and "exch_seg" in entry}
        for requested_symbol, tradable_symbol in SYMBOL_ALIASES.items():
            if tradable_symbol in exchange_map:
                exchange_map[requested_symbol] = exchange_map[tradable_symbol]
        return exchange_map
    except Exception:
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

def create_runtime_cache():
    cache_dir = Path(
        os.environ.get(
            "STOCKGENIE_CACHE_DIR",
            Path(tempfile.gettempdir()) / "stockgenie_stock_data_cache",
        )
    )
    try:
        return Cache(str(cache_dir))
    except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
        logging.warning(f"Resetting invalid stock data cache at {cache_dir}: {str(e)}")
        shutil.rmtree(cache_dir, ignore_errors=True)
        return Cache(str(cache_dir))

cache = create_runtime_cache()
smartapi_auth_error = None
smartapi_auth_lock = threading.Lock()
NIFTY_50_TOKEN = "99926000"
MIN_TOP_PICK_SCORE = 5
PUBLIC_SHARE_MIN_GRADE = "B"
PUBLIC_SHARE_ALLOWED_SWING = ["Buy", "Strong Buy"]
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
MIN_SWING_PULLBACK_ENTRY_GAP_PERCENT = 0.5
MIN_SWING_CONSOLIDATION_CANDLES = 3
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
HOLDING_PERIOD_DAYS = [1, 2, 3, 5, 10, 20]
MIN_HOLDING_PERIOD_SAMPLE_SIZE = 3
SETUP_EVIDENCE_MEDIUM_SAMPLE_SIZE = 15
SETUP_EVIDENCE_HIGH_SAMPLE_SIZE = 50
MAX_HISTORICAL_EXPECTANCY_RANKING_ADJUSTMENT = 0.6
MAX_SETUP_EXPECTANCY_RANKING_ADJUSTMENT = 0.5
WEAK_MARKET_REGIME_SCORE_MULTIPLIER = 0.90
MARKET_REGIME_BULL_BREADTH_THRESHOLD = 60.0
MARKET_REGIME_WEAK_BREADTH_THRESHOLD = 50.0
STRICT_WEAK_MARKET_SIGNAL_BREADTH_THRESHOLD = 30.0
WEAK_INDUSTRY_ADVANCE_RATIO_THRESHOLD = 0.20
WEAK_INDUSTRY_SECTOR_LEADER_MULTIPLIER = 0.50
MAX_WEAK_INDUSTRY_SECTOR_LEADER_ADJUSTMENT = 0.20
WEAK_INDUSTRY_BREADTH_PENALTY = -0.30
BANK_WEAK_MARKET_SECTOR_SCORE_PENALTY = -0.40
PROBABILITY_TARGET_LEVELS = [2, 4, 6]
DEFAULT_OPTIMAL_HOLD_DAYS_BY_SETUP = {
    "fresh_breakout": 5,
    "sector_leader_continuation": 8,
    "mean_reversion_bounce": 3,
    "high_rvol_explosive": 2,
    "slow_institutional_trend": 15,
    "trend_continuation": 8,
}
SETUP_TYPE_LABELS = {
    "fresh_breakout": "Fresh Breakout",
    "sector_leader_continuation": "Sector Leader",
    "high_rvol_explosive": "High RVOL",
    "trend_continuation": "Trend Continuation",
    "mean_reversion_bounce": "Mean Reversion Bounce",
    "slow_institutional_trend": "Slow Institutional Trend",
}
MARKET_STATS_INDUSTRY_ALIASES = {
    "Auto": "Automobile and Auto Components",
    "Bank": "Financial Services",
    "ConstructionMaterials": "Construction Materials",
    "CapitalGoods": "Capital Goods",
    "Chemicals": "Chemicals",
    "FMCG": "Fast Moving Consumer Goods",
    "Healthcare": "Healthcare",
    "IT": "Information Technology",
    "ConsumerDurables": "Consumer Durables",
    "Jewellery": "Consumer Durables",
    "Electricals": "Capital Goods",
    "Agri": "Agricultural Food & other Products",
    "Hospitality": "Consumer Services",
    "Textiles": "Textiles",
    "Industrial_Gases_Fuels": "Oil Gas & Consumable Fuels",
    "Logistics": "Services",
    "Alcohol": "Fast Moving Consumer Goods",
    "Plastic": "Chemicals",
    "ShipBuilding": "Capital Goods",
    "Media": "Media Entertainment & Publication",
    "Metals": "Metals & Mining",
    "OilGas": "Oil Gas & Consumable Fuels",
    "Power": "Power",
    "RealEstate": "Realty",
    "Telecom": "Telecommunication",
}
EXIT_STATUS_PRIORITY = {
    "HOLD": 0,
    "TRAIL_SL": 1,
    "BOOK_PARTIAL": 2,
    "EXIT_WARNING": 3,
    "EXIT": 4,
}

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
        "HDFCBANK-EQ", "ICICIBANK-EQ", "SBIN-EQ", "AXISBANK-EQ", "KOTAKBANK-EQ",
        "BANKBARODA-EQ", "UNIONBANK-EQ", "CANBK-EQ", "PNB-EQ", "INDIANB-EQ",
        "IDBI-EQ", "FEDERALBNK-EQ", "YESBANK-EQ", "AUBANK-EQ", "INDUSINDBK-EQ",
        "BANKINDIA-EQ", "IOB-EQ", "IDFCFIRSTB-EQ", "MAHABANK-EQ", "BANDHANBNK-EQ",
        "UCOBANK-EQ", "CENTRALBK-EQ", "KARURVYSYA-EQ", "RBLBANK-EQ", "CUB-EQ",
        "PSB-EQ", "J&KBANK-EQ", "TMB-EQ", "SOUTHBANK-EQ", "UJJIVANSFB-EQ",
        "KTKBANK-EQ", "EQUITASBNK-EQ", "CSBBANK-EQ", "DCBBANK-EQ", "JSFB-EQ",
        "UTKARSHBNK-EQ", "SURYODAY-EQ", "ESAFSFB-EQ", "DHANBANK-EQ", "CAPITALSFB-EQ",
        "FINOPB-EQ"
    ],
    "IT": [
        "TCS-EQ", "INFY-EQ", "HCLTECH-EQ", "WIPRO-EQ", "TECHM-EQ",
        "LTIM-EQ", "OFSS-EQ", "PERSISTENT-EQ", "COFORGE-EQ", "MPHASIS-EQ",
        "HEXAWARE-EQ", "TATAELXSI-EQ", "KPITTECH-EQ", "AFFLE-EQ", "FSL-EQ",
        "ECLERX-EQ", "ZENSARTECH-EQ", "INTELLECT-EQ", "CYIENT-EQ", "BSOFT-EQ",
        "RATEGAIN-EQ", "SONATSOFTW-EQ", "NEWGEN-EQ", "TANLA-EQ", "LATENTVIEW-EQ",
        "HAPPSTMNDS-EQ", "MASTEK-EQ", "CMSINFO-EQ", "DATAMATICS-EQ", "MAPMYINDIA-EQ",
        "JUSTDIAL-EQ", "AURIONPRO-EQ", "EMUDHRA-EQ", "SASKEN-EQ", "ROUTE-EQ",
        "NIITMTS-EQ", "RSYSTEMS-EQ", "NETWEB-EQ", "PROTEAN-EQ", "RAMCOSYS-EQ",
        "NUCLEUS-EQ", "INFOBEAN-EQ", "DYNPRO-EQ", "IZMO-EQ", "EXPLEOSOL-EQ",
        "NIITLTD-EQ", "GENESYS-EQ", "QUICKHEAL-EQ", "ONWARDTEC-EQ", "ALLDIGI-EQ",
        "MOSCHIP-EQ", "SAKSOFT-EQ", "63MOONS-EQ", "ZAGGLE-EQ", "BLS-EQ",
        "CEINSYSTECH-EQ", "TAC-EQ", "CYBERTECH-EQ", "MINDTECK-EQ", "KSOLVES-EQ",
        "APTECHT-EQ", "IRIS-EQ", "TECHD-EQ", "ALLIEDDIGI-EQ"
    ],
    "Finance": [
        "BAJFINANCE-EQ", "BAJAJFINSV-EQ", "SHRIRAMFIN-EQ", "JIOFIN-EQ", "PFC-EQ",
        "CHOLAFIN-EQ", "MUTHOOTFIN-EQ", "IRFC-EQ", "BSE-EQ", "HDFCAMC-EQ",
        "ABCAPITAL-EQ", "RECLTD-EQ", "MCX-EQ", "NAM-INDIA-EQ", "BAJAJHFL-EQ",
        "LTF-EQ", "SBICARD-EQ", "MAXFIN-EQ", "MOTILALOFS-EQ", "SUNDARMFIN-EQ",
        "PIRAMAL-EQ", "360ONE-EQ", "HUDCO-EQ", "M&MFIN-EQ", "POONAWALLA-EQ",
        "ANGELONE-EQ", "ABSLAMC-EQ", "LICHSGFIN-EQ", "NUVAMA-EQ", "MANAPPURAM-EQ",
        "PNBHOUSING-EQ", "CDSL-EQ", "IIFL-EQ", "IFCI-EQ", "AADHARHFC-EQ",
        "CREDITACC-EQ", "CAMS-EQ", "NSDL-EQ", "KFINTECH-EQ", "CHOICEIN-EQ",
        "APTUS-EQ", "UTIAMC-EQ", "JMFINANCIL-EQ", "HOMEFIRST-EQ", "EDELWEISS-EQ",
        "CANFINHOME-EQ", "IIFLCAPS-EQ", "SBFC-EQ", "AAVAS-EQ", "INDIASHLTR-EQ",
        "FEDFINA-EQ", "MASFIN-EQ", "NORTHERNARC-EQ", "SGFIN-EQ", "AYEFIN-EQ",
        "MUTHOOTMF-EQ", "SHAREINDIA-EQ", "FUSION-EQ", "SATIN-EQ", "MUFIN-EQ",
        "REPCOHOME-EQ", "GEOJITFSL-EQ", "PNBGILTS-EQ", "ARMANFIN-EQ", "UGROCAP-EQ",
        "FINOPB-EQ"
    ],
    "Auto": [
        "MARUTI-EQ", "M&M-EQ", "BAJAJ-AUTO-EQ", "EICHERMOT-EQ", "TVSMOTOR-EQ",
        "CUMMINSIND-EQ", "HYUNDAI-EQ", "MOTHERSON-EQ", "TATAMOTORS-EQ", "BOSCHLTD-EQ",
        "HEROMOTOCO-EQ", "ASHOKLEY-EQ", "SCHAEFFLER-EQ", "UNOMINDA-EQ", "TIINDIA-EQ",
        "MRF-EQ", "BALKRISIND-EQ", "ATHERENERG-EQ", "SONACOMS-EQ", "ENDURANCE-EQ",
        "EXIDEIND-EQ", "ESCORTS-EQ", "TIMKEN-EQ", "ZFCVINDIA-EQ", "APOLLOTYRE-EQ",
        "FORCEMOT-EQ", "SANSERA-EQ", "CASTROLIND-EQ", "CIEINDIA-EQ", "JBMA-EQ",
        "ARE&M-EQ", "MINDACORP-EQ", "SHRIPISTON-EQ", "GABRIEL-EQ", "CEATLTD-EQ",
        "SKFINDIA-EQ", "LUMAXTECH-EQ", "JKTYRE-EQ", "RKFORGE-EQ", "BANCOINDIA-EQ",
        "VARROC-EQ", "ASKAUTOLTD-EQ", "PRICOLLTD-EQ", "SUPRAJIT-EQ", "SJS-EQ",
        "FIEMIND-EQ", "SMLISUZU-EQ", "LUMAXIND-EQ", "LGBBROSLTD-EQ", "JAMNAAUTO-EQ",
        "SUBROS-EQ", "SHARDAMOTR-EQ", "SWARAJENG-EQ", "GULFOILLUB-EQ", "SANDHAR-EQ",
        "VSTTILLERS-EQ", "NRBBEARING-EQ", "WHEELS-EQ", "JTEKTINDIA-EQ", "SHANTIGEAR-EQ",
        "TVSSRICHAK-EQ", "UNIPARTS-EQ", "AUTOAXLES-EQ", "RML-EQ", "RAJRATAN-EQ",
        "PIXTRANS-EQ", "MMFL-EQ", "TALBROAUTO-EQ", "INDNIPPON-EQ", "RICOAUTO-EQ",
        "GOODYEAR-EQ", "GNAXLES-EQ", "RACLGEAR-EQ", "PRECAM-EQ", "JAYBARMARU-EQ",
        "ATULAUTO-EQ", "BHARATSE-EQ", "KROSS-EQ", "ALICON-EQ", "MUNJALAU-EQ",
        "MENONBE-EQ", "INDOFARM-EQ", "KINETICENG-EQ", "MUNJALSHOW-EQ", "TOLINS-EQ",
        "JAYUSHIN-EQ", "REMSONSIND-EQ", "PPAP-EQ", "BHARATGEAR-EQ", "KRANTI-EQ",
        "AMBAAUTO-EQ", "IPRINGLTD-EQ"
    ],
    "Healthcare": [
        "SUNPHARMA-EQ", "DIVISLAB-EQ", "TORNTPHARM-EQ", "APOLLOHOSP-EQ", "CIPLA-EQ",
        "ZYDUSLIFE-EQ", "DRREDDY-EQ", "LUPIN-EQ", "MANKIND-EQ", "MAXHEALTH-EQ",
        "AUROPHARMA-EQ", "LAURUSLABS-EQ", "FORTIS-EQ", "BIOCON-EQ", "ALKEM-EQ",
        "GLENMARK-EQ", "ABBOTINDIA-EQ", "IPCALAB-EQ", "NH-EQ", "ASTERDM-EQ",
        "GLAND-EQ", "AJANTPHARM-EQ", "GLAXO-EQ", "JBCHEPHARM-EQ", "MEDANTA-EQ",
        "EMCURE-EQ", "KIMS-EQ", "WOCKPHARMA-EQ", "LALPATHLAB-EQ", "PPLPHARMA-EQ",
        "PFIZER-EQ", "GRANULES-EQ", "ERIS-EQ", "SYNGENE-EQ", "NATCOPHARM-EQ",
        "JUBLPHARMA-EQ", "CAPLIPOINT-EQ", "APLLTD-EQ", "POLYMED-EQ", "VIJAYA-EQ",
        "RAINBOW-EQ", "INDGN-EQ", "METROPOLIS-EQ", "MARKSANS-EQ", "SHILPAMED-EQ",
        "STARHEALTH-EQ", "THYROCARE-EQ", "SUPRIYA-EQ", "FDC-EQ", "KOVAI-EQ",
        "AARTIPHARM-EQ", "BLISSGVS-EQ", "ADVENZYMES-EQ", "IOLCP-EQ", "GUFICBIO-EQ",
        "INDRAMEDCO-EQ", "AARTIDRUGS-EQ", "PANACEABIO-EQ", "MEDIASSIST-EQ", "UNICHEMLAB-EQ",
        "VIMTALABS-EQ", "HIKAL-EQ", "SHALBY-EQ", "KRSNAA-EQ", "WINDLAS-EQ",
        "HESTERBIO-EQ", "TTKHLTCARE-EQ", "LINCOLN-EQ", "GPTHEALTH-EQ", "KOPRAN-EQ",
        "INDOCO-EQ", "JAGSNPHARM-EQ"
    ],
    "Metals": [
        "JSWSTEEL-EQ", "COALINDIA-EQ", "TATASTEEL-EQ", "HINDALCO-EQ", "HINDZINC-EQ",
        "VEDL-EQ", "JSL-EQ", "LLOYDSME-EQ", "NMDC-EQ", "SAIL-EQ",
        "NATIONALUM-EQ", "HINDCOPPER-EQ", "APLAPOLLO-EQ", "WELCORP-EQ",
        "SHYAMMETL-EQ", "KIOCL-EQ", "GMDC-EQ", "SARDAEN-EQ", "GPIL-EQ",
        "RATNAMANI-EQ", "JINDALSAW-EQ", "USHAMART-EQ", "GRAVITA-EQ", "JAYNECOIND-EQ",
        "MAHSEAMLES-EQ", "MIDHANI-EQ", "IMFA-EQ", "KIRLFER-EQ", "ASHAPURMIN-EQ",
        "SUNFLAG-EQ", "MOIL-EQ", "TECHNOE-EQ", "BANSALWIRE-EQ", "DEEDEV-EQ",
        "ELECTCAST-EQ", "GOODLUCK-EQ", "KSL-EQ", "DECGOLD-EQ", "SAMBHV-EQ",
        "VENUSPIPES-EQ", "MAITHANALL-EQ", "JTLIND-EQ", "PRAKASH-EQ", "PENIND-EQ",
        "MUKANDLTD-EQ", "JAICORPLTD-EQ", "HITECH-EQ", "ARFIN-EQ", "HARIOMPIPE-EQ",
        "NELCAST-EQ", "RATNAVEER-EQ", "MANAKALUCO-EQ", "SHERA-EQ", "BEDMUTHA-EQ"
    ],
    "FMCG": [
        "HINDUNILVR-EQ", "NESTLEIND-EQ", "VBL-EQ", "BRITANNIA-EQ", "MARICO-EQ",
        "GODREJCP-EQ", "DABUR-EQ", "VMM-EQ", "COLPAL-EQ", "GODFRYPHLP-EQ",
        "PGHH-EQ", "JUBLFOOD-EQ", "GILLETTE-EQ", "AWL-EQ", "HATSUN-EQ",
        "EMAMILTD-EQ", "BIKAJI-EQ", "ZYDUSWELL-EQ", "DEVYANI-EQ", "HONASA-EQ",
        "LTFOODS-EQ", "GODREJAGRO-EQ", "ORKLAIND-EQ", "EUREKAFORB-EQ", "CELLO-EQ",
        "SAFARI-EQ", "JYOTHYLAB-EQ", "BAJAJCON-EQ", "SFL-EQ", "DODLA-EQ",
        "HNDFDS-EQ", "BECTORFOOD-EQ", "RBA-EQ", "VIPIND-EQ", "VSTIND-EQ",
        "VADILALIND-EQ", "WAKEFIT-EQ", "ORIENTELEC-EQ", "GOPAL-EQ", "ADFFOODS-EQ",
        "FLAIR-EQ", "HERITGFOOD-EQ", "PARAGMILK-EQ", "DIAMONDYD-EQ", "TASTYBITE-EQ",
        "UNITDSPR-EQ", "GULPOLY-EQ", "KRISHIVAL-EQ", "LOTUSCHOCO-EQ", "IFBAGRO-EQ",
        "KSE-EQ", "SUKHJITS-EQ", "SHEETAL-EQ", "SHREEOSFM-EQ", "MEGASTAR-EQ",
        "RAMDEVBABA-EQ", "CHATHA-EQ", "MILKFOOD-EQ", "BAMBINO-EQ", "SRIVARU-EQ"
    ],
    "Power": [
        "ADANIPOWER-EQ", "NTPC-EQ", "POWERGRID-EQ", "ADANIENSOL-EQ", "TATAPOWER-EQ",
        "JSWENERGY-EQ", "NTPCGREEN-EQ", "NHPC-EQ", "TORNTPOWER-EQ", "NLCINDIA-EQ",
        "IREDA-EQ", "SJVN-EQ", "ACMESOLAR-EQ", "CESC-EQ", "NAVA-EQ",
        "IEX-EQ", "GMRP&UI-EQ", "PTC-EQ", "SWSOLAR-EQ", "KIRLOSIND-EQ",
        "RELINFRA-EQ", "GIPCL-EQ", "BFUTILITIE-EQ", "RPOWER-EQ", "REPL-EQ",
        "DANISHPOWER-EQ", "APS-EQ", "GITARENEW-EQ"
    ],
    "CapitalGoods": [
        "ABB-EQ", "SIEMENS-EQ", "BHEL-EQ", "CGPOWER-EQ", "HAVELLS-EQ",
        "THERMAX-EQ", "APARINDS-EQ", "AIAENG-EQ", "LTTS-EQ", "SCHNEIDER-EQ",
        "KIRLOSKAROIL-EQ", "SYRMA-EQ", "TRITURBINE-EQ", "TDPOWERSYS-EQ", "ELGIEQUIP-EQ",
        "INOXWIND-EQ", "BEML-EQ", "KSB-EQ", "TEGA-EQ", "GRAPHITE-EQ",
        "KIRLOSBROS-EQ", "VGUARD-EQ", "ENGINERSIN-EQ", "KEC-EQ", "INGERRAND-EQ",
        "ELECON-EQ", "ACE-EQ", "KIRLPNU-EQ", "VOLTAMP-EQ", "GENUSPOWER-EQ",
        "KPIGREEN-EQ", "ISGEC-EQ", "SHAKTIPUMP-EQ", "PRAJIND-EQ", "SKIPPER-EQ",
        "IONEXCHANG-EQ", "GREAVESCOT-EQ", "IDEAFORGE-EQ", "HIRECT-EQ", "HARSHA-EQ",
        "CYIENTDLM-EQ", "GMMPFAUDLR-EQ", "PITTIENG-EQ", "BHARATBIJLEE-EQ", "INDOTECH-EQ",
        "HLEGLAS-EQ", "HONDAPOWER-EQ", "BAJEL-EQ", "HPL-EQ", "RISHABH-EQ",
        "ADORWELD-EQ", "THEJO-EQ", "EMSLIMITED-EQ", "IFGLEXPOR-EQ", "EVERESTIND-EQ",
        "ROTOPUMP-EQ", "TEMBO-EQ", "YUKEN-EQ", "WPIL-EQ"
    ],
    "OilGas": [
        "RELIANCE-EQ", "ONGC-EQ", "IOC-EQ", "BPCL-EQ", "HINDPETRO-EQ",
        "OIL-EQ", "PETRONET-EQ", "MRPL-EQ", "CHENNPETRO-EQ", "DEEPINDS-EQ",
        "ANTELOPE-EQ", "VEEDOL-EQ", "HOEC-EQ", "PRABHA-EQ", "ASIANENE-EQ",
        "JINDRILL-EQ", "DOLPHIN-EQ", "GANDHAR-EQ", "GUJGASLTD-EQ", "IRMENERGY-EQ",
        "CONTPETRO-EQ"
    ],
    "Chemicals": [
        "ASIANPAINT-EQ", "SOLARINDS-EQ", "PIDILITIND-EQ", "BERGEPAINT-EQ", "FACT-EQ",
        "UPL-EQ", "COROMANDEL-EQ", "PIIND-EQ", "FLUOROCHEM-EQ", "NAVINFLUOR-EQ",
        "GODREJIND-EQ", "HSCL-EQ", "SUMICHEM-EQ", "DEEPAKNTR-EQ", "ATUL-EQ",
        "BAYERCROP-EQ", "CHAMBLFERT-EQ", "TATACHEM-EQ", "DEEPAKFERT-EQ", "KANSAINER-EQ",
        "AARTIIND-EQ", "ANURAS-EQ", "AETHER-EQ", "BASF-EQ", "FINEORG-EQ",
        "PARADEEP-EQ", "SUPPETRO-EQ", "PRIVISCL-EQ", "PCBL-EQ",
        "JUBLINGREA-EQ", "ALKYLAMINE-EQ", "CLEAN-EQ", "SHARDACROP-EQ", "GNFC-EQ",
        "SUDARSCHEM-EQ", "RCF-EQ", "RAIN-EQ", "BALAMINES-EQ", "INDIAGLYCO-EQ",
        "GSFC-EQ", "AWHCL-EQ", "GALAXYSURF-EQ", "NEOGEN-EQ", "EPIGRAL-EQ",
        "DHANUKA-EQ", "INDIGOPNTS-EQ", "RALLIS-EQ", "VISHNU-EQ", "TANFACIND-EQ",
        "STYRENIX-EQ", "NFL-EQ", "PONDYOXIDE-EQ", "ROSSARI-EQ", "NOCIL-EQ",
        "CAMLINFINE-EQ", "BHANSALIENG-EQ", "SIRCA-EQ", "BHARATRAS-EQ", "INSECTICID-EQ"
    ],
    "Telecom": [
        "BHARTIARTL-EQ", "INDUSTOWER-EQ", "BHARTIHEXA-EQ", "TATACOMM-EQ", "ITI-EQ",
        "HFCL-EQ", "BBOX-EQ", "TEJASNET-EQ", "RAILTEL-EQ", "AVANTEL-EQ",
        "OPTIEMUS-EQ", "EXICOM-EQ", "NELCO-EQ", "VINDHYATEL-EQ"
    ],
    "Infrastructure": [
        "LT-EQ", "ADANIPORTS-EQ", "GMRAIRPORT-EQ", "JSWINFRA-EQ", "RVNL-EQ",
        "KPIL-EQ", "IRCON-EQ", "TECHNOE-EQ", "AFCONS-EQ", "RITES-EQ",
        "WABAG-EQ", "NCC-EQ", "GRINFRA-EQ", "POWERMECH-EQ", "CEIGALL-EQ",
        "GEPIL-EQ", "PNCINFRA-EQ", "MANINFRA-EQ", "KNRCON-EQ", "JKIL-EQ",
        "HGINFRA-EQ", "ASHOKA-EQ", "SANGHVIMOV-EQ", "RAMKY-EQ", "BONDADA-EQ",
        "BGRENERGY-EQ", "ADVAIT-EQ", "JNKINDIA-EQ", "KPEL-EQ", "EPACK-EQ",
        "SPMLINFRA-EQ", "INDIANHUME-EQ", "GPTINFRA-EQ", "RELINFRA-EQ", "LIKHITHA-EQ",
        "OMINFRA-EQ", "GANESHGREEN-EQ", "DENTA-EQ", "OMPOWER-EQ", "GOEL-EQ",
        "BRAHMINFRA-EQ", "MARKOLINES-EQ", "GANESHINFRA-EQ", "KINGSINFRA-EQ", "VVIPIL-EQ",
        "SAVY-EQ", "UNIVASTU-EQ", "AVPINFRA-EQ", "DESCO-EQ", "AKANKSHA-EQ",
        "SAFETY-EQ", "EXPO-EQ", "WINSOL-EQ", "HECINFRA-EQ", "OVAL-EQ",
        "RUDRABHISHEK-EQ", "KAYCEE-EQ", "LAKSHYA-EQ", "RAMATELE-EQ", "SRINIBAS-EQ",
        "TELGE-EQ"
    ],
    "Insurance": [
        "LICI-EQ", "SBILIFE-EQ", "HDFCLIFE-EQ", "ICICIGI-EQ", "POLICYBZR-EQ",
        "ICICIPRULI-EQ", "GICRE-EQ", "GODIGIT-EQ", "NIACL-EQ", "NIVABUPA-EQ",
        "CANFINHOME-EQ"
    ],
    "Diversified": [
        "ITC-EQ", "GRASIM-EQ", "SRF-EQ", "LINDEINDIA-EQ", "ADITYAINFO-EQ",
        "3MINDIA-EQ", "TATATECH-EQ", "STARHEALTH-EQ", "DCMSHRIRAM-EQ", "BIRLACORPN-EQ",
        "PRSMJOHNSN-EQ", "SURYAROSNI-EQ", "TRUALTBIO-EQ", "BALMLAWRIE-EQ", "JASH-EQ",
        "SHK-EQ", "GODAVARIB-EQ", "TEXINFRA-EQ", "ORICONENT-EQ", "EXHICON-EQ",
        "SAAKSHI-EQ", "SKP-EQ", "EKI-EQ", "GILLANDERS-EQ", "INFOLLION-EQ",
        "HINDPRAKASH-EQ", "KAIRA-EQ", "DEFRAIL-EQ"
    ],
    "ConstructionMaterials": [
        "ULTRACEMCO-EQ", "AMBUJACEM-EQ", "SHREECEM-EQ", "JKCEMENT-EQ", "DALBHARAT-EQ",
        "ACC-EQ", "ASAHIINDIA-EQ", "RAMCOCEM-EQ", "JSWCEMENT-EQ", "KAJARIACER-EQ",
        "CENTURYPLY-EQ", "INDIACEM-EQ", "NUVOCO-EQ", "STARCEMENT-EQ", "JKLAKSHMI-EQ",
        "BORORENEW-EQ", "CERA-EQ", "GREENLAM-EQ", "STYLAMIND-EQ", "GREENPLY-EQ",
        "HEIDELBERG-EQ", "ORIENTCEM-EQ", "BOROLTD-EQ", "POKARNA-EQ", "GREENPANEL-EQ",
        "SAGCEM-EQ", "MANGLMCEM-EQ", "NITCO-EQ", "SOMANYCERA-EQ", "KCP-EQ",
        "LAOPALA-EQ", "GARUDA-EQ", "SAINTGOBAIN-EQ", "SHREDIGCEM-EQ", "ORIENTCER-EQ",
        "DECCANCE-EQ", "SEJALLTD-EQ", "NCLIND-EQ", "VISAKAIND-EQ", "EVERESTIND-EQ",
        "HALDYNGL-EQ", "ORIENTBELL-EQ", "ANJANICEM-EQ", "SAHYADRI-EQ", "KESHAV-EQ",
        "MARBLECITY-EQ", "AGARWALTUF-EQ", "ARCHIDPLY-EQ", "DUROPLY-EQ", "KAKATCEM-EQ",
        "PACIFICI-EQ", "PANYAMCEM-EQ", "BANARBEADS-EQ", "AIROLAM-EQ", "ALFAICA-EQ",
        "RAVILEELA-EQ", "EMPIREIND-EQ", "HRSALUGLA-EQ"
    ],
    "RealEstate": [
        "DLF-EQ", "LODHA-EQ", "PHOENIXLTD-EQ", "PRESTIGE-EQ", "OBEROIRLTY-EQ",
        "KNOWLEDGE-EQ", "GODREJPROP-EQ", "NBCC-EQ", "ANANTRAJ-EQ", "BRIGADE-EQ",
        "SOBHA-EQ", "ABREL-EQ", "SIGNATURE-EQ", "WEWORK-EQ", "WELENT-EQ",
        "MAXESTATES-EQ", "MAHLIFE-EQ", "DBL-EQ", "GANESHHOUC-EQ", "KALPATARU-EQ",
        "VALORESTATE-EQ", "AHLUCONT-EQ", "PURVA-EQ", "RUSTOMJEE-EQ", "AGIIL-EQ",
        "SUNTECK-EQ", "HEMIPROP-EQ", "RAYMONDREAL-EQ", "TARC-EQ", "ASHIANA-EQ",
        "PSPPROJECT-EQ", "RAYMOND-EQ", "KOLTEPATIL-EQ", "HUBTOWN-EQ", "MARATHON-EQ",
        "ARVSMART-EQ", "AJMERA-EQ", "ARKADE-EQ", "CAPACITE-EQ", "SHRIRAMPPS-EQ",
        "OMAXE-EQ", "ARIHANTSUP-EQ", "SURAJEST-EQ", "ELDEHSG-EQ", "GEECEE-EQ",
        "EMAMIREAL-EQ", "RDBREAL-EQ", "DEEPAKBLD-EQ", "CHAVDA-EQ", "SVGLOBAL-EQ",
        "SABTNL-EQ", "VEER-EQ", "SAMOR-EQ", "HBESTATE-EQ", "ALPHAGEO-EQ",
        "GARNET-EQ", "THAKDEV-EQ", "RODIUM-EQ", "YURANUS-EQ", "HOMESFY-EQ"
    ],
    "Aviation": [
        "BEL-EQ", "HAL-EQ", "INDIGO-EQ", "BDL-EQ", "DATAPATTNS-EQ",
        "MTARTECH-EQ", "APOLLO-EQ", "AZAD-EQ", "AXISCADES-EQ", "UNIMECH-EQ",
        "ROSSTECH-EQ", "NIBE-EQ", "JAYKAY-EQ", "SIKA-EQ", "TAALENT-EQ",
        "TAANEJA-EQ", "FLYSBS-EQ", "C2C-EQ", "GLOBALVECT-EQ"
    ],
    "Miscellaneous": [
        "ADANIGREEN-EQ", "NAUKRI-EQ", "CRISIL-EQ", "EMMVEE-EQ", "CUMMINSIND-EQ",
        "ZENTEC-EQ", "JYOTICNC-EQ", "INOXINDIA-EQ", "ASTRAMICRO-EQ", "DOMS-EQ",
        "WAAREERTL-EQ", "PARAS-EQ", "SIS-EQ", "SAATVIK-EQ", "SMARTWORKS-EQ",
        "ICRA-EQ", "CARERATING-EQ", "KNRCON-EQ", "DBCORP-EQ", "QUESS-EQ",
        "INDIQUBE-EQ", "NAVNETEDUL-EQ", "MPSLTD-EQ", "GKENERGY-EQ", "TEAMLEASE-EQ",
        "AWFIS-EQ", "VERANDA-EQ", "SANSTAR-EQ", "MOBAVENUE-EQ", "KRISHNADEF-EQ",
        "SOLARWORLD-EQ", "JAGRAN-EQ", "BOROSCI-EQ", "UDS-EQ", "ANTONYWASTE-EQ",
        "JAROEDU-EQ", "SOLEX-EQ", "CREATIVE-EQ", "MAMATA-EQ", "KOKUYOCMLN-EQ",
        "KRYSTAL-EQ", "IIRM-EQ", "SANDESH-EQ", "HINDMEDIA-EQ", "LINC-EQ",
        "SCHAND-EQ", "RKSWAMY-EQ", "ZODIAC-EQ", "KOTYARK-EQ", "CAREERPOINT-EQ"
    ],
    "Retail": [
        "DMART-EQ", "TRENT-EQ", "LENSKART-EQ", "MEESHO-EQ", "ABLBL-EQ",
        "INDIAMART-EQ", "MEDPLUS-EQ", "MANYAVAR-EQ", "V2RETAIL-EQ", "ABFRL-EQ",
        "AVL-EQ", "VMART-EQ", "SHOPERSTOP-EQ", "GOCOLORS-EQ", "CANTABIL-EQ",
        "SAISILKS-EQ", "PNGSGARGI-EQ", "FOCE-EQ", "PATELRETAIL-EQ", "PURPLEUTED-EQ",
        "ENCOMPASS-EQ", "BRANDMAN-EQ", "TRIDENT-EQ", "SILGO-EQ", "ACETECH-EQ",
        "SDRETAIL-EQ", "JAYJALARAM-EQ", "GUJJUBHAI-EQ", "WOMANCART-EQ", "ONDOOR-EQ",
        "LUXURYTIME-EQ"
    ],
    "Jewellery": [
        "TITAN-EQ", "KALYANKJIL-EQ", "THANGAMAYL-EQ", "IGIL-EQ", "BLUESTONE-EQ",
        "SKYGOLD-EQ", "PNGJL-EQ", "SENCO-EQ", "GOLDIAM-EQ", "VAIBHAVGBL-EQ",
        "RAJESHEXPO-EQ", "DPABHUSHAN-EQ", "SHRINGAR-EQ", "SHANTIGOLD-EQ", "KHAZANCHI-EQ",
        "PNGSREVA-EQ", "TRIBHOVDAS-EQ", "RHL-EQ", "ASIANSTAR-EQ", "MVGJL-EQ",
        "UTSSAV-EQ", "RBZJEWEL-EQ", "UDAYJEW-EQ", "DIVINEHIRA-EQ", "PUSHPAJWL-EQ",
        "GROVJEW-EQ", "GOLKUNDA-EQ", "EYANTRA-EQ", "YAASHVI-EQ", "KABRAJEWELS-EQ",
        "CEENIK-EQ", "NANAVATI-EQ", "AJCJEWEL-EQ", "SHOORA-EQ"
    ],
    "Trading": [
        "ADANIENT-EQ", "REDINGTON-EQ", "SUNDRMFAST-EQ", "FIRSTCRY-EQ", "MMTC-EQ",
        "LLOYDSENT-EQ", "MSTC-EQ", "EFC-EQ", "STYLEBAAZA-EQ", "LANDMARK-EQ",
        "TCC-EQ", "MACFOS-EQ", "AMIRCHAND-EQ", "AEROFLEX-EQ", "HEXATRADEX-EQ",
        "BIZOTIC-EQ", "UNIPHOS-EQ", "STCINDIA-EQ", "VASA-EQ", "SMT-EQ",
        "SUNLITE-EQ", "OCCL-EQ", "NUPURRE-EQ", "KOTHARIPRO-EQ", "VERITAS-EQ",
        "JINDWORLD-EQ", "MANBRO-EQ", "ASTAL-EQ", "AGRIBIO-EQ", "BALGOPAL-EQ",
        "MESON-EQ", "INDIANCHAIN-EQ", "COMPAUTO-EQ", "REVATHI-EQ", "MACOBS-EQ",
        "RACE-EQ", "AUSOMENT-EQ", "CHEMKART-EQ", "RAMAVISION-EQ", "THACKER-EQ",
        "POSITRON-EQ", "DHOOTIN-EQ", "WHBRADY-EQ", "ROCKINGDCE-EQ", "TCIIND-EQ",
        "CHEMTECH-EQ", "ZENITHEXPO-EQ", "MRPAGRO-EQ", "WOL3D-EQ", "AUREATETEC-EQ",
        "GAUTAMEXIM-EQ", "BOMCYC-EQ", "KHAITANLTD-EQ", "FORTUNE-EQ", "CHPL-EQ",
        "LELAVOIR-EQ"
    ],
    "Media": [
        "ZEEL-EQ", "SUNTV-EQ", "TVTODAY-EQ",
        "SAREGAMA-EQ"
    ],
    "ConsumerDurables": [
        "LGELECTRON-EQ", "DIXON-EQ", "VOLTAS-EQ", "BLUESTARCO-EQ", "HONAUT-EQ",
        "AMBER-EQ", "NETWEB-EQ", "CGPOWER-EQ", "PGEL-EQ", "WHIRLPOOL-EQ",
        "TTKPRESTIG-EQ", "ETHOSLTD-EQ", "IFBIND-EQ", "GNG-EQ", "SYMPHONY-EQ",
        "TIMEX-EQ", "EMIL-EQ", "HAWKINCOOK-EQ", "BOSCHLTD-EQ", "RPTECH-EQ",
        "BAJAJELEC-EQ", "KDDL-EQ", "EPACK-EQ", "STOVEKRAFT-EQ", "HINDWAREAP-EQ",
        "DLINKINDIA-EQ", "WONDERLA-EQ", "ICEMAKE-EQ", "BUTTERFLY-EQ", "VIRTUOSO-EQ",
        "CONTROLPR-EQ", "TVSELECT-EQ", "OSELDEVICE-EQ", "PRIZOR-EQ", "CWD-EQ",
        "SINGER-EQ", "ARHAM-EQ", "PELATRO-EQ", "EMMFORCE-EQ", "EPWINDIA-EQ",
        "UMIYA-EQ", "ADTECH-EQ"
    ],
    "Electricals": [
        "POLYCAB-EQ", "KEI-EQ", "STLTECH-EQ", "RRKABEL-EQ", "KAYNES-EQ",
        "FINCABLES-EQ", "AVALON-EQ", "DIACABS-EQ", "OLECTRA-EQ", "PRECWIRE-EQ",
        "KSH-EQ", "CENTUM-EQ", "WEBELSOLAR-EQ", "UNIVCABLES-EQ", "VMARCIND-EQ",
        "MARINE-EQ", "AIMTRON-EQ", "INA-EQ", "DCXINDIA-EQ", "PARACABLES-EQ",
        "VIDYAWIRES-EQ", "INDOSOLAR-EQ", "VINYAS-EQ", "DYCL-EQ", "QUADFUTURE-EQ",
        "RIR-EQ", "IKIO-EQ", "SALZERELEC-EQ", "SWELECTES-EQ", "VILAS-EQ",
        "SAHASRA-EQ", "SPEL-EQ", "BIRLACABLE-EQ", "SUPREMEPWR-EQ", "JDCABLES-EQ",
        "QUALITEK-EQ", "DELTON-EQ", "DYNAMIC-EQ", "CORDSCABLE-EQ", "BCCFUBA-EQ",
        "DHANASHREE-EQ", "JASCH-EQ", "COSMOFE-EQ", "PRIMECABLE-EQ", "EPPELTONE-EQ",
        "BHADORA-EQ", "INCAP-EQ", "MAGNUS-EQ", "FINELINE-EQ", "SWITCHTECH-EQ"
    ],
    "Agri": [
        "TATACONSUM-EQ", "PATANJALI-EQ", "AVANTIFEED-EQ", "CCL-EQ", "EIDPARRY-EQ",
        "BALRAMCHIN-EQ", "BBTC-EQ", "MANORAMA-EQ", "KRBL-EQ", "TRIVENI-EQ",
        "GAEL-EQ", "GOKULAGRO-EQ", "BANARISUG-EQ", "KSCL-EQ", "GRMOVER-EQ",
        "SUNDROP-EQ", "MVKAGRO-EQ", "VINCOFE-EQ", "VENKEYS-EQ", "CLSEL-EQ",
        "SKMEGGPROD-EQ", "APEX-EQ", "AVTNPL-EQ", "BOMDYEING-EQ", "UTTAMSUGAR-EQ",
        "DHAMPURSUG-EQ", "AVADHSUGAR-EQ", "REGAAL-EQ", "APISINDUS-EQ", "MCLEODRUSS-EQ",
        "DBOL-EQ", "MAGADSUGAR-EQ", "MODINATUR-EQ", "VENREF-EQ", "KNAGRI-EQ",
        "PAJSONS-EQ", "KRITINUT-EQ", "MAWANASUG-EQ", "GOODRICKE-EQ", "HARRMALAYA-EQ",
        "TRANSCHEM-EQ", "SAYAJIIND-EQ", "NATHBIOGEN-EQ", "HALDER-EQ", "PONNIERODE-EQ",
        "HOACFOODS-EQ", "UNITEDTEA-EQ", "INTEGPROT-EQ", "PERIAKARAM-EQ", "JAYSREETEA-EQ",
        "BHARATRAS-EQ", "SHIVAAGRO-EQ", "ROSSELLIND-EQ", "VIJAYSOLV-EQ", "NEELAMALAI-EQ",
        "INDOUS-EQ", "OVOBELE-EQ", "JKAGRIGEN-EQ", "DHUNTEAIND-EQ", "ELFIN-EQ",
        "INDIASUCRO-EQ", "MANGALSEEDS-EQ", "MAYANKCATTLE-EQ", "BANDA-EQ", "JAMESWARREN-EQ",
        "VISTAAR-EQ", "RUKMADEVI-EQ", "KLRF-EQ", "NATURITE-EQ", "GUJAPOLLO-EQ",
        "AGRITECH-EQ", "SIMRANFARMS-EQ", "TERAITEA-EQ", "UNIQUEORG-EQ", "NKIND-EQ",
        "POONADAL-EQ", "PEARLGREEN-EQ", "KANCOTEA-EQ", "TYROONTEA-EQ", "SUNILAGRO-EQ",
        "COCHMAL-EQ", "TGIFAGRO-EQ"
    ],
    "Hospitality": [
        "INDHOTEL-EQ", "ITCHOTELS-EQ", "EIHOTEL-EQ", "CHALET-EQ", "TRAVELFOOD-EQ",
        "VENTIVE-EQ", "SCHLOSS-EQ", "TBOTEK-EQ", "BLS-EQ", "LEMONTREE-EQ",
        "WESTLIFE-EQ", "IXIGO-EQ", "INDTOUR-EQ", "JUNIPER-EQ", "THOMASCOOK-EQ",
        "SAMHI-EQ", "WONDERLA-EQ", "BRIGADEHOT-EQ", "TAJGVK-EQ",
        "ORIENTHOT-EQ", "EIHAHOTELS-EQ", "YATRA-EQ", "BENARAS-EQ", "ASIANHOTNR-EQ",
        "VICEROY-EQ", "ROHLTD-EQ", "UPHOTELS-EQ", "ECOSMOBLTY-EQ", "ADVANIHOTR-EQ",
        "KAMATHOTEL-EQ", "SPECIALITY-EQ", "SAYAJIHOTL-EQ", "SINCLAIR-EQ", "NICCOPARK-EQ",
        "APOLSINHOT-EQ", "GRANDCONT-EQ", "ROBUSTHOT-EQ", "SUBHOTELS-EQ", "ASIANENE-EQ",
        "EMERALD-EQ", "SAYAJIPUNE-EQ", "ITDCEM-EQ", "VEGORAMA-EQ", "WISETRAVEL-EQ",
        "HBGHOTELS-EQ", "SAVERA-EQ", "KHFM-EQ", "GUJHOTELS-EQ", "JINDHOT-EQ",
        "YAANENT-EQ", "WESTLEISURE-EQ"
    ],
    "Textiles": [
        "PAGEIND-EQ", "KPRMILL-EQ", "VTL-EQ", "ARVIND-EQ", "PGIL-EQ",
        "INDOCOUNT-EQ", "GOKEX-EQ", "PDSL-EQ", "MAYURUNIQ-EQ", "SIYSIL-EQ",
        "SANGAMIND-EQ", "SPAL-EQ", "RUPA-EQ", "AMBIKCO-EQ"
    ],
    "Industrial_Gases_Fuels": [
        "ATGL-EQ", "GUJGASLTD-EQ", "IGL-EQ", "MGL-EQ", "REFEX-EQ",
        "CONFIPET-EQ", "AGARIND-EQ", "ELLEN-EQ"
    ],
    "Logistics": [
        "CONCOR-EQ", "DELHIVERY-EQ", "AEGISLOG-EQ", "AEGISVOPAK-EQ", "GESHIP-EQ",
        "SCI-EQ", "BLUEDART-EQ", "GPPL-EQ", "TCI-EQ", "TVSSCS-EQ",
        "VRLLOG-EQ", "SEAMECLTD-EQ", "MAHLOG-EQ", "DREDGECORP-EQ", "TCIEXP-EQ",
        "NAVKARCORP-EQ", "RITCO-EQ", "GLOTTIS-EQ", "SICALLOG-EQ", "IWARE-EQ",
        "AVG-EQ", "OMFREIGHT-EQ", "SADHAV-EQ", "ASPINWALL-EQ", "RAPIDFLEET-EQ",
        "CROWN-EQ", "SANCO-EQ", "KESARTERM-EQ", "REPONO-EQ", "HARIYANA-EQ",
        "ABCINDQ-EQ"
    ],
    "Alcohol": [
        "MCDOWELL-N-EQ", "RADICO-EQ", "UBL-EQ", "ABDL-EQ", "TI-EQ",
        "PICCADIL-EQ", "GLOBUSSPR-EQ", "GMBREW-EQ", "SDBL-EQ", "ASALCBR-EQ",
        "SULA-EQ", "VINATIORGA-EQ", "JAGAJITIND-EQ", "MONIKA-EQ", "NORSPIRITS-EQ",
        "ALCOBEX-EQ", "SILVEROAK-EQ"
    ],
    "Plastic": [
        "SUPREMEIND-EQ", "ASTRAL-EQ", "SHAILY-EQ", "GARFIBRES-EQ", "FINPIPE-EQ",
        "TIMETECHNO-EQ", "KINGFA-EQ", "EPL-EQ", "CARYSIL-EQ", "PRINCEPIPE-EQ",
        "DDEVPLSTIK-EQ", "MOLDTKPAC-EQ", "APOLLOPIPE-EQ", "NILKAMAL-EQ", "PLATIND-EQ",
        "ARROWGREEN-EQ"
    ],
    "ShipBuilding": [
        "MAZDOCK-EQ", "COCHINSHIP-EQ", "GRSE-EQ"
    ],
    "Defence": [
        "HAL-EQ", "BEL-EQ", "BDL-EQ", "PARAS-EQ", "BEML-EQ",
        "DATAPATTNS-EQ", "AZAD-EQ", "SOLARINDS-EQ"
    ],
    "Consumer Services": [
        "ETERNAL-EQ", "NYKAA-EQ", "ADANIPORTS-EQ", "IRCTC-EQ", "PAYTM-EQ", "INDHOTEL-EQ", "NAUKRI-EQ",
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
        exchange = load_symbol_exchange_map().get(symbol, "NSE")

        # Enforce rate limit before making the API call
        enforce_rate_limit()

        # Retry logic for API instability
        for attempt in range(3):
            try:
                if not smart_api or not hasattr(smart_api, "getCandleData"):
                    logging.error(f"SmartAPI client unavailable for {symbol}; skipping candle request.")
                    return pd.DataFrame()

                historical_data = smart_api.getCandleData({
                    "exchange": exchange,
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
                    if not smart_api:
                        logging.error(f"SmartAPI re-login failed after invalid token for {symbol}; skipping candle request.")
                        return pd.DataFrame()
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

def nifty_regime_snapshot_from_closes(closes, source):
    close_series = pd.to_numeric(pd.Series(closes), errors="coerce").dropna()
    if len(close_series) < 50:
        return {}

    close = float(close_series.iloc[-1])
    ema20 = float(close_series.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(close_series.ewm(span=50, adjust=False).mean().iloc[-1])
    return {
        "nifty_close": close,
        "nifty_ema20": ema20,
        "nifty_ema50": ema50,
        "nifty_above_ema20": close > ema20,
        "nifty_above_ema50": close > ema50,
        "nifty_regime_source": source,
    }

@st.cache_data(ttl=1800)
def fetch_yahoo_nifty_regime_snapshot():
    try:
        response = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI",
            params={"range": "6mo", "interval": "1d"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        result = (
            payload.get("chart", {})
            .get("result", [{}])[0]
        )
        closes = (
            result.get("indicators", {})
            .get("quote", [{}])[0]
            .get("close", [])
        )
        return nifty_regime_snapshot_from_closes(closes, "Yahoo")
    except Exception as e:
        logging.warning(f"Failed to compute fallback Yahoo NIFTY market regime: {str(e)}")
        return {}

@st.cache_data(ttl=1800)
def fetch_nifty_regime_snapshot():
    try:
        auth_error = get_smartapi_auth_error()
        if auth_error:
            logging.warning(f"Skipping NIFTY regime request: {auth_error}")
            return fetch_yahoo_nifty_regime_snapshot()

        smart_api = get_smartapi_session(API_KEYS["Historical"], CLIENT_ID, PASSWORD, TOTP_SECRET)
        if not smart_api:
            return fetch_yahoo_nifty_regime_snapshot()

        end_date = datetime.now()
        start_date = end_date - timedelta(days=120)
        enforce_rate_limit()
        historical_data = smart_api.getCandleData({
            "exchange": "NSE",
            "symboltoken": NIFTY_50_TOKEN,
            "interval": "ONE_DAY",
            "fromdate": start_date.strftime("%Y-%m-%d %H:%M"),
            "todate": end_date.strftime("%Y-%m-%d %H:%M")
        })

        if not historical_data or not isinstance(historical_data, dict) or not historical_data.get("data"):
            return fetch_yahoo_nifty_regime_snapshot()

        data = pd.DataFrame(historical_data["data"], columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        snapshot = nifty_regime_snapshot_from_closes(data["Close"], "SmartAPI")
        return snapshot or fetch_yahoo_nifty_regime_snapshot()
    except Exception as e:
        logging.warning(f"Failed to compute NIFTY market regime: {str(e)}")
        return fetch_yahoo_nifty_regime_snapshot()

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

def classify_setup_type(row):
    fresh_breakout_bonus_value = to_number_or_none(row.get("Fresh Breakout Bonus")) or 0.0
    sector_leader_adjustment_value = to_number_or_none(row.get("Sector Leader Adjustment")) or 0.0
    trend_persistence = to_number_or_none(row.get("Trend Persistence")) or 0.0
    rvol = to_number_or_none(row.get("RVOL")) or 0.0
    avg_volume_value = to_number_or_none(row.get("Avg Volume Value")) or 0.0
    mean_reversion = str(row.get("Mean_Reversion") or row.get("Mean Reversion") or "")

    if "Buy" in mean_reversion:
        return "mean_reversion_bounce"
    if rvol >= EXHAUSTION_RVOL_THRESHOLD:
        return "high_rvol_explosive"
    if fresh_breakout_bonus_value > 0:
        return "fresh_breakout"
    if sector_leader_adjustment_value > 0.25:
        return "sector_leader_continuation"
    if trend_persistence >= 75 and avg_volume_value >= 100_000_000:
        return "slow_institutional_trend"
    return "trend_continuation"

def optimal_hold_days_for_setup(setup_type):
    return DEFAULT_OPTIMAL_HOLD_DAYS_BY_SETUP.get(setup_type, 8)

def exit_review_days_for_setup(setup_type, expected_hold_days=None):
    expected_hold_days = int(expected_hold_days or optimal_hold_days_for_setup(setup_type))
    review_days = sorted({
        min(3, expected_hold_days),
        max(1, min(5, expected_hold_days)),
        expected_hold_days,
    })
    return [day for day in review_days if day > 0]

def exit_review_schedule_text(setup_type, expected_hold_days=None):
    return " / ".join(f"Day {day}" for day in exit_review_days_for_setup(setup_type, expected_hold_days))

def stronger_exit_status(current_status, candidate_status):
    return (
        candidate_status
        if EXIT_STATUS_PRIORITY.get(candidate_status, 0) > EXIT_STATUS_PRIORITY.get(current_status, 0)
        else current_status
    )

def db_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value

def setup_holding_metrics(history_df, min_trades=MIN_HOLDING_PERIOD_SAMPLE_SIZE):
    pnl_columns = [f"pnl_day_{day}" for day in HOLDING_PERIOD_DAYS]
    if history_df.empty or "setup_type" not in history_df.columns:
        return pd.DataFrame()

    rows = []
    grouped = history_df.dropna(subset=["setup_type"]).groupby("setup_type")
    for setup_type, setup_df in grouped:
        expectancy_by_day = {}
        median_by_day = {}
        trades_by_day = {}
        for hold_days in HOLDING_PERIOD_DAYS:
            column = f"pnl_day_{hold_days}"
            if column not in setup_df.columns:
                continue
            returns = pd.to_numeric(setup_df[column], errors="coerce").dropna()
            if len(returns) < min_trades:
                continue
            expectancy_by_day[hold_days] = float(returns.mean())
            median_by_day[hold_days] = float(returns.median())
            trades_by_day[hold_days] = len(returns)

        if not expectancy_by_day:
            continue

        optimal_exit_day = max(expectancy_by_day, key=expectancy_by_day.get)
        optimal_returns = pd.to_numeric(setup_df[f"pnl_day_{optimal_exit_day}"], errors="coerce").dropna()
        wins = optimal_returns[optimal_returns > 0]
        losses = optimal_returns[optimal_returns <= 0]
        win_rate = float(len(wins) / len(optimal_returns)) if len(optimal_returns) else 0.0
        loss_rate = 1.0 - win_rate
        avg_win = float(wins.mean()) if not wins.empty else 0.0
        avg_loss = abs(float(losses.mean())) if not losses.empty else 0.0
        historical_expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        entry_prices = pd.to_numeric(setup_df.get("entry_price"), errors="coerce")
        highest_prices = pd.to_numeric(setup_df.get("highest_price_after_entry"), errors="coerce")
        mfe = ((highest_prices - entry_prices) / entry_prices * 100).replace([np.inf, -np.inf], np.nan).dropna()
        mae = pd.to_numeric(setup_df.get("max_drawdown_after_entry"), errors="coerce").dropna()
        post_peak_decay = pd.to_numeric(setup_df.get("post_peak_decay_pct"), errors="coerce").dropna()
        exit_efficiency = pd.to_numeric(setup_df.get("exit_efficiency_score"), errors="coerce").dropna()
        peak_days = pd.to_numeric(setup_df.get("days_to_peak"), errors="coerce").dropna()
        early_column = "pnl_day_2" if "pnl_day_2" in setup_df.columns else "pnl_day_1"
        early_returns = pd.to_numeric(setup_df.get(early_column), errors="coerce").dropna()
        five_day_returns = pd.to_numeric(setup_df.get("pnl_day_5"), errors="coerce").dropna()
        five_day_wins = five_day_returns[five_day_returns > 0]
        five_day_losses = five_day_returns[five_day_returns <= 0]
        five_day_win_rate = float((five_day_returns > 0).mean() * 100) if not five_day_returns.empty else None
        five_day_avg_win = float(five_day_wins.mean()) if not five_day_wins.empty else 0.0
        five_day_avg_loss = abs(float(five_day_losses.mean())) if not five_day_losses.empty else 0.0
        five_day_loss_rate = 1.0 - ((five_day_win_rate or 0.0) / 100)
        setup_expectancy = (
            ((five_day_win_rate or 0.0) / 100) * five_day_avg_win
            - five_day_loss_rate * five_day_avg_loss
        ) if len(five_day_returns) >= min_trades else None
        target_probabilities = {}
        for target_pct in PROBABILITY_TARGET_LEVELS:
            target_probabilities[target_pct] = (
                float((mfe >= target_pct).mean() * 100)
                if len(mfe) >= min_trades
                else None
            )

        rows.append({
            "Setup Type": setup_type,
            "Trades": int(len(optimal_returns)),
            "Win Rate %": round(win_rate * 100, 1),
            "Avg Return %": round(float(optimal_returns.mean()), 2),
            "Median Return %": round(float(optimal_returns.median()), 2),
            "Avg 5D Return %": round(float(five_day_returns.mean()), 2) if not five_day_returns.empty else None,
            "5D Win Rate %": round(five_day_win_rate, 1) if five_day_win_rate is not None else None,
            "Avg DD %": round(float(mae.mean()), 2) if not mae.empty else None,
            "Setup Expectancy %": round(setup_expectancy, 2) if setup_expectancy is not None else None,
            "Avg Peak Day": round(float(peak_days.mean()), 1) if not peak_days.empty else None,
            "Median Hold Return %": round(median_by_day[optimal_exit_day], 2),
            "Early Failure Rate %": round(float((early_returns < 0).mean() * 100), 1) if not early_returns.empty else None,
            "Avg MFE %": round(float(mfe.mean()), 2) if not mfe.empty else None,
            "Avg MAE %": round(float(mae.mean()), 2) if not mae.empty else None,
            "Avg Post-Peak Decay %": round(float(post_peak_decay.mean()), 2) if not post_peak_decay.empty else None,
            "Avg Exit Efficiency %": round(float(exit_efficiency.mean()), 1) if not exit_efficiency.empty else None,
            "Optimal Exit Day": int(optimal_exit_day),
            "Optimal Expectancy %": round(expectancy_by_day[optimal_exit_day], 2),
            "Historical Expectancy %": round(historical_expectancy, 2),
            "Avg Win %": round(avg_win, 2),
            "Avg Loss %": round(avg_loss, 2),
            "Target +2% Probability %": round(target_probabilities[2], 1) if target_probabilities[2] is not None else None,
            "Target +4% Probability %": round(target_probabilities[4], 1) if target_probabilities[4] is not None else None,
            "Target +6% Probability %": round(target_probabilities[6], 1) if target_probabilities[6] is not None else None,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Optimal Expectancy %", ascending=False)

def setup_type_win_rate_table(history_df):
    metrics_df = setup_holding_metrics(history_df)
    if metrics_df.empty:
        return pd.DataFrame()

    metrics_df["Confidence"] = metrics_df["Win Rate %"].apply(setup_confidence_grade)
    display_columns = ["Setup Type", "Trades", "Win Rate %", "Confidence", "Avg Return %", "Median Return %"]
    table_df = metrics_df[[col for col in display_columns if col in metrics_df.columns]].copy()
    if "Setup Type" in table_df.columns:
        table_df["Setup Type"] = table_df["Setup Type"].apply(setup_type_display_name)
    return table_df.sort_values(["Win Rate %", "Trades"], ascending=[False, False])

def setup_confidence_grade(win_rate):
    win_rate = to_number_or_none(win_rate)
    if win_rate is None:
        return "N/A"
    if win_rate > 70:
        return "A+"
    if win_rate >= 60:
        return "A"
    if win_rate >= 50:
        return "B"
    return "C"

def setup_evidence_level(sample_count):
    sample_count = to_number_or_none(sample_count)
    if sample_count is None:
        sample_count = 0
    if sample_count >= SETUP_EVIDENCE_HIGH_SAMPLE_SIZE:
        return "High"
    if sample_count >= SETUP_EVIDENCE_MEDIUM_SAMPLE_SIZE:
        return "Medium"
    return "Low"

def learned_hold_days_lookup(history_df):
    metrics_df = setup_holding_metrics(history_df)
    if metrics_df.empty:
        return {}
    return {
        row["Setup Type"]: int(row["Optimal Exit Day"])
        for _, row in metrics_df.iterrows()
        if pd.notna(row.get("Optimal Exit Day"))
    }

def load_learned_hold_days_lookup():
    try:
        conn = get_db_connection()
        history_df = pd.read_sql_query("SELECT * FROM daily_picks WHERE pick_type = 'daily'", conn)
        conn.close()
        return learned_hold_days_lookup(history_df)
    except Exception as e:
        logging.warning(f"Failed to load learned holding periods: {str(e)}")
        return {}

def expected_hold_days_for_setup(setup_type, learned_lookup=None):
    learned_lookup = learned_lookup or {}
    return int(learned_lookup.get(setup_type, optimal_hold_days_for_setup(setup_type)))

def historical_expectancy_lookup(history_df):
    metrics_df = setup_holding_metrics(history_df)
    if metrics_df.empty:
        return {}
    return {
        row["Setup Type"]: {
            "expectancy": to_number_or_none(row.get("Historical Expectancy %")) or 0.0,
            "trades": int(row.get("Trades") or 0),
            "optimal_exit_day": int(row.get("Optimal Exit Day") or 0),
        }
        for _, row in metrics_df.iterrows()
        if pd.notna(row.get("Historical Expectancy %"))
    }

def load_historical_expectancy_lookup():
    try:
        conn = get_db_connection()
        history_df = pd.read_sql_query("SELECT * FROM daily_picks WHERE pick_type = 'daily'", conn)
        conn.close()
        return historical_expectancy_lookup(history_df)
    except Exception as e:
        logging.warning(f"Failed to load historical expectancy: {str(e)}")
        return {}

def historical_expectancy_adjustment(row, expectancy_lookup):
    if not expectancy_lookup:
        return 0.0
    setup_type = row.get("setup_type") or classify_setup_type(row)
    setup_stats = expectancy_lookup.get(setup_type)
    if not setup_stats or setup_stats.get("trades", 0) < MIN_HOLDING_PERIOD_SAMPLE_SIZE:
        return 0.0
    expectancy = setup_stats.get("expectancy", 0.0)
    adjustment = expectancy / 10.0
    return round(
        max(
            -MAX_HISTORICAL_EXPECTANCY_RANKING_ADJUSTMENT,
            min(MAX_HISTORICAL_EXPECTANCY_RANKING_ADJUSTMENT, adjustment),
        ),
        2,
    )

def setup_expectancy_lookup(history_df):
    metrics_df = setup_holding_metrics(history_df)
    if metrics_df.empty:
        return {}
    return {
        row["Setup Type"]: {
            "setup_expectancy": to_number_or_none(row.get("Setup Expectancy %")) or 0.0,
            "avg_return": to_number_or_none(row.get("Avg Return %")) or 0.0,
            "avg_5d_return": to_number_or_none(row.get("Avg 5D Return %")) or 0.0,
            "median_return": to_number_or_none(row.get("Median Return %")) or 0.0,
            "win_rate": to_number_or_none(row.get("Win Rate %")) or 0.0,
            "avg_dd": to_number_or_none(row.get("Avg DD %")) or 0.0,
            "trades": int(row.get("Trades") or 0),
        }
        for _, row in metrics_df.iterrows()
        if pd.notna(row.get("Setup Expectancy %"))
    }

def persisted_setup_expectancy_lookup():
    conn = get_db_connection()
    try:
        rows = conn.execute('''
            SELECT setup_type, trades, win_rate, avg_return, avg_drawdown,
                   setup_expectancy, setup_expectancy_bonus, avg_5d_return,
                   optimal_exit_day, target_2_prob, target_4_prob, target_6_prob,
                   median_return
            FROM setup_expectancy
        ''').fetchall()
    finally:
        conn.close()

    lookup = {}
    for (
        setup_type,
        trades,
        win_rate,
        avg_return,
        avg_drawdown,
        setup_expectancy,
        expectancy_bonus,
        avg_5d_return,
        optimal_exit_day,
        target_2_prob,
        target_4_prob,
        target_6_prob,
        median_return,
    ) in rows:
        lookup[setup_type] = {
            "setup_expectancy": to_number_or_none(setup_expectancy) or 0.0,
            "avg_5d_return": to_number_or_none(avg_5d_return) or to_number_or_none(avg_return) or 0.0,
            "avg_return": to_number_or_none(avg_return) or 0.0,
            "median_return": to_number_or_none(median_return) or 0.0,
            "win_rate": to_number_or_none(win_rate) or 0.0,
            "avg_dd": to_number_or_none(avg_drawdown) or 0.0,
            "trades": int(trades or 0),
            "expectancy_bonus": to_number_or_none(expectancy_bonus) or 0.0,
            "optimal_exit_day": int(optimal_exit_day or 0),
            "target_2_prob": to_number_or_none(target_2_prob),
            "target_4_prob": to_number_or_none(target_4_prob),
            "target_6_prob": to_number_or_none(target_6_prob),
        }
    return lookup

def load_setup_expectancy_lookup():
    try:
        persisted_lookup = persisted_setup_expectancy_lookup()
        if persisted_lookup:
            return persisted_lookup
        conn = get_db_connection()
        history_df = pd.read_sql_query("SELECT * FROM daily_picks WHERE pick_type = 'daily'", conn)
        conn.close()
        return setup_expectancy_lookup(history_df)
    except Exception as e:
        logging.warning(f"Failed to load setup expectancy: {str(e)}")
        return {}

def setup_expectancy_adjustment(row, setup_expectancy_stats):
    if not setup_expectancy_stats:
        return 0.0
    setup_type = row.get("Setup Type") or row.get("setup_type") or classify_setup_type(row)
    stats = setup_expectancy_stats.get(setup_type)
    if not stats or stats.get("trades", 0) < MIN_HOLDING_PERIOD_SAMPLE_SIZE:
        return 0.0

    expectancy = stats.get("setup_expectancy", 0.0)
    avg_dd = stats.get("avg_dd", 0.0)
    if "expectancy_bonus" in stats:
        return round(
            max(
                -MAX_SETUP_EXPECTANCY_RANKING_ADJUSTMENT,
                min(MAX_SETUP_EXPECTANCY_RANKING_ADJUSTMENT, stats.get("expectancy_bonus", 0.0)),
            ),
            2,
        )
    return setup_expectancy_bonus(expectancy, avg_dd)

def setup_expectancy_bonus(setup_expectancy, avg_dd):
    setup_expectancy = to_number_or_none(setup_expectancy) or 0.0
    avg_dd = to_number_or_none(avg_dd) or 0.0
    risk_adjusted_expectancy = setup_expectancy + (avg_dd * 0.20)
    return round(
        max(
            -MAX_SETUP_EXPECTANCY_RANKING_ADJUSTMENT,
            min(MAX_SETUP_EXPECTANCY_RANKING_ADJUSTMENT, risk_adjusted_expectancy / 10.0),
        ),
        2,
    )

def refresh_setup_expectancy_database():
    conn = get_db_connection()
    try:
        history_df = pd.read_sql_query("SELECT * FROM daily_picks WHERE pick_type = 'daily'", conn)
        metrics_df = setup_holding_metrics(history_df)
        if metrics_df.empty:
            return 0

        updated_at = app_timestamp_string()
        rows = []
        for _, row in metrics_df.iterrows():
            setup_type = row.get("Setup Type")
            setup_expectancy = to_number_or_none(row.get("Setup Expectancy %"))
            avg_drawdown = to_number_or_none(row.get("Avg DD %"))
            if not setup_type or setup_expectancy is None:
                continue
            rows.append((
                setup_type,
                int(row.get("Trades") or 0),
                db_value(row.get("Win Rate %")),
                db_value(row.get("Avg Return %")),
                db_value(row.get("Optimal Exit Day")),
                db_value(avg_drawdown),
                db_value(setup_expectancy),
                setup_expectancy_bonus(setup_expectancy, avg_drawdown),
                db_value(row.get("Avg 5D Return %")),
                db_value(row.get("Optimal Exit Day")),
                db_value(row.get("Avg Peak Day")),
                db_value(row.get("Early Failure Rate %")),
                db_value(row.get("Avg MFE %")),
                db_value(row.get("Avg MAE %")),
                db_value(row.get("Avg Post-Peak Decay %")),
                db_value(row.get("Avg Exit Efficiency %")),
                db_value(row.get("Target +2% Probability %")),
                db_value(row.get("Target +4% Probability %")),
                db_value(row.get("Target +6% Probability %")),
                db_value(row.get("Median Return %")),
                updated_at,
            ))

        conn.executemany('''
            INSERT INTO setup_expectancy (
                setup_type, trades, win_rate, avg_return, avg_hold_days,
                avg_drawdown, setup_expectancy, setup_expectancy_bonus,
                avg_5d_return, optimal_exit_day, avg_peak_day,
                early_failure_rate, avg_mfe, avg_mae, avg_post_peak_decay,
                avg_exit_efficiency, target_2_prob, target_4_prob,
                target_6_prob, median_return, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(setup_type) DO UPDATE SET
                trades = excluded.trades,
                win_rate = excluded.win_rate,
                avg_return = excluded.avg_return,
                avg_hold_days = excluded.avg_hold_days,
                avg_drawdown = excluded.avg_drawdown,
                setup_expectancy = excluded.setup_expectancy,
                setup_expectancy_bonus = excluded.setup_expectancy_bonus,
                avg_5d_return = excluded.avg_5d_return,
                optimal_exit_day = excluded.optimal_exit_day,
                avg_peak_day = excluded.avg_peak_day,
                early_failure_rate = excluded.early_failure_rate,
                avg_mfe = excluded.avg_mfe,
                avg_mae = excluded.avg_mae,
                avg_post_peak_decay = excluded.avg_post_peak_decay,
                avg_exit_efficiency = excluded.avg_exit_efficiency,
                target_2_prob = excluded.target_2_prob,
                target_4_prob = excluded.target_4_prob,
                target_6_prob = excluded.target_6_prob,
                median_return = excluded.median_return,
                updated_at = excluded.updated_at
        ''', rows)
        conn.commit()
        conn.close()
        conn = None
        sync_history_backup()
        return len(rows)
    except sqlite3.OperationalError as e:
        if conn is not None:
            conn.rollback()
        logging.exception(f"Failed to refresh setup expectancy database: {str(e)}")
        return 0
    finally:
        if conn is not None:
            conn.close()

def quote_identifier(identifier):
    return '"' + str(identifier).replace('"', '""') + '"'

DAILY_PICK_LIFECYCLE_COLUMNS = {
    "pick_type": "TEXT",
    "entry_date": "TEXT",
    "entry_price": "REAL",
    "setup_type": "TEXT",
    "sector_relative_strength": "REAL",
    "optimal_hold_days": "INTEGER",
    "expected_hold_days": "INTEGER",
    "exit_review_day": "TEXT",
    "exit_status": "TEXT",
    "exit_reason": "TEXT",
    "exit_advice_updated_at": "TEXT",
    "highest_price_after_entry": "REAL",
    "days_to_peak": "INTEGER",
    "pnl_day_1": "REAL",
    "pnl_day_2": "REAL",
    "pnl_day_3": "REAL",
    "pnl_day_5": "REAL",
    "pnl_day_10": "REAL",
    "pnl_day_20": "REAL",
    "max_drawdown_after_entry": "REAL",
    "post_peak_decay_pct": "REAL",
    "exit_efficiency_score": "REAL",
    "outcome_updated_at": "TEXT",
}

def ensure_table_columns(conn, table_name, expected_columns):
    existing_columns = {
        row[1] for row in conn.execute(f"PRAGMA table_info({quote_identifier(table_name)})").fetchall()
    }
    for column, column_type in expected_columns.items():
        if column not in existing_columns:
            conn.execute(
                f"ALTER TABLE {quote_identifier(table_name)} "
                f"ADD COLUMN {quote_identifier(column)} {column_type}"
            )

def ensure_daily_pick_lifecycle_columns(conn):
    ensure_table_columns(conn, "daily_picks", DAILY_PICK_LIFECYCLE_COLUMNS)

def migrate_daily_picks_primary_key(conn):
    table_info = conn.execute("PRAGMA table_info(daily_picks)").fetchall()
    if not table_info:
        return

    pk_columns = [
        row[1]
        for row in sorted((row for row in table_info if row[5]), key=lambda row: row[5])
    ]
    if pk_columns == ["date", "symbol", "pick_type"]:
        return

    conn.execute("UPDATE daily_picks SET pick_type = 'daily' WHERE pick_type IS NULL OR pick_type = ''")
    columns = [row[1] for row in table_info]
    column_defs = [
        f"{quote_identifier(row[1])} {row[2] or 'TEXT'}"
        for row in table_info
    ]
    column_list = ", ".join(quote_identifier(column) for column in columns)
    conn.execute("ALTER TABLE daily_picks RENAME TO daily_picks_old")
    conn.execute(f'''
        CREATE TABLE daily_picks (
            {", ".join(column_defs)},
            PRIMARY KEY (date, symbol, pick_type)
        )
    ''')
    conn.execute(f'''
        INSERT OR REPLACE INTO daily_picks ({column_list})
        SELECT {column_list}
        FROM daily_picks_old
    ''')
    conn.execute("DROP TABLE daily_picks_old")

def init_database(allow_restore=True):
    conn = get_db_connection()
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
    conn.execute('''
        CREATE TABLE IF NOT EXISTS setup_expectancy (
            setup_type TEXT PRIMARY KEY,
            trades INTEGER,
            win_rate REAL,
            avg_return REAL,
            avg_hold_days REAL,
            avg_drawdown REAL,
            median_return REAL,
            setup_expectancy REAL,
            setup_expectancy_bonus REAL,
            avg_5d_return REAL,
            optimal_exit_day INTEGER,
            avg_peak_day REAL,
            early_failure_rate REAL,
            avg_mfe REAL,
            avg_mae REAL,
            avg_post_peak_decay REAL,
            avg_exit_efficiency REAL,
            target_2_prob REAL,
            target_4_prob REAL,
            target_6_prob REAL,
            updated_at TEXT
        )
    ''')
    setup_expectancy_columns = {
        "target_2_prob": "REAL",
        "target_4_prob": "REAL",
        "target_6_prob": "REAL",
        "median_return": "REAL",
    }
    existing_setup_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(setup_expectancy)").fetchall()
    }
    for column, column_type in setup_expectancy_columns.items():
        if column not in existing_setup_columns:
            conn.execute(f"ALTER TABLE setup_expectancy ADD COLUMN {column} {column_type}")
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
        "entry_date": "TEXT",
        "entry_price": "REAL",
        "setup_type": "TEXT",
        "sector": "TEXT",
        "relative_strength": "REAL",
        "sector_relative_strength": "REAL",
        "trend_persistence": "REAL",
        "rvol": "REAL",
        "liquidity_value": "REAL",
        "breakout_age": "REAL",
        "ema20_distance": "REAL",
        "sector_leader_score": "REAL",
        "sector_leader_adjustment": "REAL",
        "optimal_hold_days": "INTEGER",
        "expected_hold_days": "INTEGER",
        "exit_review_day": "TEXT",
        "exit_status": "TEXT",
        "exit_reason": "TEXT",
        "exit_advice_updated_at": "TEXT",
        "highest_price_after_entry": "REAL",
        "days_to_peak": "INTEGER",
        "pnl_day_1": "REAL",
        "pnl_day_2": "REAL",
        "pnl_day_3": "REAL",
        "pnl_day_5": "REAL",
        "pnl_day_10": "REAL",
        "pnl_day_20": "REAL",
        "max_drawdown_after_entry": "REAL",
        "post_peak_decay_pct": "REAL",
        "exit_efficiency_score": "REAL",
        "outcome_updated_at": "TEXT",
    }
    existing_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(daily_picks)").fetchall()
    }
    for column, column_type in expected_columns.items():
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE daily_picks ADD COLUMN {column} {column_type}")
    migrate_daily_picks_primary_key(conn)
    conn.commit()
    conn.close()
    if allow_restore and restore_history_backup_if_empty():
        init_database(allow_restore=False)

def insert_top_picks(results_df, pick_type="daily"):
    if results_df is None or results_df.empty:
        return 0
    conn = get_db_connection()
    cursor = conn.cursor()
    history_df = pd.read_sql_query("SELECT * FROM daily_picks WHERE pick_type = 'daily'", conn)
    learned_lookup = learned_hold_days_lookup(history_df)
    
    data_to_insert = []
    for _, row in results_df.head(5).iterrows():
        setup_type = classify_setup_type(row)
        expected_hold_days = expected_hold_days_for_setup(setup_type, learned_lookup)
        exit_review_day = exit_review_schedule_text(setup_type, expected_hold_days)
        data_to_insert.append((
            app_date_string(),
            row.get('Symbol'),
            db_value(row.get('Score', 0)),
            db_value(row.get('Current Price')),
            db_value(row.get('Buy At')),
            db_value(row.get('Stop Loss')),
            db_value(row.get('Target')),
            row.get('Intraday'),
            row.get('Swing'),
            row.get('Short-Term'),
            row.get('Long-Term'),
            row.get('Mean_Reversion'),
            row.get('Breakout'),
            row.get('Ichimoku_Trend'),
            row.get('Recommendation'),
            row.get('Regime'),
            db_value(row.get('Position Size')),
            db_value(row.get('Trailing Stop')),
            row.get('Reason'),
            pick_type,
            app_date_string(),
            db_value(row.get('Buy At') or row.get('Current Price')),
            setup_type,
            row.get('Sector'),
            db_value(row.get('Relative Strength')),
            db_value(row.get('Sector Relative Strength %')),
            db_value(row.get('Trend Persistence')),
            db_value(row.get('RVOL')),
            db_value(row.get('Avg Volume Value')),
            db_value(row.get('Fresh Breakout Age')),
            db_value(row.get('EMA20 Distance %')),
            db_value(row.get('Sector Leader Score')),
            db_value(row.get('Sector Leader Adjustment')),
            expected_hold_days,
            expected_hold_days,
            exit_review_day,
            "HOLD",
            f"Initial advisory hold for {setup_type}; review on {exit_review_day}."
        ))

    cursor.executemany('''
        INSERT OR REPLACE INTO daily_picks (
            date, symbol, score, current_price, buy_at, stop_loss, target,
            intraday, swing, short_term, long_term, mean_reversion, breakout,
            ichimoku_trend, recommendation, regime, position_size, trailing_stop,
            reason, pick_type, entry_date, entry_price, setup_type, sector,
            relative_strength, sector_relative_strength, trend_persistence,
            rvol, liquidity_value, breakout_age, ema20_distance,
            sector_leader_score, sector_leader_adjustment, optimal_hold_days,
            expected_hold_days, exit_review_day, exit_status, exit_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', data_to_insert)
    
    conn.commit()
    conn.close()
    sync_history_backup()
    return len(data_to_insert)

def calculate_holding_period_outcome(symbol, entry_date, entry_price):
    entry_price = to_float_or_none(entry_price)
    if not symbol or not entry_date or not entry_price:
        return None

    data = fetch_stock_data_cached(symbol, period="3mo", interval="1d")
    if data.empty or not {"High", "Low", "Close"}.issubset(data.columns):
        return None

    entry_date = pd.to_datetime(entry_date).date()
    post_entry = data[data.index.date > entry_date].copy()
    if post_entry.empty:
        return None

    highs = pd.to_numeric(post_entry["High"], errors="coerce").dropna()
    lows = pd.to_numeric(post_entry["Low"], errors="coerce").dropna()
    closes = pd.to_numeric(post_entry["Close"], errors="coerce")
    if highs.empty or lows.empty or closes.dropna().empty:
        return None

    highest_price = float(highs.max())
    days_to_peak = int(post_entry.index.get_loc(highs.idxmax()) + 1)
    latest_close = to_float_or_none(closes.dropna().iloc[-1])
    peak_gain_pct = ((highest_price - entry_price) / entry_price) * 100
    latest_gain_pct = ((latest_close - entry_price) / entry_price) * 100 if latest_close else None
    outcome = {
        "highest_price_after_entry": highest_price,
        "days_to_peak": days_to_peak,
        "max_drawdown_after_entry": ((float(lows.min()) - entry_price) / entry_price) * 100,
        "post_peak_decay_pct": latest_gain_pct - peak_gain_pct if latest_gain_pct is not None else None,
        "outcome_updated_at": app_timestamp_string(),
    }

    for hold_days in HOLDING_PERIOD_DAYS:
        column = f"pnl_day_{hold_days}"
        if len(post_entry) >= hold_days:
            close_price = to_float_or_none(closes.iloc[hold_days - 1])
            outcome[column] = ((close_price - entry_price) / entry_price) * 100 if close_price else None
        else:
            outcome[column] = None

    available_returns = [
        outcome.get(f"pnl_day_{hold_days}")
        for hold_days in HOLDING_PERIOD_DAYS
        if outcome.get(f"pnl_day_{hold_days}") is not None
    ]
    realized_return = available_returns[-1] if available_returns else latest_gain_pct
    if peak_gain_pct > 0 and realized_return is not None:
        outcome["exit_efficiency_score"] = max(0.0, min(100.0, (realized_return / peak_gain_pct) * 100))
    else:
        outcome["exit_efficiency_score"] = None

    return outcome

def calculate_exit_advice(
    symbol,
    entry_date,
    entry_price,
    setup_type,
    expected_hold_days,
    sector_relative_strength,
):
    entry_price = to_float_or_none(entry_price)
    expected_hold_days = int(expected_hold_days or optimal_hold_days_for_setup(setup_type))
    if not symbol or not entry_date or not entry_price:
        return None

    data = fetch_stock_data_cached(symbol, period="3mo", interval="1d")
    if data.empty or not {"Close", "High"}.issubset(data.columns):
        return None

    entry_date = pd.to_datetime(entry_date).date()
    post_entry = data[data.index.date > entry_date].copy()
    if post_entry.empty:
        return {
            "exit_status": "HOLD",
            "exit_reason": "Entry day; no post-entry candle available yet.",
            "exit_advice_updated_at": app_timestamp_string(),
        }

    close = pd.to_numeric(data["Close"], errors="coerce")
    latest_close = to_float_or_none(close.iloc[-1])
    if latest_close is None:
        return None

    elapsed_days = len(post_entry)
    latest_pnl = ((latest_close - entry_price) / entry_price) * 100
    max_gain = ((float(pd.to_numeric(post_entry["High"], errors="coerce").max()) - entry_price) / entry_price) * 100
    status = "HOLD"
    reasons = []

    if max_gain >= 6 and elapsed_days < expected_hold_days:
        status = stronger_exit_status(status, "TRAIL_SL")
        reasons.append(f"Reached +{max_gain:.1f}% before expected hold; trail stop loss.")

    if max_gain >= 10 and elapsed_days <= max(3, expected_hold_days // 2):
        status = stronger_exit_status(status, "BOOK_PARTIAL")
        reasons.append(f"Reached +{max_gain:.1f}% quickly; consider booking partial.")

    if elapsed_days >= 3 and latest_pnl <= 0:
        status = stronger_exit_status(status, "EXIT_WARNING")
        reasons.append(f"No profit after {elapsed_days} trading days; review exit.")

    if len(close.dropna()) >= 20:
        ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        if latest_close < ema20:
            status = stronger_exit_status(status, "EXIT_WARNING")
            reasons.append("Latest close is below EMA20.")

    sector_relative_strength = to_number_or_none(sector_relative_strength)
    if sector_relative_strength is not None and sector_relative_strength < 0:
        status = stronger_exit_status(status, "EXIT_WARNING")
        reasons.append("Sector relative strength is negative.")

    if elapsed_days >= expected_hold_days:
        status = stronger_exit_status(status, "EXIT")
        reasons.append(f"Reached expected hold day {expected_hold_days}; review/book.")

    if not reasons:
        reasons.append(f"Within expected hold window; current PnL {latest_pnl:.1f}% after {elapsed_days} trading days.")

    return {
        "exit_status": status,
        "exit_reason": " ".join(reasons),
        "exit_advice_updated_at": app_timestamp_string(),
    }

def update_exit_advice(limit=50):
    conn = get_db_connection()
    try:
        ensure_daily_pick_lifecycle_columns(conn)
        history_df = pd.read_sql_query("SELECT * FROM daily_picks WHERE pick_type = 'daily'", conn)
        learned_lookup = learned_hold_days_lookup(history_df)
        rows = conn.execute('''
            SELECT rowid, symbol, entry_date, date, entry_price, buy_at, current_price,
                   setup_type, expected_hold_days, optimal_hold_days, sector_relative_strength
            FROM daily_picks
            WHERE pick_type = 'daily'
            ORDER BY date DESC
            LIMIT ?
        ''', (limit,)).fetchall()

        updated = 0
        for (
            rowid,
            symbol,
            entry_date,
            pick_date,
            entry_price,
            buy_at,
            current_price,
            setup_type,
            expected_hold_days,
            optimal_hold_days,
            sector_relative_strength,
        ) in rows:
            setup_type = setup_type or "trend_continuation"
            expected_hold_days = expected_hold_days_for_setup(setup_type, learned_lookup)
            advice = calculate_exit_advice(
                symbol,
                entry_date or pick_date,
                entry_price or buy_at or current_price,
                setup_type,
                expected_hold_days,
                sector_relative_strength,
            )
            if not advice:
                continue
            conn.execute('''
                UPDATE daily_picks
                SET expected_hold_days = ?,
                    exit_review_day = ?,
                    exit_status = ?,
                    exit_reason = ?,
                    exit_advice_updated_at = ?
                WHERE rowid = ?
            ''', (
                int(expected_hold_days),
                exit_review_schedule_text(setup_type, expected_hold_days),
                advice.get("exit_status"),
                advice.get("exit_reason"),
                advice.get("exit_advice_updated_at"),
                rowid,
            ))
            updated += 1

        conn.commit()
        conn.close()
        conn = None
        if updated:
            sync_history_backup()
        return updated
    except sqlite3.OperationalError as e:
        if conn is not None:
            conn.rollback()
        logging.exception(f"Failed to update exit advice: {str(e)}")
        return 0
    finally:
        if conn is not None:
            conn.close()

def update_holding_period_outcomes(limit=50):
    conn = get_db_connection()
    try:
        ensure_daily_pick_lifecycle_columns(conn)
        rows = conn.execute('''
            SELECT rowid, symbol, entry_date, date, entry_price, buy_at, current_price
            FROM daily_picks
            WHERE pick_type = 'daily'
              AND (
                outcome_updated_at IS NULL
                OR pnl_day_2 IS NULL
                OR pnl_day_20 IS NULL
              )
            ORDER BY date DESC
            LIMIT ?
        ''', (limit,)).fetchall()

        updated = 0
        for rowid, symbol, entry_date, pick_date, entry_price, buy_at, current_price in rows:
            outcome = calculate_holding_period_outcome(
                symbol,
                entry_date or pick_date,
                entry_price or buy_at or current_price,
            )
            if not outcome:
                continue

            conn.execute('''
                UPDATE daily_picks
                SET highest_price_after_entry = ?,
                    days_to_peak = ?,
                    pnl_day_1 = ?,
                    pnl_day_2 = ?,
                    pnl_day_3 = ?,
                    pnl_day_5 = ?,
                    pnl_day_10 = ?,
                    pnl_day_20 = ?,
                    max_drawdown_after_entry = ?,
                    post_peak_decay_pct = ?,
                    exit_efficiency_score = ?,
                    outcome_updated_at = ?
                WHERE rowid = ?
            ''', (
                db_value(outcome.get("highest_price_after_entry")),
                db_value(outcome.get("days_to_peak")),
                db_value(outcome.get("pnl_day_1")),
                db_value(outcome.get("pnl_day_2")),
                db_value(outcome.get("pnl_day_3")),
                db_value(outcome.get("pnl_day_5")),
                db_value(outcome.get("pnl_day_10")),
                db_value(outcome.get("pnl_day_20")),
                db_value(outcome.get("max_drawdown_after_entry")),
                db_value(outcome.get("post_peak_decay_pct")),
                db_value(outcome.get("exit_efficiency_score")),
                outcome.get("outcome_updated_at"),
                rowid,
            ))
            updated += 1

        conn.commit()
        conn.close()
        conn = None
        if updated:
            sync_history_backup()
        return updated
    except sqlite3.OperationalError as e:
        if conn is not None:
            conn.rollback()
        logging.exception(f"Failed to update holding-period outcomes: {str(e)}")
        return 0
    finally:
        if conn is not None:
            conn.close()

def holding_period_expectancy(history_df):
    pnl_columns = [f"pnl_day_{day}" for day in HOLDING_PERIOD_DAYS]
    available_columns = [col for col in pnl_columns if col in history_df.columns]
    if history_df.empty or not available_columns or "setup_type" not in history_df.columns:
        return pd.DataFrame()

    rows = []
    for setup_type, setup_df in history_df.dropna(subset=["setup_type"]).groupby("setup_type"):
        for hold_days in HOLDING_PERIOD_DAYS:
            column = f"pnl_day_{hold_days}"
            if column not in setup_df.columns:
                continue
            returns = pd.to_numeric(setup_df[column], errors="coerce").dropna()
            if returns.empty:
                continue
            rows.append({
                "Setup Type": setup_type,
                "Hold Days": hold_days,
                "Trades": len(returns),
                "Average Return %": round(float(returns.mean()), 2),
                "Median Return %": round(float(returns.median()), 2),
                "Win Rate %": round(float((returns > 0).mean() * 100), 1),
                "Expectancy %": round(float(returns.mean()), 2),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Setup Type", "Expectancy %"], ascending=[True, False])

def expected_hold_text(row):
    setup_type = row.get("setup_type") or row.get("Setup Type") or classify_setup_type(row)
    learned_lookup = load_learned_hold_days_lookup()
    setup_expectancy_stats = load_setup_expectancy_lookup()
    expected_hold_days = (
        to_number_or_none(row.get("expected_hold_days"))
        or expected_hold_days_for_setup(setup_type, learned_lookup)
    )
    exit_review_day = row.get("exit_review_day") or exit_review_schedule_text(setup_type, expected_hold_days)
    exit_status = row.get("exit_status") or "HOLD"
    exit_reason = row.get("exit_reason")
    similar_setup_stats = setup_expectancy_stats.get(setup_type, {})
    row_sample_count = to_number_or_none(row.get("Setup Sample Size"))
    setup_sample_count = int(
        row_sample_count
        if row_sample_count is not None
        else similar_setup_stats.get("trades") or 0
    )
    setup_evidence = row.get("Setup Evidence") or setup_evidence_level(setup_sample_count)
    has_setup_history = setup_sample_count >= MIN_HOLDING_PERIOD_SAMPLE_SIZE
    historical_win_rate = similar_setup_stats.get("win_rate") if has_setup_history else None
    historical_win_rate_text = (
        format_percent(historical_win_rate, 0)
        if has_setup_history
        else "Collecting Data"
    )
    setup_confidence = (
        setup_confidence_grade(historical_win_rate)
        if has_setup_history
        else "Collecting Data"
    )
    text = (
        f"Setup Type: {setup_type}  \n"
        f"Historical Performance  \n"
        f"Win Rate: {historical_win_rate_text}  \n"
        f"Confidence: {setup_confidence}  \n"
        f"Evidence: {setup_evidence}  \n"
        f"Sample Size: {setup_sample_count}  \n"
        f"Expected Hold: {int(expected_hold_days)} trading days  \n"
        f"Exit Review: {exit_review_day}"
    )
    if exit_status:
        text += f"  \nExit Status: {exit_status}"
    if exit_reason:
        text += f"  \nExit Reason: {exit_reason}"
    if has_setup_history:
        text += (
            f"  \nAvg Return: {format_percent(similar_setup_stats.get('avg_return'), 1)}"
        )
    probabilities = similar_setup_stats
    probability_parts = []
    for target_pct in PROBABILITY_TARGET_LEVELS:
        probability = probabilities.get(f"target_{target_pct}_prob")
        if probability is not None:
            probability_parts.append(f"+{target_pct}% : {probability:.0f}%")
    if probability_parts:
        text += "  \nHistorical Similar Setups"
        for probability_part in probability_parts:
            text += f"  \n{probability_part}"
    return text

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
        consolidation_candles = calculate_entry_consolidation_candles(data, fresh_breakout_age)
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
                "Consolidation Candles": consolidation_candles,
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
                "Consolidation Candles": consolidation_candles,
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
        "EMA20 Distance %", "Fresh Breakout Age", "Consolidation Candles", "RVOL", "Avg Volume Value",
        "Buy At", "Stop Loss", "Target", "Recommendation", "Intraday", "Swing", "Short-Term",
        "Long-Term", "Mean_Reversion", "Breakout", "Ichimoku_Trend", "Major Trend Conflict",
        "Status", "Error"
    ]
    for col in expected_cols:
         if col not in results_df.columns:
             results_df[col] = None

    # Filter for Top Picks (Success only)
    success_df = results_df[results_df["Status"] == "Success"].copy()
    if success_df.empty:
        return pd.DataFrame(), results_df
    sector_momentum = calculate_sector_momentum_map(success_df)
    sector_breadth = calculate_sector_breadth_map(success_df)
    market_stats = fetch_market_stats()
    regime_snapshot = market_regime_snapshot(sector_breadth, market_stats=market_stats)
    nifty_5d_return = fetch_nifty_5d_return()
    success_df["Score"] = pd.to_numeric(success_df["Score"], errors="coerce").fillna(0)
    success_df = success_df[success_df.apply(is_actionable_entry, axis=1)]
    if success_df.empty:
        return pd.DataFrame(), results_df
    success_df = success_df[success_df["Score"] >= MIN_TOP_PICK_SCORE]
    if success_df.empty:
        return pd.DataFrame(), results_df
    ranked_success_df = add_entry_quality_columns(
        success_df,
        sector_momentum,
        nifty_5d_return,
        sector_breadth=sector_breadth,
        market_regime=regime_snapshot,
    )

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

    if st.session_state.get("show_buy_above_cmp_only", False) and not top_picks_df.empty:
        top_picks_df = top_picks_df[top_picks_df.apply(is_buy_above_cmp_setup, axis=1)]

    if not top_picks_df.empty:
        top_picks_df = top_picks_df[top_picks_df.apply(is_swing_quality_setup, axis=1)]
    if not top_picks_df.empty:
        top_picks_df = top_picks_df[top_picks_df.apply(is_public_share_swing_pick, axis=1)]
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
        "Previous Day Move %", "Overnight Gap %", "EMA20 Distance %", "Fresh Breakout Age", "Consolidation Candles", "RVOL", "Avg Volume Value",
        "Intraday", "Recommendation", "Buy At", "Stop Loss", "Target",
        "Ichimoku_Trend", "Major Trend Conflict", "Entry Type"
    ]
    for col in expected_cols:
        if col not in results_df.columns:
            results_df[col] = None 

    if "Score" not in results_df.columns:
        results_df["Score"] = 0

    sector_momentum = calculate_sector_momentum_map(results_df)
    sector_breadth = calculate_sector_breadth_map(results_df)
    market_stats = fetch_market_stats()
    regime_snapshot = market_regime_snapshot(sector_breadth, market_stats=market_stats)
    nifty_5d_return = fetch_nifty_intraday_return()
    results_df["Score"] = pd.to_numeric(results_df["Score"], errors="coerce").fillna(0)
    results_df = results_df[results_df.apply(is_actionable_entry, axis=1)]
    if results_df.empty:
        return pd.DataFrame()
    results_df = results_df[results_df["Score"] >= MIN_TOP_PICK_SCORE]
    if results_df.empty:
        return pd.DataFrame()
        
    recommendation_mode = st.session_state.get('recommendation_mode', 'Standard')
    if recommendation_mode == "Adaptive":
        results_df = results_df[results_df["Recommendation"].str.contains("Buy", na=False)]
    else:
        results_df = results_df[results_df["Intraday"].str.contains("Buy", na=False)]
    if st.session_state.get("show_buy_above_cmp_only", False) and not results_df.empty:
        results_df = results_df[results_df.apply(is_buy_above_cmp_setup, axis=1)]
    results_df = add_entry_quality_columns(
        results_df,
        sector_momentum,
        nifty_5d_return,
        sector_breadth=sector_breadth,
        market_regime=regime_snapshot,
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

def swing_signal_for_grade(grade):
    grade = str(grade or "").strip().upper()
    return {
        "A+": "Strong Buy",
        "A": "Strong Buy",
        "B+": "Buy",
        "B": "Buy",
        "C+": "WATCHLIST",
        "C": "AVOID",
        "D": "AVOID",
    }.get(grade, "WATCHLIST")

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

def grade_meets_minimum(grade, minimum_grade):
    grade_order = ["D", "C", "C+", "B", "B+", "A", "A+"]
    grade = str(grade or "").strip().upper()
    minimum_grade = str(minimum_grade or "").strip().upper()
    if grade not in grade_order or minimum_grade not in grade_order:
        return False
    return grade_order.index(grade) >= grade_order.index(minimum_grade)

def is_public_share_swing_pick(row):
    grade = row.get("Confidence Grade") or confidence_grade(row)
    displayed_swing_signal = swing_signal_for_grade(grade)
    return (
        grade_meets_minimum(grade, PUBLIC_SHARE_MIN_GRADE)
        and displayed_swing_signal in PUBLIC_SHARE_ALLOWED_SWING
    )

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

def is_buy_above_cmp_setup(row):
    current = to_float_or_none(row.get("Current Price"))
    entry = to_float_or_none(row.get("Buy At"))
    return current is not None and entry is not None and entry > current

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

def entry_display_details(current_price, buy_at, entry_type="Standard", include_breakout_context=False):
    if isinstance(buy_at, tuple):
        buy_at, tuple_entry_type = buy_at
        entry_type = tuple_entry_type or entry_type

    normalized_type = str(entry_type or "").strip().lower()
    current = to_float_or_none(current_price)
    entry = to_float_or_none(buy_at)
    breakout_label = "Buy Above (Breakout)" if include_breakout_context else "Buy Above"

    if normalized_type == "choppy":
        return "⚠️", "No Trade", "Choppy"

    if current is not None and entry is not None:
        if entry < current:
            return "🔵", "Wait for Pullback", buy_at
        if entry > current:
            return "🟢", breakout_label, buy_at
        return "", "Buy Near CMP", buy_at

    if normalized_type == "breakout":
        return "🟢", breakout_label, buy_at
    if normalized_type == "pullback":
        return "🔵", "Wait for Pullback", buy_at
    return "", "Buy At", buy_at

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

def calculate_consolidation_candles(data, lookback=8, max_range_pct=3.0, end_offset=0):
    if data.empty or not {"High", "Low", "Close"}.issubset(data.columns):
        return np.nan

    end_position = len(data) - int(end_offset or 0)
    if end_position < lookback:
        return np.nan

    window = data.iloc[:end_position].tail(lookback)
    highs = pd.to_numeric(window["High"], errors="coerce")
    lows = pd.to_numeric(window["Low"], errors="coerce")
    closes = pd.to_numeric(window["Close"], errors="coerce")
    if highs.isna().any() or lows.isna().any() or closes.isna().any():
        return np.nan

    count = 0
    for high, low, close in zip(reversed(highs.tolist()), reversed(lows.tolist()), reversed(closes.tolist())):
        if close <= 0:
            break
        candle_range_pct = ((high - low) / close) * 100
        if candle_range_pct <= max_range_pct:
            count += 1
        else:
            break
    return count

def calculate_entry_consolidation_candles(data, breakout_age):
    breakout_age = to_number_or_none(breakout_age)
    if breakout_age is not None and 1 <= breakout_age <= FRESH_BREAKOUT_MAX_AGE:
        return calculate_consolidation_candles(data, end_offset=int(breakout_age))
    return calculate_consolidation_candles(data)

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

def calculate_sector_breadth_map(df):
    if df.empty or "EMA20 Distance %" not in df.columns:
        return {}
    breadth_df = df.copy()
    if "Sector" not in breadth_df.columns:
        breadth_df["Sector"] = breadth_df["Symbol"].apply(get_stock_sector)
    breadth_df["EMA20 Distance %"] = pd.to_numeric(
        breadth_df["EMA20 Distance %"],
        errors="coerce",
    )
    breadth_df = breadth_df.dropna(subset=["EMA20 Distance %"])
    if breadth_df.empty:
        return {}

    breadth = {}
    for sector, sector_df in breadth_df.groupby("Sector", dropna=False):
        total = len(sector_df)
        above_ema20 = int((sector_df["EMA20 Distance %"] > 0).sum())
        breadth[sector] = {
            "above": above_ema20,
            "total": int(total),
            "pct": (above_ema20 / total * 100) if total else None,
        }
    return breadth

@st.cache_data(ttl=900)
def fetch_market_stats():
    try:
        response = requests.get(
            "https://brkpoint.in/api/market-stats",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return {}
        return payload
    except Exception as e:
        logging.warning(f"Failed to fetch market stats breadth: {str(e)}")
        return {}

def market_stats_breadth_summary(market_stats):
    breadth = (market_stats or {}).get("breadth") or {}
    total = int(breadth.get("total") or 0)
    advancing = int(breadth.get("advancing") or 0)
    if total <= 0:
        return {"above": 0, "total": 0, "pct": None, "source": "scan_ema20"}
    return {
        "above": advancing,
        "total": total,
        "pct": advancing / total * 100,
        "source": "brkpoint_market_stats",
    }

def market_stats_industry_lookup(market_stats):
    industry_rows = (market_stats or {}).get("industry") or []
    lookup = {}
    for row in industry_rows:
        industry = row.get("Industry")
        if industry:
            lookup[industry] = row
    return lookup

def market_stats_for_sector(sector, market_stats):
    industry_name = MARKET_STATS_INDUSTRY_ALIASES.get(sector, sector)
    return market_stats_industry_lookup(market_stats).get(industry_name, {})

def market_stats_sector_text(sector, market_stats):
    stats = market_stats_for_sector(sector, market_stats)
    total = int(stats.get("total") or 0)
    advancing = int(stats.get("advancing") or 0)
    if total <= 0:
        return None
    avg_change = to_number_or_none(stats.get("avgChange"))
    avg_change_text = f", Avg: {avg_change:.2f}%" if avg_change is not None else ""
    return f"{sector} Advance Breadth: {advancing}/{total}{avg_change_text}"

def market_stats_sector_advance_ratio(sector, market_stats):
    stats = market_stats_for_sector(sector, market_stats)
    total = to_number_or_none(stats.get("total"))
    advancing = to_number_or_none(stats.get("advancing"))
    if total is None or total <= 0 or advancing is None:
        return np.nan
    return advancing / total

def industry_breadth_penalty(advance_ratio):
    advance_ratio = to_number_or_none(advance_ratio)
    if advance_ratio is None:
        return 0.0
    if advance_ratio < WEAK_INDUSTRY_ADVANCE_RATIO_THRESHOLD:
        return WEAK_INDUSTRY_BREADTH_PENALTY
    return 0.0

def sector_breadth_summary(sector_breadth):
    if not sector_breadth:
        return {"above": 0, "total": 0, "pct": None}
    total = sum(int(stats.get("total") or 0) for stats in sector_breadth.values())
    above = sum(int(stats.get("above") or 0) for stats in sector_breadth.values())
    return {
        "above": above,
        "total": total,
        "pct": (above / total * 100) if total else None,
    }

def sector_breadth_text(row):
    above = to_number_or_none(row.get("Sector Breadth Above EMA20"))
    total = to_number_or_none(row.get("Sector Breadth Total"))
    if above is None or total is None or total <= 0:
        return "N/A"
    sector = row.get("Sector") or "Sector"
    breadth_pct = above / total * 100
    return f"{sector} Breadth: {breadth_pct:.0f}% ({int(above)}/{int(total)})"

def market_regime_from_inputs(nifty_snapshot, breadth_summary):
    nifty_snapshot = nifty_snapshot or {}
    breadth_summary = breadth_summary or {}
    above_ema20 = bool(nifty_snapshot.get("nifty_above_ema20"))
    above_ema50 = bool(nifty_snapshot.get("nifty_above_ema50"))
    breadth_pct = to_number_or_none(breadth_summary.get("pct"))

    if not nifty_snapshot:
        if breadth_pct is not None and breadth_pct < MARKET_REGIME_WEAK_BREADTH_THRESHOLD:
            return "Weak"
        return "Unknown"
    if (
        above_ema20
        and above_ema50
        and (breadth_pct is None or breadth_pct >= MARKET_REGIME_BULL_BREADTH_THRESHOLD)
    ):
        return "Bull"
    if (
        (not above_ema20 and not above_ema50)
        or (breadth_pct is not None and breadth_pct < MARKET_REGIME_WEAK_BREADTH_THRESHOLD)
    ):
        return "Weak"
    return "Neutral"

def market_regime_snapshot(sector_breadth, market_stats=None):
    market_stats = market_stats if market_stats is not None else fetch_market_stats()
    breadth = market_stats_breadth_summary(market_stats)
    if breadth.get("pct") is None:
        breadth = sector_breadth_summary(sector_breadth)
        breadth["source"] = "scan_ema20"
    nifty_snapshot = fetch_nifty_regime_snapshot()
    return {
        **nifty_snapshot,
        "market_regime": market_regime_from_inputs(nifty_snapshot, breadth),
        "market_breadth_above_ema20": breadth.get("above"),
        "market_breadth_total": breadth.get("total"),
        "market_breadth_pct": breadth.get("pct"),
        "market_breadth_source": breadth.get("source"),
        "market_stats": market_stats,
    }

def market_regime_score_multiplier(market_regime):
    return WEAK_MARKET_REGIME_SCORE_MULTIPLIER if market_regime == "Weak" else 1.0

def is_explicit_false(value):
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"false", "no", "0"}
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    if isinstance(value, (bool, np.bool_)):
        return not bool(value)
    return False

def is_explicit_true(value):
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return False

def is_strict_weak_market_context(row):
    breadth_pct = to_number_or_none(row.get("Market Breadth %"))
    if breadth_pct is None or breadth_pct >= STRICT_WEAK_MARKET_SIGNAL_BREADTH_THRESHOLD:
        return False
    return (
        is_explicit_false(row.get("Nifty Above EMA20"))
        and is_explicit_false(row.get("Nifty Above EMA50"))
    )

def bank_weak_market_sector_penalty(row):
    if str(row.get("Sector") or "").strip().lower() != "bank":
        return 0.0
    if (
        is_explicit_false(row.get("Nifty Above EMA20"))
        and is_explicit_false(row.get("Nifty Above EMA50"))
    ):
        return BANK_WEAK_MARKET_SECTOR_SCORE_PENALTY
    return 0.0

def downgrade_buy_signal_for_weak_market(value):
    if not isinstance(value, str):
        return value
    signal = value.strip()
    if signal == "Strong Buy":
        return "Buy"
    if signal == "Buy":
        return "Hold"
    return value

def downgrade_intraday_signal_for_weak_market(value):
    if not isinstance(value, str):
        return value
    if value.strip() == "Strong Buy":
        return "Buy"
    return value

def apply_strict_weak_market_signal_downgrades(ranked_df):
    ranked_df["Weak Market Signal Downgrade"] = False
    if ranked_df.empty:
        return ranked_df
    downgrade_mask = ranked_df.apply(is_strict_weak_market_context, axis=1)
    if not downgrade_mask.any():
        return ranked_df
    ranked_df.loc[downgrade_mask, "Weak Market Signal Downgrade"] = True
    if "Intraday" in ranked_df.columns:
        ranked_df.loc[downgrade_mask, "Intraday"] = ranked_df.loc[
            downgrade_mask,
            "Intraday",
        ].apply(downgrade_intraday_signal_for_weak_market)
    for column in ("Swing", "Long-Term"):
        if column in ranked_df.columns:
            ranked_df.loc[downgrade_mask, column] = ranked_df.loc[
                downgrade_mask,
                column,
            ].apply(downgrade_buy_signal_for_weak_market)
    return ranked_df

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
        return 0.8
    if distance_pct <= 2.5:
        return 0.2
    if distance_pct <= 3:
        return 0.0
    return 0.0

def liquidity_adjustment(avg_volume_value):
    avg_volume_value = to_number_or_none(avg_volume_value)
    if avg_volume_value is None or avg_volume_value <= 0:
        return 0.0
    turnover_cr = avg_volume_value / 10_000_000
    if turnover_cr < 20:
        return 0.0
    if turnover_cr < 50:
        return 0.5
    if turnover_cr < 100:
        return 1.0
    if turnover_cr < 250:
        return 1.5
    return 2.2

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

def absolute_sector_leader_score(row):
    relative_strength = to_number_or_none(row.get("Relative Strength")) or 0.0
    avg_volume_value = to_number_or_none(row.get("Avg Volume Value")) or 0.0
    trend_persistence = to_number_or_none(row.get("Trend Persistence")) or 0.0
    turnover_cr = avg_volume_value / 10_000_000
    rs_score = (
        1.0 if relative_strength >= 6
        else 0.9 if relative_strength >= 3
        else 0.7 if relative_strength >= 1
        else 0.5 if relative_strength > 0
        else 0.0
    )
    liquidity_score = (
        1.0 if turnover_cr >= 100
        else 0.8 if turnover_cr >= 50
        else 0.6 if turnover_cr >= 20
        else 0.4 if turnover_cr >= 10
        else 0.0
    )
    persistence_score = (
        1.0 if trend_persistence >= 75
        else 0.7 if trend_persistence >= 60
        else 0.5 if trend_persistence >= 50
        else 0.0
    )
    return (rs_score + liquidity_score + persistence_score) / 3

def sector_leader_adjustment_columns(ranked_df):
    ranked_df["Sector Leader Score"] = 0.5
    ranked_df["Sector Leader Adjustment"] = 0.0
    if ranked_df.empty:
        return ranked_df

    metrics = ["Relative Strength", "Avg Volume Value", "Trend Persistence"]
    for metric in metrics:
        ranked_df[metric] = pd.to_numeric(ranked_df[metric], errors="coerce")
    if "Sector Relative Strength %" not in ranked_df.columns:
        ranked_df["Sector Relative Strength %"] = 0.0
    ranked_df["Sector Relative Strength %"] = pd.to_numeric(
        ranked_df["Sector Relative Strength %"],
        errors="coerce",
    ).fillna(0.0)
    if "Industry Advance Ratio" not in ranked_df.columns:
        ranked_df["Industry Advance Ratio"] = np.nan
    ranked_df["Industry Advance Ratio"] = pd.to_numeric(
        ranked_df["Industry Advance Ratio"],
        errors="coerce",
    )

    for _, sector_df in ranked_df.groupby("Sector", dropna=False):
        if len(sector_df) < 2:
            index = sector_df.index[0]
            row = sector_df.iloc[0]
            singleton_score = absolute_sector_leader_score(row)
            singleton_adjustment = max(
                0.0,
                (singleton_score - 0.5) * 2 * MAX_SECTOR_LEADER_RANKING_ADJUSTMENT,
            )
            if (to_number_or_none(row.get("Sector Relative Strength %")) or 0.0) < 0:
                singleton_adjustment *= 0.5
            industry_advance_ratio = to_number_or_none(row.get("Industry Advance Ratio"))
            if (
                industry_advance_ratio is not None
                and industry_advance_ratio < WEAK_INDUSTRY_ADVANCE_RATIO_THRESHOLD
                and singleton_adjustment > 0
            ):
                singleton_adjustment = min(
                    singleton_adjustment * WEAK_INDUSTRY_SECTOR_LEADER_MULTIPLIER,
                    MAX_WEAK_INDUSTRY_SECTOR_LEADER_ADJUSTMENT,
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

        relative_leader_score = sum(metric_ranks) / len(metric_ranks)
        absolute_leader_score = sector_df.apply(absolute_sector_leader_score, axis=1)
        leader_score = relative_leader_score
        if len(sector_df) <= 5:
            leader_score = pd.concat(
                [relative_leader_score, absolute_leader_score],
                axis=1,
            ).max(axis=1)
        leader_adjustment = (
            (leader_score - 0.5)
            * 2
            * MAX_SECTOR_LEADER_RANKING_ADJUSTMENT
        ).clip(
            lower=-MAX_SECTOR_LEADER_RANKING_ADJUSTMENT,
            upper=MAX_SECTOR_LEADER_RANKING_ADJUSTMENT,
        )
        weak_sector_leader = (
            (sector_df["Sector Relative Strength %"] < 0)
            & (leader_adjustment > 0)
        )
        leader_adjustment = leader_adjustment.mask(
            weak_sector_leader,
            leader_adjustment * 0.5,
        )
        weak_industry_leader = (
            (sector_df["Industry Advance Ratio"] < WEAK_INDUSTRY_ADVANCE_RATIO_THRESHOLD)
            & (leader_adjustment > 0)
        )
        leader_adjustment = leader_adjustment.mask(
            weak_industry_leader,
            (leader_adjustment * WEAK_INDUSTRY_SECTOR_LEADER_MULTIPLIER).clip(
                upper=MAX_WEAK_INDUSTRY_SECTOR_LEADER_ADJUSTMENT,
            ),
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
    if weak_liquidity and weak_sector:
        return False

    entry_gap = to_number_or_none(row.get("Entry Distance %"))
    consolidation_candles = to_number_or_none(row.get("Consolidation Candles")) or 0
    has_consolidation = consolidation_candles >= MIN_SWING_CONSOLIDATION_CANDLES
    if (
        entry_gap is not None
        and entry_gap < MIN_SWING_PULLBACK_ENTRY_GAP_PERCENT
        and not has_consolidation
    ):
        return False

    return True

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
    sector_breadth=None,
    market_regime=None,
    ranking_weights=None,
    intraday=False,
):
    sector_momentum = sector_momentum or {}
    sector_breadth = sector_breadth or {}
    market_regime = market_regime or {"market_regime": "Unknown"}
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
        ranked_df["Sector Breadth Above EMA20"] = np.nan
        ranked_df["Sector Breadth Total"] = np.nan
        ranked_df["Sector Breadth %"] = np.nan
        ranked_df["Sector Breadth Text"] = None
        ranked_df["Market Regime"] = market_regime.get("market_regime", "Unknown")
        ranked_df["Market Breadth %"] = market_regime.get("market_breadth_pct")
        ranked_df["Market Breadth Source"] = market_regime.get("market_breadth_source")
        ranked_df["Industry Breadth Text"] = None
        ranked_df["Industry Advance Ratio"] = np.nan
        ranked_df["Industry Breadth Penalty"] = 0.0
        ranked_df["Weak Market Signal Downgrade"] = False
        ranked_df["Bank Weak Market Penalty"] = 0.0
        ranked_df["Market Regime Multiplier"] = market_regime_score_multiplier(
            market_regime.get("market_regime", "Unknown")
        )
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
        ranked_df["Consolidation Candles"] = np.nan
        ranked_df["Sector Exhaustion Penalty"] = 0.0
        ranked_df["Trend Persistence Adjustment"] = 0.0
        ranked_df["Sector Leader Score"] = 0.5
        ranked_df["Sector Leader Adjustment"] = 0.0
        ranked_df["Setup Type"] = pd.Series(dtype=object)
        ranked_df["Setup Sample Size"] = pd.Series(dtype=int)
        ranked_df["Setup Evidence"] = pd.Series(dtype=object)
        ranked_df["Historical Expectancy Adjustment"] = 0.0
        ranked_df["Setup Expectancy Adjustment"] = 0.0
        ranked_df["Exhaustion Penalty"] = 0.0
        ranked_df["Raw Ranking Score"] = pd.Series(dtype=float)
        ranked_df["Ranking Score"] = pd.Series(dtype=float)
        ranked_df["Confidence Grade"] = pd.Series(dtype=object)
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
    ranked_df["Sector Breadth Above EMA20"] = ranked_df["Sector"].map(
        lambda sector: sector_breadth.get(sector, {}).get("above")
    )
    ranked_df["Sector Breadth Total"] = ranked_df["Sector"].map(
        lambda sector: sector_breadth.get(sector, {}).get("total")
    )
    ranked_df["Sector Breadth %"] = ranked_df["Sector"].map(
        lambda sector: sector_breadth.get(sector, {}).get("pct")
    )
    ranked_df["Sector Breadth Text"] = ranked_df.apply(sector_breadth_text, axis=1)
    ranked_df["Market Regime"] = market_regime.get("market_regime", "Unknown")
    ranked_df["Market Breadth Above EMA20"] = market_regime.get("market_breadth_above_ema20")
    ranked_df["Market Breadth Total"] = market_regime.get("market_breadth_total")
    ranked_df["Market Breadth %"] = market_regime.get("market_breadth_pct")
    ranked_df["Market Breadth Source"] = market_regime.get("market_breadth_source")
    ranked_df["Industry Breadth Text"] = ranked_df["Sector"].map(
        lambda sector: market_stats_sector_text(sector, market_regime.get("market_stats"))
    )
    ranked_df["Industry Advance Ratio"] = ranked_df["Sector"].map(
        lambda sector: market_stats_sector_advance_ratio(sector, market_regime.get("market_stats"))
    )
    ranked_df["Industry Breadth Penalty"] = ranked_df["Industry Advance Ratio"].apply(
        industry_breadth_penalty
    )
    ranked_df["Nifty Above EMA20"] = market_regime.get("nifty_above_ema20")
    ranked_df["Nifty Above EMA50"] = market_regime.get("nifty_above_ema50")
    ranked_df["Market Regime Multiplier"] = market_regime_score_multiplier(
        market_regime.get("market_regime", "Unknown")
    )
    ranked_df["Sector Relative Strength %"] = ranked_df["Sector Performance %"] - nifty_5d_return
    ranked_df["Sector Momentum Score"] = ranked_df["Sector Relative Strength %"].apply(sector_momentum_adjustment)
    ranked_df["Bank Weak Market Penalty"] = ranked_df.apply(bank_weak_market_sector_penalty, axis=1)
    ranked_df["Sector Momentum Score"] = (
        ranked_df["Sector Momentum Score"] + ranked_df["Bank Weak Market Penalty"]
    )
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
    if "Consolidation Candles" not in ranked_df.columns:
        ranked_df["Consolidation Candles"] = np.nan
    ranked_df["Latest Move %"] = pd.to_numeric(ranked_df["Latest Move %"], errors="coerce")
    ranked_df["EMA20 Distance %"] = pd.to_numeric(ranked_df["EMA20 Distance %"], errors="coerce")
    ranked_df["Fresh Breakout Age"] = pd.to_numeric(ranked_df["Fresh Breakout Age"], errors="coerce")
    ranked_df["Consolidation Candles"] = pd.to_numeric(ranked_df["Consolidation Candles"], errors="coerce")
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
    ranked_df["Setup Type"] = ranked_df.apply(classify_setup_type, axis=1)
    expectancy_lookup = {} if intraday else load_historical_expectancy_lookup()
    setup_expectancy_stats = {} if intraday else load_setup_expectancy_lookup()
    ranked_df["Setup Sample Size"] = ranked_df["Setup Type"].map(
        lambda setup_type: int(setup_expectancy_stats.get(setup_type, {}).get("trades") or 0)
    )
    ranked_df["Setup Evidence"] = ranked_df["Setup Sample Size"].apply(setup_evidence_level)
    ranked_df["Historical Expectancy Adjustment"] = ranked_df.apply(
        lambda row: historical_expectancy_adjustment(row, expectancy_lookup),
        axis=1,
    )
    ranked_df["Setup Expectancy Adjustment"] = ranked_df.apply(
        lambda row: setup_expectancy_adjustment(row, setup_expectancy_stats),
        axis=1,
    )
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
        + ranked_df["Historical Expectancy Adjustment"]
        + ranked_df["Setup Expectancy Adjustment"]
        + ranked_df["Sector Exhaustion Penalty"]
        + ranked_df["Industry Breadth Penalty"]
        + ranked_df["Exhaustion Penalty"]
        + ranked_df["Gap Risk Penalty"]
    )
    ranked_df["Raw Ranking Score"] = raw_opportunity_score.round(3)
    base_ranking_score = normalize_opportunity_score(raw_opportunity_score)
    ranked_df["Ranking Score Before Regime"] = base_ranking_score.round(1)
    ranked_df["Ranking Score"] = (
        base_ranking_score * ranked_df["Market Regime Multiplier"]
    ).round(1)
    ranked_df = apply_strict_weak_market_signal_downgrades(ranked_df)
    ranked_df["Confidence Grade"] = ranked_df.apply(confidence_grade, axis=1)
    return ranked_df

def confidence_grade(row):
    opportunity_score = to_number_or_none(row.get("Ranking Score"))
    if opportunity_score is None:
        return "C"
    market_regime = row.get("Market Regime") or "Unknown"
    if opportunity_score >= 90:
        grade = "A+"
    elif opportunity_score >= 85:
        grade = "A"
    elif opportunity_score >= 80:
        grade = "B+"
    elif opportunity_score >= 70:
        grade = "B"
    elif opportunity_score >= 60:
        grade = "C+"
    else:
        grade = "C"

    grade_order = ["C", "C+", "B", "B+", "A", "A+"]
    regime_caps = {
        "Neutral": "A",
        "Weak": "B+",
    }
    grade_cap = regime_caps.get(str(market_regime).strip().title())
    if grade_cap and grade_order.index(grade) > grade_order.index(grade_cap):
        return grade_cap
    return grade

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

def setup_type_display_name(setup_type):
    if not setup_type:
        return "Unknown"
    return SETUP_TYPE_LABELS.get(
        setup_type,
        str(setup_type).replace("_", " ").title(),
    )

def format_compact_currency(value):
    value = to_number_or_none(value)
    if value is None:
        return "N/A"
    if abs(value) >= 10_000_000:
        return f"₹{value / 10_000_000:.2f}Cr"
    if abs(value) >= 100_000:
        return f"₹{value / 100_000:.2f}L"
    return f"₹{value:.0f}"

def format_yes_no(value):
    if value is None:
        return "N/A"
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1"}:
            return "Yes"
        if text in {"false", "no", "0"}:
            return "No"
        return "N/A"
    return "Yes" if bool(value) else "No"

def ranking_audit_text(row):
    grade = row.get("Confidence Grade") or confidence_grade(row)
    market_regime = row.get("Market Regime") or "Unknown"
    market_multiplier = to_number_or_none(row.get("Market Regime Multiplier")) or 1.0
    market_breadth_pct = to_number_or_none(row.get("Market Breadth %"))
    market_breadth_source = row.get("Market Breadth Source")
    nifty_above_ema20 = row.get("Nifty Above EMA20")
    nifty_above_ema50 = row.get("Nifty Above EMA50")
    weak_market_signal_downgrade = is_explicit_true(row.get("Weak Market Signal Downgrade"))
    exhaustion_penalty = to_number_or_none(row.get("Exhaustion Penalty")) or 0.0
    gap_risk_penalty = to_number_or_none(row.get("Gap Risk Penalty")) or 0.0
    fresh_breakout_bonus_value = to_number_or_none(row.get("Fresh Breakout Bonus")) or 0.0
    sector_exhaustion_penalty_value = to_number_or_none(row.get("Sector Exhaustion Penalty")) or 0.0
    trend_persistence_adjustment_value = to_number_or_none(row.get("Trend Persistence Adjustment")) or 0.0
    sector_leader_adjustment_value = to_number_or_none(row.get("Sector Leader Adjustment")) or 0.0
    industry_breadth_penalty_value = to_number_or_none(row.get("Industry Breadth Penalty")) or 0.0
    bank_weak_market_penalty_value = to_number_or_none(row.get("Bank Weak Market Penalty")) or 0.0
    historical_expectancy_adjustment_value = to_number_or_none(row.get("Historical Expectancy Adjustment")) or 0.0
    setup_expectancy_adjustment_value = to_number_or_none(row.get("Setup Expectancy Adjustment")) or 0.0
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

    consolidation_text = ""
    consolidation_candles = to_number_or_none(row.get("Consolidation Candles"))
    if consolidation_candles is not None and consolidation_candles >= MIN_SWING_CONSOLIDATION_CANDLES:
        consolidation_text = f" | Consolidation: {format_number(consolidation_candles, 0)} candles"

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

    industry_breadth_penalty_text = ""
    if industry_breadth_penalty_value < 0:
        industry_breadth_penalty_text = (
            f" | Industry Breadth Penalty: {format_number(industry_breadth_penalty_value, 1)}"
        )

    bank_weak_market_penalty_text = ""
    if bank_weak_market_penalty_value < 0:
        bank_weak_market_penalty_text = (
            f" | Bank Weak Market: {format_number(bank_weak_market_penalty_value, 1)}"
        )

    historical_expectancy_text = ""
    if abs(historical_expectancy_adjustment_value) > 0:
        historical_expectancy_text = (
            f" | Historical EV: {format_number(historical_expectancy_adjustment_value, 1)} "
            f"({row.get('Setup Type') or 'setup'})"
        )

    setup_expectancy_text = ""
    if abs(setup_expectancy_adjustment_value) > 0:
        setup_expectancy_text = (
            f" | Setup Expectancy: {format_number(setup_expectancy_adjustment_value, 1)} "
            f"({row.get('Setup Type') or 'setup'})"
        )

    market_regime_text = (
        f"Market Regime: {market_regime}  \n"
        f"Breadth: {format_percent(market_breadth_pct, 0)}  \n"
        f"Nifty Above EMA20: {format_yes_no(nifty_above_ema20)}  \n"
        f"Nifty Above EMA50: {format_yes_no(nifty_above_ema50)}"
    )
    if market_breadth_source == "brkpoint_market_stats":
        market_regime_text += "  \nBreadth Source: Market Stats"
    if market_multiplier < 1:
        market_regime_text += f"  \nRegime Score Multiplier: {market_multiplier:.2f}"
    if weak_market_signal_downgrade:
        market_regime_text += "  \nSignal Downgrade: Strict Weak Market"

    industry_breadth_text = clean_display_text(row.get("Industry Breadth Text"), fallback="")
    if industry_breadth_text:
        industry_breadth_text = f" | {industry_breadth_text} "
    else:
        industry_breadth_text = ""

    return (
        f"Grade: {grade}  \n"
        f"{market_regime_text}  \n"
        f"Opportunity Score: {format_number(row.get('Ranking Score'))} | "
        f"RS: {format_percent(row.get('Relative Strength'))} "
        f"({format_number(row.get('Relative Strength Score'), 1)}) | "
        f"RVOL: {format_number(row.get('RVOL'))} "
        f"({format_number(row.get('RVOL Score'), 1)}{effective_rvol_text}) | "
        f"Sector: {row.get('Sector', 'Other')} "
        f"{format_percent(row.get('Sector Performance %'))} "
        f"(Rel: {format_percent(row.get('Sector Relative Strength %'))}) "
        f" | {row.get('Sector Breadth Text') or sector_breadth_text(row)} "
        f"{industry_breadth_text}"
        f"({format_number(row.get('Sector Momentum Score'), 1)}) | "
        f"RR: {format_number(row.get('Reward/Risk'))} | "
        f"Entry Gap: {format_percent(row.get('Entry Distance %'))} "
        f"({format_number(row.get('Entry Distance Score'), 1)}) | "
        f"Liquidity: {format_compact_currency(row.get('Avg Volume Value'))} "
        f"({format_number(row.get('Liquidity Score'), 1)})"
        f"{fresh_breakout_text}"
        f"{consolidation_text}"
        f"{trend_persistence_text}"
        f"{sector_leader_text}"
        f"{industry_breadth_penalty_text}"
        f"{bank_weak_market_penalty_text}"
        f"{historical_expectancy_text}"
        f"{setup_expectancy_text}"
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
    st.subheader(f"📅 Analysis for {app_now().strftime('%d %b %Y')}")

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
    if 'show_buy_above_cmp_only' not in st.session_state:
        st.session_state.show_buy_above_cmp_only = False
    st.sidebar.checkbox(
        "Only show Buy Above CMP setups",
        key="show_buy_above_cmp_only",
        help="Hide entries below or at current price, including Wait for Pullback setups."
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
        
        saved_picks_count = 0
        if not top_picks_df.empty:
            saved_picks_count = insert_top_picks(top_picks_df, pick_type="daily")
            st.session_state.last_daily_top_picks = top_picks_df.copy()
            
        progress_bar.empty()
        loading_text.empty()
        
        # Display Top 5
        if not top_picks_df.empty:
            st.subheader("🏆 Today's Top 5 Stocks")
            st.caption(f"Saved {saved_picks_count} picks to historical database: {DB_PATH}")
            st.caption(history_storage_notice())
            for _, row in top_picks_df.iterrows():
                grade = row.get("Confidence Grade") or confidence_grade(row)
                swing_signal = swing_signal_for_grade(grade)
                with st.expander(f"{row['Symbol']} - Grade {grade} - {tooltip('Score', TOOLTIPS['Score'])}: {row['Score']}/7"):
                    current_price = row.get('Current Price', 'N/A')
                    buy_at = row.get('Buy At', 'N/A')
                    stop_loss = row.get('Stop Loss', 'N/A')
                    target = row.get('Target', 'N/A')
                    hold_advice = expected_hold_text(row)
                    buy_icon, buy_label, buy_display_value = entry_display_details(
                        current_price,
                        buy_at,
                        row.get('Entry Type', 'Standard')
                    )
                    if st.session_state.recommendation_mode == "Adaptive":
                        st.markdown(f"""
                        {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: {format_currency(current_price)}  
                        {buy_icon} {buy_label}: {format_currency(buy_display_value)} | Stop Loss: {format_currency(stop_loss)}  
                        Target: {format_currency(target)}  
                        **Confidence Grade**: {grade}
                        **Audit**: {ranking_audit_text(row)}
                        **Hold Plan**:
                        {hold_advice}
                        Recommendation: {colored_recommendation(row.get('Recommendation', 'N/A'))}  
                        Regime: {row.get('Regime', 'N/A')}  
                        Position Size (₹): {row.get('Position Size', 'N/A')}  
                        Trailing Stop: ₹{row.get('Trailing Stop', 'N/A')}  
                        Reason: {row.get('Reason', 'N/A')}
                        """)
                    else:
                        st.markdown(f"""
                        {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: {format_currency(current_price)}  
                        {buy_icon} {buy_label}: {format_currency(buy_display_value)} | Stop Loss: {format_currency(stop_loss)}  
                        Target: {format_currency(target)}  
                        **Confidence Grade**: {grade}
                        **Audit**: {ranking_audit_text(row)}
                        **Hold Plan**:
                        {hold_advice}
                        Intraday: {colored_recommendation(row.get('Intraday', 'N/A'))}  
                        Swing: {colored_recommendation(swing_signal)}  
                        Short-Term: {colored_recommendation(row.get('Short-Term', 'N/A'))}  
                        Long-Term: {colored_recommendation(row.get('Long-Term', 'N/A'))}  
                        Mean Reversion: {colored_recommendation(row.get('Mean_Reversion', 'N/A'))}  
                        Breakout: {colored_recommendation(row.get('Breakout', 'N/A'))}  
                        Ichimoku Trend: {colored_recommendation(row.get('Ichimoku_Trend', 'N/A'))}
                        """)
        else:
            successful_df = (
                full_report_df[full_report_df["Status"] == "Success"].copy()
                if not full_report_df.empty and "Status" in full_report_df.columns
                else pd.DataFrame()
            )
            if successful_df.empty:
                st.warning("No top picks available because no stocks completed successfully.")
            elif st.session_state.get("show_buy_above_cmp_only", False):
                st.warning(
                    "No valid Buy Above CMP swing picks today. "
                    "Market weak. Wait for better setup."
                )
            else:
                st.warning(
                    "No valid swing picks today. "
                    "Market weak. Wait for better setup."
                )
            
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
                    file_name=f"strategy_report_{app_now().strftime('%Y%m%d')}.csv",
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
        saved_intraday_count = insert_top_picks(intraday_results, pick_type="intraday")
        if not intraday_results.empty:
            st.session_state.last_intraday_top_picks = intraday_results.copy()
        progress_bar.empty()
        loading_text.empty()
        if not intraday_results.empty:
            st.subheader("🏆 Top 5 Intraday Stocks (⚡ Fast Exit)")
            st.caption(f"Saved {saved_intraday_count} picks to historical database: {DB_PATH}")
            st.caption(history_storage_notice())
            for _, row in intraday_results.iterrows():
                grade = row.get("Confidence Grade") or confidence_grade(row)
                with st.expander(f"{row['Symbol']} - Grade {grade} - {tooltip('Score', TOOLTIPS['Score'])}: {row['Score']}/7"):
                    current_price = row.get('Current Price', 'N/A')
                    buy_at = row.get('Buy At', 'N/A')
                    stop_loss = row.get('Stop Loss', 'N/A')
                    target = row.get('Target', 'N/A')
                    buy_icon, buy_label, buy_display_value = entry_display_details(
                        current_price,
                        buy_at,
                        row.get('Entry Type', 'Standard'),
                        include_breakout_context=True
                    )
                    if st.session_state.recommendation_mode == "Adaptive":
                        st.markdown(f"""
                        {tooltip('Current Price', TOOLTIPS['Stop Loss'])}: {format_currency(current_price)}  
                        {buy_icon} {buy_label}: {format_currency(buy_display_value)} | Stop Loss: {format_currency(stop_loss)}  
                        Target: {format_currency(target)}  
                        **Confidence Grade**: {grade}
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
                        {buy_icon} {buy_label}: {format_currency(buy_display_value)} | Stop Loss: {format_currency(stop_loss)}  
                        Target: {format_currency(target)}  
                        **Confidence Grade**: {grade}
                        **Audit**: {ranking_audit_text(row)}
                        Intraday: {colored_recommendation(row.get('Intraday', 'N/A'))}
                        
                        **Strategy Notes:**
                        {clean_display_text(row.get('Pattern Notes'))}
                        
                        **Entry Advice:**
                        {clean_display_text(row.get('Entry Strategy'))}
                        """)
        else:
            if st.session_state.get("show_buy_above_cmp_only", False):
                st.warning(
                    "No Buy Above CMP intraday setups matched the current filters. "
                    "Turn off 'Only show Buy Above CMP setups' to include Wait for Pullback candidates."
                )
            else:
                st.warning("⚠️ No intraday picks available due to data issues.")

    # Historical picks button
    if st.button("📜 View Historical Picks"):
        with st.spinner("Updating holding-period outcomes..."):
            updated_outcomes = update_holding_period_outcomes()
            updated_exit_advice = update_exit_advice()
            updated_setup_expectancy = refresh_setup_expectancy_database()
        conn = get_db_connection()
        history_df = pd.read_sql_query("SELECT * FROM daily_picks ORDER BY date DESC", conn)
        setup_expectancy_df = pd.read_sql_query("SELECT * FROM setup_expectancy ORDER BY setup_expectancy DESC", conn)
        conn.close()
        if not history_df.empty:
            st.subheader("📜 Historical Top Picks")
            st.caption(f"Database: {DB_PATH}")
            st.caption(history_storage_notice())
            if updated_outcomes:
                st.caption(f"Updated holding-period outcomes for {updated_outcomes} picks.")
            if updated_exit_advice:
                st.caption(f"Updated exit advice for {updated_exit_advice} picks.")
            if updated_setup_expectancy:
                st.caption(f"Updated setup expectancy database for {updated_setup_expectancy} setup types.")
            all_dates = sorted(history_df['date'].unique(), reverse=True)
            date_filter = st.selectbox("Filter by Date", ["All"] + all_dates)
            pick_type_filter = st.selectbox("Filter by Pick Type", ["All", "daily", "intraday"])
            filtered_df = history_df.copy()
            if pick_type_filter != "All":
                filtered_df = filtered_df[filtered_df['pick_type'] == pick_type_filter]
            if date_filter != "All":
                filtered_df = filtered_df[filtered_df['date'] == date_filter]
            expectancy_df = holding_period_expectancy(history_df[history_df["pick_type"] == "daily"])
            setup_metrics_df = setup_holding_metrics(history_df[history_df["pick_type"] == "daily"])
            setup_win_rate_df = setup_type_win_rate_table(history_df[history_df["pick_type"] == "daily"])
            if not setup_win_rate_df.empty:
                with st.expander("Setup-Type Win Rate", expanded=True):
                    st.dataframe(setup_win_rate_df, use_container_width=True)
            if not setup_expectancy_df.empty:
                with st.expander("Setup Expectancy Database", expanded=True):
                    st.dataframe(setup_expectancy_df, use_container_width=True)
            if not setup_metrics_df.empty:
                with st.expander("Setup Holding Metrics", expanded=False):
                    st.dataframe(setup_metrics_df, use_container_width=True)
            if not expectancy_df.empty:
                with st.expander("Holding Period Expectancy", expanded=False):
                    st.dataframe(expectancy_df, use_container_width=True)
            st.dataframe(filtered_df)
        else:
            st.warning("⚠️ No historical data available in the SQLite database.")
            st.caption(f"Database: {DB_PATH}")
            st.info(history_storage_notice())
            session_frames = []
            if isinstance(st.session_state.get("last_daily_top_picks"), pd.DataFrame):
                latest_daily = st.session_state.last_daily_top_picks.copy()
                latest_daily["pick_type"] = "daily"
                latest_daily["date"] = app_date_string()
                session_frames.append(latest_daily)
            if isinstance(st.session_state.get("last_intraday_top_picks"), pd.DataFrame):
                latest_intraday = st.session_state.last_intraday_top_picks.copy()
                latest_intraday["pick_type"] = "intraday"
                latest_intraday["date"] = app_date_string()
                session_frames.append(latest_intraday)
            if session_frames:
                st.info("Showing current-session generated picks. The historical SQLite database did not retain rows.")
                st.dataframe(pd.concat(session_frames, ignore_index=True))

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
                buy_icon, buy_label, buy_display_value = entry_display_details(
                    current_price,
                    buy_at,
                    entry_type
                )
                label = f"{buy_icon} {buy_label}".strip()
                
                st.metric(label, format_currency(buy_display_value))
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
