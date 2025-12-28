def calculate_sector_performance():
    """
    Calculates the real-time performance of each sector based on constituent stocks.
    Returns a DataFrame sorted by % Change.
    """
    sector_performance = []
    
    # Flatten all symbols to fetch data in one batch for efficiency
    all_symbols = []
    for sector, symbols in SECTORS.items():
        all_symbols.extend(symbols)
    
    # Remove duplicates
    all_symbols = list(set(all_symbols))
    
    # Batch fetch live data (Lightweight fetch: LTP & Close only)
    # We use a simplified fetch here or just reuse fetch_stock_data if cache hits are high
    # For speed, we will use fetch_stock_data in parallel but only keep necessary cols
    
    live_data = {}
    
    # We can use ThreadPool for speed
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_symbol = {executor.submit(fetch_stock_data, symbol, "1d"): symbol for symbol in all_symbols}
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                data = future.result()
                if not data.empty:
                     # Calculate % Change
                    current = data['Close'].iloc[-1]
                    prev_close = data['Open'].iloc[-1] # Approximation if 'Previous Close' not explicit, or use Close[-2]
                    if len(data) > 1:
                        prev_close = data['Close'].iloc[-2]
                    
                    change = ((current - prev_close) / prev_close) * 100
                    live_data[symbol] = change
            except Exception as e:
                pass

    # Aggregate by Sector
    for sector, symbols in SECTORS.items():
        sector_changes = []
        for symbol in symbols:
            if symbol in live_data:
                sector_changes.append(live_data[symbol])
        
        if sector_changes:
            avg_change = sum(sector_changes) / len(sector_changes)
            sentiment = "Bullish" if avg_change > 0 else "Bearish"
            if avg_change > 1.0: sentiment = "Strong Bullish"
            if avg_change < -1.0: sentiment = "Strong Bearish"
            
            sector_performance.append({
                "Sector": sector,
                "% Change": round(avg_change, 2),
                "Sentiment": sentiment
            })
    
    if not sector_performance:
        return pd.DataFrame()
        
    df = pd.DataFrame(sector_performance)
    return df.sort_values(by="% Change", ascending=False)
