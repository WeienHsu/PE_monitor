"""
Hard-coded mapping of common sector/industry ETFs → Damodaran industry name.

The Damodaran spreadsheet publishes a trailing P/E for each US industry.
This map lets us look up an ETF's "industry benchmark PE" so the ETF's own
trailing PE can be expressed as a ratio (cheap if < 1, expensive if > 1).

Names on the right must match Damodaran's Industry Name exactly (whitespace
is normalised at lookup time, so trailing spaces in the source are OK).
"""

ETF_TO_INDUSTRY: dict[str, str] = {
    # Select Sector SPDRs
    "XLK": "Software (System & Application)",
    "XLF": "Banks (Regional)",
    "XLV": "Healthcare Products",
    "XLY": "Retail (General)",
    "XLP": "Food Processing",
    "XLE": "Oil/Gas (Production and Exploration)",
    "XLI": "Machinery",
    "XLB": "Chemical (Basic)",
    "XLU": "Utility (General)",
    "XLRE": "R.E.I.T.",
    "XLC": "Telecom. Services",
    # iShares / Vanguard / SPDR sector ETFs
    "VGT": "Software (System & Application)",
    "VHT": "Healthcare Products",
    "VFH": "Banks (Regional)",
    "VCR": "Retail (General)",
    "VDC": "Food Processing",
    "VDE": "Oil/Gas (Production and Exploration)",
    "VIS": "Machinery",
    "VAW": "Chemical (Basic)",
    "VPU": "Utility (General)",
    "VNQ": "R.E.I.T.",
    "VOX": "Telecom. Services",
    # Semiconductor
    "SOXX": "Semiconductor",
    "SMH": "Semiconductor",
    # Cybersecurity / Software
    "CIBR": "Software (Internet)",
    "HACK": "Software (Internet)",
    "IGV": "Software (System & Application)",
    # Biotech
    "IBB": "Drugs (Biotechnology)",
    "XBI": "Drugs (Biotechnology)",
    # Homebuilders / Transports
    "ITB": "Homebuilding",
    "XHB": "Homebuilding",
    "IYT": "Transportation",
    # Gold miners (different from GLD — miners are equities)
    "GDX": "Metals & Mining",
    "GDXJ": "Metals & Mining",
    # REITs
    "IYR": "R.E.I.T.",
    "REM": "R.E.I.T.",
    # Financials / Banks
    "KBE": "Banks (Regional)",
    "KRE": "Banks (Regional)",
    # Energy subsegments
    "XOP": "Oil/Gas (Production and Exploration)",
    "OIH": "Oilfield Svcs/Equip.",
    # Consumer
    "XRT": "Retail (General)",
    # Clean energy / EVs (best-fit industry; Damodaran has no direct match)
    "ICLN": "Power",
    "TAN": "Power",
}


def get_industry_for_etf(ticker: str) -> str | None:
    """Return the Damodaran industry name for a sector/industry ETF, or None."""
    return ETF_TO_INDUSTRY.get(ticker.upper())


# yfinance .info["industry"] is much more granular than Damodaran's table.
# Map the most common yfinance industries onto the closest Damodaran entry.
# Case-insensitive lookup via get_damodaran_for_yfinance_industry().
YFINANCE_TO_DAMODARAN: dict[str, str] = {
    # Technology
    "consumer electronics": "Computers/Peripherals",
    "semiconductors": "Semiconductor",
    "semiconductor equipment & materials": "Semiconductor Equip",
    "software - application": "Software (System & Application)",
    "software - infrastructure": "Software (System & Application)",
    "information technology services": "Information Services",
    "communication equipment": "Telecom. Equipment",
    "computer hardware": "Computers/Peripherals",
    "electronic components": "Electronics (General)",
    "internet content & information": "Software (Internet)",
    "internet retail": "Retail (Online)",
    # Healthcare
    "drug manufacturers - general": "Drugs (Pharmaceutical)",
    "drug manufacturers - specialty & generic": "Drugs (Pharmaceutical)",
    "biotechnology": "Drugs (Biotechnology)",
    "medical devices": "Healthcare Products",
    "medical instruments & supplies": "Healthcare Products",
    "healthcare plans": "Healthcare Support Services",
    "diagnostics & research": "Healthcare Support Services",
    # Financials
    "banks - diversified": "Bank (Money Center)",
    "banks - regional": "Banks (Regional)",
    "insurance - diversified": "Insurance (General)",
    "insurance - life": "Insurance (Life)",
    "insurance - property & casualty": "Insurance (Prop/Cas.)",
    "asset management": "Investments & Asset Management",
    "capital markets": "Brokerage & Investment Banking",
    "credit services": "Financial Svcs. (Non-bank & Insurance)",
    # Energy
    "oil & gas integrated": "Oil/Gas (Integrated)",
    "oil & gas e&p": "Oil/Gas (Production and Exploration)",
    "oil & gas refining & marketing": "Oil/Gas Distribution",
    "oil & gas equipment & services": "Oilfield Svcs/Equip.",
    "oil & gas midstream": "Oil/Gas Distribution",
    # Consumer
    "beverages - non-alcoholic": "Beverage (Soft)",
    "beverages - wineries & distilleries": "Beverage (Alcoholic)",
    "tobacco": "Tobacco",
    "packaged foods": "Food Processing",
    "confectioners": "Food Processing",
    "household & personal products": "Household Products",
    "apparel retail": "Apparel",
    "specialty retail": "Retail (Special Lines)",
    "department stores": "Retail (General)",
    "auto manufacturers": "Auto & Truck",
    "auto parts": "Auto Parts",
    "restaurants": "Restaurant/Dining",
    "lodging": "Hotel/Gaming",
    # Industrials
    "aerospace & defense": "Aerospace/Defense",
    "industrial distribution": "Machinery",
    "specialty industrial machinery": "Machinery",
    "farm & heavy construction machinery": "Machinery",
    "engineering & construction": "Engineering/Construction",
    "airlines": "Air Transport",
    "trucking": "Trucking",
    "railroads": "Transportation",
    "integrated freight & logistics": "Transportation",
    # Materials
    "specialty chemicals": "Chemical (Specialty)",
    "chemicals": "Chemical (Basic)",
    "steel": "Steel",
    "other industrial metals & mining": "Metals & Mining",
    "copper": "Metals & Mining",
    "gold": "Precious Metals",
    # Utilities & REITs
    "utilities - regulated electric": "Utility (General)",
    "utilities - diversified": "Utility (General)",
    "utilities - renewable": "Power",
    "reit - retail": "R.E.I.T.",
    "reit - residential": "R.E.I.T.",
    "reit - office": "R.E.I.T.",
    "reit - industrial": "R.E.I.T.",
    "reit - healthcare facilities": "R.E.I.T.",
    "reit - diversified": "R.E.I.T.",
    # Communication
    "telecom services": "Telecom. Services",
    "entertainment": "Entertainment",
    "broadcasting": "Broadcasting",
    "advertising agencies": "Advertising",
}


def get_damodaran_for_yfinance_industry(yf_industry: str | None) -> str | None:
    """Map a yfinance .info["industry"] string to the closest Damodaran industry."""
    if not yf_industry:
        return None
    return YFINANCE_TO_DAMODARAN.get(yf_industry.strip().lower())
