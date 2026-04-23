"""Shared fixtures for PE Monitor unit tests."""

import pytest


@pytest.fixture
def mock_stable_info():
    return {
        "beta": 0.75,
        "revenueGrowth": 0.04,
        "dividendYield": 0.028,
        "priceToSalesTrailing12Months": 5.2,
        "bookValue": 25.0,
        "longName": "Stable Test Corp",
        "quoteType": "EQUITY",
        "operatingCashflow": 50_000_000,
        "freeCashflow": 40_000_000,
        "pegRatio": 1.5,
        "forwardPE": 18.0,
        "forwardEps": 5.5,
        "enterpriseToEbitda": 12.0,
        "sharesOutstanding": 1_000_000,
        "marketCap": 100_000_000,
    }


@pytest.fixture
def mock_growth_info():
    return {
        "beta": 1.35,
        "revenueGrowth": 0.25,
        "dividendYield": None,
        "priceToSalesTrailing12Months": 10.5,
        "bookValue": 15.0,
        "longName": "Growth Test Corp",
        "quoteType": "EQUITY",
        "operatingCashflow": 20_000_000,
        "freeCashflow": 5_000_000,
        "pegRatio": 0.8,
        "forwardPE": 25.0,
        "forwardEps": 4.0,
        "enterpriseToEbitda": 20.0,
        "sharesOutstanding": 1_000_000,
        "marketCap": 200_000_000,
    }
