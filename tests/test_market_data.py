"""Market data tests with a fake downloader: no internet needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import portfolio_assistant as pa
from portfolio_assistant.market_data import fetch_prices, fetch_ticker


class FakeYahoo:
    """Serves synthetic histories and records every request."""

    def __init__(self, histories: dict[str, pd.Series]):
        self.histories = histories
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, ticker, start, end):
        self.calls.append((ticker, start))
        series = self.histories.get(ticker, pd.Series(dtype="float64"))
        if start is not None:
            series = series.loc[start:]
        return series.rename(ticker)


def _series(start, n, seed, freq="B"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)), index=idx)


def test_second_call_reads_the_cache(tmp_path):
    yahoo = FakeYahoo({"AAA": _series("2015-01-01", 500, 1)})
    first = fetch_ticker("AAA", "2015-01-01", cache_dir=tmp_path, downloader=yahoo)
    second = fetch_ticker("AAA", "2015-01-01", cache_dir=tmp_path, downloader=yahoo, refresh=False)
    assert len(yahoo.calls) == 1
    pd.testing.assert_series_equal(first, second, check_freq=False)


def test_refresh_only_adds_the_new_days(tmp_path):
    full = _series("2015-01-01", 500, 2)
    yahoo = FakeYahoo({"AAA": full.iloc[:400]})
    fetch_ticker("AAA", "2015-01-01", cache_dir=tmp_path, downloader=yahoo)
    yahoo.histories["AAA"] = full
    out = fetch_ticker("AAA", "2015-01-01", cache_dir=tmp_path, downloader=yahoo)
    assert len(out) == 500
    assert yahoo.calls[-1][1] == full.index[390].strftime("%Y-%m-%d")  # 10-day overlap only


def test_a_dividend_rewrite_reloads_the_whole_history(tmp_path):
    full = _series("2015-01-01", 500, 3)
    yahoo = FakeYahoo({"AAA": full.iloc[:400]})
    fetch_ticker("AAA", "2015-01-01", cache_dir=tmp_path, downloader=yahoo)
    yahoo.histories["AAA"] = full * 0.98  # Yahoo re-adjusted every past price
    out = fetch_ticker("AAA", "2015-01-01", cache_dir=tmp_path, downloader=yahoo)
    np.testing.assert_allclose(out.to_numpy(), (full * 0.98).to_numpy())


def test_unknown_ticker_is_named_in_the_error(tmp_path):
    yahoo = FakeYahoo({"AAA": _series("2015-01-01", 100, 4)})
    with pytest.raises(pa.DataError, match="ZZZ"):
        fetch_prices(["AAA", "ZZZ"], "2015-01-01", cache_dir=tmp_path, downloader=yahoo,
                     verbose=False)


def test_panel_starts_with_the_youngest_ticker_and_drops_closed_days(tmp_path):
    old = _series("2015-01-01", 1200, 5)
    young = _series("2016-01-01", 600, 6)
    holidays = young.index[[10, 20]]
    young = young.drop(holidays)                        # a holiday on the other exchange
    yahoo = FakeYahoo({"OLD": old, "YOUNG": young})
    panel = fetch_prices(["OLD", "YOUNG"], "2015-01-01", cache_dir=tmp_path,
                         downloader=yahoo, verbose=False)
    assert panel.index[0] == young.index[0]
    assert not panel.isna().any().any()
    assert not panel.index.isin(holidays).any()
    assert len(panel) == len(young)


def test_the_panel_runs_through_the_engine(tmp_path):
    yahoo = FakeYahoo({t: _series("2015-01-01", 900, i) for i, t in enumerate(["A", "B", "C"])})
    panel = fetch_prices(["A", "B", "C"], "2015-01-01", cache_dir=tmp_path,
                         downloader=yahoo, verbose=False)
    result = pa.run_backtest(panel, pa.equal_weight(), pa.BacktestConfig(rebalance="monthly"))
    assert result.n_decisions > 30 and result.nav.iloc[-1] > 0


def test_todays_live_price_is_never_taken_for_a_close():
    from portfolio_assistant.market_data import final_closes
    s = pd.Series([1.0, 2.0, 3.0], pd.DatetimeIndex(["2026-10-07", "2026-10-08", "2026-10-09"]))
    at_lunch = pd.Timestamp("2026-10-09 14:12", tz="Europe/Paris")
    assert final_closes(s, at_lunch).index[-1] == pd.Timestamp("2026-10-08")
    next_morning = pd.Timestamp("2026-10-10 08:00", tz="Europe/Paris")
    assert final_closes(s, next_morning).index[-1] == pd.Timestamp("2026-10-09")
    london = s.copy()
    london.index = london.index.tz_localize("Europe/London")
    assert len(final_closes(london, at_lunch)) == 2


def test_a_live_price_already_in_the_cache_is_removed(tmp_path):
    today = pd.Timestamp.now(tz="Europe/Paris").tz_localize(None).normalize()
    idx = pd.bdate_range(end=today - pd.Timedelta(days=1), periods=30).append(pd.DatetimeIndex([today]))
    pd.Series(np.linspace(100, 110, len(idx)), idx, name="X").to_frame().to_csv(tmp_path / "X.csv")
    fresh = pd.Series(np.linspace(100, 110, len(idx))[:-1], idx[:-1])
    out = fetch_ticker("X", str(idx[0].date()), cache_dir=tmp_path, downloader=lambda t, s, e: fresh.loc[s:])
    assert out.index[-1] < today
    assert pd.read_csv(tmp_path / "X.csv", index_col=0, parse_dates=True).index[-1] < today


def test_a_day_yahoo_skipped_is_filled_when_it_comes_back(tmp_path):
    idx = pd.bdate_range("2026-09-01", "2026-10-08")
    full = pd.Series(np.linspace(100, 110, len(idx)), idx)
    holed = full.drop(pd.Timestamp("2026-10-07"))
    holed.rename("X").to_frame().to_csv(tmp_path / "X.csv")       # an earlier refresh missed the 7th
    out = fetch_ticker("X", "2026-09-01", cache_dir=tmp_path, downloader=lambda t, s, e: full.loc[s:])
    assert pd.Timestamp("2026-10-07") in out.index and out.index.is_monotonic_increasing
    assert len(out) == len(full)


def test_young_fund_keeps_its_cache_when_the_source_is_down(tmp_path):
    import pandas as pd
    from portfolio_assistant.market_data import fetch_ticker
    idx = pd.bdate_range("2015-01-05", periods=30)
    fetch_ticker("YNG.PA", "2005-01-01", cache_dir=tmp_path,
                 downloader=lambda t, s, e: pd.Series(range(1, 31), index=idx, dtype="float64"))
    out = fetch_ticker("YNG.PA", "2005-01-01", cache_dir=tmp_path,
                       downloader=lambda t, s, e: pd.Series(dtype="float64"))
    assert len(out) == 30 and out.iloc[-1] == 30.0


def test_session_ranges_follow_the_panel_currency_and_drop_bad_rows(tmp_path):
    import numpy as np
    from portfolio_assistant.market_data import fetch_ranges
    idx = pd.bdate_range("2020-01-01", periods=5)
    closes = pd.DataFrame({"A": [10.0, 11, 12, 13, 14], "B": [1.0, 1, 1, 1, 1]}, index=idx)

    def source(ticker, start):
        if ticker == "B":
            raise ConnectionError("no network")
        return pd.DataFrame({"o": 1.0, "h": [1.02, 1.02, 0.99, 1.02, 1.02], "l": 0.97}, index=idx)

    frames, missing = fetch_ranges(closes, cache_dir=tmp_path, downloader=source)
    assert missing == ["B"] and frames["high"]["A"].iloc[0] == pytest.approx(10.2)
    assert np.isnan(frames["high"]["A"].iloc[2])                      # a high under its close is dropped
    again, _ = fetch_ranges(closes, cache_dir=tmp_path, downloader=lambda t, s: (_ for _ in ()).throw(OSError()))
    assert again["low"]["A"].iloc[1] == pytest.approx(11 * 0.97)        # read back from the cache
