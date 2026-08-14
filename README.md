# Leveraged-ETF desk (personal)

Same-session watchlist and a tiny trade journal. **Not a live broker. Not a bot.**

**Site:** https://murtaza7863.github.io/trading/

Log trades in the browser. They stay in `localStorage` on your machine and are not stored in this repo. The Webull CSV is gitignored.

## Use the site

Open the URL above. **Refresh board** reloads the latest committed snapshot (a GitHub Action updates it on weekday sessions).

For a live Yahoo pull, run the desk on your laptop:

```bash
python run_web.py         # http://127.0.0.1:8787
```

Then click **Refresh board**. That scores Yahoo daily plus 5-minute premarket / cash / night bars.

Fuel ranks names that are moving *now* (range vs ATR, rvol, extended hours), not names that were merely wild last month. Lean is a tape read, not a forecast.
