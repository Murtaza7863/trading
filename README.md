# Leveraged-ETF desk (personal)

Same-session watchlist and a tiny trade journal. **Not a live broker. Not a bot.**

**Site:** https://murtaza7863.github.io/trading/

Log trades in the browser. They stay in `localStorage` on your machine and are not stored in this repo. The Webull CSV is gitignored.

## Use the site

Open the URL above. Refreshing the Yahoo board only works on your laptop:

```bash
python run_watchlist.py   # updates docs/watchlist.json
python run_web.py         # local desk at http://127.0.0.1:8787
```

Then commit `docs/watchlist.json` if you want the hosted board updated.
