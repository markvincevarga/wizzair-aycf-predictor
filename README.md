# WizzAir All You Can Fly Availability Data

A machine learning system that attempts to help predict when AYCF tickets are will be
available, for every route in the in the network.

WizzAir created a subscription service called [All You Can Fly](https://www.wizzair.com/en-gb/information-and-services/memberships/all-you-can-fly), where subscribers can pay a nominal fix price (10 EUR) for flight tickets booked at most 3 days in advance. However, not all seats or all flights are eligible for the cheap tickets.

From the All You Can Fly **FAQ**:
> [...] flight and seat availability seen in the normal booking flow does not guarantee that the available flights and seats for All You Can Fly members will be the same.

## Data Sources

- [wizzair-aycf-availability](https://github.com/markvincevarga/wizzair-aycf-availability/) Daily availability of WizzAir All You Can Fly deal flight tickets. This is the basis of our data. Made by Márk.
- [OpenHolidays API](https://www.openholidaysapi.org/) To determine how close the specific flight is to the next/previous holiday in either country. People fly more around holidays, so we think that fewer AYCF tickets will be available close to holidays.
- Narrow Index NEER (Nominal Effective Exchange Rate) data: relative value change of different currencies daily from [BIS Data Portal](https://data.bis.org/). NEER data of the currency related to a departure or destination country is correlated with the strength of a certain currency. We think that when their home currency is strong, people are more likely to travel abroad, thereby reducing the number of available AYCF tickets.

## Architecture

### Components

- Controller (in this repository): performs data processing
- Feature Store: SQL database to store the data gathered from external sources
and the predictions
- Model Store: S3 bucket storing the model weights
- UI (in the `docs/` directory): presents the predictions

### Runbook Steps

All steps below require the following environment variables set:

```bash
# write access to the database
export CLOUDFLARE_API_TOKEN=...
```

**Fill** updates the database with the latest data. Creates the tables if necessary.

```bash
uv run fill.py --db wizz
```

**Train** uses all the available data in the database to train the model.

```bash
uv run train.py --db wizz
```

**Predict** uses the latest available data to put predictions in the database.

```bash
uv run predict.py --db wizz
```

**UI** starts a http server to present the predictions from the database.

```bash
uv run streamlit run app.py --db wizz
```