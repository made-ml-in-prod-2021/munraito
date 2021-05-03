import pandas as pd
from pandas_profiling import ProfileReport
from pathlib import Path
from constants import DATA_PATH

# DATA_PATH = Path(__file__).resolve().parent.parent.joinpath("data/raw/heart.csv")
df = pd.read_csv(DATA_PATH)
ProfileReport(df, title="Pandas Profiling Report") \
    .to_file(Path(__file__).resolve().parent.parent.joinpath("reports/EDA_pandas_profiling.html"))
