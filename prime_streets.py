import os
import pandas as pd
from deflactor import Deflactor
from dotenv import load_dotenv
from difflib import get_close_matches
import unicodedata
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import logging

class PrimeStreetsProcessor:
    """
    Encapsulates the full pipeline:
      1. Load extracted‐streets and cadastral data
      2. Clean and normalize street names & zones
      3. Match extracted names to cadastral names
      4. Deflate prices by date/zone & normalize per m²
      5. Compute street vs. zone averages & market share
      6. Select top and bottom streets per zone
      7. Export results to CSV and PDF
    """
    def __init__(
        self,
        streets_csv: str,
        catastro_csv: str,
        output_dir: str = ".",
        min_ads: int = 10,
        top_n: int = 10,
    ):
        load_dotenv()
        os.makedirs(output_dir, exist_ok=True)
        self.streets_csv = streets_csv
        self.catastro_csv = catastro_csv
        self.output_dir = output_dir
        self.min_ads = min_ads
        self.top_n = top_n

        # placeholders
        self.df = None
        self.df_cat = None
        self.valid_streets = {}
        self.normalized_df = None
        self.price_comparison = None
        self.market_share = None
        self.final_table = None
        self.top_streets = None
        self.bottom_streets = None

        logging.basicConfig(level=logging.INFO)

    def load_data(self):
        logging.info("Loading data...")
        self.df = pd.read_csv(self.streets_csv, low_memory=False)
        self.df_cat = pd.read_csv(self.catastro_csv, sep=";", low_memory=False)

    def clean_and_prepare(self):
        logging.info("Cleaning cadastral and extracted data...")
        # Cadastral
        self.df_cat["nombre_via_clean"] = (
            self.df_cat["nombre_via"].astype(str).str.strip().str.lower()
        )
        self.df_cat["zona_clean"] = (
            self.df_cat["zona"]
            .astype(str)
            .str.split("-")
            .str[-1]
            .str.strip()
            .str.lower()
        )
        # Extracted
        self.df["street_name_clean"] = (
            self.df["street_name_to_extract"].astype(str).str.strip().str.lower()
        )
        self.df["zone_clean"] = self.df["zone"].astype(str).str.strip().str.lower()

        # Build valid streets dict
        self.valid_streets = (
            self.df_cat.groupby("zona_clean")["nombre_via_clean"]
            .unique()
            .to_dict()
        )

    @staticmethod
    def _impute_unknown_street(series: pd.Series) -> pd.Series:
        empty = ["", " ", "Unknown", "N/A", "Not Available", "nan", "None"]
        return (
            series.replace(empty, pd.NA)
            .str.strip()
            .fillna("Unknown Street")
        )

    @staticmethod
    def _normalize_str(s: str) -> str:
        s = unicodedata.normalize("NFKD", str(s))
        s = s.encode("ascii", errors="ignore").decode("utf-8")
        return s.strip().lower()

    def match_streets(self):
        logging.info("Matching extracted street names to cadastral names...")
        def best_match(row):
            street = row["street_name_clean"]
            zone = row["zone_clean"]
            if pd.isna(street) or pd.isna(zone):
                return None
            candidates = self.valid_streets.get(zone, [])
            matches = get_close_matches(street, candidates, cutoff=0.7, n=1)
            return matches[0] if matches else None

        self.df["matched_street"] = self.df.apply(best_match, axis=1)
        # fallback to cleaned extract if no match
        self.df["street_name"] = (
            self.df["matched_street"]
            .fillna(self.df["street_name_clean"])
        )

    def normalize_prices(self):
        logging.info("Deflating and normalizing prices...")
        # rename so reflactor knows columns
        df = self.df.copy()
        df["first_appearance"] = pd.to_datetime(df["first_appearance"])
        processor = Deflactor(
            df=df,
            start_date_column="first_appearance",
            end_date_column="last_appearance",
            zone_column="zona",
            price_column="local_price",
            area_column="area",
            save_df_as_reference=False,
        )
        normalized = processor.deflactor()
        normalized.dropna(subset=["deflated_price"], inplace=True)
        # rename matched column
        normalized.rename(
            columns={"matched_street": "street_name"}, inplace=True
        )
        self.normalized_df = normalized

    def calculate_price_comparison(self):
        logging.info("Calculating price comparisons...")
        df = self.normalized_df.copy()
        # normalized €/m²
        df["normalized_€/m2"] = df["deflated_price"] / df["area"]

        # zone averages
        df["zone_avg_price_per_m2"] = df.groupby("zone")[
            "normalized_€/m2"
        ].transform("mean")

        # zone avg excluding street
        def excl_avg(row):
            zone_df = df[df["zone"] == row["zone"]]
            return (
                zone_df["normalized_€/m2"]
                .loc[zone_df["street_name"] != row["street_name"]]
                .mean()
            )

        df["zone_avg_price_per_m2_excluding_street"] = df.apply(
            excl_avg, axis=1
        )

        # street averages
        street_avg = (
            df.groupby(["zone", "street_name"])["normalized_€/m2"]
            .mean()
            .rename("street_avg_price_per_m2")
        )
        df = df.merge(
            street_avg, on=["zone", "street_name"], how="left"
        )

        # factor
        df["street_factor"] = (
            df["street_avg_price_per_m2"]
            / df["zone_avg_price_per_m2_excluding_street"]
            - 1
        ) * 100

        self.price_comparison = df

    def calculate_market_share(self):
        logging.info("Calculating market share...")
        df = self.price_comparison.copy()
        # counts
        df["ads_count"] = df["street_name"].map(
            df["street_name"].value_counts()
        )
        df["zone_ads_count"] = df["zone"].map(
            df["zone"].value_counts()
        )
        df["zone_ads_excluding_street"] = (
            df["zone_ads_count"] - df["ads_count"]
        )
        # market share
        df["ads_market_share"] = df.apply(
            lambda r: (r["ads_count"] / r["zone_ads_excluding_street"]) * 100
            if r["zone_ads_excluding_street"] > 0
            else 0,
            axis=1,
        )
        self.market_share = df

    def generate_final_table(self):
        logging.info("Generating final summary table...")
        cols = [
            "zone",
            "zone_avg_price_per_m2",
            "zone_avg_price_per_m2_excluding_street",
            "street_name",
            "street_avg_price_per_m2",
            "street_factor",
            "ads_market_share",
            "ads_count",
        ]
        self.final_table = self.market_share[cols].drop_duplicates()

    def select_top_bottom(self):
        logging.info("Selecting top/bottom streets per zone...")
        grp = (
            self.final_table.groupby(["zone", "street_name"])
            .agg(
                {
                    "ads_count": "sum",
                    "street_avg_price_per_m2": "mean",
                    "zone_avg_price_per_m2_excluding_street": "mean",
                    "street_factor": "max",
                }
            )
            .reset_index()
        )
        filtered = grp[grp["ads_count"] >= self.min_ads]

        self.bottom_streets = (
            filtered.groupby("zone")
            .apply(lambda x: x.nsmallest(self.top_n, "street_factor"))
            .reset_index(drop=True)
        )
        self.top_streets = (
            filtered.groupby("zone")
            .apply(lambda x: x.nlargest(self.top_n, "street_factor"))
            .reset_index(drop=True)
        )

    def save_csvs(self):
        logging.info("Saving CSV outputs...")
        bottom_path = os.path.join(self.output_dir, "prime_streets_bottom_final.csv")
        top_path = os.path.join(self.output_dir, "prime_streets_top_final.csv")
        self.bottom_streets.to_csv(bottom_path, index=False)
        self.top_streets.to_csv(top_path, index=False)

    @staticmethod
    def _df_to_pdf(df: pd.DataFrame, pdf_path: str):
        c = canvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        y = height - 40
        for _, row in df.iterrows():
            text = " | ".join(map(str, row.values))
            c.drawString(40, y, text)
            y -= 20
            if y < 40:
                c.showPage()
                y = height - 40
        c.save()

    def generate_pdf(self, which: str = "top"):
        """
        which: "top" or "bottom"
        """
        logging.info(f"Generating {which} streets PDF report...")
        if which == "top":
            df = self.top_streets
            fname = "prime_streets_top_final.pdf"
        else:
            df = self.bottom_streets
            fname = "prime_streets_bottom_final.pdf"
        path = os.path.join(self.output_dir, fname)
        self._df_to_pdf(df, path)

    def run(self):
        self.load_data()
        self.clean_and_prepare()
        self.match_streets()
        self.normalize_prices()
        # format street names
        self.normalized_df["street_names_fx"] = self.normalized_df[
            "street_name"
        ].apply(self._normalize_str)
        self.normalized_df["street_names_fx"] = self._impute_unknown_street(
            self.normalized_df["street_names_fx"]
        )
        self.calculate_price_comparison()
        self.calculate_market_share()
        self.generate_final_table()
        self.select_top_bottom()
        self.save_csvs()
        # if you want PDFs:
        self.generate_pdf("top")
        self.generate_pdf("bottom")
        logging.info("Pipeline complete.")


if __name__ == "__main__":
    proc = PrimeStreetsProcessor(
        streets_csv="df_all_extracted_streets.csv",
        catastro_csv="catastro_sophiq_20241210.csv",
        output_dir="outputs",
        min_ads=10,
        top_n=10,
    )
    proc.run()
