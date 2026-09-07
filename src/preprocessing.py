import pandas as pd
from pathlib import Path
import json
from datetime import date
from IPython.display import display

GOOD = ["Paid Off Loan"]
BAD = [
    "External Collection",
    "Internal Collection",
    "Charged Off",
    "Charged Off Paid Off",
    "Settled Bankruptcy",
    "Settlement Paid Off",
]
UNRESOLVED = [
    "New Loan",
    "Pending Paid Off",
    "Settlement Pending Paid Off",
    "Returned Item",
    "Test",
]


class EDA:
    def __init__(self, df: pd.DataFrame, keys: list) -> None:
        self.df = df
        self.keys = keys

    def run(self):
        print("Shape:")
        print(self.df.shape)
        print("\nTypes:")
        print(self.df.dtypes)
        print("\nSummary:")
        print(display(self.df.describe(include="all", percentiles=[0.01, 0.99]).T))

        self.print_head()

        self.null()

        self.check_duplicates()
        self.analyze_discrete()

    def null(self):
        # check for nulls
        print("\nNull-check:")
        print(self.df.isna().sum())

    def print_head(self):
        print("\nFirst 5 rows:")
        print(display(self.df.head().T))

    def analyze_discrete(self):
        # check counts for categorical columns
        print("\nDiscrete Cols Value Counts")
        dis_cols = self.df.select_dtypes(
            include=["category", "object", "string", "bool"]
        ).columns
        for col in dis_cols:
            distinct = self.df[col].nunique()
            if distinct < 50:
                print(self.df[col].value_counts(dropna=False))
            else:
                print(f"{col}: {distinct} distinct")

    def check_duplicates(self):
        # check for duplicates
        print("\nDuplicates:")
        print(self.df.duplicated().sum())

        # check for duplicates in ID
        print("\nDuplicates in ID:")
        for key in self.keys:
            print(f"{self.df[key].name}: {self.df[key].dropna().duplicated().sum()}")


class Preprocess:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def change_date_types(self, date_cols: list) -> pd.DataFrame:
        for col in date_cols:
            before = self.df[col].notna().sum()

            # adding format resolved the raw date precision issue
            self.df[col] = pd.to_datetime(
                self.df[col], errors="coerce", format="ISO8601"
            )
            lost = before - self.df[col].notna().sum()
            if lost:
                print(f"{col}: {lost} values failed to parse.")

        return self.df

    def change_clarity_types(
        self, csv_path: str, key_col: str = "underwritingid"
    ) -> pd.DataFrame:
        dic = pd.read_csv(csv_path)
        dic.columns = [c.strip() for c in dic.columns]
        declared = dict(
            zip(
                dic["fieldName_in_file"].str.strip().str.lower(),
                dic["Type"].astype(str).str.strip(),
            )
        )

        for col in self.df.columns:
            if col == key_col:
                continue

            kind = declared.get(col.lower())
            if kind == "Number":
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
            else:
                values = set(self.df[col].dropna().astype(str).str.lower().unique())
                if values <= {"true", "false"}:
                    self.df[col] = (
                        self.df[col]
                        .astype(str)
                        .str.lower()
                        .map({"true": True, "false": False})
                        .astype("boolean")
                    )
                else:
                    self.df[col] = self.df[col].astype("category")

        return self.df

    def update_model_df(self, df: pd.DataFrame):
        self.df = df

    def add_target(self) -> pd.DataFrame:
        self.df["target"] = self.df["loanStatus"].isin(BAD).astype(int)

        return self.df

    def extract_funded(self) -> pd.DataFrame:
        funded = self.df[self.df["isFunded"] == 1]

        model_df = funded[~funded["loanStatus"].isin(UNRESOLVED)].copy()
        self.update_model_df(model_df)

        return self.df

    def merge_data(
        self, payment_df: pd.DataFrame, clarity_df: pd.DataFrame, loan_df: pd.DataFrame
    ) -> pd.DataFrame:
        agg_payment = payment_df.merge(
            loan_df[["loanId", "anon_ssn"]], on="loanId", how="inner"
        )
        agg_payment["failed"] = agg_payment["paymentStatus"].isin(
            ["Rejected", "Rejected Awaiting Retry"]
        )
        agg_payment = agg_payment.rename(columns={"loanId": "pay_loanId"})

        # attach each application with its payments by the customer
        history = pd.merge(
            agg_payment,
            self.df[["loanId", "anon_ssn", "applicationDate"]],
            left_on="anon_ssn",
            right_on="anon_ssn",
            how="inner",
        )
        # only take payments that have already happened and not from this application
        history = history[
            (history["paymentDate"] < history["applicationDate"])
            & (history["pay_loanId"] != history["loanId"])
        ]

        # only one row per payment
        agg_payment = history.groupby(["loanId"]).agg(
            previous_payment_count=("paymentAmount", "size"),
            previous_failed_payment=("failed", "sum"),
            previous_total_payment=("paymentAmount", "sum"),
        )
        data = self.df.merge(
            agg_payment, on="loanId", how="left", suffixes=("_loan", "_pay")
        )

        data = pd.merge(
            data,
            clarity_df,
            left_on="clarityFraudId",
            right_on="underwritingid",
            how="left",
        )

        self.update_model_df(data)

        return self.df
