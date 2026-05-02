import json
from pathlib import Path
from textwrap import dedent


OUTPUT = Path(r"C:\Users\thear\Documents\New project\threat_intel_ml_research.ipynb")


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    md(
        r"""
        # Threat Intelligence ML Research Notebook

        This notebook performs a single, end-to-end machine-learning research analysis across OTX threat pulses, CISA KEV-style CVE data, malicious domains, and malicious IP intelligence.

        **Design goals**

        - Load the four CSVs directly from `W:\archive`.
        - Use `pandas` parsing rather than raw line counts, because WHOIS fields contain embedded newlines.
        - Normalize `Unknown`, `{}`, blanks, and similar placeholders before analysis.
        - Engineer features for text, categorical, temporal, and reputation/vote signals.
        - Run reproducible ML workflows for topic discovery, classification, anomaly detection, and risk prioritization.
        - Avoid artificial joins across datasets where no shared identifier exists.

        **Important research caveat**

        These datasets appear to mix confirmed threat intelligence, enrichment metadata, and scanner/vote output. The models below should be treated as analytical triage aids, not as production detection logic or ground truth.
        """
    ),
    md(
        r"""
        ## 1. Environment Setup

        Run this cell first. It checks the requested ML/visualization stack and installs anything missing into the active Python environment.
        """
    ),
    code(
        r"""
        import importlib.util
        import subprocess
        import sys

        REQUIRED_PACKAGES = {
            "pandas": "pandas",
            "numpy": "numpy",
            "matplotlib": "matplotlib",
            "seaborn": "seaborn",
            "sklearn": "scikit-learn",
            "plotly": "plotly",
        }

        missing = [
            package_name
            for import_name, package_name in REQUIRED_PACKAGES.items()
            if importlib.util.find_spec(import_name) is None
        ]

        if missing:
            print("Installing missing packages:", ", ".join(missing))
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
            print("Install complete. If imports still fail, restart the kernel and rerun from this cell.")
        else:
            print("All required packages are available.")
        """
    ),
    code(
        r"""
        from __future__ import annotations

        import json
        import math
        import re
        import warnings
        from pathlib import Path

        import numpy as np
        import pandas as pd

        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px
        import plotly.graph_objects as go

        from IPython.display import Markdown, display

        from sklearn.compose import ColumnTransformer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.ensemble import IsolationForest, RandomForestClassifier
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            balanced_accuracy_score,
            classification_report,
            confusion_matrix,
            silhouette_score,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.cluster import KMeans

        warnings.filterwarnings("ignore", category=FutureWarning)

        RANDOM_STATE = 42
        pd.set_option("display.max_columns", 120)
        pd.set_option("display.max_colwidth", 160)
        pd.set_option("display.width", 160)

        sns.set_theme(style="whitegrid", context="notebook")
        plt.rcParams["figure.figsize"] = (11, 5)
        plt.rcParams["axes.titleweight"] = "bold"

        try:
            import plotly.io as pio
            pio.renderers.default = "notebook_connected"
        except Exception:
            pass
        """
    ),
    md(
        r"""
        ## 2. Configuration And Source Paths

        All source files are configured in one place so this notebook can be rerun after the CSVs are refreshed.
        """
    ),
    code(
        r"""
        DATA_PATHS = {
            "otx": Path(r"W:\archive\1_otx_threat_intel.csv"),
            "cve": Path(r"W:\archive\2_cve_vulnerabilities.csv"),
            "domains": Path(r"W:\archive\3_malicious_domains.csv"),
            "ips": Path(r"W:\archive\4_malicious_ips.csv"),
        }

        EXPECTED_ROWS = {
            "otx": 2365,
            "cve": 1585,
            "domains": 162,
            "ips": 200,
        }

        REQUIRED_COLUMNS = {
            "otx": [
                "Pulse_ID", "Title", "Description", "Author", "Created", "Modified", "TLP",
                "Tags", "Malware_Families", "Attack_IDs", "Industries", "Countries",
                "Indicators_Count", "Subscribers"
            ],
            "cve": [
                "cveID", "vendorProject", "product", "vulnerabilityName", "dateAdded",
                "shortDescription", "requiredAction", "dueDate",
                "knownRansomwareCampaignUse", "cwes"
            ],
            "domains": [
                "Domain", "TLD", "Domain_Length", "Has_Numbers", "Has_Hyphen",
                "Registrar", "Creation_Date", "Last_Update_Date", "Reputation",
                "Malicious_Votes", "Suspicious_Votes", "Harmless_Votes",
                "Undetected_Votes", "Total_Engines", "Threat_Severity", "Categories",
                "Popularity_Rank", "Last_Analysis_Date", "WHOIS_Summary", "Data_Source"
            ],
            "ips": [
                "IP", "Country", "Continent", "ASN", "Owner", "Network",
                "Malicious_Votes", "Suspicious_Votes", "Harmless_Votes",
                "Undetected_Votes", "Total_Reports", "Reputation_Score",
                "Threat_Label", "Threat_Category", "Regional_Registry",
                "WHOIS_Summary", "TOR_Node", "Times_Submitted",
                "Last_Analysis_Date", "Threat_Severity"
            ],
        }

        for name, path in DATA_PATHS.items():
            if not path.exists():
                raise FileNotFoundError(f"{name} source file not found: {path}")

        print("Configured source files:")
        for name, path in DATA_PATHS.items():
            print(f" - {name:8s} {path}")
        """
    ),
    md(
        r"""
        ## 3. Utility Functions

        These helpers keep cleaning, validation, feature engineering, and visualizations consistent across the four datasets.
        """
    ),
    code(
        r"""
        PLACEHOLDER_VALUES = {"", "unknown", "none", "nan", "null", "n/a", "na", "{}"}


        def normalize_scalar(value):
            if pd.isna(value):
                return np.nan
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned.lower() in PLACEHOLDER_VALUES:
                    return np.nan
                return cleaned
            return value


        def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            for col in out.select_dtypes(include=["object"]).columns:
                out[col] = out[col].map(normalize_scalar)
            return out


        def split_multi(value) -> list[str]:
            value = normalize_scalar(value)
            if pd.isna(value):
                return []
            return [
                part.strip()
                for part in str(value).split(",")
                if part.strip() and part.strip().lower() not in PLACEHOLDER_VALUES
            ]


        def explode_counts(df: pd.DataFrame, column: str, top_n: int = 20, lower: bool = False) -> pd.Series:
            values = df[column].apply(split_multi).explode()
            values = values.dropna()
            if lower:
                values = values.str.lower()
            return values.value_counts().head(top_n)


        def to_number(series: pd.Series) -> pd.Series:
            return pd.to_numeric(series, errors="coerce")


        def parse_datetime(series: pd.Series) -> pd.Series:
            return pd.to_datetime(series, errors="coerce")


        def parse_unix_datetime(series: pd.Series) -> pd.Series:
            numeric = pd.to_numeric(series, errors="coerce")
            return pd.to_datetime(numeric, unit="s", errors="coerce")


        def bool_from_yes_no(series: pd.Series) -> pd.Series:
            return (
                series.astype(str)
                .str.strip()
                .str.lower()
                .map({"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0})
                .astype("float")
            )


        def validate_columns(df: pd.DataFrame, name: str) -> None:
            missing = sorted(set(REQUIRED_COLUMNS[name]) - set(df.columns))
            if missing:
                raise ValueError(f"{name} is missing required columns: {missing}")


        def missing_profile(df: pd.DataFrame) -> pd.DataFrame:
            rows = []
            for col in df.columns:
                raw = df[col]
                placeholder = raw.astype(str).str.strip().str.lower().isin(PLACEHOLDER_VALUES).sum()
                missing = raw.isna().sum()
                total = int(missing + placeholder)
                rows.append(
                    {
                        "column": col,
                        "missing_or_placeholder": total,
                        "pct": total / len(df) if len(df) else np.nan,
                        "dtype": str(raw.dtype),
                        "unique_non_null": raw.dropna().nunique(),
                    }
                )
            return pd.DataFrame(rows).sort_values(["missing_or_placeholder", "column"], ascending=[False, True])


        def show_top(series: pd.Series, title: str, x_label: str = "Count", top_n: int = 15):
            data = series.head(top_n).sort_values(ascending=True)
            fig = px.bar(
                data,
                x=data.values,
                y=data.index,
                orientation="h",
                labels={"x": x_label, "y": ""},
                title=title,
                template="plotly_white",
            )
            fig.update_layout(height=max(420, 28 * len(data)), title_x=0.02)
            fig.show()
            return data.sort_values(ascending=False)


        def severity_ordered_counts(df: pd.DataFrame, column: str = "Threat_Severity") -> pd.Series:
            order = ["Low", "Medium", "High", "Critical"]
            counts = df[column].fillna("Missing").value_counts()
            ordered = [item for item in order if item in counts.index] + [
                item for item in counts.index if item not in order
            ]
            return counts.loc[ordered]


        def safe_train_test_split(X, y, test_size=0.25):
            counts = pd.Series(y).value_counts()
            stratify = y if len(counts) > 1 and counts.min() >= 2 else None
            return train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=RANDOM_STATE,
                stratify=stratify,
            )


        def model_report(name: str, y_true, y_pred, labels=None) -> dict:
            report = {
                "model": name,
                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            }
            display(Markdown(f"### {name}"))
            display(pd.DataFrame([report]))
            print(classification_report(y_true, y_pred, zero_division=0))
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            plt.figure(figsize=(6, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels if labels is not None else "auto",
                yticklabels=labels if labels is not None else "auto",
            )
            plt.title(f"{name} Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.show()
            return report


        def numeric_anomaly_detection(df: pd.DataFrame, feature_cols: list[str], id_col: str, contamination: float = 0.08) -> pd.DataFrame:
            model_df = df[[id_col] + feature_cols].copy()
            for col in feature_cols:
                model_df[col] = pd.to_numeric(model_df[col], errors="coerce")
            pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", IsolationForest(contamination=contamination, random_state=RANDOM_STATE)),
                ]
            )
            predictions = pipe.fit_predict(model_df[feature_cols])
            scores = pipe.named_steps["model"].decision_function(
                pipe.named_steps["scaler"].transform(
                    pipe.named_steps["imputer"].transform(model_df[feature_cols])
                )
            )
            result = model_df.copy()
            result["anomaly_flag"] = predictions
            result["anomaly_score"] = scores
            return result.sort_values("anomaly_score")
        """
    ),
    md(
        r"""
        ## 4. Load Data

        The parser uses `pandas.read_csv`, which correctly handles quoted embedded newlines in WHOIS fields.
        """
    ),
    code(
        r"""
        raw = {name: pd.read_csv(path) for name, path in DATA_PATHS.items()}

        for name, df in raw.items():
            validate_columns(df, name)

        row_summary = pd.DataFrame(
            [
                {
                    "dataset": name,
                    "parsed_rows": len(df),
                    "expected_rows": EXPECTED_ROWS[name],
                    "columns": df.shape[1],
                    "matches_expected": len(df) == EXPECTED_ROWS[name],
                }
                for name, df in raw.items()
            ]
        )
        display(row_summary)

        if not row_summary["matches_expected"].all():
            display(Markdown("**Note:** one or more row counts differ from the original baseline. Continue, but inspect source refreshes."))

        samples = {name: df.head(3) for name, df in raw.items()}
        for name, sample in samples.items():
            display(Markdown(f"### Sample: {name}"))
            display(sample)
        """
    ),
    md(
        r"""
        ## 5. Data Quality Profile

        Before cleaning, profile row counts, duplicate keys, placeholder-heavy columns, and date parseability.
        """
    ),
    code(
        r"""
        key_columns = {
            "otx": "Pulse_ID",
            "cve": "cveID",
            "domains": "Domain",
            "ips": "IP",
        }

        quality_rows = []
        for name, df in raw.items():
            key = key_columns[name]
            quality_rows.append(
                {
                    "dataset": name,
                    "rows": len(df),
                    "columns": df.shape[1],
                    "duplicate_full_rows": int(df.duplicated().sum()),
                    f"duplicate_{key}": int(df[key].duplicated().sum()),
                    "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 3),
                }
            )

        display(pd.DataFrame(quality_rows))

        for name, df in raw.items():
            display(Markdown(f"### Missing / Placeholder Profile: {name}"))
            display(missing_profile(df).head(12))
        """
    ),
    md(
        r"""
        ## 6. Clean And Type Data

        Cleaning is intentionally transparent: it preserves original columns while adding parsed and engineered fields.
        """
    ),
    code(
        r"""
        otx = normalize_frame(raw["otx"])
        cve = normalize_frame(raw["cve"])
        domains = normalize_frame(raw["domains"])
        ips = normalize_frame(raw["ips"])

        # OTX: keep the latest modified copy for repeated Pulse_ID values.
        otx["Created_dt"] = parse_datetime(otx["Created"])
        otx["Modified_dt"] = parse_datetime(otx["Modified"])
        otx["Indicators_Count_num"] = to_number(otx["Indicators_Count"])
        otx["Subscribers_num"] = to_number(otx["Subscribers"])
        otx = (
            otx.sort_values(["Pulse_ID", "Modified_dt"], ascending=[True, False])
            .drop_duplicates("Pulse_ID", keep="first")
            .reset_index(drop=True)
        )
        otx["created_month"] = otx["Created_dt"].dt.to_period("M").astype(str)
        otx["created_year"] = otx["Created_dt"].dt.year
        otx["text"] = (
            otx[["Title", "Description", "Tags", "Malware_Families", "Attack_IDs"]]
            .fillna("")
            .agg(" ".join, axis=1)
        )

        # CVE.
        cve["dateAdded_dt"] = parse_datetime(cve["dateAdded"])
        cve["dueDate_dt"] = parse_datetime(cve["dueDate"])
        cve["days_to_due"] = (cve["dueDate_dt"] - cve["dateAdded_dt"]).dt.days
        cve["ransomware_known"] = cve["knownRansomwareCampaignUse"].eq("Known").astype(int)
        cve["text"] = (
            cve[["vendorProject", "product", "vulnerabilityName", "shortDescription", "requiredAction", "cwes"]]
            .fillna("")
            .agg(" ".join, axis=1)
        )
        cve["cwes_clean"] = cve["cwes"].fillna("")

        # Domains.
        numeric_domain_cols = [
            "Domain_Length", "Creation_Date", "Last_Update_Date", "Reputation",
            "Malicious_Votes", "Suspicious_Votes", "Harmless_Votes",
            "Undetected_Votes", "Total_Engines", "Popularity_Rank", "Last_Analysis_Date"
        ]
        for col in numeric_domain_cols:
            domains[col + "_num"] = to_number(domains[col])
        domains["Has_Numbers_bin"] = bool_from_yes_no(domains["Has_Numbers"])
        domains["Has_Hyphen_bin"] = bool_from_yes_no(domains["Has_Hyphen"])
        domains["Creation_Date_dt"] = parse_unix_datetime(domains["Creation_Date_num"])
        domains["Last_Update_Date_dt"] = parse_unix_datetime(domains["Last_Update_Date_num"])
        domains["Last_Analysis_Date_dt"] = parse_unix_datetime(domains["Last_Analysis_Date_num"])
        domains["domain_age_days"] = (
            domains["Last_Analysis_Date_dt"] - domains["Creation_Date_dt"]
        ).dt.days
        domains["malicious_rate"] = domains["Malicious_Votes_num"] / domains["Total_Engines_num"].replace(0, np.nan)
        domains["suspicious_rate"] = domains["Suspicious_Votes_num"] / domains["Total_Engines_num"].replace(0, np.nan)
        domains["signal_rate"] = (
            domains["Malicious_Votes_num"] + domains["Suspicious_Votes_num"]
        ) / domains["Total_Engines_num"].replace(0, np.nan)

        # IPs.
        numeric_ip_cols = [
            "ASN", "Malicious_Votes", "Suspicious_Votes", "Harmless_Votes",
            "Undetected_Votes", "Total_Reports", "Reputation_Score",
            "Times_Submitted", "Last_Analysis_Date"
        ]
        for col in numeric_ip_cols:
            ips[col + "_num"] = to_number(ips[col])
        ips["TOR_Node_bin"] = bool_from_yes_no(ips["TOR_Node"])
        ips["Last_Analysis_Date_dt"] = parse_unix_datetime(ips["Last_Analysis_Date_num"])
        ips["malicious_rate"] = ips["Malicious_Votes_num"] / ips["Total_Reports_num"].replace(0, np.nan)
        ips["suspicious_rate"] = ips["Suspicious_Votes_num"] / ips["Total_Reports_num"].replace(0, np.nan)
        ips["signal_rate"] = (
            ips["Malicious_Votes_num"] + ips["Suspicious_Votes_num"]
        ) / ips["Total_Reports_num"].replace(0, np.nan)

        clean_summary = pd.DataFrame(
            [
                {"dataset": "otx", "rows_after_cleaning": len(otx), "dedupe_key": "Pulse_ID"},
                {"dataset": "cve", "rows_after_cleaning": len(cve), "dedupe_key": "cveID"},
                {"dataset": "domains", "rows_after_cleaning": len(domains), "dedupe_key": "Domain"},
                {"dataset": "ips", "rows_after_cleaning": len(ips), "dedupe_key": "IP"},
            ]
        )
        display(clean_summary)
        """
    ),
    md(
        r"""
        ## 7. OTX Threat Pulse Analysis

        This section profiles threat themes, MITRE ATT&CK techniques, malware families, affected sectors, and temporal behavior.
        """
    ),
    code(
        r"""
        display(Markdown("### OTX Timeline"))
        otx_timeline = otx.dropna(subset=["Created_dt"]).groupby(pd.Grouper(key="Created_dt", freq="M")).size().reset_index(name="pulses")
        fig = px.line(
            otx_timeline,
            x="Created_dt",
            y="pulses",
            title="OTX Threat Pulses Over Time",
            markers=True,
            template="plotly_white",
        )
        fig.update_layout(title_x=0.02)
        fig.show()

        display(Markdown("### Top Tags"))
        top_tags = explode_counts(otx, "Tags", top_n=20, lower=True)
        display(top_tags.to_frame("count"))
        show_top(top_tags, "Top OTX Tags", top_n=20)

        display(Markdown("### Top MITRE ATT&CK Techniques"))
        top_attack_ids = explode_counts(otx, "Attack_IDs", top_n=25)
        display(top_attack_ids.to_frame("count"))
        show_top(top_attack_ids, "Top MITRE ATT&CK Technique IDs", top_n=25)

        display(Markdown("### Malware Families And Targeting"))
        display(explode_counts(otx, "Malware_Families", top_n=20).to_frame("count"))
        display(explode_counts(otx, "Industries", top_n=15).to_frame("count"))
        display(explode_counts(otx, "Countries", top_n=15).to_frame("count"))
        """
    ),
    code(
        r"""
        display(Markdown("### OTX Topic Discovery With TF-IDF, KMeans, And SVD"))

        otx_topic_df = otx.dropna(subset=["text"]).copy()
        vectorizer = TfidfVectorizer(
            max_features=3000,
            min_df=2,
            ngram_range=(1, 2),
            stop_words="english",
        )
        X_otx_text = vectorizer.fit_transform(otx_topic_df["text"])

        n_clusters = min(7, max(3, len(otx_topic_df) // 250))
        kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
        otx_topic_df["cluster"] = kmeans.fit_predict(X_otx_text)

        svd = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
        coords = svd.fit_transform(X_otx_text)
        otx_topic_df["svd_x"] = coords[:, 0]
        otx_topic_df["svd_y"] = coords[:, 1]

        if n_clusters > 1:
            silhouette = silhouette_score(coords, otx_topic_df["cluster"])
            display(Markdown(f"Approximate silhouette score on 2D SVD projection: **{silhouette:.3f}**"))

        terms = np.array(vectorizer.get_feature_names_out())
        topic_rows = []
        for cluster_id in sorted(otx_topic_df["cluster"].unique()):
            center = kmeans.cluster_centers_[cluster_id]
            top_terms = terms[np.argsort(center)[-12:]][::-1]
            sample_titles = (
                otx_topic_df.loc[otx_topic_df["cluster"] == cluster_id, "Title"]
                .dropna()
                .head(5)
                .tolist()
            )
            topic_rows.append(
                {
                    "cluster": cluster_id,
                    "records": int((otx_topic_df["cluster"] == cluster_id).sum()),
                    "top_terms": ", ".join(top_terms),
                    "sample_titles": " | ".join(sample_titles),
                }
            )
        otx_topics = pd.DataFrame(topic_rows)
        display(otx_topics)

        fig = px.scatter(
            otx_topic_df,
            x="svd_x",
            y="svd_y",
            color=otx_topic_df["cluster"].astype(str),
            hover_data=["Title", "Created_dt", "TLP"],
            title="OTX Text Topic Clusters",
            template="plotly_white",
        )
        fig.update_layout(title_x=0.02, legend_title_text="Cluster")
        fig.show()
        """
    ),
    md(
        r"""
        ## 8. CVE Vulnerability Analysis And Ransomware Classifier

        The ransomware label is `Known` vs `Unknown`. Treat `Unknown` as an unlabeled/unknown state, not proof that ransomware actors never use the vulnerability. The classifier is still useful for learning which features resemble known ransomware-used vulnerabilities.
        """
    ),
    code(
        r"""
        display(Markdown("### CVE Portfolio"))
        display(cve[["dateAdded_dt", "dueDate_dt", "vendorProject", "product", "knownRansomwareCampaignUse", "cwes"]].head())

        cve_year = cve.dropna(subset=["dateAdded_dt"]).groupby(cve["dateAdded_dt"].dt.year).size().reset_index(name="cves")
        fig = px.bar(cve_year, x="dateAdded_dt", y="cves", title="CVEs Added By Year", template="plotly_white")
        fig.update_layout(title_x=0.02, xaxis_title="Year")
        fig.show()

        show_top(cve["vendorProject"].value_counts(), "Top Vendors / Projects", top_n=20)
        show_top(cve["product"].value_counts(), "Top Products", top_n=20)
        top_cwes = explode_counts(cve, "cwes", top_n=25)
        show_top(top_cwes, "Top CWEs", top_n=25)

        ransomware_counts = cve["knownRansomwareCampaignUse"].fillna("Missing").value_counts()
        display(ransomware_counts.to_frame("count"))
        fig = px.pie(
            ransomware_counts.reset_index(),
            names="knownRansomwareCampaignUse",
            values="count",
            title="Known Ransomware Campaign Use Label Distribution",
            hole=0.45,
            template="plotly_white",
        )
        fig.show()
        """
    ),
    code(
        r"""
        display(Markdown("### CVE Ransomware-Use Classifier"))

        cve_model_df = cve[
            ["text", "vendorProject", "product", "cwes_clean", "days_to_due", "ransomware_known"]
        ].copy()
        y = cve_model_df["ransomware_known"]
        X = cve_model_df.drop(columns=["ransomware_known"])

        X_train, X_test, y_train, y_test = safe_train_test_split(X, y, test_size=0.25)

        cve_features = ColumnTransformer(
            transformers=[
                ("text", TfidfVectorizer(max_features=2500, ngram_range=(1, 2), stop_words="english", min_df=2), "text"),
                ("cwe", TfidfVectorizer(token_pattern=r"[^,\s]+", lowercase=False), "cwes_clean"),
                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), ["vendorProject", "product"]),
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), ["days_to_due"]),
            ],
            remainder="drop",
        )
        cve_clf = Pipeline(
            steps=[
                ("features", cve_features),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        )
        cve_clf.fit(X_train, y_train)
        cve_pred = cve_clf.predict(X_test)
        cve_report = model_report("CVE Known-Ransomware Classifier", y_test, cve_pred, labels=[0, 1])

        try:
            feature_names = cve_clf.named_steps["features"].get_feature_names_out()
            coefficients = cve_clf.named_steps["model"].coef_[0]
            coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
            display(Markdown("### Features Most Associated With Known Ransomware Use"))
            display(coef_df.sort_values("coefficient", ascending=False).head(20))
            display(Markdown("### Features Least Associated With Known Ransomware Use"))
            display(coef_df.sort_values("coefficient", ascending=True).head(20))
        except Exception as exc:
            print("Feature importance extraction skipped:", exc)

        cve_scored = cve.copy()
        cve_scored["ransomware_similarity_score"] = cve_clf.predict_proba(X)[:, 1]
        display(Markdown("### CVEs With Highest Ransomware Similarity Score"))
        display(
            cve_scored.sort_values("ransomware_similarity_score", ascending=False)[
                ["cveID", "vendorProject", "product", "vulnerabilityName", "knownRansomwareCampaignUse", "cwes", "ransomware_similarity_score", "dueDate_dt"]
            ].head(25)
        )
        """
    ),
    md(
        r"""
        ## 9. Domain Intelligence Analysis, Severity Modeling, And Anomaly Detection
        """
    ),
    code(
        r"""
        display(Markdown("### Domain Severity And TLD Patterns"))
        domain_severity = severity_ordered_counts(domains)
        display(domain_severity.to_frame("count"))
        fig = px.bar(domain_severity.reset_index(), x="Threat_Severity", y="count", title="Domain Threat Severity", template="plotly_white")
        fig.update_layout(title_x=0.02)
        fig.show()

        show_top(domains["TLD"].value_counts(), "Top Domain TLDs", top_n=20)

        fig = px.scatter(
            domains,
            x="Reputation_num",
            y="signal_rate",
            color="Threat_Severity",
            size="Malicious_Votes_num",
            hover_data=["Domain", "TLD", "Registrar", "Popularity_Rank_num"],
            title="Domain Reputation vs Malicious/Suspicious Vote Rate",
            template="plotly_white",
        )
        fig.update_layout(title_x=0.02)
        fig.show()

        display(Markdown("### Highest-Signal Domains"))
        display(
            domains.sort_values(["Threat_Severity", "signal_rate", "Malicious_Votes_num"], ascending=[True, False, False])[
                ["Domain", "TLD", "Threat_Severity", "Reputation_num", "Malicious_Votes_num", "Suspicious_Votes_num", "signal_rate", "Popularity_Rank_num", "Last_Analysis_Date_dt"]
            ].head(25)
        )
        """
    ),
    code(
        r"""
        display(Markdown("### Domain Threat Severity Classifier"))

        domain_feature_cols_num = [
            "Domain_Length_num", "Has_Numbers_bin", "Has_Hyphen_bin", "Reputation_num",
            "Malicious_Votes_num", "Suspicious_Votes_num", "Harmless_Votes_num",
            "Undetected_Votes_num", "Total_Engines_num", "Popularity_Rank_num",
            "domain_age_days", "malicious_rate", "suspicious_rate", "signal_rate"
        ]
        domain_feature_cols_cat = ["TLD", "Registrar", "Data_Source"]

        domain_model_df = domains.dropna(subset=["Threat_Severity"]).copy()
        X = domain_model_df[domain_feature_cols_num + domain_feature_cols_cat]
        y = domain_model_df["Threat_Severity"]
        X_train, X_test, y_train, y_test = safe_train_test_split(X, y, test_size=0.30)

        domain_preprocess = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), domain_feature_cols_num),
                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), domain_feature_cols_cat),
            ]
        )
        domain_clf = Pipeline(
            steps=[
                ("features", domain_preprocess),
                ("model", RandomForestClassifier(
                    n_estimators=400,
                    random_state=RANDOM_STATE,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                )),
            ]
        )
        domain_clf.fit(X_train, y_train)
        domain_pred = domain_clf.predict(X_test)
        domain_labels = sorted(y.unique())
        domain_report = model_report("Domain Severity Classifier", y_test, domain_pred, labels=domain_labels)

        # Feature importance for transformed feature space.
        try:
            domain_feature_names = domain_clf.named_steps["features"].get_feature_names_out()
            domain_importances = domain_clf.named_steps["model"].feature_importances_
            domain_importance_df = pd.DataFrame(
                {"feature": domain_feature_names, "importance": domain_importances}
            ).sort_values("importance", ascending=False)
            display(domain_importance_df.head(25))
            fig = px.bar(
                domain_importance_df.head(20).sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                title="Domain Severity Model: Top Feature Importances",
                template="plotly_white",
            )
            fig.update_layout(title_x=0.02, height=620)
            fig.show()
        except Exception as exc:
            print("Domain feature importance extraction skipped:", exc)

        domain_anomalies = numeric_anomaly_detection(
            domains,
            feature_cols=domain_feature_cols_num,
            id_col="Domain",
            contamination=0.08,
        )
        domains_scored = domains.merge(
            domain_anomalies[["Domain", "anomaly_flag", "anomaly_score"]],
            on="Domain",
            how="left",
        )
        display(Markdown("### Most Anomalous Domains"))
        display(
            domains_scored.sort_values("anomaly_score")[
                ["Domain", "TLD", "Threat_Severity", "Reputation_num", "Malicious_Votes_num", "signal_rate", "anomaly_score", "Registrar"]
            ].head(20)
        )

        fig = px.scatter(
            domains_scored,
            x="signal_rate",
            y="anomaly_score",
            color="Threat_Severity",
            hover_data=["Domain", "TLD", "Registrar"],
            title="Domain Anomaly Score vs Vote Signal Rate",
            template="plotly_white",
        )
        fig.update_layout(title_x=0.02)
        fig.show()
        """
    ),
    md(
        r"""
        ## 10. IP Intelligence Analysis, Severity Modeling, And Anomaly Detection
        """
    ),
    code(
        r"""
        display(Markdown("### IP Severity, Geography, Registry, And TOR Patterns"))
        ip_severity = severity_ordered_counts(ips)
        display(ip_severity.to_frame("count"))
        fig = px.bar(ip_severity.reset_index(), x="Threat_Severity", y="count", title="IP Threat Severity", template="plotly_white")
        fig.update_layout(title_x=0.02)
        fig.show()

        show_top(ips["Country"].fillna("Missing").value_counts(), "Top IP Countries", top_n=20)
        show_top(ips["Owner"].fillna("Missing").value_counts(), "Top IP Owners", top_n=20)
        show_top(ips["Regional_Registry"].fillna("Missing").value_counts(), "Top Regional Registries", top_n=10)

        tor_counts = ips["TOR_Node"].fillna("Missing").value_counts()
        display(tor_counts.to_frame("count"))

        fig = px.scatter(
            ips,
            x="Reputation_Score_num",
            y="signal_rate",
            color="Threat_Severity",
            size="Malicious_Votes_num",
            hover_data=["IP", "Country", "ASN_num", "Owner", "TOR_Node"],
            title="IP Reputation vs Malicious/Suspicious Vote Rate",
            template="plotly_white",
        )
        fig.update_layout(title_x=0.02)
        fig.show()
        """
    ),
    code(
        r"""
        display(Markdown("### IP Threat Severity Classifier"))

        ip_feature_cols_num = [
            "ASN_num", "Malicious_Votes_num", "Suspicious_Votes_num", "Harmless_Votes_num",
            "Undetected_Votes_num", "Total_Reports_num", "Reputation_Score_num",
            "Times_Submitted_num", "TOR_Node_bin", "malicious_rate", "suspicious_rate", "signal_rate"
        ]
        ip_feature_cols_cat = ["Country", "Continent", "Owner", "Regional_Registry", "Threat_Label", "Threat_Category"]

        ip_model_df = ips.dropna(subset=["Threat_Severity"]).copy()
        X = ip_model_df[ip_feature_cols_num + ip_feature_cols_cat]
        y = ip_model_df["Threat_Severity"]
        X_train, X_test, y_train, y_test = safe_train_test_split(X, y, test_size=0.30)

        ip_preprocess = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), ip_feature_cols_num),
                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), ip_feature_cols_cat),
            ]
        )
        ip_clf = Pipeline(
            steps=[
                ("features", ip_preprocess),
                ("model", RandomForestClassifier(
                    n_estimators=400,
                    random_state=RANDOM_STATE,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                )),
            ]
        )
        ip_clf.fit(X_train, y_train)
        ip_pred = ip_clf.predict(X_test)
        ip_labels = sorted(y.unique())
        ip_report = model_report("IP Severity Classifier", y_test, ip_pred, labels=ip_labels)

        try:
            ip_feature_names = ip_clf.named_steps["features"].get_feature_names_out()
            ip_importances = ip_clf.named_steps["model"].feature_importances_
            ip_importance_df = pd.DataFrame(
                {"feature": ip_feature_names, "importance": ip_importances}
            ).sort_values("importance", ascending=False)
            display(ip_importance_df.head(25))
            fig = px.bar(
                ip_importance_df.head(20).sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                title="IP Severity Model: Top Feature Importances",
                template="plotly_white",
            )
            fig.update_layout(title_x=0.02, height=620)
            fig.show()
        except Exception as exc:
            print("IP feature importance extraction skipped:", exc)

        ip_anomalies = numeric_anomaly_detection(
            ips,
            feature_cols=ip_feature_cols_num,
            id_col="IP",
            contamination=0.08,
        )
        ips_scored = ips.merge(
            ip_anomalies[["IP", "anomaly_flag", "anomaly_score"]],
            on="IP",
            how="left",
        )
        display(Markdown("### Most Anomalous IPs"))
        display(
            ips_scored.sort_values("anomaly_score")[
                ["IP", "Country", "Owner", "Threat_Severity", "Reputation_Score_num", "Malicious_Votes_num", "signal_rate", "TOR_Node", "anomaly_score"]
            ].head(20)
        )

        fig = px.scatter(
            ips_scored,
            x="signal_rate",
            y="anomaly_score",
            color="Threat_Severity",
            symbol="TOR_Node",
            hover_data=["IP", "Country", "Owner", "ASN_num"],
            title="IP Anomaly Score vs Vote Signal Rate",
            template="plotly_white",
        )
        fig.update_layout(title_x=0.02)
        fig.show()
        """
    ),
    md(
        r"""
        ## 11. Cross-Dataset Synthesis

        There is no reliable shared key across these four datasets. This section therefore avoids fake joins and instead creates an analyst synthesis from comparable signals: ransomware themes, MITRE techniques, CVE exposure, high-severity domains, and high-severity IPs.
        """
    ),
    code(
        r"""
        ransomware_otx = otx[
            otx[["Title", "Description", "Tags", "Malware_Families"]]
            .fillna("")
            .agg(" ".join, axis=1)
            .str.contains("ransomware", case=False, na=False)
        ].copy()

        phishing_otx = otx[
            otx[["Title", "Description", "Tags"]]
            .fillna("")
            .agg(" ".join, axis=1)
            .str.contains("phishing|credential|stealer|infostealer", case=False, regex=True, na=False)
        ].copy()

        known_ransomware_cves = cve_scored[cve_scored["knownRansomwareCampaignUse"].eq("Known")].copy()

        high_domain = domains_scored[domains_scored["Threat_Severity"].isin(["High", "Critical"])].copy()
        high_ip = ips_scored[ips_scored["Threat_Severity"].isin(["High", "Critical"])].copy()

        synthesis = pd.DataFrame(
            [
                {"signal": "OTX ransomware-themed pulses", "count": len(ransomware_otx)},
                {"signal": "OTX phishing / credential theft / stealer-themed pulses", "count": len(phishing_otx)},
                {"signal": "CVEs labeled known ransomware campaign use", "count": len(known_ransomware_cves)},
                {"signal": "High/Critical malicious domains", "count": len(high_domain)},
                {"signal": "High/Critical malicious IPs", "count": len(high_ip)},
            ]
        )
        display(synthesis)

        display(Markdown("### Priority CVEs"))
        display(
            cve_scored.sort_values(["ransomware_known", "ransomware_similarity_score", "dateAdded_dt"], ascending=[False, False, False])[
                ["cveID", "vendorProject", "product", "vulnerabilityName", "knownRansomwareCampaignUse", "ransomware_similarity_score", "dateAdded_dt", "dueDate_dt", "cwes"]
            ].head(30)
        )

        display(Markdown("### Priority Domains"))
        display(
            domains_scored.sort_values(["Threat_Severity", "signal_rate", "anomaly_score"], ascending=[True, False, True])[
                ["Domain", "TLD", "Threat_Severity", "Reputation_num", "Malicious_Votes_num", "Suspicious_Votes_num", "signal_rate", "anomaly_score", "Registrar"]
            ].head(30)
        )

        display(Markdown("### Priority IPs"))
        display(
            ips_scored.sort_values(["Threat_Severity", "signal_rate", "anomaly_score"], ascending=[True, False, True])[
                ["IP", "Country", "ASN_num", "Owner", "Threat_Severity", "Reputation_Score_num", "Malicious_Votes_num", "signal_rate", "TOR_Node", "anomaly_score"]
            ].head(30)
        )
        """
    ),
    md(
        r"""
        ## 12. Final Findings Generator

        This cell produces a concise, data-driven research summary from the computed outputs.
        """
    ),
    code(
        r"""
        newest_otx = otx["Created_dt"].max()
        newest_cve = cve["dateAdded_dt"].max()
        newest_domain = domains["Last_Analysis_Date_dt"].max()
        newest_ip = ips["Last_Analysis_Date_dt"].max()

        top_attack = top_attack_ids.index[0] if len(top_attack_ids) else "n/a"
        top_tag = top_tags.index[0] if len(top_tags) else "n/a"
        top_cwe = top_cwes.index[0] if len(top_cwes) else "n/a"

        final_markdown = f'''
        ## Executive Research Summary

        **Coverage.** The notebook analyzed {len(otx):,} de-duplicated OTX pulses, {len(cve):,} CVEs, {len(domains):,} domains, and {len(ips):,} IPs.

        **Freshness.** Latest observed dates: OTX `{newest_otx}`, CVE additions `{newest_cve}`, domain analysis `{newest_domain}`, and IP analysis `{newest_ip}`.

        **Threat themes.** The strongest OTX tag signal is **{top_tag}**, and the most frequent MITRE ATT&CK technique ID is **{top_attack}**.

        **Vulnerability exposure.** The leading CWE signal is **{top_cwe}**. CVE ransomware modeling should be read as similarity-based triage because `Unknown` is not a verified negative class.

        **Infrastructure risk.** Domain/IP severity and anomaly models identify high-signal outliers using scanner votes, reputation, geography/registry ownership, and structural features.

        **Analyst action.** Prioritize the generated CVE/domain/IP tables for validation, enrichment, and defensive control mapping. Do not directly operationalize model predictions without human review and current external threat-intel confirmation.
        '''
        display(Markdown(final_markdown))

        model_scorecard = pd.DataFrame([cve_report, domain_report, ip_report])
        display(Markdown("### Model Scorecard"))
        display(model_scorecard)
        """
    ),
    md(
        r"""
        ## 13. Reproducibility Checklist

        - Source paths are centralized in `DATA_PATHS`.
        - Expected parsed row counts are encoded in `EXPECTED_ROWS`.
        - Randomized ML steps use `RANDOM_STATE = 42`.
        - Original CSV files are read only and are not modified.
        - Cross-dataset conclusions are synthesized without unsupported joins.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


OUTPUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {OUTPUT}")
