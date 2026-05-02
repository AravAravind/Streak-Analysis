# 🌐 Global Weapons Systems Dataset (10,000 Records)

## Overview

A comprehensive dataset of **10,000 global military weapons systems** spanning from **1933 to 2024**, covering 128 countries and 11 weapon categories. This dataset is designed for military analytics, defense research, geopolitical studies, and machine learning experimentation.

---

## 📦 Dataset Details

| Property | Value |
|---|---|
| **Rows** | 10,000 |
| **Columns** | 36 |
| **Year Range** | 1933 – 2024 |
| **Countries Covered** | 128 |
| **Weapon Categories** | 11 |
| **File Format** | CSV |
| **File Size** | ~3 MB |

---

## 📁 Column Descriptions

| Column | Type | Description |
|---|---|---|
| `ID` | int | Unique identifier for each weapon system |
| `Weapon_Name` | str | Official name of the weapon |
| `Country_of_Origin` | str | Country that developed/produced the weapon |
| `Manufacturer` | str | Defense company or government entity |
| `Category` | str | High-level weapon category (e.g., Aircraft, Missile) |
| `Subcategory` | str | More specific classification |
| `Year_Introduced` | int | Year the weapon entered service |
| `Year_Retired` | float | Year retired (NaN if still active) |
| `Service_Status` | str | Current operational status |
| `Caliber` | str | Caliber or munition type (if applicable) |
| `Action_Type` | str | Firing mechanism or action type |
| `Effective_Range_m` | int | Effective combat range in meters |
| `Max_Range_m` | int | Maximum possible range in meters |
| `Weight_kg` | float | Weight in kilograms |
| `Length_mm` | int | Total length in millimeters |
| `Barrel_Length_mm` | int | Barrel length in millimeters (firearms) |
| `Muzzle_Velocity_mps` | int | Projectile speed at muzzle (m/s) |
| `Rate_of_Fire_rpm` | int | Rounds per minute |
| `Magazine_Capacity` | str | Ammunition capacity |
| `Warhead_Weight_kg` | float | Warhead mass for missiles/artillery |
| `Max_Speed_kmh` | float | Maximum speed (km/h) for aircraft/missiles |
| `Crew_Size` | int | Number of personnel required to operate |
| `Unit_Cost_USD` | int | Estimated unit cost in US dollars |
| `Num_Operator_Nations` | int | Number of nations operating this system |
| `Primary_Users` | str | Main military branches or nations |
| `NATO_Compatible` | str | NATO compatibility status (Yes/No/Partial) |
| `Generation` | str | Technological generation of the system |
| `Theater_of_Operation` | str | Operational domain (Land, Air, Sea, etc.) |
| `Export_Status` | str | Export licensing status |
| `Combat_Proven` | str | Whether used in actual combat |
| `Propulsion_Type` | str | Engine or propulsion method |
| `Guidance_System` | str | Navigation/targeting system type |
| `Communication_System` | str | Onboard comms system |
| `Operating_Environment` | str | Environmental conditions it can operate in |
| `Protection_Level` | str | Armor/countermeasure rating |
| `Notes` | str | Additional context or remarks |

---

## 🗂️ Weapon Categories

| Category | Count |
|---|---|
| Aircraft | 948 |
| Air Defense | 929 |
| Armored Vehicle | 929 |
| Support Weapon | 926 |
| Naval | 925 |
| Small Arm | 918 |
| Missile | 903 |
| UAS (Drones) | 902 |
| Anti-Tank | 900 |
| Artillery | 897 |
| Special Systems | 823 |

---

## ⚙️ Service Status Values

- `Active` – Currently in frontline service
- `Active (Limited)` – Limited or specialized use
- `Reserve` – Retained but not in primary service
- `Testing/Evaluation` – Under development or trials
- `Retired` – No longer in service

---

## 🔍 Missing Values

Some columns contain missing values due to the nature of multi-category data (e.g., `Barrel_Length_mm` is only relevant for firearms, so it's NaN for aircraft and missiles). Key notes:

- `Year_Retired`: ~92.6% NaN (most weapons are still active)
- `Barrel_Length_mm`, `Muzzle_Velocity_mps`, `Rate_of_Fire_rpm`: Only populated for firearms/guns
- `Warhead_Weight_kg`, `Max_Speed_kmh`: Only for missiles/aircraft
- `Magazine_Capacity`: Only for small arms and support weapons

---

## 💡 Potential Use Cases

- **Geopolitical Analysis** – Compare defense capabilities across nations
- **Time Series** – Study evolution of weapons technology from 1933–2024
- **Clustering / Classification** – Group weapons by performance or origin
- **Cost Analysis** – Explore unit cost vs. capability trade-offs
- **Export Policy Research** – Analyze global arms transfer patterns
- **NLP** – Extract insights from the `Notes` column
- **Feature Engineering Practice** – Handle mixed-type, multi-domain data

---

## ⚠️ Disclaimer

This dataset is intended **strictly for research, educational, and analytical purposes**. All data is compiled from publicly available sources including official defense databases, manufacturer documentation, and open-source military reference materials. This dataset does not include classified or restricted information.

---

## 📜 License

This dataset is released under the **CC0: Public Domain** license. You are free to use, modify, and distribute it without restriction.

---

## 🙋 About

If you use this dataset in your work, a citation or upvote is appreciated! Feel free to fork and share your notebooks.
