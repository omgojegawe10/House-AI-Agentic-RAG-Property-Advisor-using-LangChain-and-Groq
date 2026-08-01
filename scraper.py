from bs4 import BeautifulSoup
import requests
import pandas as pd
import re

class MagicBricks:
    def __init__(self):
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/137.0 Safari/537.36"
            )
        }

    def get_ready_to_move_flats(self, city):
        details = [
            "Carpet Area",
            "Status",
            "Floor",
            "Transaction",
            "Furnishing",
            "Facing",
            "Overlooking",
            "Society",
            "Bathroom",
            "Balcony",
            "Car Parking",
            "Ownership",
            "Super Area",
            "Dimensions",
            "Plot Area",
        ]

        city_slug = city.replace(" ", "-").lower()
        base_url = (
            f"https://www.magicbricks.com/"
            f"ready-to-move-flats-in-{city_slug}-pppfs"
        )

        try:
            response = requests.get(base_url, headers=self.headers)
            if response.status_code != 200:
                print("Failed to fetch first page.")
                return None

            soup = BeautifulSoup(response.text, "lxml")
            container = soup.find("div", {"class": "mb-srp__left"})
            if not container:
                print("No listings found.")
                return None

            tab = container.find(
                "li",
                {"class": "mb-srp__tabs__list--item"}
            )

            if not tab:
                print("Could not determine total results.")
                return None

            total_results_text = tab.get_text(strip=True)
            matches = re.findall(r"\d{1,3}(?:,\d{3})*", total_results_text)
            if not matches:
                print("Could not parse total results.")
                return None

            total_results = int(matches[0].replace(",", ""))
            num_of_pages = (total_results + 29) // 30
            print(f"Found {total_results} properties.")
            print(f"Scraping {num_of_pages} pages...\n")

            flats_data = []
            for page in range(1, num_of_pages + 1):
                if page == 1:
                    url = base_url
                else:
                    url = f"{base_url}/page-{page}"

                print(f"Scraping page {page}...")
                response = requests.get(url, headers=self.headers)
                if response.status_code != 200:
                    continue

                soup = BeautifulSoup(response.text, "lxml")
                for item in soup.find_all(
                    "div",
                    {"class": "mb-srp__list"}
                ):

                    title_tag = item.find("h2")
                    title = (
                        title_tag.get_text(strip=True)
                        if title_tag
                        else None
                    )

                    labels = {key: None for key in details}
                    for detail_item in item.find_all(
                        "div",
                        {
                            "class":
                            "mb-srp__card__summary__list--item"
                        }
                    ):

                        label_tag = detail_item.find(
                            "div",
                            {
                                "class":
                                "mb-srp__card__summary--label"
                            }
                        )

                        value_tag = detail_item.find(
                            "div",
                            {
                                "class":
                                "mb-srp__card__summary--value"
                            }
                        )

                        if label_tag and value_tag:
                            label = label_tag.get_text(strip=True)
                            value = value_tag.get_text(strip=True)

                            if label in labels:
                                labels[label] = value

                    desc_tag = item.find(
                        "div",
                        {
                            "class":
                            "mb-srp__card--desc--text"
                        }
                    )

                    description = (
                        desc_tag.get_text(strip=True)
                        if desc_tag
                        else None
                    )

                    amount_tag = item.find(
                        "div",
                        {
                            "class":
                            "mb-srp__card__price--amount"
                        }
                    )

                    amount = (
                        amount_tag.get_text(strip=True)
                        if amount_tag
                        else None
                    )

                    sqft_tag = item.find(
                        "div",
                        {
                            "class":
                            "mb-srp__card__price--size"
                        }
                    )

                    price_per_sqft = (
                        sqft_tag.get_text(strip=True)
                        if sqft_tag
                        else None
                    )

                    flat_data = [
                        title,
                        description,
                        amount,
                        price_per_sqft,
                        city,
                    ] + list(labels.values())

                    flats_data.append(flat_data)

            return flats_data

        except Exception as e:
            print(f"Error: {e}")
            return None


# ==========================
# SCRAPE ONE CITY
# ==========================

city = "Nanded"

scraper = MagicBricks()
flats_info = scraper.get_ready_to_move_flats(city)

if flats_info:

    columns = [
        "Title",
        "Description",
        "Amount",
        "Price_per_sqft",
        "City",
        "Carpet Area",
        "Status",
        "Floor",
        "Transaction",
        "Furnishing",
        "Facing",
        "Overlooking",
        "Society",
        "Bathroom",
        "Balcony",
        "Car Parking",
        "Ownership",
        "Super Area",
        "Dimensions",
        "Plot Area",
    ]

    df = pd.DataFrame(flats_info, columns=columns)

    print(df.head())

    df.to_csv(f"{city}_House_Price.csv", index=False)
    print(f"\nSaved {len(df)} properties to {city}_House_Price.csv")

else:
    print(f"No data found for {city}")