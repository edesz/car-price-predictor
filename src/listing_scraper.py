import re

from numpy import nan as np_nan


def pandas_clean_data(
    df,
    pcols=["msrp", "dealer_price"],
    cols_to_move_to_front=["link", "listing_number", "page_number"],
):
    # Append target column as minimum of two price columns
    df[pcols] = df[pcols].replace("[\$,]", "", regex=True).astype(float)
    df["price"] = df[pcols].min(axis=1, skipna=True)

    # Move some columns to front of df
    cols = df.columns.tolist()
    for c in cols_to_move_to_front:
        cols.insert(0, cols.pop(cols.index(c)))
    df = df.reindex(columns=cols)
    return df


def scrape_single_listing(soup, page_number=1, listing_number=1, state="TX"):
    """Scrape a single listing"""
    d = {}
    e = {}
    for k in [
        "Fuel Type",
        "City MPG",
        "Highway MPG",
        "Drivetrain",
        "Engine",
        "Mileage",
        "Exterior Color",
        "Interior Color",
        "Stock",
        "Transmission",
        "VIN",
        "seller_address",
        "seller_zip",
        "seller_rating",
        "seller_reviews",
        "type",
        "title",
        "miles",
        "dealer_price",
        "msrp",
        "consumer_stars",
        "consumer_reviews",
        "Comfort",
        "Performance",
        "Exterior Styling",
        "Interior Design",
        "Value for the Money",
        "Reliability",
    ]:
        d[k] = np_nan

    # Add page number, listing number, url to dictionary
    d["page_number"] = page_number
    d["listing_number"] = listing_number
    try:
        url_link = "https://www.cars.com/vehicledetail/detail/" + (
            soup.find("section", {"class": "vehicle-info"}).find("a")["href"]
        )
        d["link"] = url_link.split("/shopping/")[-1] + "overview"
    except Exception as e:
        e["link"] = [
            f"page_{page_number}",
            f"listing_{listing_number}",
            "link\n" + str(e),
        ]

    # Get car features listed under Basics section
    try:
        basics_items = soup.find(
            "div", {"class": "vdp-details-basics"}
        ).find_all("li", {"class": "vdp-details-basics__item"})
        # print(listing_number)
        for basics_field in basics_items:
            key = basics_field.find("strong").text.replace(":", "")
            value = basics_field.find("span").text.strip()
            d[key] = value
    except Exception as e:
        page_num_str = f"page_{page_number}"
        listing_str = f"listing_{listing_number}"
        e["link"] = [page_num_str, listing_str, str(e)]

    # Get seller address and zipcode
    try:
        seller_addr = soup.find(
            "div", {"class": "seller-details-location"}
        ).text.strip()
        d["seller_address"] = seller_addr
    except Exception as e:
        e["seller_address"] = [
            f"page_{page_number}",
            f"listing_{listing_number}",
            str(e),
        ]
        e["seller_zip"] = [
            f"page_{page_number}",
            f"listing_{listing_number}",
            np_nan,
        ]
    d["seller_zip"] = d["seller_address"].split(state + " ")[-1]

    # Get seller rating and reviews
    try:
        seller_rat = soup.find(
            "p", {"class": "rating__link rating__link--has-reviews"}
        ).text.strip()
        d["seller_rating"] = seller_rat.split(")")[0].replace("(", "")
        d["seller_reviews"] = (
            seller_rat.split(")")[-1].replace(" Reviews", "").strip()
        )
    except Exception as e:
        e["seller_rating"] = [
            f"page_{page_number}",
            f"listing_{listing_number}",
            str(e),
        ]
        e["seller_reviews"] = [
            f"page_{page_number}",
            f"listing_{listing_number}",
            np_nan,
        ]

    # Get vehicle type (new or old) and title of listing
    title_d = soup.find("div", {"class": "vehicle-info__title-container"})
    d["type"] = title_d.find("h1", {"class": "vehicle-info__stock-type"}).text
    d["title"] = title_d.find(
        "h1", {"class": "cui-heading-2--secondary vehicle-info__title"}
    ).text.strip()

    # Get miles driven
    try:
        miles_cont = soup.find(
            "div",
            {"class": "vdp-cap-price__mileage--mobile vehicle-info__mileage"},
        )
        # print(listing_number, miles_cont)
        d["miles"] = miles_cont.text if miles_cont else None
        # print(listing_number, d["miles"])
    except Exception as e:
        e["miles"] = [
            f"page_{page_number}",
            f"listing_{listing_number}",
            str(e),
        ]

    # Get dealer price (if listed)
    dealer_price = soup.find(
        "div",
        {"class": "vehicle-info__price vehicle-info__price--dealer-price"},
    )
    # print(listing_number, dealer_price.text)
    if not dealer_price:
        price_class_str = (
            "vehicle-info__price-display "
            "vehicle-info__price-display--dealer cui-heading-2"
        )
        dealer_price = soup.find("span", {"class": price_class_str})
        # print(dealer_price)
        if dealer_price:
            d["dealer_price"] = dealer_price.text
    elif dealer_price.find_all("span"):
        dp = dealer_price.find_all("span")
        # print(listing_number, dp)
        dp_value = [
            m.text if "Dealer Price " not in m.text else np_nan for m in dp
        ]
        # print(listing_number, dp_value)
        d["dealer_price"] = next(
            (item for item in dp_value if item is not np_nan), np_nan
        )
    # print(listing_number, d["dealer_price"])

    # Get MSRP (if listed)
    msrp = soup.find("div", {"class": "vehicle-info__price--msrp"})
    # print(listing_number, msrp)
    if msrp:
        msrp_cont = msrp.find_all("span")
        # print(listing_number, msrp_cont)
        msrp_value = [
            m.text if "MSRP" not in m.text else np_nan for m in msrp_cont
        ]
        d["msrp"] = next(
            (item for item in msrp_value if item is not np_nan), np_nan
        )
    # print(listing_number, d["msrp"])

    # Get dealer reviews (these are dealer reviews left by other customers)
    cons_star = soup.find("div", {"class": "overall-review-stars"})
    d["consumer_stars"] = cons_star.text.strip() if cons_star else np_nan
    # print(listing_number, d["consumer_stars"])
    cons_reviews = soup.find("div", {"class": "review-stars-average"})
    d["consumer_reviews"] = (
        cons_reviews.text.strip()
        .replace("Average based on", "")
        .strip()
        .replace(" reviews", "")
        if cons_reviews
        else np_nan
    )
    # print(listing_number, d["consumer_reviews"])

    # Get review ratings by category
    try:
        rev__cont_str = "review-rating-breakdown"
        reviews_ratings = soup.find("div", {"class": rev__cont_str})
        if reviews_ratings:
            rev_cols = reviews_ratings.find_all(
                "div", {"class": "review-column"}
            )
            for rev_col in rev_cols:
                rev_rows = rev_col.find_all("div", {"class": "review-row"})
                for r in rev_rows:
                    fields_values = r.findAll("p")
                    key = ""
                    stars = ""
                    for field_value in fields_values:
                        if not field_value.find("strong"):
                            key = key + field_value.text
                        else:
                            stars = stars + field_value.find("strong").text
                        d[key] = stars
    except Exception as e:
        e["reviews_ratings"] = [
            f"page_{page_number}",
            f"listing_{listing_number}",
            str(e),
        ]

    #     print(listing_number)
    #     for k, v in d.items():
    #         print(k, v)
    #     print("\n")

    # Get APR
    try:
        apr_matches_type1 = re.findall(
            "months at \d+\.\d+% APR", soup.prettify()
        )
        apr_matches_type2 = re.findall("\d+\.\d+% APR for", soup.prettify())
        matches = (
            apr_matches_type1 + apr_matches_type2
            if not apr_matches_type1
            else apr_matches_type1
        )
        # print(matches)
        d["APR"] = min(re.findall("\d+\.\d+", ", ".join(matches)))
        # print(listing_number, d["APR"], len(set(matches)))
    except Exception as e:
        e["APR"] = [f"page_{page_number}", f"listing_{listing_number}", str(e)]

    # Get lowest per month financing option
    # # First type of element location
    try:
        pmonth_match_1 = soup.find_all(
            "div", {"class": "online-shopper-v2-payments__price"}
        )
        # print(pmonth_match_1)
        per_month1 = []
        if isinstance(pmonth_match_1, list):
            for pmonth_match1 in pmonth_match_1:
                # print(p)
                p_val = pmonth_match1.text.replace("\n", "").replace("$", "")
                if p_val != "":
                    per_month1.append(int(p_val))
                else:
                    per_month1.append(None)
        else:
            per_month1 = [None]
    except Exception as e:
        e["per_month_min_method1"] = [
            f"page_{page_number}",
            f"listing_{listing_number}",
            str(e),
        ]
    # print(i, per_month1)
    # # Second type of element location
    try:
        pmonth_match_2 = soup.find(
            "span", {"class": "cui-heading-1 monthly-payment"}
        )
        pmonth_match_2 = (
            int(pmonth_match_2.text.replace("$", ""))
            if pmonth_match_2
            else None
        )
        # # Combine two types of elements
        pmonth_combined = [
            x for x in per_month1 + [pmonth_match_2] if x is not None
        ]
        # print(pmonth_combined)
        # # Get minimum of both types of elements to use as per month financing
        pmonth_final = min(pmonth_combined) if pmonth_combined else None
        # print(i, pmonth_combined, pmonth_final)
        d["per_month_min"] = pmonth_final
    except Exception as e:
        e["per_month_min_method2"] = [
            f"page_{page_number}",
            f"listing_{listing_number}",
            str(e),
        ]
    return d, e
