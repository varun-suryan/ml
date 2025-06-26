from bs4 import BeautifulSoup
import requests, time
import http.client, urllib

conn = http.client.HTTPSConnection("api.pushover.net:443")
url = "https://www.cefcu.com/rates.html?tab=mortgage"  # Replace with the URL of the webpage you want to read


def extract_mortgage_sections(html):
    soup = BeautifulSoup(html, "html.parser")
    data = {}

    # Define the mortgage types we care about
    mortgage_types = ["Fixed-Rate Mortgages", "Adjustable-Rate Mortgages"]

    for mt in mortgage_types:
        # Find header matching the exact visible text
        header = soup.find(lambda tag: tag.name in ["h2", "h3", "h4"] and mt in tag.get_text())
        if not header:
            print(f"[WARN] '{mt}' header not found")
            continue

        # Expect the table directly *following* the header
        table = header.find_next("table")
        if not table:
            print(f"[WARN] '{mt}' table not found")
            continue

        # Parse table rows
        rows = []
        for tr in table.find_all("tr"):
            cells = [cell.get_text(strip=True) for cell in tr.find_all(["th", "td"]) if cell.get_text(strip=True)]
            if cells:
                rows.append(cells)

        if rows:
            data[mt] = rows

    return data


while True:
    response = requests.get(url)
    if response.status_code == 200:
        # Proceed to get HTML content
        pass
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
    sections = extract_mortgage_sections(response.text)

    for mt, rows in sections.items():
        headers = rows[0]
        for row in rows[1:]:
            if row[0] == '7/1 ARM - 30 Year Term':
                if float(row[1][:-1]) < 6.375:
                    conn.request("POST", "/1/messages.json",
                                 urllib.parse.urlencode({
                                     "token": "a9q7dnaras8bzxon5n933h7brnk84u",
                                     "user": "uho96vgxkye3gtj9wz5d8wun3miosa",
                                     "message": r"Good News! 7/1 ARM - 30 Year Term has dropped to {}%".format(
                                         float(row[1][:-1])),
                                 }), {"Content-type": "application/x-www-form-urlencoded"})

            if row[0] == '30 Year - Fixed Rate':
                if float(row[1][:-1]) < 6.625:
                    conn.request("POST", "/1/messages.json",
                                 urllib.parse.urlencode({
                                     "token": "a9q7dnaras8bzxon5n933h7brnk84u",
                                     "user": "uho96vgxkye3gtj9wz5d8wun3miosa",
                                     "message": r"Good News! 30 Year - Fixed Rate has dropped to {}%".format(
                                         float(row[1][:-1])),
                                 }), {"Content-type": "application/x-www-form-urlencoded"})

            # print(dict(zip(headers, row)))
    time.sleep(1800)
