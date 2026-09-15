from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "munich": "Sunny, 22°C",
        "berlin": "Cloudy, 19°C",
        "vienna": "Partly cloudy, 21°C",
    }

    normalized_city = city.split(",")[0].strip().lower()

    return weather_data.get(
        normalized_city,
        f"Weather data for {city} is not available.",
    )


@tool
def get_attractions(city: str) -> str:
    """Get popular attractions for a city."""
    attractions_data = {
        "munich": (
            "Marienplatz, English Garden, Nymphenburg Palace, "
            "BMW Museum, and Deutsches Museum."
        ),
        "berlin": (
            "Brandenburg Gate, Museum Island, Reichstag Building, "
            "East Side Gallery, and Berlin Cathedral."
        ),
        "vienna": (
            "Schönbrunn Palace, St. Stephen's Cathedral, "
            "Belvedere Palace, Hofburg Palace, and MuseumsQuartier."
        ),
    }

    normalized_city = city.split(",")[0].strip().lower()

    return attractions_data.get(
        normalized_city,
        f"Attraction data for {city} is not available.",
    )


@tool
def get_restaurants(city: str, cuisine: str) -> str:
    """Get restaurant recommendations for a city and cuisine."""
    restaurant_data = {
        ("munich", "italian"): (
            "Hippocampus, Vi Vadi, and Pizzesco."
        ),
        ("munich", "german"): (
            "Augustiner-Keller, Haxnbauer, and Wirtshaus in der Au."
        ),
        ("berlin", "italian"): (
            "Cecconi's Berlin, Lavanderia Vecchia, and Bocca di Bacco."
        ),
        ("berlin", "german"): (
            "Zur letzten Instanz, Max und Moritz, and Marjellchen."
        ),
        ("vienna", "italian"): (
            "Il Sestante, Trattoria Martinelli, and Regina Margherita."
        ),
        ("vienna", "austrian"): (
            "Figlmüller, Plachutta, and Gasthaus Pöschl."
        ),
    }

    normalized_city = city.split(",")[0].strip().lower()
    normalized_cuisine = cuisine.strip().lower()

    return restaurant_data.get(
        (normalized_city, normalized_cuisine),
        (
            f"Restaurant data for {cuisine} cuisine "
            f"in {city} is not available."
        ),
    )