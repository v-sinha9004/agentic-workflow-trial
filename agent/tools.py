"""
Tool registry and built-in educational tools for flight and travel planning.
Automatically converts Python function signatures and docstrings into OpenAI tool schemas.
"""

import ast
import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    schema: Dict[str, Any]
    requires_confirmation: bool = False

    def execute(self, **kwargs) -> str:
        """Executes the underlying function with provided arguments, handling errors gracefully."""
        try:
            result = self.func(**kwargs)
            if isinstance(result, (dict, list)):
                return json.dumps(result, indent=2)
            return str(result)
        except Exception as e:
            return f"Error executing tool '{self.name}': {type(e).__name__}: {str(e)}"


class ToolRegistry:
    """Registry that manages available tools and formats them for OpenAI."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(
        self,
        func: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        requires_confirmation: bool = False,
    ) -> Tool:
        # Check explicit arguments first, then function attributes attached by decorator, then reflection
        tool_name = name or getattr(func, "_tool_name", None) or func.__name__
        docstring = inspect.getdoc(func) or ""
        tool_description = (
            description
            or getattr(func, "_tool_description", None)
            or (docstring.split("\n\n")[0] if docstring else f"Function {tool_name}")
        )
        req_conf = requires_confirmation or getattr(func, "_requires_confirmation", False)

        schema = self._generate_openai_schema(func, tool_name, tool_description, docstring)
        tool_obj = Tool(
            name=tool_name,
            description=tool_description,
            func=func,
            schema=schema,
            requires_confirmation=req_conf,
        )
        self._tools[tool_name] = tool_obj
        return tool_obj

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Returns OpenAI-compatible tools schema list."""
        return [t.schema for t in self._tools.values()]

    def _generate_openai_schema(self, func: Callable, name: str, description: str, docstring: str) -> Dict[str, Any]:
        """Inspects function signatures and docstrings to build OpenAI function parameters JSON schema."""
        sig = inspect.signature(func)
        properties: Dict[str, Any] = {}
        required: List[str] = []

        # Parse param descriptions from docstring if present (e.g., ":param origin: City or airport code")
        param_docs: Dict[str, str] = {}
        for line in docstring.split("\n"):
            line = line.strip()
            param_match = re.match(r"^:param\s+([a-zA-Z0-9_]+):\s*(.+)$", line) or re.match(r"^-\s*([a-zA-Z0-9_]+):\s*(.+)$", line)
            if param_match:
                param_docs[param_match.group(1)] = param_match.group(2)

        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = "string"
            if param.annotation != inspect.Parameter.empty and param.annotation in type_mapping:
                param_type = type_mapping[param.annotation]

            param_info = {
                "type": param_type,
                "description": param_docs.get(param_name, f"Parameter '{param_name}'"),
            }
            properties[param_name] = param_info

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


# Global registry instance
registry = ToolRegistry()


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    target_registry: Optional[ToolRegistry] = None,
    requires_confirmation: bool = False,
):
    """Decorator to register a function as an agent tool."""
    def decorator(func: Callable):
        func._tool_name = name or func.__name__
        func._tool_description = description
        func._requires_confirmation = requires_confirmation
        reg = target_registry or registry
        reg.register(func, name=name, description=description, requires_confirmation=requires_confirmation)
        return func
    return decorator


# =====================================================================
# Built-in Educational Tools: Flight & Travel Planning Domain
# =====================================================================

_MOCK_FLIGHTS = [
    {
        "flight_number": "BA 178",
        "airline": "British Airways",
        "origin": "NYC",
        "destination": "LON",
        "departure": "08:15 AM",
        "arrival": "08:10 PM",
        "price": 450.00,
        "class": "Economy",
        "stops": 0,
    },
    {
        "flight_number": "VS 004",
        "airline": "Virgin Atlantic",
        "origin": "NYC",
        "destination": "LON",
        "departure": "07:30 PM",
        "arrival": "07:25 AM (+1)",
        "price": 495.00,
        "class": "Economy",
        "stops": 0,
    },
    {
        "flight_number": "DL 001",
        "airline": "Delta Air Lines",
        "origin": "NYC",
        "destination": "LON",
        "departure": "06:00 PM",
        "arrival": "06:30 AM (+1)",
        "price": 520.00,
        "class": "Economy",
        "stops": 0,
    },
    {
        "flight_number": "AF 023",
        "airline": "Air France",
        "origin": "NYC",
        "destination": "PAR",
        "departure": "04:30 PM",
        "arrival": "06:00 AM (+1)",
        "price": 510.00,
        "class": "Economy",
        "stops": 0,
    },
    {
        "flight_number": "UA 906",
        "airline": "United Airlines",
        "origin": "SFO",
        "destination": "NRT",
        "departure": "11:20 AM",
        "arrival": "03:15 PM (+1)",
        "price": 780.00,
        "class": "Economy",
        "stops": 0,
    },
]

_MOCK_BAGGAGE_POLICIES = {
    "british airways": {
        "airline": "British Airways",
        "carry_on": "1 personal item + 1 standard carry-on included (free)",
        "first_checked_bag_fee": 60.00,
        "second_checked_bag_fee": 90.00,
        "max_weight_kg": 23,
    },
    "virgin atlantic": {
        "airline": "Virgin Atlantic",
        "carry_on": "1 standard carry-on up to 10kg included (free)",
        "first_checked_bag_fee": 75.00,
        "second_checked_bag_fee": 100.00,
        "max_weight_kg": 23,
    },
    "delta air lines": {
        "airline": "Delta Air Lines",
        "carry_on": "1 carry-on bag + 1 personal item included (free)",
        "first_checked_bag_fee": 70.00,
        "second_checked_bag_fee": 100.00,
        "max_weight_kg": 23,
    },
    "air france": {
        "airline": "Air France",
        "carry_on": "1 hand baggage + 1 personal item (total 12kg) included",
        "first_checked_bag_fee": 65.00,
        "second_checked_bag_fee": 85.00,
        "max_weight_kg": 23,
    },
    "united airlines": {
        "airline": "United Airlines",
        "carry_on": "1 full-sized carry-on + 1 personal item included",
        "first_checked_bag_fee": 75.00,
        "second_checked_bag_fee": 105.00,
        "max_weight_kg": 23,
    },
}

_MOCK_WEATHER = {
    "london": {"temperature_c": 14, "condition": "Rainy and overcast", "recommendation": "Pack an umbrella and a waterproof jacket."},
    "paris": {"temperature_c": 18, "condition": "Partly cloudy with mild breeze", "recommendation": "Light layers and comfortable walking shoes."},
    "tokyo": {"temperature_c": 22, "condition": "Sunny and clear", "recommendation": "Sunglasses and breathable clothing."},
    "new york": {"temperature_c": 20, "condition": "Sunny with light wind", "recommendation": "Mild weather, casual spring attire."},
    "san francisco": {"temperature_c": 16, "condition": "Foggy morning clearing to sun", "recommendation": "Layering is essential due to shifting fog."},
}


@tool(name="search_flights", description="Search available flights between origin and destination cities or airport codes.")
def search_flights(origin: str, destination: str, date: str) -> str:
    """
    Search available flights for a given route and date.
    :param origin: Departure city or airport code (e.g. NYC, SFO, JFK)
    :param destination: Arrival city or airport code (e.g. LON, PAR, NRT, LHR)
    :param date: Departure date (e.g. 2026-09-25 or 'next Friday')
    """
    orig_clean = origin.strip().upper()
    dest_clean = destination.strip().upper()

    city_map = {
        "NEW YORK": "NYC", "JFK": "NYC", "EWR": "NYC", "LGA": "NYC",
        "LONDON": "LON", "LHR": "LON", "LGW": "LON",
        "PARIS": "PAR", "CDG": "PAR",
        "SAN FRANCISCO": "SFO",
        "TOKYO": "NRT", "HND": "NRT",
    }
    orig_code = city_map.get(orig_clean, orig_clean)
    dest_code = city_map.get(dest_clean, dest_clean)

    matched = [
        f for f in _MOCK_FLIGHTS
        if (f["origin"] == orig_code or orig_code in f["origin"]) and (f["destination"] == dest_code or dest_code in f["destination"])
    ]

    if not matched:
        return (
            f"No direct flights found matching {origin} -> {destination} on {date}. "
            f"Currently indexed routes include: NYC -> LON, NYC -> PAR, SFO -> NRT."
        )

    return json.dumps({
        "search_route": f"{origin} -> {destination}",
        "date": date,
        "flights_found": matched,
    }, indent=2)


@tool(name="get_baggage_policy", description="Look up baggage rules, carry-on limits, and checked bag fees for an airline.")
def get_baggage_policy(airline: str) -> str:
    """
    Look up baggage allowance and fees for an airline.
    :param airline: Name of the airline (e.g. British Airways, Delta, Virgin Atlantic)
    """
    clean_name = airline.strip().lower()

    for key, policy in _MOCK_BAGGAGE_POLICIES.items():
        if key in clean_name or clean_name in key:
            return json.dumps(policy, indent=2)

    return (
        f"Baggage policy for '{airline}' not found in database. "
        f"Available airlines: {', '.join(k.title() for k in _MOCK_BAGGAGE_POLICIES.keys())}."
    )


@tool(name="calculator", description="Safely evaluate a mathematical calculation expression to compute costs, fees, or totals.")
def calculator(expression: str) -> str:
    """
    Safely calculates a mathematical expression.
    :param expression: Math expression to compute (e.g. '(2 * 450) + (2 * 60)')
    """
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )
    try:
        clean_expr = expression.strip().replace("$", "").replace(",", "")
        tree = ast.parse(clean_expr, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return f"Error: Expression contains unsupported or unsafe operator '{type(node).__name__}'."
        result = eval(compile(tree, "<string>", "eval"), {"__builtins__": None}, {})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression '{expression}': {str(e)}"


@tool(name="get_city_weather", description="Get weather conditions and travel packing recommendations for a destination city.")
def get_city_weather(city: str, date: str = "") -> str:
    """
    Get weather forecast and packing advice for a city.
    :param city: Destination city name (e.g. London, Paris, Tokyo)
    :param date: Optional travel date
    """
    clean_city = city.strip().lower()
    for key, weather in _MOCK_WEATHER.items():
        if key in clean_city or clean_city in key:
            res = dict(weather)
            res["city"] = key.title()
            if date:
                res["date"] = date
            return json.dumps(res, indent=2)

    return f"Weather data currently unavailable for '{city}'. General recommendation: check local forecast before departure."


# =====================================================================
# In-Memory Booking Storage & Human-in-the-Loop Booking Tool
# =====================================================================

_MOCK_BOOKINGS: List[Dict[str, Any]] = []


@tool(
    name="book_flight",
    description="Book a confirmed flight ticket after user verification. Requires flight_number, passenger_name, and travel date.",
    requires_confirmation=True,
)
def book_flight(flight_number: str, passenger_name: str = "Passenger", date: str = "") -> str:
    """
    Book a confirmed flight ticket and generate a booking reservation reference.
    :param flight_number: Flight code to book (e.g. 'BA 178', 'VS 004', 'DL 001')
    :param passenger_name: Name of passenger or passengers
    :param date: Flight travel date (e.g. '2026-09-25')
    """
    clean_num = flight_number.strip().upper()
    flight = next((f for f in _MOCK_FLIGHTS if f["flight_number"].upper() == clean_num), None)

    ref_suffix = abs(hash(f"{clean_num}-{passenger_name}-{date}")) % 90000 + 10000
    booking_ref = f"BK-{clean_num.replace(' ', '')}-{ref_suffix}"

    booking_record = {
        "status": "CONFIRMED",
        "booking_reference": booking_ref,
        "flight_number": clean_num,
        "airline": flight["airline"] if flight else "Partner Airline",
        "route": f"{flight['origin']} -> {flight['destination']}" if flight else "Confirmed Route",
        "passenger_name": passenger_name,
        "date": date or "Confirmed Date",
        "price_paid": flight["price"] if flight else "Standard Fare",
        "message": f"Successfully booked flight {clean_num} for {passenger_name}. Booking reference: {booking_ref}.",
    }
    _MOCK_BOOKINGS.append(booking_record)
    return json.dumps(booking_record, indent=2)

